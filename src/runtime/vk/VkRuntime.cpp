#include <algorithm>
#include <atomic>
#include <mutex>

#include "vuk/Exception.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/runtime/Cache.hpp"
#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/runtime/vk/DeviceVkResource.hpp"
#include "vuk/runtime/vk/Program.hpp"
#include "vuk/runtime/vk/Query.hpp"
#include "vuk/runtime/vk/VkQueueExecutor.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"
#include "vuk/runtime/vk/VkSwapchain.hpp"

#include <plf_colony.h>
#include <robin_hood.h>

namespace {
	/* TODO: I am currently unaware of any use case that would make supporting static loading worthwhile
	void load_pfns_static(vuk::RuntimeCreateParameters::FunctionPointers& pfns) {
#define VUK_X(name)                                                                                                                                            \
	if (pfns.name == nullptr) {                                                                                                                                  \
	  pfns.name = (PFN_##name)name;                                                                                                                              \
	}
#define VUK_Y(name)                                                                                                                                            \
	if (pfns.name == nullptr) {                                                                                                                                  \
	  pfns.name = (PFN_##name)name;                                                                                                                              \
	}
#include "vuk/VulkanPFNOptional.hpp"
#include "vuk/VulkanPFNRequired.hpp"
#undef VUK_X
#undef VUK_Y
	}*/

	void load_pfns_dynamic(VkInstance instance, VkDevice device, vuk::FunctionPointers& pfns) {
		pfns.vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)pfns.vkGetInstanceProcAddr(instance, "vkGetDeviceProcAddr");
#define VUK_X(name)                                                                                                                                            \
	if (pfns.name == nullptr) {                                                                                                                                  \
		pfns.name = (PFN_##name)pfns.vkGetDeviceProcAddr(device, #name);                                                                                           \
	}
#define VUK_Y(name)                                                                                                                                            \
	if (pfns.name == nullptr) {                                                                                                                                  \
		pfns.name = (PFN_##name)pfns.vkGetInstanceProcAddr(instance, #name);                                                                                       \
	}
#include "vuk/runtime/vk/VkPFNOptional.hpp"
#include "vuk/runtime/vk/VkPFNRequired.hpp"
#undef VUK_X
#undef VUK_Y
	}
} // namespace

namespace vuk {
	bool FunctionPointers::check_pfns() {
		bool valid = true;
#define VUK_X(name) valid = valid && name;
#define VUK_Y(name) valid = valid && name;
#include "vuk/runtime/vk/VkPFNRequired.hpp"
#undef VUK_X
#undef VUK_Y
		return valid;
	}

	vuk::Result<void> FunctionPointers::load_pfns(VkInstance instance, VkDevice device, bool allow_dynamic_loading_of_vk_function_pointers) {
		// PFN loading
		// if the user passes in PFNs, those will be used, always
		if (check_pfns()) {
			return { vuk::expected_value };
		}
		// we don't have all the PFNs, so we will load them if this is allowed
		if (vkGetInstanceProcAddr && allow_dynamic_loading_of_vk_function_pointers) {
			load_pfns_dynamic(instance, device, *this);
			if (!check_pfns()) {
				return { vuk::expected_error,
					       vuk::RequiredPFNMissingException{ "A Vulkan PFN is required, but was not provided and dynamic loading could not load it." } };
			}
		} else {
			return { vuk::expected_error, vuk::RequiredPFNMissingException{ "A Vulkan PFN is required, but was not provided and dynamic loading was not allowed." } };
		}
		return { vuk::expected_value };
	}

	Result<std::vector<uint32_t>> compile_glsl(const create_info_t<ShaderModule>& cinfo, uint32_t shader_compiler_target_version);
	Result<std::vector<uint32_t>> compile_hlsl(const create_info_t<ShaderModule>& cinfo, uint32_t shader_compiler_target_version);
	Result<std::vector<uint32_t>> compile_slang(const create_info_t<ShaderModule>& cinfo, uint32_t shader_compiler_target_version);
	Result<std::vector<uint32_t>> compile_c(const create_info_t<ShaderModule>& cinfo, uint32_t shader_compiler_target_version);

	struct ContextImpl {
		template<class T>
		struct FN {
			static T create_fn(void* ctx, const create_info_t<T>& ci) {
				return reinterpret_cast<Runtime*>(ctx)->create(ci);
			}

			static void destroy_fn(void* ctx, const T& v) {
				return reinterpret_cast<Runtime*>(ctx)->destroy(v);
			}
		};

		VkDevice device;

		std::unique_ptr<DeviceVkResource> device_vk_resource;
		Allocator direct_allocator;

		std::vector<std::unique_ptr<Executor>> executors;

		Cache<PipelineBaseInfo> pipelinebase_cache;
		Cache<DescriptorPool> pool_cache;
		Cache<Sampler> sampler_cache;
		Cache<ShaderModule> shader_modules;
		Cache<DescriptorSetLayoutAllocInfo> descriptor_set_layouts;
		Cache<VkPipelineLayout> pipeline_layouts;

		std::mutex begin_frame_lock;

		std::atomic<size_t> frame_counter = 0;
		std::atomic<size_t> unique_handle_id_counter = 0;

		std::mutex named_pipelines_lock;
		robin_hood::unordered_flat_map<Name, PipelineBaseInfo*> named_pipelines;

		std::atomic<uint64_t> query_id_counter = 0;
		VkPhysicalDeviceProperties physical_device_properties;

		std::mutex swapchains_lock;
		plf::colony<Swapchain> swapchains;

		std::mutex query_lock;
		robin_hood::unordered_map<Query, uint64_t> timestamp_result_map;

		void collect(uint64_t absolute_frame) {
			// collect rarer resources
			static constexpr uint32_t cache_collection_frequency = 16;
			auto remainder = absolute_frame % cache_collection_frequency;
			switch (remainder) {
				/*case 3:
				  ptc.impl->sampler_cache.collect(cache_collection_frequency); break;*/ // sampler cache can't be collected due to persistent descriptor sets
			case 4:
				pipeline_layouts.collect(absolute_frame, cache_collection_frequency);
				break;
			/* case 5:
				pipelinebase_cache.collect(absolute_frame, cache_collection_frequency);
				break;*/ // can't be collected since we keep the pointer around in PipelineInfos
			case 6:
				pool_cache.collect(absolute_frame, cache_collection_frequency);
				break;
			}
		}

		ContextImpl(Runtime& ctx) :
		    device(ctx.device),
		    device_vk_resource(std::make_unique<DeviceVkResource>(ctx)),
		    direct_allocator(*device_vk_resource.get()),
		    pipelinebase_cache(&ctx, &FN<struct PipelineBaseInfo>::create_fn, &FN<struct PipelineBaseInfo>::destroy_fn),
		    pool_cache(&ctx, &FN<struct DescriptorPool>::create_fn, &FN<struct DescriptorPool>::destroy_fn),
		    sampler_cache(&ctx, &FN<Sampler>::create_fn, &FN<Sampler>::destroy_fn),
		    shader_modules(&ctx, &FN<struct ShaderModule>::create_fn, &FN<struct ShaderModule>::destroy_fn),
		    descriptor_set_layouts(&ctx, &FN<struct DescriptorSetLayoutAllocInfo>::create_fn, &FN<struct DescriptorSetLayoutAllocInfo>::destroy_fn),
		    pipeline_layouts(&ctx, &FN<VkPipelineLayout>::create_fn, &FN<VkPipelineLayout>::destroy_fn) {
			ctx.vkGetPhysicalDeviceProperties(ctx.physical_device, &physical_device_properties);
		}
	};

	Runtime::Runtime(RuntimeCreateParameters params) :
	    FunctionPointers(params.pointers),
	    instance(params.instance),
	    device(params.device),
	    physical_device(params.physical_device) {
		assert(check_pfns());

		impl = new ContextImpl(*this);
		impl->executors = std::move(params.executors);
		for (auto& exe : impl->executors) {
			if (exe->type == Executor::Type::eVulkanDeviceQueue) {
				all_queue_families.push_back(static_cast<QueueExecutor*>(exe.get())->get_queue_family_index());
			}
		}
		this->vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);
		min_buffer_alignment =
		    std::max(physical_device_properties.limits.minUniformBufferOffsetAlignment, physical_device_properties.limits.minStorageBufferOffsetAlignment);
		VkPhysicalDeviceProperties2 prop2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
		if (this->vkCmdBuildAccelerationStructuresKHR) {
			prop2.pNext = &rt_properties;
			rt_properties.pNext = &as_properties;
		}
		this->vkGetPhysicalDeviceProperties2(physical_device, &prop2);
		// if we haved push descriptors available, use it
		if (params.pointers.vkCmdPushDescriptorSetKHR) {
			default_descriptor_set_strategy = DescriptorSetStrategyFlagBits::ePushDescriptor;
		}
	}

	Executor* Runtime::get_executor(ExecutorTag tag) {
		auto it = std::find_if(impl->executors.begin(), impl->executors.end(), [=](auto& exe) { return exe->tag == tag; });
		if (it != impl->executors.end()) {
			return it->get();
		} else {
			return nullptr;
		}
	}

	Executor* Runtime::get_executor(DomainFlagBits domain) {
		auto it = std::find_if(impl->executors.begin(), impl->executors.end(), [=](auto& exe) { return exe->tag.domain == domain; });
		if (it != impl->executors.end()) {
			return it->get();
		} else {
			return nullptr;
		}
	}

	std::vector<Executor*> Runtime::get_executors() {
		std::vector<Executor*> executors;
		for (auto& exe : impl->executors) {
			executors.push_back(exe.get());
		}
		return executors;
	}

	bool Runtime::debug_enabled() const {
		return this->vkSetDebugUtilsObjectNameEXT != nullptr;
	}

	void Runtime::begin_region(const VkCommandBuffer& cb, Name name, std::array<float, 4> color) {
		if (!debug_enabled())
			return;
		VkDebugUtilsLabelEXT label = { .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
		label.pLabelName = name.c_str();
		::memcpy(label.color, color.data(), sizeof(float) * 4);
		this->vkCmdBeginDebugUtilsLabelEXT(cb, &label);
	}

	void Runtime::end_region(const VkCommandBuffer& cb) {
		if (!debug_enabled())
			return;
		this->vkCmdEndDebugUtilsLabelEXT(cb);
	}

	void PersistentDescriptorSet::update_combined_image_sampler(unsigned binding, unsigned array_index, ImageView iv, Sampler sampler, ImageLayout layout) {
		assert(binding < descriptor_bindings.size());
		assert(array_index < descriptor_bindings[binding].size());
		descriptor_bindings[binding][array_index].image = DescriptorImageInfo(sampler, iv, layout);
		descriptor_bindings[binding][array_index].type = (DescriptorType)((uint8_t)DescriptorType::eCombinedImageSampler | (uint8_t)DescriptorType::ePendingWrite);
	}

	void PersistentDescriptorSet::update_storage_image(unsigned binding, unsigned array_index, ImageView iv) {
		assert(binding < descriptor_bindings.size());
		assert(array_index < descriptor_bindings[binding].size());
		descriptor_bindings[binding][array_index].image = DescriptorImageInfo({}, iv, ImageLayout::eGeneral);
		descriptor_bindings[binding][array_index].type = (DescriptorType)((uint8_t)DescriptorType::eStorageImage | (uint8_t)DescriptorType::ePendingWrite);
	}

	void PersistentDescriptorSet::update_uniform_buffer(unsigned binding, unsigned array_index, Buffer buffer) {
		assert(binding < descriptor_bindings.size());
		assert(array_index < descriptor_bindings[binding].size());
		descriptor_bindings[binding][array_index].buffer = VkDescriptorBufferInfo{ buffer.buffer, buffer.offset, buffer.size };
		descriptor_bindings[binding][array_index].type = (DescriptorType)((uint8_t)DescriptorType::eUniformBuffer | (uint8_t)DescriptorType::ePendingWrite);
	}

	void PersistentDescriptorSet::update_storage_buffer(unsigned binding, unsigned array_index, Buffer buffer) {
		assert(binding < descriptor_bindings.size());
		assert(array_index < descriptor_bindings[binding].size());
		descriptor_bindings[binding][array_index].buffer = VkDescriptorBufferInfo{ buffer.buffer, buffer.offset, buffer.size };
		descriptor_bindings[binding][array_index].type = (DescriptorType)((uint8_t)DescriptorType::eStorageBuffer | (uint8_t)DescriptorType::ePendingWrite);
	}

	void PersistentDescriptorSet::update_sampler(unsigned binding, unsigned array_index, Sampler sampler) {
		assert(binding < descriptor_bindings.size());
		assert(array_index < descriptor_bindings[binding].size());
		descriptor_bindings[binding][array_index].image = DescriptorImageInfo(sampler, {}, {});
		descriptor_bindings[binding][array_index].type = (DescriptorType)((uint8_t)DescriptorType::eSampler | (uint8_t)DescriptorType::ePendingWrite);
	}

	void PersistentDescriptorSet::update_sampled_image(unsigned binding, unsigned array_index, ImageView iv, ImageLayout layout) {
		assert(binding < descriptor_bindings.size());
		assert(array_index < descriptor_bindings[binding].size());
		descriptor_bindings[binding][array_index].image = DescriptorImageInfo({}, iv, layout);
		descriptor_bindings[binding][array_index].type = (DescriptorType)((uint8_t)DescriptorType::eSampledImage | (uint8_t)DescriptorType::ePendingWrite);
	}

	void PersistentDescriptorSet::update_acceleration_structure(unsigned binding, unsigned array_index, VkAccelerationStructureKHR as) {
		assert(binding < descriptor_bindings.size());
		assert(array_index < descriptor_bindings[binding].size());
		descriptor_bindings[binding][array_index].as.as = as;
		descriptor_bindings[binding][array_index].type =
		    (DescriptorType)((uint8_t)DescriptorType::eAccelerationStructureKHR | (uint8_t)DescriptorType::ePendingWrite);
	}

	void PersistentDescriptorSet::commit(Runtime& ctx) {
		wdss.clear();
		for (unsigned i = 0; i < descriptor_bindings.size(); i++) {
			auto& db = descriptor_bindings[i];
			for (unsigned j = 0; j < db.size(); j++) {
				if ((uint8_t)db[j].type & (uint8_t)DescriptorType::ePendingWrite) { // clear pending write
					db[j].type = (DescriptorType)((uint8_t)db[j].type & ~(uint8_t)DescriptorType::ePendingWrite);
					if (db[j].type == DescriptorType::eAccelerationStructureKHR) {
						db[j].as.wds = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
						db[j].as.wds.accelerationStructureCount = 1;
						db[j].as.wds.pAccelerationStructures = &db[j].as.as;
					}
					wdss.push_back(VkWriteDescriptorSet{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					                                     .pNext = db[j].type == DescriptorType::eAccelerationStructureKHR ? &db[j].as.wds : nullptr,
					                                     .dstSet = backing_set,
					                                     .dstBinding = i,
					                                     .dstArrayElement = j,
					                                     .descriptorCount = 1,
					                                     .descriptorType = DescriptorBinding::vk_descriptor_type(db[j].type),
					                                     .pImageInfo = &db[j].image.dii,
					                                     .pBufferInfo = &db[j].buffer });
				}
			}
		}
		ctx.vkUpdateDescriptorSets(ctx.device, (uint32_t)wdss.size(), wdss.data(), 0, nullptr);
	}

	ShaderModule Runtime::create(const create_info_t<ShaderModule>& cinfo) {
		std::vector<uint32_t> spirv;
		const uint32_t* spirv_ptr = nullptr;
		size_t size = 0;

		switch (cinfo.source.language) {
		case ShaderSourceLanguage::eGlsl: {
#if VUK_USE_SHADERC
			spirv = *compile_glsl(cinfo, shader_compiler_target_version);
			spirv_ptr = spirv.data();
			size = spirv.size();
#endif
			break;
		}
		case ShaderSourceLanguage::eHlsl: {
#if VUK_USE_DXC
			spirv = *compile_hlsl(cinfo, shader_compiler_target_version);
			spirv_ptr = spirv.data();
			size = spirv.size();
#endif
			break;
		}
		case ShaderSourceLanguage::eC: {
#if VUK_USE_VCC
			spirv = *compile_c(cinfo, shader_compiler_target_version);
			spirv_ptr = spirv.data();
			size = spirv.size();
#endif
			break;
		}
		case ShaderSourceLanguage::eSlang: {
#if VUK_USE_SLANG
			spirv = *compile_slang(cinfo, shader_compiler_target_version);
			spirv_ptr = spirv.data();
			size = spirv.size();
#endif
			break;
		}
		case ShaderSourceLanguage::eSpirv: {
			spirv_ptr = cinfo.source.data_ptr;
			size = cinfo.source.size;
			break;
		}
		default:
			assert(0);
		}

		Program p;
		auto stage = p.introspect(spirv_ptr, size);

		VkShaderModuleCreateInfo moduleCreateInfo{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
		moduleCreateInfo.codeSize = size * sizeof(uint32_t);
		moduleCreateInfo.pCode = spirv_ptr;
		VkShaderModule sm;
		this->vkCreateShaderModule(device, &moduleCreateInfo, nullptr, &sm);
		std::string name = "ShaderModule: " + cinfo.filename;
		set_name(sm, Name(name));
		// WORKAROUND: slang doesn't seem to preserve entry point name
		bool override_entry_point_name = cinfo.source.language == ShaderSourceLanguage::eSlang;
		return { sm, std::move(p), stage, override_entry_point_name };
	}

	PipelineBaseInfo Runtime::create(const create_info_t<PipelineBaseInfo>& cinfo) {
		std::vector<VkPipelineShaderStageCreateInfo> psscis;
		std::vector<std::string> entry_point_names;

		// accumulate descriptors from all stages
		Program accumulated_reflection;
		std::string pipe_name = "Pipeline:";
		for (auto i = 0; i < cinfo.shaders.size(); i++) {
			auto& source = cinfo.shaders[i];
			if (source.data_ptr == nullptr) {
				continue;
			}
			auto& sm = impl->shader_modules.acquire({ source, cinfo.shader_paths[i], cinfo.defines, cinfo.compile_options });
			VkPipelineShaderStageCreateInfo shader_stage{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
			shader_stage.pSpecializationInfo = nullptr;
			shader_stage.stage = sm.stage;
			shader_stage.module = sm.shader_module;
			if (sm.override_entry_point_name_to_main) {
				entry_point_names.push_back("main");
			} else {
				entry_point_names.push_back(source.entry_point);
			}
			psscis.push_back(shader_stage);
			accumulated_reflection.append(sm.reflection_info);
			pipe_name += cinfo.shader_paths[i] + "+";
		}
		pipe_name = pipe_name.substr(0, pipe_name.size() - 1); // trim off last "+"

		// acquire descriptor set layouts (1 per set)
		// acquire pipeline layout
		PipelineLayoutCreateInfo plci;
		plci.dslcis = PipelineBaseCreateInfo::build_descriptor_layouts(accumulated_reflection, cinfo);
		// use explicit descriptor layouts if there are any
		for (auto& l : cinfo.explicit_set_layouts) {
			plci.dslcis[l.index] = l;
		}
		plci.pcrs.insert(plci.pcrs.begin(), accumulated_reflection.push_constant_ranges.begin(), accumulated_reflection.push_constant_ranges.end());
		plci.plci.pushConstantRangeCount = (uint32_t)accumulated_reflection.push_constant_ranges.size();
		plci.plci.pPushConstantRanges = accumulated_reflection.push_constant_ranges.data();
		std::array<DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> dslai = {};
		std::vector<VkDescriptorSetLayout> dsls;
		bool use_pd = default_descriptor_set_strategy == DescriptorSetStrategyFlagBits::ePushDescriptor;
		size_t index = 0;
		for (auto& dsl : plci.dslcis) {
			dsl.dslci.bindingCount = (uint32_t)dsl.bindings.size();
			dsl.dslci.pBindings = dsl.bindings.data();
			VkDescriptorSetLayoutBindingFlagsCreateInfo dslbfci{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
			if (dsl.flags.size() > 0) {
				dslbfci.bindingCount = (uint32_t)dsl.bindings.size();
				dslbfci.pBindingFlags = dsl.flags.data();
				dsl.dslci.pNext = &dslbfci;
			} else if (use_pd && (index == plci.dslcis.size() - 1)) {
				dsl.dslci.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
			}
			auto descset_layout_alloc_info = impl->descriptor_set_layouts.acquire(dsl);
			dslai[dsl.index] = descset_layout_alloc_info;
			dsls.push_back(dslai[dsl.index].layout);
			if (dsl.bindings.size() == 0) {
				dslai[dsl.index].layout = VK_NULL_HANDLE;
			}
			dslai[dsl.index].push = dsl.dslci.flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
			index++;
		}
		plci.plci.pSetLayouts = dsls.data();
		plci.plci.setLayoutCount = (uint32_t)dsls.size();

		PipelineBaseInfo pbi;
		pbi.psscis = std::move(psscis);
		pbi.entry_point_names = std::move(entry_point_names);
		pbi.layout_info = dslai;
		pbi.pipeline_layout = impl->pipeline_layouts.acquire(plci);
		pbi.dslcis = std::move(plci.dslcis);
		for (auto& dslci : pbi.dslcis) {
			std::sort(dslci.bindings.begin(), dslci.bindings.end(), [](auto& a, auto& b) { return a.binding < b.binding; });
		}
		pbi.pipeline_name = Name(pipe_name);
		pbi.reflection_info = std::move(accumulated_reflection);
		pbi.binding_flags = cinfo.binding_flags;
		pbi.variable_count_max = cinfo.variable_count_max;
		pbi.hit_groups = cinfo.hit_groups;
		pbi.max_ray_recursion_depth = cinfo.max_ray_recursion_depth;
		return pbi;
	}

	bool Runtime::load_pipeline_cache(std::span<std::byte> data) {
		VkPipelineCacheCreateInfo pcci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = data.size_bytes(), .pInitialData = data.data() };
		this->vkDestroyPipelineCache(device, vk_pipeline_cache, nullptr);
		this->vkCreatePipelineCache(device, &pcci, nullptr, &vk_pipeline_cache);
		return true;
	}

	std::vector<std::byte> Runtime::save_pipeline_cache() {
		size_t size;
		std::vector<std::byte> data;
		this->vkGetPipelineCacheData(device, vk_pipeline_cache, &size, nullptr);
		data.resize(size);
		this->vkGetPipelineCacheData(device, vk_pipeline_cache, &size, data.data());
		return data;
	}

	Query Runtime::create_timestamp_query() {
		return { impl->query_id_counter++ };
	}

	DeviceVkResource& Runtime::get_vk_resource() {
		return *impl->device_vk_resource;
	}

	DescriptorSetLayoutAllocInfo Runtime::create(const create_info_t<DescriptorSetLayoutAllocInfo>& cinfo) {
		DescriptorSetLayoutAllocInfo ret;
		auto cinfo_mod = cinfo;
		for (auto& b : cinfo_mod.bindings) {
			b.descriptorType = DescriptorBinding::vk_descriptor_type((vuk::DescriptorType)b.descriptorType);
		}
		cinfo_mod.dslci.pBindings = cinfo_mod.bindings.data();
		this->vkCreateDescriptorSetLayout(device, &cinfo_mod.dslci, nullptr, &ret.layout);
		for (size_t i = 0; i < cinfo_mod.bindings.size(); i++) {
			auto& b = cinfo_mod.bindings[i];
			// if this is not a variable count binding, add it to the descriptor count
			if (cinfo_mod.flags.size() <= i || !(cinfo_mod.flags[i] & to_integral(DescriptorBindingFlagBits::eVariableDescriptorCount))) {
				auto index = b.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR ? 11 : to_integral(b.descriptorType);
				ret.descriptor_counts[index] += b.descriptorCount;
			} else { // a variable count binding
				ret.variable_count_binding = (uint32_t)i;
				ret.variable_count_binding_type = DescriptorType(b.descriptorType);
				ret.variable_count_binding_max_size = b.descriptorCount;
			}
		}
		return ret;
	}

	VkPipelineLayout Runtime::create(const create_info_t<VkPipelineLayout>& cinfo) {
		VkPipelineLayout pl;
		this->vkCreatePipelineLayout(device, &cinfo.plci, nullptr, &pl);
		return pl;
	}

	uint64_t Runtime::get_frame_count() const {
		return impl->frame_counter;
	}

	void Runtime::create_named_pipeline(Name name, PipelineBaseCreateInfo ci) {
		auto pbi = &impl->pipelinebase_cache.acquire(std::move(ci));
		std::lock_guard _(impl->named_pipelines_lock);
		impl->named_pipelines.insert_or_assign(name, pbi);
	}

	PipelineBaseInfo* Runtime::get_named_pipeline(Name name) {
		std::lock_guard _(impl->named_pipelines_lock);
		return impl->named_pipelines.at(name);
	}

	bool Runtime::is_pipeline_available(Name name) const {
		std::lock_guard _(impl->named_pipelines_lock);
		return impl->named_pipelines.contains(name);
	}

	PipelineBaseInfo* Runtime::get_pipeline(const PipelineBaseCreateInfo& pbci) {
		return &impl->pipelinebase_cache.acquire(pbci);
	}

	Program Runtime::get_pipeline_reflection_info(const PipelineBaseCreateInfo& pci) {
		auto& res = impl->pipelinebase_cache.acquire(pci);
		return res.reflection_info;
	}

	ShaderModule Runtime::compile_shader(ShaderSource source, std::string path) {
		ShaderModuleCreateInfo sci;
		sci.filename = std::move(path);
		sci.source = std::move(source);
		auto sm = impl->shader_modules.remove(sci);
		if (sm) {
			this->vkDestroyShaderModule(device, sm->shader_module, nullptr);
		}
		return impl->shader_modules.acquire(sci);
	}

	void Runtime::set_shader_target_version(const uint32_t target_version) {
		assert((target_version >= VK_API_VERSION_1_0 && target_version < VK_MAKE_API_VERSION(0, 1, 5, 0)) && "Invalid target version was passed.");
		shader_compiler_target_version = target_version;
	}

	void Runtime::destroy(const DescriptorPool& dp) {
		dp.destroy(*this, device);
	}

	void Runtime::destroy(const ShaderModule& sm) {
		this->vkDestroyShaderModule(device, sm.shader_module, nullptr);
	}

	void Runtime::destroy(const DescriptorSetLayoutAllocInfo& ds) {
		this->vkDestroyDescriptorSetLayout(device, ds.layout, nullptr);
	}

	void Runtime::destroy(const VkPipelineLayout& pl) {
		this->vkDestroyPipelineLayout(device, pl, nullptr);
	}

	void Runtime::destroy(const DescriptorSet&) {
		// no-op, we destroy the pools
	}

	void Runtime::destroy(const Sampler& sa) {
		this->vkDestroySampler(device, sa.payload, nullptr);
	}

	void Runtime::destroy(const PipelineBaseInfo& pbi) {
		// no-op, we don't own device objects
	}

	Runtime::~Runtime() {
		if (impl) {
			this->vkDeviceWaitIdle(device);

			this->vkDestroyPipelineCache(device, vk_pipeline_cache, nullptr);

			delete impl;
		}
	}

	uint64_t Runtime::get_unique_handle_id() {
		return impl->unique_handle_id_counter++;
	}

	void Runtime::next_frame() {
		impl->frame_counter++;
		collect(impl->frame_counter);
	}

	Result<void> Runtime::wait_idle() {
		std::unique_lock<std::recursive_mutex> graphics_lock;
		for (auto& exe : impl->executors) {
			exe->lock();
		}
		VkResult result = this->vkDeviceWaitIdle(device);

		for (auto& exe : impl->executors) {
			exe->unlock();
		}
		if (result < 0) {
			return { expected_error, VkException{ result } };
		}
		return { expected_value };
	}

	void Runtime::collect(uint64_t frame) {
		impl->collect(frame);
	}

	Unique<PersistentDescriptorSet>
	Runtime::create_persistent_descriptorset(Allocator& allocator, DescriptorSetLayoutCreateInfo dslci, unsigned num_descriptors) {
		dslci.dslci.bindingCount = (uint32_t)dslci.bindings.size();
		dslci.dslci.pBindings = dslci.bindings.data();
		VkDescriptorSetLayoutBindingFlagsCreateInfo dslbfci{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
		if (dslci.flags.size() > 0) {
			dslbfci.bindingCount = (uint32_t)dslci.bindings.size();
			dslbfci.pBindingFlags = dslci.flags.data();
			dslci.dslci.pNext = &dslbfci;
		}
		auto& dslai = impl->descriptor_set_layouts.acquire(dslci, impl->frame_counter);
		return create_persistent_descriptorset(allocator, { dslai, dslci, num_descriptors });
	}

	Unique<PersistentDescriptorSet> Runtime::create_persistent_descriptorset(Allocator& allocator, const PersistentDescriptorSetCreateInfo& ci) {
		Unique<PersistentDescriptorSet> pds(allocator);
		allocator.allocate_persistent_descriptor_sets(std::span{ &*pds, 1 }, std::span{ &ci, 1 });
		return pds;
	}

	Unique<PersistentDescriptorSet>
	Runtime::create_persistent_descriptorset(Allocator& allocator, const PipelineBaseInfo& base, unsigned set, unsigned num_descriptors) {
		return create_persistent_descriptorset(allocator, { base.layout_info[set], base.dslcis[set], num_descriptors });
	}

	Sampler Runtime::create(const create_info_t<Sampler>& cinfo) {
		VkSampler s;
		this->vkCreateSampler(device, (VkSamplerCreateInfo*)&cinfo, nullptr, &s);
		return wrap(s);
	}

	DescriptorPool Runtime::create(const create_info_t<DescriptorPool>& cinfo) {
		return DescriptorPool{};
	}

	Sampler Runtime::acquire_sampler(const SamplerCreateInfo& sci, uint64_t absolute_frame) {
		return impl->sampler_cache.acquire(sci, absolute_frame);
	}

	DescriptorPool& Runtime::acquire_descriptor_pool(const DescriptorSetLayoutAllocInfo& dslai, uint64_t absolute_frame) {
		return impl->pool_cache.acquire(dslai, absolute_frame);
	}

	bool Runtime::is_timestamp_available(Query q) {
		std::scoped_lock _(impl->query_lock);
		auto it = impl->timestamp_result_map.find(q);
		return (it != impl->timestamp_result_map.end());
	}

	std::optional<uint64_t> Runtime::retrieve_timestamp(Query q) {
		std::scoped_lock _(impl->query_lock);
		auto it = impl->timestamp_result_map.find(q);
		if (it != impl->timestamp_result_map.end()) {
			uint64_t res = it->second;
			impl->timestamp_result_map.erase(it);
			return res;
		}
		return {};
	}

	std::optional<double> Runtime::retrieve_duration(Query q1, Query q2) {
		if (!is_timestamp_available(q1)) {
			return {};
		}
		auto r1 = retrieve_timestamp(q1);
		auto r2 = retrieve_timestamp(q2);
		if (!r2) {
			return {};
		}

		auto ns = impl->physical_device_properties.limits.timestampPeriod * (r2.value() - r1.value());
		return ns * 1e-9;
	}

	Result<void> Runtime::make_timestamp_results_available(std::span<const TimestampQueryPool> pools) {
		std::scoped_lock _(impl->query_lock);
		std::array<uint64_t, TimestampQueryPool::num_queries> host_values;

		for (auto& pool : pools) {
			if (pool.count == 0) {
				continue;
			}
			auto result = this->vkGetQueryPoolResults(device,
			                                          pool.pool,
			                                          0,
			                                          pool.count,
			                                          sizeof(uint64_t) * pool.count,
			                                          host_values.data(),
			                                          sizeof(uint64_t),
			                                          VkQueryResultFlagBits::VK_QUERY_RESULT_64_BIT | VkQueryResultFlagBits::VK_QUERY_RESULT_WAIT_BIT);
			if (result != VK_SUCCESS) {
				return { expected_error, AllocateException{ result } };
			}

			for (uint64_t i = 0; i < pool.count; i++) {
				impl->timestamp_result_map.emplace(pool.queries[i], host_values[i]);
			}
		}

		return { expected_value };
	}

	Result<void> Runtime::wait_for_domains(std::span<SyncPoint> queue_waits) {
		std::array<uint32_t, 3> domain_to_sema_index = { ~0u, ~0u, ~0u };
		std::array<VkSemaphore, 3> queue_timeline_semaphores;
		std::array<uint64_t, 3> values = {};

		uint32_t count = 0;
		for (auto& [executor, v] : queue_waits) {
			assert(executor->type == Executor::Type::eVulkanDeviceQueue);
			auto vkq = static_cast<QueueExecutor*>(executor);
			auto idx = vkq->get_queue_family_index();
			auto& mapping = domain_to_sema_index[idx];
			if (mapping == -1) {
				mapping = count++;
			}
			queue_timeline_semaphores[mapping] = vkq->get_semaphore();
			values[mapping] = values[mapping] > v ? values[mapping] : v;
		}

		VkSemaphoreWaitInfo swi{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
		swi.pSemaphores = queue_timeline_semaphores.data();
		swi.pValues = values.data();
		swi.semaphoreCount = count;
		VkResult result = this->vkWaitSemaphores(device, &swi, UINT64_MAX);
		if (result != VK_SUCCESS) {
			return { expected_error, VkException{ result } };
		}
		return { expected_value };
	}

	Result<bool> Runtime::sync_point_ready(SyncPoint sp) {
		auto& [executor, v] = sp;
		assert(executor->type == Executor::Type::eVulkanDeviceQueue);
		auto vkq = static_cast<QueueExecutor*>(executor);
		auto idx = vkq->get_queue_family_index();
		auto val = vkq->get_sync_value();
		if (!val) {
			return val;
		}
		return { expected_value, *val >= v };
	}

	Swapchain::Swapchain(Allocator alloc, size_t image_count) : allocator(alloc) {
		semaphores.resize(image_count);
		allocator.allocate_semaphores(std::span(semaphores));
	}

	Swapchain::Swapchain(Allocator alloc,
	                     size_t image_count,
	                     VkSwapchainKHR swapchain,
	                     VkSurfaceKHR surface,
	                     VkExtent2D extent,
	                     VkFormat format,
	                     std::vector<VkImage> images,
	                     std::vector<VkImageView> views) :
	    allocator(alloc),
	    swapchain(swapchain),
	    surface(surface) {
		semaphores.resize(image_count);
		allocator.allocate_semaphores(std::span(semaphores));
		for (auto i = 0; i < images.size(); i++) {
			ImageAttachment ia;
			ia.extent = { extent.width, extent.height, 1 };
			ia.format = (Format)format;
			ia.image = Image{ images[i], nullptr };
			ia.image_view = ImageView{ { 0 }, views[i] };
			ia.view_type = ImageViewType::e2D;
			ia.sample_count = Samples::e1;
			ia.base_level = ia.base_layer = 0;
			ia.level_count = ia.layer_count = 1;
			this->images.push_back(ia);
		}
	}

	Swapchain::~Swapchain() {
		if (swapchain != VK_NULL_HANDLE) {
			allocator.deallocate(std::span{ &swapchain, 1 });
		}
		for (auto& i : images) {
			allocator.deallocate(std::span{ &i.image_view, 1 });
		}
		allocator.deallocate(std::span(semaphores));
	}

	Swapchain::Swapchain(Swapchain&& o) noexcept :
	    allocator(o.allocator),
	    swapchain(std::exchange(o.swapchain, VK_NULL_HANDLE)),
	    semaphores(std::move(o.semaphores)) {
		images = std::move(o.images);
		surface = o.surface;
		image_index = o.image_index;
		acquire_result = o.acquire_result;
	}

	Swapchain& Swapchain::operator=(Swapchain&& o) noexcept {
		swapchain = std::exchange(o.swapchain, VK_NULL_HANDLE);
		semaphores = std::move(o.semaphores);
		allocator = o.allocator;
		images = std::move(o.images);
		surface = o.surface;
		image_index = o.image_index;
		acquire_result = o.acquire_result;

		return *this;
	}
} // namespace vuk
