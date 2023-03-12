#if VUK_USE_SHADERC
#include "../src/ShadercIncluder.hpp"
#include <shaderc/shaderc.hpp>
#endif
#if VUK_USE_DXC
#ifdef _WIN32
// dxcapi.h expects the COM API to be present on Windows.
// On other platforms, the Vulkan SDK will have WinAdapter.h alongside dxcapi.h that is automatically included to stand in for the COM API.
#include <atlbase.h>
#endif
#include <dxc/dxcapi.h>
#define DXC_HR(hr, msg)                                                                                                                                        \
	if (FAILED(hr)) {                                                                                                                                            \
		throw ShaderCompilationException{ msg };                                                                                                                   \
	}
#endif
#include <algorithm>
#include <atomic>

#include "../src/ContextImpl.hpp"
#include "vuk/Allocator.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Context.hpp"
#include "vuk/Exception.hpp"
#include "vuk/Program.hpp"
#include "vuk/Query.hpp"
#include "vuk/RenderGraph.hpp"

namespace {
	/* TODO: I am currently unaware of any use case that would make supporting static loading worthwhile
	void load_pfns_static(vuk::ContextCreateParameters::FunctionPointers& pfns) {
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

	void load_pfns_dynamic(VkInstance instance, VkDevice device, vuk::ContextCreateParameters::FunctionPointers& pfns) {
#define VUK_X(name)                                                                                                                                            \
	if (pfns.name == nullptr) {                                                                                                                                  \
		pfns.name = (PFN_##name)pfns.vkGetDeviceProcAddr(device, #name);                                                                                           \
	}
#define VUK_Y(name)                                                                                                                                            \
	if (pfns.name == nullptr) {                                                                                                                                  \
		pfns.name = (PFN_##name)pfns.vkGetInstanceProcAddr(instance, #name);                                                                                       \
	}
#include "vuk/VulkanPFNOptional.hpp"
#include "vuk/VulkanPFNRequired.hpp"
#undef VUK_X
#undef VUK_Y
	}

	bool check_pfns(vuk::ContextCreateParameters::FunctionPointers& pfns) {
		bool valid = true;
#define VUK_X(name) valid = valid && pfns.name;
#define VUK_Y(name) valid = valid && pfns.name;
#include "vuk/VulkanPFNRequired.hpp"
#undef VUK_X
#undef VUK_Y
		return valid;
	}

	bool load_pfns(vuk::ContextCreateParameters params, vuk::ContextCreateParameters::FunctionPointers& pfns) {
		// PFN loading
		// if the user passes in PFNs, those will be used, always
		if (check_pfns(pfns)) {
			return true;
		}
		// we don't have all the PFNs, so we will load them if this is allowed
		if (pfns.vkGetInstanceProcAddr && pfns.vkGetDeviceProcAddr && params.allow_dynamic_loading_of_vk_function_pointers) {
			load_pfns_dynamic(params.instance, params.device, pfns);
			return check_pfns(pfns);
		} else {
			return false;
		}
	}
} // namespace

namespace vuk {
	Context::Context(ContextCreateParameters params) :
	    ContextCreateParameters::FunctionPointers(params.pointers),
	    instance(params.instance),
	    device(params.device),
	    physical_device(params.physical_device),
	    graphics_queue_family_index(params.graphics_queue_family_index),
	    compute_queue_family_index(params.compute_queue_family_index),
	    transfer_queue_family_index(params.transfer_queue_family_index) {

		// TODO: conversion to static factory fn
		bool pfn_load_success = load_pfns(params, *this);
		assert(pfn_load_success);

		auto queueSubmit2KHR = (PFN_vkQueueSubmit2KHR)vkGetDeviceProcAddr(device, "vkQueueSubmit2KHR");
		assert(queueSubmit2KHR != nullptr);

		[[maybe_unused]] bool dedicated_graphics_queue_ = false;
		bool dedicated_compute_queue_ = false;
		bool dedicated_transfer_queue_ = false;

		if (params.graphics_queue != VK_NULL_HANDLE && params.graphics_queue_family_index != VK_QUEUE_FAMILY_IGNORED) {
			dedicated_graphics_queue_ = true;
		}

		if (params.compute_queue != VK_NULL_HANDLE && params.compute_queue_family_index != VK_QUEUE_FAMILY_IGNORED) {
			dedicated_compute_queue_ = true;
		} else {
			compute_queue_family_index = params.graphics_queue_family_index;
		}

		if (params.transfer_queue != VK_NULL_HANDLE && params.transfer_queue_family_index != VK_QUEUE_FAMILY_IGNORED) {
			dedicated_transfer_queue_ = true;
		} else {
			transfer_queue_family_index = compute_queue ? params.compute_queue_family_index : params.graphics_queue_family_index;
		}
		impl = new ContextImpl(*this);

		{
			TimelineSemaphore ts;
			impl->device_vk_resource->allocate_timeline_semaphores(std::span{ &ts, 1 }, {});
			dedicated_graphics_queue.emplace(this->vkQueueSubmit, queueSubmit2KHR, params.graphics_queue, params.graphics_queue_family_index, ts);
			graphics_queue = &dedicated_graphics_queue.value();
		}
		if (dedicated_compute_queue_) {
			TimelineSemaphore ts;
			impl->device_vk_resource->allocate_timeline_semaphores(std::span{ &ts, 1 }, {});
			dedicated_compute_queue.emplace(this->vkQueueSubmit, queueSubmit2KHR, params.compute_queue, params.compute_queue_family_index, ts);
			compute_queue = &dedicated_compute_queue.value();
		} else {
			compute_queue = graphics_queue;
		}
		if (dedicated_transfer_queue_) {
			TimelineSemaphore ts;
			impl->device_vk_resource->allocate_timeline_semaphores(std::span{ &ts, 1 }, {});
			dedicated_transfer_queue.emplace(this->vkQueueSubmit, queueSubmit2KHR, params.transfer_queue, params.transfer_queue_family_index, ts);
			transfer_queue = &dedicated_transfer_queue.value();
		} else {
			transfer_queue = compute_queue ? compute_queue : graphics_queue;
		}

		this->vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);
		min_buffer_alignment =
		    std::max(physical_device_properties.limits.minUniformBufferOffsetAlignment, physical_device_properties.limits.minStorageBufferOffsetAlignment);
		VkPhysicalDeviceProperties2 prop2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
		prop2.pNext = &rt_properties;
		this->vkGetPhysicalDeviceProperties2(physical_device, &prop2);
	}

	Context::Context(Context&& o) noexcept : impl(std::exchange(o.impl, nullptr)) {
		instance = o.instance;
		device = o.device;
		physical_device = o.physical_device;
		graphics_queue_family_index = o.graphics_queue_family_index;
		compute_queue_family_index = o.compute_queue_family_index;
		transfer_queue_family_index = o.transfer_queue_family_index;
		dedicated_graphics_queue = std::move(o.dedicated_graphics_queue);
		graphics_queue = &dedicated_graphics_queue.value();
		dedicated_compute_queue = std::move(o.dedicated_compute_queue);
		if (dedicated_compute_queue) {
			compute_queue = &o.dedicated_compute_queue.value();
		} else {
			compute_queue = graphics_queue;
		}
		dedicated_transfer_queue = std::move(o.dedicated_transfer_queue);
		if (dedicated_transfer_queue) {
			transfer_queue = &dedicated_transfer_queue.value();
		} else {
			transfer_queue = compute_queue ? compute_queue : graphics_queue;
		}
		rt_properties = o.rt_properties;

		impl->pipelinebase_cache.allocator = this;
		impl->pipeline_cache.allocator = this;
		impl->compute_pipeline_cache.allocator = this;
		impl->renderpass_cache.allocator = this;
		impl->transient_images.allocator = this;
		impl->pool_cache.allocator = this;
		impl->sampler_cache.allocator = this;
		impl->shader_modules.allocator = this;
		impl->descriptor_set_layouts.allocator = this;
		impl->pipeline_layouts.allocator = this;
		impl->device_vk_resource->ctx = this;
	}

	Context& Context::operator=(Context&& o) noexcept {
		impl = std::exchange(o.impl, nullptr);
		instance = o.instance;
		device = o.device;
		physical_device = o.physical_device;
		graphics_queue_family_index = o.graphics_queue_family_index;
		compute_queue_family_index = o.compute_queue_family_index;
		transfer_queue_family_index = o.transfer_queue_family_index;
		dedicated_graphics_queue = std::move(o.dedicated_graphics_queue);
		graphics_queue = &dedicated_graphics_queue.value();
		dedicated_compute_queue = std::move(o.dedicated_compute_queue);
		if (dedicated_compute_queue) {
			compute_queue = &o.dedicated_compute_queue.value();
		} else {
			compute_queue = graphics_queue;
		}
		dedicated_transfer_queue = std::move(o.dedicated_transfer_queue);
		if (dedicated_transfer_queue) {
			transfer_queue = &dedicated_transfer_queue.value();
		} else {
			transfer_queue = compute_queue ? compute_queue : graphics_queue;
		}

		impl->pipelinebase_cache.allocator = this;
		impl->pipeline_cache.allocator = this;
		impl->compute_pipeline_cache.allocator = this;
		impl->renderpass_cache.allocator = this;
		impl->transient_images.allocator = this;
		impl->pool_cache.allocator = this;
		impl->sampler_cache.allocator = this;
		impl->shader_modules.allocator = this;
		impl->descriptor_set_layouts.allocator = this;
		impl->pipeline_layouts.allocator = this;
		impl->device_vk_resource->ctx = this;

		return *this;
	}

	bool Context::debug_enabled() const {
		return this->vkSetDebugUtilsObjectNameEXT != nullptr;
	}

	void Context::set_name(const Texture& tex, Name name) {
		if (!debug_enabled())
			return;
		set_name(tex.image->image, name);
		set_name(tex.view->payload, name);
	}

	void Context::begin_region(const VkCommandBuffer& cb, Name name, std::array<float, 4> color) {
		if (!debug_enabled())
			return;
		VkDebugUtilsLabelEXT label = { .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
		label.pLabelName = name.c_str();
		::memcpy(label.color, color.data(), sizeof(float) * 4);
		this->vkCmdBeginDebugUtilsLabelEXT(cb, &label);
	}

	void Context::end_region(const VkCommandBuffer& cb) {
		if (!debug_enabled())
			return;
		this->vkCmdEndDebugUtilsLabelEXT(cb);
	}

	Result<void> Context::submit_graphics(std::span<VkSubmitInfo> sis, VkFence fence) {
		return graphics_queue->submit(sis, fence);
	}

	Result<void> Context::submit_graphics(std::span<VkSubmitInfo2KHR> sis) {
		return graphics_queue->submit(sis, VK_NULL_HANDLE);
	}

	Result<void> Context::submit_transfer(std::span<VkSubmitInfo> sis, VkFence fence) {
		return transfer_queue->submit(sis, fence);
	}

	Result<void> Context::submit_transfer(std::span<VkSubmitInfo2KHR> sis) {
		return transfer_queue->submit(sis, VK_NULL_HANDLE);
	}

	void PersistentDescriptorSet::update_combined_image_sampler(Context& ctx,
	                                                            unsigned binding,
	                                                            unsigned array_index,
	                                                            ImageView iv,
	                                                            SamplerCreateInfo sci,
	                                                            ImageLayout layout) {
		descriptor_bindings[binding][array_index].image = DescriptorImageInfo(ctx.acquire_sampler(sci, ctx.get_frame_count()), iv, layout);
		descriptor_bindings[binding][array_index].type = DescriptorType::eCombinedImageSampler;
		VkWriteDescriptorSet wds = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
		wds.descriptorCount = 1;
		wds.descriptorType = (VkDescriptorType)DescriptorType::eCombinedImageSampler;
		wds.dstArrayElement = array_index;
		wds.dstBinding = binding;
		wds.pImageInfo = &descriptor_bindings[binding][array_index].image.dii;
		wds.dstSet = backing_set;
		pending_writes.push_back(wds);
	}

	void PersistentDescriptorSet::update_storage_image(Context& ctx, unsigned binding, unsigned array_index, ImageView iv) {
		descriptor_bindings[binding][array_index].image = DescriptorImageInfo({}, iv, ImageLayout::eGeneral);
		descriptor_bindings[binding][array_index].type = DescriptorType::eStorageImage;
		VkWriteDescriptorSet wds = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
		wds.descriptorCount = 1;
		wds.descriptorType = (VkDescriptorType)DescriptorType::eStorageImage;
		wds.dstArrayElement = array_index;
		wds.dstBinding = binding;
		wds.pImageInfo = &descriptor_bindings[binding][array_index].image.dii;
		wds.dstSet = backing_set;
		pending_writes.push_back(wds);
	}

	void PersistentDescriptorSet::update_uniform_buffer(Context& ctx, unsigned binding, unsigned array_index, Buffer buffer) {
		descriptor_bindings[binding][array_index].buffer = VkDescriptorBufferInfo{ buffer.buffer, buffer.offset, buffer.size };
		descriptor_bindings[binding][array_index].type = DescriptorType::eUniformBuffer;
		VkWriteDescriptorSet wds = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
		wds.descriptorCount = 1;
		wds.descriptorType = (VkDescriptorType)DescriptorType::eUniformBuffer;
		wds.dstArrayElement = 0;
		wds.dstBinding = binding;
		wds.pBufferInfo = &descriptor_bindings[binding][array_index].buffer;
		wds.dstSet = backing_set;
		pending_writes.push_back(wds);
	}

	void PersistentDescriptorSet::update_storage_buffer(Context& ctx, unsigned binding, unsigned array_index, Buffer buffer) {
		descriptor_bindings[binding][array_index].buffer = VkDescriptorBufferInfo{ buffer.buffer, buffer.offset, buffer.size };
		descriptor_bindings[binding][array_index].type = DescriptorType::eStorageBuffer;
		VkWriteDescriptorSet wds = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
		wds.descriptorCount = 1;
		wds.descriptorType = (VkDescriptorType)DescriptorType::eStorageBuffer;
		wds.dstArrayElement = 0;
		wds.dstBinding = binding;
		wds.pBufferInfo = &descriptor_bindings[binding][array_index].buffer;
		wds.dstSet = backing_set;
		pending_writes.push_back(wds);
	}

	ShaderModule Context::create(const create_info_t<ShaderModule>& cinfo) {
		std::vector<uint32_t> spirv;
		const uint32_t* spirv_ptr = nullptr;
		size_t size = 0;

		switch (cinfo.source.language) {
#if VUK_USE_SHADERC
		case ShaderSourceLanguage::eGlsl: {
			shaderc::Compiler compiler;
			shaderc::CompileOptions options;
			options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
			options.SetIncluder(std::make_unique<ShadercDefaultIncluder>());
			const auto result = compiler.CompileGlslToSpv(cinfo.source.as_c_str(), shaderc_glsl_infer_from_source, cinfo.filename.c_str(), options);

			if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
				std::string message = result.GetErrorMessage().c_str();
				throw ShaderCompilationException{ message };
			}

			spirv = std::vector<uint32_t>{ result.cbegin(), result.cend() };
			spirv_ptr = spirv.data();
			size = spirv.size();
			break;
		}
#endif
#if VUK_USE_DXC
		case ShaderSourceLanguage::eHlsl: {
			std::vector<LPCWSTR> arguments;
			arguments.push_back(L"-E");
			arguments.push_back(L"main");
			arguments.push_back(L"-spirv");
			arguments.push_back(L"-fspv-target-env=vulkan1.1");
			arguments.push_back(L"-fvk-use-gl-layout");
			arguments.push_back(L"-no-warnings");

			static const std::pair<const char*, HlslShaderStage> inferred[] = {
				{ ".vert.", HlslShaderStage::eVertex },   { ".frag.", HlslShaderStage::ePixel },       { ".comp.", HlslShaderStage::eCompute },
				{ ".geom.", HlslShaderStage::eGeometry }, { ".mesh.", HlslShaderStage::eMesh },        { ".hull.", HlslShaderStage::eHull },
				{ ".dom.", HlslShaderStage::eDomain },    { ".amp.", HlslShaderStage::eAmplification }
			};

			static const std::unordered_map<HlslShaderStage, LPCWSTR> stage_mappings{
				{ HlslShaderStage::eVertex, L"vs_6_7" },   { HlslShaderStage::ePixel, L"ps_6_7" },        { HlslShaderStage::eCompute, L"cs_6_7" },
				{ HlslShaderStage::eGeometry, L"gs_6_7" }, { HlslShaderStage::eMesh, L"ms_6_7" },         { HlslShaderStage::eHull, L"hs_6_7" },
				{ HlslShaderStage::eDomain, L"ds_6_7" },   { HlslShaderStage::eAmplification, L"as_6_7" }
			};

			HlslShaderStage shader_stage = cinfo.source.hlsl_stage;
			if (shader_stage == HlslShaderStage::eInferred) {
				for (const auto& [ext, stage] : inferred) {
					if (cinfo.filename.find(ext) != std::string::npos) {
						shader_stage = stage;
						break;
					}
				}
			}

			assert((shader_stage != HlslShaderStage::eInferred) && "Failed to infer HLSL shader stage");

			arguments.push_back(L"-T");
			arguments.push_back(stage_mappings.at(shader_stage));

			DxcBuffer source_buf;
			source_buf.Ptr = cinfo.source.as_c_str();
			source_buf.Size = cinfo.source.data.size() * 4;
			source_buf.Encoding = 0;

			CComPtr<IDxcCompiler3> compiler = nullptr;
			DXC_HR(DxcCreateInstance(CLSID_DxcCompiler, __uuidof(IDxcCompiler3), (void**)&compiler), "Failed to create DXC compiler");

			CComPtr<IDxcUtils> utils = nullptr;
			DXC_HR(DxcCreateInstance(CLSID_DxcUtils, __uuidof(IDxcUtils), (void**)&utils), "Failed to create DXC utils");

			CComPtr<IDxcIncludeHandler> include_handler = nullptr;
			DXC_HR(utils->CreateDefaultIncludeHandler(&include_handler), "Failed to create include handler");

			CComPtr<IDxcResult> result = nullptr;
			DXC_HR(compiler->Compile(&source_buf, arguments.data(), arguments.size(), &*include_handler, __uuidof(IDxcResult), (void**)&result),
			       "Failed to compile with DXC");

			CComPtr<IDxcBlobUtf8> errors = nullptr;
			DXC_HR(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr), "Failed to get DXC compile errors");
			if (errors && errors->GetStringLength() > 0) {
				std::string message = errors->GetStringPointer();
				throw ShaderCompilationException{ message };
			}

			CComPtr<IDxcBlob> output = nullptr;
			DXC_HR(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&output), nullptr), "Failed to get DXC output");
			assert(output != nullptr);

			const uint32_t* begin = (const uint32_t*)output->GetBufferPointer();
			const uint32_t* end = begin + (output->GetBufferSize() / 4);

			spirv = std::vector<uint32_t>{ begin, end };
			spirv_ptr = spirv.data();
			size = spirv.size();
			break;
		}
#endif
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
		return { sm, p, stage };
	}

	PipelineBaseInfo Context::create(const create_info_t<PipelineBaseInfo>& cinfo) {
		std::vector<VkPipelineShaderStageCreateInfo> psscis;

		// accumulate descriptors from all stages
		Program accumulated_reflection;
		std::string pipe_name = "Pipeline:";
		for (auto i = 0; i < cinfo.shaders.size(); i++) {
			auto& contents = cinfo.shaders[i];
			if (contents.data_ptr == nullptr) {
				continue;
			}
			auto& sm = impl->shader_modules.acquire({ contents, cinfo.shader_paths[i] });
			VkPipelineShaderStageCreateInfo shader_stage{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
			shader_stage.pSpecializationInfo = nullptr;
			shader_stage.stage = sm.stage;
			shader_stage.module = sm.shader_module;
			shader_stage.pName = "main"; // TODO: make param
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
		for (auto& dsl : plci.dslcis) {
			dsl.dslci.bindingCount = (uint32_t)dsl.bindings.size();
			dsl.dslci.pBindings = dsl.bindings.data();
			VkDescriptorSetLayoutBindingFlagsCreateInfo dslbfci{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
			if (dsl.flags.size() > 0) {
				dslbfci.bindingCount = (uint32_t)dsl.bindings.size();
				dslbfci.pBindingFlags = dsl.flags.data();
				dsl.dslci.pNext = &dslbfci;
			}
			auto descset_layout_alloc_info = impl->descriptor_set_layouts.acquire(dsl);
			dslai[dsl.index] = descset_layout_alloc_info;
			dsls.push_back(dslai[dsl.index].layout);
		}
		plci.plci.pSetLayouts = dsls.data();
		plci.plci.setLayoutCount = (uint32_t)dsls.size();

		PipelineBaseInfo pbi;
		pbi.psscis = std::move(psscis);
		pbi.layout_info = dslai;
		pbi.pipeline_layout = impl->pipeline_layouts.acquire(plci);
		pbi.dslcis = std::move(plci.dslcis);
		for (auto& dslci : pbi.dslcis) {
			std::sort(dslci.bindings.begin(), dslci.bindings.end(), [](auto& a, auto& b) { return a.binding < b.binding; });
		}
		pbi.pipeline_name = Name(pipe_name);
		pbi.reflection_info = accumulated_reflection;
		pbi.binding_flags = cinfo.binding_flags;
		pbi.variable_count_max = cinfo.variable_count_max;
		pbi.hit_groups = cinfo.hit_groups;
		pbi.max_ray_recursion_depth = cinfo.max_ray_recursion_depth;
		return pbi;
	}

	bool Context::load_pipeline_cache(std::span<std::byte> data) {
		VkPipelineCacheCreateInfo pcci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = data.size_bytes(), .pInitialData = data.data() };
		this->vkDestroyPipelineCache(device, impl->vk_pipeline_cache, nullptr);
		this->vkCreatePipelineCache(device, &pcci, nullptr, &impl->vk_pipeline_cache);
		return true;
	}

	std::vector<std::byte> Context::save_pipeline_cache() {
		size_t size;
		std::vector<std::byte> data;
		this->vkGetPipelineCacheData(device, impl->vk_pipeline_cache, &size, nullptr);
		data.resize(size);
		this->vkGetPipelineCacheData(device, impl->vk_pipeline_cache, &size, data.data());
		return data;
	}

	Queue& Context::domain_to_queue(DomainFlags domain) const {
		auto queue_only = (DomainFlagBits)(domain & DomainFlagBits::eQueueMask).m_mask;
		switch (queue_only) {
		case DomainFlagBits::eGraphicsQueue:
			return *graphics_queue;
		case DomainFlagBits::eComputeQueue:
			return *compute_queue;
		case DomainFlagBits::eTransferQueue:
			return *transfer_queue;
		default:
			assert(0);
			return *transfer_queue;
		}
	};

	uint32_t Context::domain_to_queue_index(DomainFlags domain) const {
		auto queue_only = (DomainFlagBits)(domain & DomainFlagBits::eQueueMask).m_mask;
		switch (queue_only) {
		case DomainFlagBits::eGraphicsQueue:
			return graphics_queue_family_index;
		case DomainFlagBits::eComputeQueue:
			return compute_queue_family_index;
		case DomainFlagBits::eTransferQueue:
			return transfer_queue_family_index;
		default:
			assert(0);
			return 0;
		}
	};

	uint32_t Context::domain_to_queue_family_index(DomainFlags domain) const {
		return domain_to_queue_index(domain);
	}

	Query Context::create_timestamp_query() {
		return { impl->query_id_counter++ };
	}

	DeviceVkResource& Context::get_vk_resource() {
		return *impl->device_vk_resource;
	}

	DescriptorSetLayoutAllocInfo Context::create(const create_info_t<DescriptorSetLayoutAllocInfo>& cinfo) {
		DescriptorSetLayoutAllocInfo ret;
		this->vkCreateDescriptorSetLayout(device, &cinfo.dslci, nullptr, &ret.layout);
		for (size_t i = 0; i < cinfo.bindings.size(); i++) {
			auto& b = cinfo.bindings[i];
			// if this is not a variable count binding, add it to the descriptor count
			if (cinfo.flags.size() <= i || !(cinfo.flags[i] & to_integral(DescriptorBindingFlagBits::eVariableDescriptorCount))) {
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

	VkPipelineLayout Context::create(const create_info_t<VkPipelineLayout>& cinfo) {
		VkPipelineLayout pl;
		this->vkCreatePipelineLayout(device, &cinfo.plci, nullptr, &pl);
		return pl;
	}

	SwapchainRef Context::add_swapchain(Swapchain sw) {
		std::lock_guard _(impl->swapchains_lock);
		return &*impl->swapchains.emplace(sw);
	}

	void Context::remove_swapchain(SwapchainRef sw) {
		std::lock_guard _(impl->swapchains_lock);
		for (auto it = impl->swapchains.begin(); it != impl->swapchains.end(); it++) {
			if (&*it == sw) {
				impl->swapchains.erase(it);
				return;
			}
		}
	}

	uint64_t Context::get_frame_count() const {
		return impl->frame_counter;
	}

	void Context::create_named_pipeline(Name name, PipelineBaseCreateInfo ci) {
		std::lock_guard _(impl->named_pipelines_lock);
		impl->named_pipelines.insert_or_assign(name, &impl->pipelinebase_cache.acquire(std::move(ci)));
	}

	PipelineBaseInfo* Context::get_named_pipeline(Name name) {
		std::lock_guard _(impl->named_pipelines_lock);
		return impl->named_pipelines.at(name);
	}

	PipelineBaseInfo* Context::get_pipeline(const PipelineBaseCreateInfo& pbci) {
		return &impl->pipelinebase_cache.acquire(pbci);
	}

	Program Context::get_pipeline_reflection_info(const PipelineBaseCreateInfo& pci) {
		auto& res = impl->pipelinebase_cache.acquire(pci);
		return res.reflection_info;
	}

	ShaderModule Context::compile_shader(ShaderSource source, std::string path) {
		ShaderModuleCreateInfo sci;
		sci.filename = std::move(path);
		sci.source = std::move(source);
		auto sm = impl->shader_modules.remove(sci);
		if (sm) {
			this->vkDestroyShaderModule(device, sm->shader_module, nullptr);
		}
		return impl->shader_modules.acquire(sci);
	}

	Texture Context::allocate_texture(Allocator& allocator, ImageCreateInfo ici) {
		ici.imageType = ici.extent.depth > 1 ? ImageType::e3D : ici.extent.height > 1 ? ImageType::e2D : ImageType::e1D;
		Unique<Image> dst = allocate_image(allocator, ici).value(); // TODO: dropping error
		ImageViewCreateInfo ivci;
		ivci.format = ici.format;
		ivci.image = dst->image;
		ivci.subresourceRange.aspectMask = format_to_aspect(ici.format);
		ivci.subresourceRange.baseArrayLayer = 0;
		ivci.subresourceRange.baseMipLevel = 0;
		ivci.subresourceRange.layerCount = 1;
		ivci.subresourceRange.levelCount = ici.mipLevels;
		ivci.viewType = ici.imageType == ImageType::e3D ? ImageViewType::e3D : ici.imageType == ImageType::e2D ? ImageViewType::e2D : ImageViewType::e1D;
		Texture tex{ std::move(dst), allocate_image_view(allocator, ivci).value() }; // TODO: dropping error
		tex.extent = ici.extent;
		tex.format = ici.format;
		tex.sample_count = ici.samples;
		tex.layer_count = 1;
		tex.level_count = ici.mipLevels;
		return tex;
	}

	void Context::destroy(const RGImage& image) {
		deallocate(impl->direct_allocator, image.image);
	}

	void Context::destroy(const DescriptorPool& dp) {
		dp.destroy(*this, device);
	}

	void Context::destroy(const PipelineInfo& pi) {
		this->vkDestroyPipeline(device, pi.pipeline, nullptr);
	}

	void Context::destroy(const ComputePipelineInfo& pi) {
		this->vkDestroyPipeline(device, pi.pipeline, nullptr);
	}

	void Context::destroy(const RayTracingPipelineInfo& pi) {
		impl->device_vk_resource->deallocate_buffers(std::span{ &pi.sbt, 1 });
		this->vkDestroyPipeline(device, pi.pipeline, nullptr);
	}

	void Context::destroy(const ShaderModule& sm) {
		this->vkDestroyShaderModule(device, sm.shader_module, nullptr);
	}

	void Context::destroy(const DescriptorSetLayoutAllocInfo& ds) {
		this->vkDestroyDescriptorSetLayout(device, ds.layout, nullptr);
	}

	void Context::destroy(const VkPipelineLayout& pl) {
		this->vkDestroyPipelineLayout(device, pl, nullptr);
	}

	void Context::destroy(const VkRenderPass& rp) {
		this->vkDestroyRenderPass(device, rp, nullptr);
	}

	void Context::destroy(const DescriptorSet&) {
		// no-op, we destroy the pools
	}

	void Context::destroy(const VkFramebuffer& fb) {
		this->vkDestroyFramebuffer(device, fb, nullptr);
	}

	void Context::destroy(const Sampler& sa) {
		this->vkDestroySampler(device, sa.payload, nullptr);
	}

	void Context::destroy(const PipelineBaseInfo& pbi) {
		// no-op, we don't own device objects
	}

	Context::~Context() {
		if (impl) {
			this->vkDeviceWaitIdle(device);

			for (auto& s : impl->swapchains) {
				for (auto& swiv : s.image_views) {
					this->vkDestroyImageView(device, swiv.payload, nullptr);
				}
				this->vkDestroySwapchainKHR(device, s.swapchain, nullptr);
			}

			this->vkDestroyPipelineCache(device, impl->vk_pipeline_cache, nullptr);

			if (dedicated_graphics_queue) {
				impl->device_vk_resource->deallocate_timeline_semaphores(std::span{ &dedicated_graphics_queue->get_submit_sync(), 1 });
			}

			if (dedicated_compute_queue) {
				impl->device_vk_resource->deallocate_timeline_semaphores(std::span{ &dedicated_compute_queue->get_submit_sync(), 1 });
			}

			if (dedicated_transfer_queue) {
				impl->device_vk_resource->deallocate_timeline_semaphores(std::span{ &dedicated_transfer_queue->get_submit_sync(), 1 });
			}

			delete impl;
		}
	}

	uint64_t Context::get_unique_handle_id() {
		return impl->unique_handle_id_counter++;
	}

	void Context::next_frame() {
		impl->frame_counter++;
		collect(impl->frame_counter);
	}

	void Context::wait_idle() {
		std::unique_lock<std::recursive_mutex> graphics_lock;
		if (dedicated_graphics_queue) {
			graphics_lock = std::unique_lock{ graphics_queue->get_queue_lock() };
		}
		std::unique_lock<std::recursive_mutex> compute_lock;
		if (dedicated_compute_queue) {
			compute_lock = std::unique_lock{ compute_queue->get_queue_lock() };
		}
		std::unique_lock<std::recursive_mutex> transfer_lock;
		if (dedicated_transfer_queue) {
			transfer_lock = std::unique_lock{ transfer_queue->get_queue_lock() };
		}

		this->vkDeviceWaitIdle(device);
	}

	void Context::collect(uint64_t frame) {
		impl->collect(frame);
	}

	Unique<PersistentDescriptorSet>
	Context::create_persistent_descriptorset(Allocator& allocator, DescriptorSetLayoutCreateInfo dslci, unsigned num_descriptors) {
		dslci.dslci.bindingCount = (uint32_t)dslci.bindings.size();
		dslci.dslci.pBindings = dslci.bindings.data();
		VkDescriptorSetLayoutBindingFlagsCreateInfo dslbfci{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
		if (dslci.flags.size() > 0) {
			dslbfci.bindingCount = (uint32_t)dslci.bindings.size();
			dslbfci.pBindingFlags = dslci.flags.data();
			dslci.dslci.pNext = &dslbfci;
		}
		auto& dslai = impl->descriptor_set_layouts.acquire(dslci, impl->frame_counter);
		return create_persistent_descriptorset(allocator, { dslai, num_descriptors });
	}

	Unique<PersistentDescriptorSet> Context::create_persistent_descriptorset(Allocator& allocator, const PersistentDescriptorSetCreateInfo& ci) {
		Unique<PersistentDescriptorSet> pds(allocator);
		allocator.allocate_persistent_descriptor_sets(std::span{ &*pds, 1 }, std::span{ &ci, 1 });
		return pds;
	}

	Unique<PersistentDescriptorSet>
	Context::create_persistent_descriptorset(Allocator& allocator, const PipelineBaseInfo& base, unsigned set, unsigned num_descriptors) {
		return create_persistent_descriptorset(allocator, { base.layout_info[set], num_descriptors });
	}

	void Context::commit_persistent_descriptorset(PersistentDescriptorSet& array) {
		this->vkUpdateDescriptorSets(device, (uint32_t)array.pending_writes.size(), array.pending_writes.data(), 0, nullptr);
		array.pending_writes.clear();
	}

	RGImage Context::create(const create_info_t<RGImage>& cinfo) {
		RGImage res{};
		res.image = allocate_image(impl->direct_allocator, cinfo.ici)->release();
		std::string name = std::string("Image: RenderTarget ") + std::string(cinfo.name.to_sv());
		set_name(res.image.image, Name(name));
		return res;
	}

	VkRenderPass Context::create(const create_info_t<VkRenderPass>& cinfo) {
		VkRenderPass rp;
		this->vkCreateRenderPass(device, &cinfo, nullptr, &rp);
		return rp;
	}

	template<class T>
	T read(const std::byte*& data_ptr) {
		T t;
		memcpy(&t, data_ptr, sizeof(T));
		data_ptr += sizeof(T);
		return t;
	};

	PipelineInfo Context::create(const create_info_t<PipelineInfo>& cinfo) {
		// create gfx pipeline
		VkGraphicsPipelineCreateInfo gpci{ .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
		gpci.renderPass = cinfo.render_pass;
		gpci.layout = cinfo.base->pipeline_layout;
		auto psscis = cinfo.base->psscis;
		gpci.pStages = psscis.data();
		gpci.stageCount = (uint32_t)psscis.size();

		// read variable sized data
		const std::byte* data_ptr = cinfo.is_inline() ? cinfo.inline_data : cinfo.extended_data;

		// subpass
		if (cinfo.records.nonzero_subpass) {
			gpci.subpass = read<uint8_t>(data_ptr);
		}

		// INPUT ASSEMBLY
		VkPipelineInputAssemblyStateCreateInfo input_assembly_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			                                                           .topology = static_cast<VkPrimitiveTopology>(cinfo.topology),
			                                                           .primitiveRestartEnable = cinfo.primitive_restart_enable };
		gpci.pInputAssemblyState = &input_assembly_state;
		// VERTEX INPUT
		fixed_vector<VkVertexInputBindingDescription, VUK_MAX_ATTRIBUTES> vibds;
		fixed_vector<VkVertexInputAttributeDescription, VUK_MAX_ATTRIBUTES> viads;
		VkPipelineVertexInputStateCreateInfo vertex_input_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
		if (cinfo.records.vertex_input) {
			viads.resize(cinfo.base->reflection_info.attributes.size());
			for (auto& viad : viads) {
				auto compressed = read<PipelineInstanceCreateInfo::VertexInputAttributeDescription>(data_ptr);
				viad.binding = compressed.binding;
				viad.location = compressed.location;
				viad.format = (VkFormat)compressed.format;
				viad.offset = compressed.offset;
			}
			vertex_input_state.pVertexAttributeDescriptions = viads.data();
			vertex_input_state.vertexAttributeDescriptionCount = (uint32_t)viads.size();

			vibds.resize(read<uint8_t>(data_ptr));
			for (auto& vibd : vibds) {
				auto compressed = read<PipelineInstanceCreateInfo::VertexInputBindingDescription>(data_ptr);
				vibd.binding = compressed.binding;
				vibd.inputRate = (VkVertexInputRate)compressed.inputRate;
				vibd.stride = compressed.stride;
			}
			vertex_input_state.pVertexBindingDescriptions = vibds.data();
			vertex_input_state.vertexBindingDescriptionCount = (uint32_t)vibds.size();
		}
		gpci.pVertexInputState = &vertex_input_state;
		// PIPELINE COLOR BLEND ATTACHMENTS
		VkPipelineColorBlendStateCreateInfo color_blend_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			                                                     .attachmentCount = cinfo.attachmentCount };
		auto default_writemask = ColorComponentFlagBits::eR | ColorComponentFlagBits::eG | ColorComponentFlagBits::eB | ColorComponentFlagBits::eA;
		std::vector<VkPipelineColorBlendAttachmentState> pcbas(
		    cinfo.attachmentCount, VkPipelineColorBlendAttachmentState{ .blendEnable = false, .colorWriteMask = (VkColorComponentFlags)default_writemask });
		if (cinfo.records.color_blend_attachments) {
			if (!cinfo.records.broadcast_color_blend_attachment_0) {
				for (auto& pcba : pcbas) {
					auto compressed = read<PipelineInstanceCreateInfo::PipelineColorBlendAttachmentState>(data_ptr);
					pcba = { compressed.blendEnable,
						       (VkBlendFactor)compressed.srcColorBlendFactor,
						       (VkBlendFactor)compressed.dstColorBlendFactor,
						       (VkBlendOp)compressed.colorBlendOp,
						       (VkBlendFactor)compressed.srcAlphaBlendFactor,
						       (VkBlendFactor)compressed.dstAlphaBlendFactor,
						       (VkBlendOp)compressed.alphaBlendOp,
						       compressed.colorWriteMask };
				}
			} else { // handle broadcast
				auto compressed = read<PipelineInstanceCreateInfo::PipelineColorBlendAttachmentState>(data_ptr);
				for (auto& pcba : pcbas) {
					pcba = { compressed.blendEnable,
						       (VkBlendFactor)compressed.srcColorBlendFactor,
						       (VkBlendFactor)compressed.dstColorBlendFactor,
						       (VkBlendOp)compressed.colorBlendOp,
						       (VkBlendFactor)compressed.srcAlphaBlendFactor,
						       (VkBlendFactor)compressed.dstAlphaBlendFactor,
						       (VkBlendOp)compressed.alphaBlendOp,
						       compressed.colorWriteMask };
				}
			}
		}
		if (cinfo.records.logic_op) {
			auto compressed = read<PipelineInstanceCreateInfo::BlendStateLogicOp>(data_ptr);
			color_blend_state.logicOpEnable = true;
			color_blend_state.logicOp = static_cast<VkLogicOp>(compressed.logic_op);
		}
		if (cinfo.records.blend_constants) {
			memcpy(&color_blend_state.blendConstants, data_ptr, sizeof(float) * 4);
			data_ptr += sizeof(float) * 4;
		}

		color_blend_state.pAttachments = pcbas.data();
		color_blend_state.attachmentCount = (uint32_t)pcbas.size();
		gpci.pColorBlendState = &color_blend_state;

		// SPECIALIZATION CONSTANTS
		fixed_vector<VkSpecializationInfo, graphics_stage_count> specialization_infos;
		fixed_vector<VkSpecializationMapEntry, VUK_MAX_SPECIALIZATIONCONSTANT_RANGES> specialization_map_entries;
		uint16_t specialization_constant_data_size = 0;
		const std::byte* specialization_constant_data = nullptr;
		if (cinfo.records.specialization_constants) {
			Bitset<VUK_MAX_SPECIALIZATIONCONSTANT_RANGES> set_constants = {};
			set_constants = read<Bitset<VUK_MAX_SPECIALIZATIONCONSTANT_RANGES>>(data_ptr);
			specialization_constant_data = data_ptr;

			for (unsigned i = 0; i < cinfo.base->reflection_info.spec_constants.size(); i++) {
				auto& sc = cinfo.base->reflection_info.spec_constants[i];
				uint16_t size = sc.type == Program::Type::edouble ? (uint16_t)sizeof(double) : 4;
				if (set_constants.test(i)) {
					specialization_constant_data_size += size;
				}
			}
			data_ptr += specialization_constant_data_size;

			uint16_t entry_offset = 0;
			for (uint32_t i = 0; i < psscis.size(); i++) {
				auto& pssci = psscis[i];
				uint16_t data_offset = 0;
				uint16_t current_entry_offset = entry_offset;
				for (unsigned i = 0; i < cinfo.base->reflection_info.spec_constants.size(); i++) {
					auto& sc = cinfo.base->reflection_info.spec_constants[i];
					auto size = sc.type == Program::Type::edouble ? sizeof(double) : 4;
					if (sc.stage & pssci.stage) {
						specialization_map_entries.emplace_back(VkSpecializationMapEntry{ sc.binding, data_offset, size });
						data_offset += (uint16_t)size;
						entry_offset++;
					}
				}

				VkSpecializationInfo si;
				si.pMapEntries = specialization_map_entries.data() + current_entry_offset;
				si.mapEntryCount = (uint32_t)specialization_map_entries.size() - current_entry_offset;
				si.pData = specialization_constant_data;
				si.dataSize = specialization_constant_data_size;
				if (si.mapEntryCount > 0) {
					specialization_infos.push_back(si);
					pssci.pSpecializationInfo = &specialization_infos.back();
				}
			}
		}

		// RASTER STATE
		VkPipelineRasterizationStateCreateInfo rasterization_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			                                                          .polygonMode = VK_POLYGON_MODE_FILL,
			                                                          .cullMode = cinfo.cullMode,
			                                                          .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
			                                                          .lineWidth = 1.f };
		if (cinfo.records.non_trivial_raster_state) {
			auto rs = read<PipelineInstanceCreateInfo::RasterizationState>(data_ptr);
			rasterization_state = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				                      .depthClampEnable = rs.depthClampEnable,
				                      .rasterizerDiscardEnable = rs.rasterizerDiscardEnable,
				                      .polygonMode = (VkPolygonMode)rs.polygonMode,
				                      .cullMode = cinfo.cullMode,
				                      .frontFace = (VkFrontFace)rs.frontFace,
				                      .lineWidth = 1.f };
		}
		rasterization_state.depthBiasEnable = cinfo.records.depth_bias_enable;
		if (cinfo.records.depth_bias) {
			auto db = read<PipelineInstanceCreateInfo::DepthBias>(data_ptr);
			rasterization_state.depthBiasClamp = db.depthBiasClamp;
			rasterization_state.depthBiasConstantFactor = db.depthBiasConstantFactor;
			rasterization_state.depthBiasSlopeFactor = db.depthBiasSlopeFactor;
		}
		if (cinfo.records.line_width_not_1) {
			rasterization_state.lineWidth = read<float>(data_ptr);
		}
		gpci.pRasterizationState = &rasterization_state;

		// DEPTH - STENCIL STATE
		VkPipelineDepthStencilStateCreateInfo depth_stencil_state{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
		if (cinfo.records.depth_stencil) {
			auto d = read<PipelineInstanceCreateInfo::Depth>(data_ptr);
			depth_stencil_state.depthTestEnable = d.depthTestEnable;
			depth_stencil_state.depthWriteEnable = d.depthWriteEnable;
			depth_stencil_state.depthCompareOp = (VkCompareOp)d.depthCompareOp;
			if (cinfo.records.depth_bounds) {
				auto db = read<PipelineInstanceCreateInfo::DepthBounds>(data_ptr);
				depth_stencil_state.depthBoundsTestEnable = true;
				depth_stencil_state.minDepthBounds = db.minDepthBounds;
				depth_stencil_state.maxDepthBounds = db.maxDepthBounds;
			}
			if (cinfo.records.stencil_state) {
				auto s = read<PipelineInstanceCreateInfo::Stencil>(data_ptr);
				depth_stencil_state.stencilTestEnable = true;
				depth_stencil_state.front = s.front;
				depth_stencil_state.back = s.back;
			}
			gpci.pDepthStencilState = &depth_stencil_state;
		}

		// MULTISAMPLE STATE
		VkPipelineMultisampleStateCreateInfo multisample_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			                                                      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
		if (cinfo.records.more_than_one_sample) {
			auto ms = read<PipelineInstanceCreateInfo::Multisample>(data_ptr);
			multisample_state.rasterizationSamples = static_cast<VkSampleCountFlagBits>(ms.rasterization_samples);
			multisample_state.alphaToCoverageEnable = ms.alpha_to_coverage_enable;
			multisample_state.alphaToOneEnable = ms.alpha_to_one_enable;
			multisample_state.minSampleShading = ms.min_sample_shading;
			multisample_state.sampleShadingEnable = ms.sample_shading_enable;
			multisample_state.pSampleMask = nullptr; // not yet supported
		}
		gpci.pMultisampleState = &multisample_state;

		// VIEWPORTS
		const VkViewport* viewports = nullptr;
		uint8_t num_viewports = 1;
		if (cinfo.records.viewports) {
			num_viewports = read<uint8_t>(data_ptr);
			if (!(static_cast<vuk::DynamicStateFlags>(cinfo.dynamic_state_flags) & vuk::DynamicStateFlagBits::eViewport)) {
				viewports = reinterpret_cast<const VkViewport*>(data_ptr);
				data_ptr += num_viewports * sizeof(VkViewport);
			}
		}

		// SCISSORS
		const VkRect2D* scissors = nullptr;
		uint8_t num_scissors = 1;
		if (cinfo.records.scissors) {
			num_scissors = read<uint8_t>(data_ptr);
			if (!(static_cast<vuk::DynamicStateFlags>(cinfo.dynamic_state_flags) & vuk::DynamicStateFlagBits::eScissor)) {
				scissors = reinterpret_cast<const VkRect2D*>(data_ptr);
				data_ptr += num_scissors * sizeof(VkRect2D);
			}
		}

		VkPipelineViewportStateCreateInfo viewport_state{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
		viewport_state.pViewports = viewports;
		viewport_state.viewportCount = num_viewports;
		viewport_state.pScissors = scissors;
		viewport_state.scissorCount = num_scissors;
		gpci.pViewportState = &viewport_state;

		VkPipelineDynamicStateCreateInfo dynamic_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
		dynamic_state.dynamicStateCount = std::popcount(cinfo.dynamic_state_flags);
		fixed_vector<VkDynamicState, VkDynamicState::VK_DYNAMIC_STATE_DEPTH_BOUNDS> dyn_states;
		uint64_t dyn_state_cnt = 0;
		uint16_t mask = cinfo.dynamic_state_flags;
		while (mask > 0) {
			bool set = mask & 0x1;
			if (set) {
				dyn_states.push_back((VkDynamicState)dyn_state_cnt); // TODO: we will need a switch here instead of a cast when handling EXT
			}
			mask >>= 1;
			dyn_state_cnt++;
		}
		dynamic_state.pDynamicStates = dyn_states.data();
		gpci.pDynamicState = &dynamic_state;

		VkPipeline pipeline;
		VkResult res = this->vkCreateGraphicsPipelines(device, impl->vk_pipeline_cache, 1, &gpci, nullptr, &pipeline);
		assert(res == VK_SUCCESS);
		set_name(pipeline, cinfo.base->pipeline_name);
		return { cinfo.base, pipeline, gpci.layout, cinfo.base->layout_info };
	}

	ComputePipelineInfo Context::create(const create_info_t<ComputePipelineInfo>& cinfo) {
		// create compute pipeline
		VkComputePipelineCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
		cpci.layout = cinfo.base->pipeline_layout;
		cpci.stage = cinfo.base->psscis[0];

		VkPipeline pipeline;
		VkResult res = this->vkCreateComputePipelines(device, impl->vk_pipeline_cache, 1, &cpci, nullptr, &pipeline);
		assert(res == VK_SUCCESS);
		set_name(pipeline, cinfo.base->pipeline_name);
		return { { cinfo.base, pipeline, cpci.layout, cinfo.base->layout_info }, cinfo.base->reflection_info.local_size };
	}

	RayTracingPipelineInfo Context::create(const struct RayTracingPipelineInstanceCreateInfo& cinfo) {
		// create compute pipeline
		VkRayTracingPipelineCreateInfoKHR cpci{ .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
		cpci.layout = cinfo.base->pipeline_layout;

		std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;
		VkRayTracingShaderGroupCreateInfoKHR group{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
		group.anyHitShader = VK_SHADER_UNUSED_KHR;
		group.closestHitShader = VK_SHADER_UNUSED_KHR;
		group.generalShader = VK_SHADER_UNUSED_KHR;
		group.intersectionShader = VK_SHADER_UNUSED_KHR;

		uint32_t miss_count = 0;
		uint32_t hit_count = 0;
		uint32_t callable_count = 0;

		for (size_t i = 0; i < cinfo.base->psscis.size(); i++) {
			auto& stage = cinfo.base->psscis[i];
			if (stage.stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR) {
				group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
				group.generalShader = (uint32_t)i;
				groups.push_back(group);
			} else if (stage.stage == VK_SHADER_STAGE_MISS_BIT_KHR) {
				group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
				group.generalShader = (uint32_t)i;
				groups.push_back(group);
				miss_count++;
			} else if (stage.stage == VK_SHADER_STAGE_CALLABLE_BIT_KHR) {
				group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
				group.generalShader = (uint32_t)i;
				groups.push_back(group);
				callable_count++;
			}
		}
		for (auto& hg : cinfo.base->hit_groups) {
			group.type = (VkRayTracingShaderGroupTypeKHR)hg.type;
			group.generalShader = VK_SHADER_UNUSED_KHR;
			group.anyHitShader = hg.any_hit;
			group.intersectionShader = hg.intersection;
			group.closestHitShader = hg.closest_hit;
			groups.push_back(group);
			hit_count++;
		}

		cpci.groupCount = (uint32_t)groups.size();
		cpci.pGroups = groups.data();

		cpci.maxPipelineRayRecursionDepth = cinfo.base->max_ray_recursion_depth;
		cpci.pStages = cinfo.base->psscis.data();
		cpci.stageCount = (uint32_t)cinfo.base->psscis.size();

		VkPipeline pipeline;
		VkResult res = this->vkCreateRayTracingPipelinesKHR(device, {}, impl->vk_pipeline_cache, 1, &cpci, nullptr, &pipeline);
		assert(res == VK_SUCCESS);
		set_name(pipeline, cinfo.base->pipeline_name);

		auto handleCount = 1 + miss_count + hit_count + callable_count;
		uint32_t handleSize = rt_properties.shaderGroupHandleSize;
		// The SBT (buffer) need to have starting groups to be aligned and handles in the group to be aligned.
		uint32_t handleSizeAligned = vuk::align_up(handleSize, rt_properties.shaderGroupHandleAlignment);

		VkStridedDeviceAddressRegionKHR rgen_region{};
		VkStridedDeviceAddressRegionKHR miss_region{};
		VkStridedDeviceAddressRegionKHR hit_region{};
		VkStridedDeviceAddressRegionKHR call_region{};

		rgen_region.stride = vuk::align_up(handleSizeAligned, rt_properties.shaderGroupBaseAlignment);
		rgen_region.size = rgen_region.stride; // The size member of pRayGenShaderBindingTable must be equal to its stride member
		miss_region.stride = handleSizeAligned;
		miss_region.size = vuk::align_up(miss_count * handleSizeAligned, rt_properties.shaderGroupBaseAlignment);
		hit_region.stride = handleSizeAligned;
		hit_region.size = vuk::align_up(hit_count * handleSizeAligned, rt_properties.shaderGroupBaseAlignment);
		call_region.stride = handleSizeAligned;
		call_region.size = vuk::align_up(callable_count * handleSizeAligned, rt_properties.shaderGroupBaseAlignment);

		// Get the shader group handles
		uint32_t dataSize = handleCount * handleSize;
		std::vector<uint8_t> handles(dataSize);
		auto result = this->vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, 0, handleCount, dataSize, handles.data());
		assert(result == VK_SUCCESS);

		VkDeviceSize sbt_size = rgen_region.size + miss_region.size + hit_region.size + call_region.size;
		Buffer SBT;
		BufferCreateInfo bci{ .mem_usage = vuk::MemoryUsage::eCPUtoGPU, .size = sbt_size };
		auto buff_cr_result = impl->device_vk_resource->allocate_buffers(std::span{ &SBT, 1 }, std::span{ &bci, 1 }, {});
		assert(buff_cr_result);

		// Helper to retrieve the handle data
		auto get_handle = [&](int i) {
			return handles.data() + i * handleSize;
		};
		std::byte* pData{ nullptr };
		uint32_t handleIdx{ 0 };
		// Raygen
		pData = SBT.mapped_ptr;
		memcpy(pData, get_handle(handleIdx++), handleSize);
		// Miss
		pData = SBT.mapped_ptr + rgen_region.size;
		for (uint32_t c = 0; c < miss_count; c++) {
			memcpy(pData, get_handle(handleIdx++), handleSize);
			pData += miss_region.stride;
		}
		// Hit
		pData = SBT.mapped_ptr + rgen_region.size + miss_region.size;
		for (uint32_t c = 0; c < hit_count; c++) {
			memcpy(pData, get_handle(handleIdx++), handleSize);
			pData += hit_region.stride;
		}
		// Call
		pData = SBT.mapped_ptr + rgen_region.size + miss_region.size + hit_region.size;
		for (uint32_t c = 0; c < callable_count; c++) {
			memcpy(pData, get_handle(handleIdx++), handleSize);
			pData += call_region.stride;
		}

		auto sbtAddress = SBT.device_address;

		rgen_region.deviceAddress = sbtAddress;
		miss_region.deviceAddress = sbtAddress + rgen_region.size;
		hit_region.deviceAddress = sbtAddress + rgen_region.size + miss_region.size;
		call_region.deviceAddress = sbtAddress + rgen_region.size + miss_region.size + hit_region.size;

		return { { cinfo.base, pipeline, cpci.layout, cinfo.base->layout_info }, rgen_region, miss_region, hit_region, call_region, SBT };
	}

	Sampler Context::create(const create_info_t<Sampler>& cinfo) {
		VkSampler s;
		this->vkCreateSampler(device, (VkSamplerCreateInfo*)&cinfo, nullptr, &s);
		return wrap(s);
	}

	DescriptorPool Context::create(const create_info_t<DescriptorPool>& cinfo) {
		return DescriptorPool{};
	}

	VkRenderPass Context::acquire_renderpass(const RenderPassCreateInfo& rpci, uint64_t absolute_frame) {
		return impl->renderpass_cache.acquire(rpci, absolute_frame);
	}

	RGImage Context::acquire_rendertarget(const RGCI& rgci, uint64_t absolute_frame) {
		return impl->transient_images.acquire(rgci, absolute_frame);
	}

	Sampler Context::acquire_sampler(const SamplerCreateInfo& sci, uint64_t absolute_frame) {
		return impl->sampler_cache.acquire(sci, absolute_frame);
	}

	DescriptorPool& Context::acquire_descriptor_pool(const DescriptorSetLayoutAllocInfo& dslai, uint64_t absolute_frame) {
		return impl->pool_cache.acquire(dslai, absolute_frame);
	}

	PipelineInfo Context::acquire_pipeline(const PipelineInstanceCreateInfo& pici, uint64_t absolute_frame) {
		return impl->pipeline_cache.acquire(pici, absolute_frame);
	}

	ComputePipelineInfo Context::acquire_pipeline(const ComputePipelineInstanceCreateInfo& pici, uint64_t absolute_frame) {
		return impl->compute_pipeline_cache.acquire(pici, absolute_frame);
	}

	RayTracingPipelineInfo Context::acquire_pipeline(const RayTracingPipelineInstanceCreateInfo& ci, uint64_t absolute_frame) {
		return impl->ray_tracing_pipeline_cache.acquire(ci, absolute_frame);
	}

	bool Context::is_timestamp_available(Query q) {
		std::scoped_lock _(impl->query_lock);
		auto it = impl->timestamp_result_map.find(q);
		return (it != impl->timestamp_result_map.end());
	}

	std::optional<uint64_t> Context::retrieve_timestamp(Query q) {
		std::scoped_lock _(impl->query_lock);
		auto it = impl->timestamp_result_map.find(q);
		if (it != impl->timestamp_result_map.end()) {
			uint64_t res = it->second;
			impl->timestamp_result_map.erase(it);
			return res;
		}
		return {};
	}

	std::optional<double> Context::retrieve_duration(Query q1, Query q2) {
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

	Result<void> Context::make_timestamp_results_available(std::span<const TimestampQueryPool> pools) {
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
} // namespace vuk
