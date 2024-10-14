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
			dedicated_graphics_queue.emplace(this->vkQueueSubmit, this->vkQueueSubmit2KHR, params.graphics_queue, params.graphics_queue_family_index, ts);
			set_name(params.graphics_queue, "Graphics Queue");
			graphics_queue = &dedicated_graphics_queue.value();
		}
		if (dedicated_compute_queue_) {
			TimelineSemaphore ts;
			impl->device_vk_resource->allocate_timeline_semaphores(std::span{ &ts, 1 }, {});
			dedicated_compute_queue.emplace(this->vkQueueSubmit, this->vkQueueSubmit2KHR, params.compute_queue, params.compute_queue_family_index, ts);
			set_name(params.compute_queue, "Compute Queue");
			compute_queue = &dedicated_compute_queue.value();
		} else {
			compute_queue = graphics_queue;
		}
		if (dedicated_transfer_queue_) {
			TimelineSemaphore ts;
			impl->device_vk_resource->allocate_timeline_semaphores(std::span{ &ts, 1 }, {});
			dedicated_transfer_queue.emplace(this->vkQueueSubmit, this->vkQueueSubmit2KHR, params.transfer_queue, params.transfer_queue_family_index, ts);
			set_name(params.transfer_queue, "Transfer Queue");
			transfer_queue = &dedicated_transfer_queue.value();
		} else {
			transfer_queue = compute_queue ? compute_queue : graphics_queue;
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

	void PersistentDescriptorSet::commit(Context& ctx) {
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

	static std::wstring convert_to_wstring(const std::string& string) {
		std::vector<wchar_t> buffer(string.size());
		std::use_facet<std::ctype<wchar_t>>(std::locale()).widen(string.data(), string.data() + string.size(), buffer.data());
		return { buffer.data(), buffer.size() };
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

			static const std::unordered_map<uint32_t, uint32_t> target_version = {
				{ VK_API_VERSION_1_0, shaderc_env_version_vulkan_1_0 },
				{ VK_API_VERSION_1_1, shaderc_env_version_vulkan_1_1 },
				{ VK_API_VERSION_1_2, shaderc_env_version_vulkan_1_2 },
				{ VK_API_VERSION_1_3, shaderc_env_version_vulkan_1_3 },
			};

			options.SetTargetEnvironment(shaderc_target_env_vulkan, target_version.at(shader_compiler_target_version));

			static const std::unordered_map<ShaderCompileOptions::OptimizationLevel, shaderc_optimization_level> optimization_level = {
				{ ShaderCompileOptions::OptimizationLevel::O0, shaderc_optimization_level_zero },
				{ ShaderCompileOptions::OptimizationLevel::O1, shaderc_optimization_level_performance },
				{ ShaderCompileOptions::OptimizationLevel::O2, shaderc_optimization_level_performance },
				{ ShaderCompileOptions::OptimizationLevel::O3, shaderc_optimization_level_performance },
			};

			options.SetOptimizationLevel(optimization_level.at(cinfo.compile_options.optimization_level));

			options.SetIncluder(std::make_unique<ShadercDefaultIncluder>());
			for (auto& [k, v] : cinfo.defines) {
				options.AddMacroDefinition(k, v);
			}

			const auto result = compiler.CompileGlslToSpv(cinfo.source.as_c_str(), shaderc_glsl_infer_from_source, cinfo.filename.c_str(), options);
			if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
				std::string message = result.GetErrorMessage();
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

			auto entry_point = convert_to_wstring(cinfo.source.entry_point);
			arguments.push_back(entry_point.c_str());

			auto dir = std::filesystem::path(cinfo.filename).parent_path();
			auto include_path = fmt::format("-I {0}", dir.string());
			auto include_path_w = convert_to_wstring(include_path);
			arguments.push_back(include_path_w.c_str());

			std::vector<std::wstring> def_ws;
			for (auto [k, v] : cinfo.defines) {
				auto def = v.empty() ? fmt::format("-D{0}", k) : fmt::format("-D{0}={1}", k, v);
				arguments.push_back(def_ws.emplace_back(convert_to_wstring(def)).c_str());
			}

			// current valid options in dxc are 1.0 and 1.1
			static const std::unordered_map<uint32_t, const wchar_t*> target_version = {
				{ VK_API_VERSION_1_0, L"-fspv-target-env=vulkan1.0"},
				{ VK_API_VERSION_1_1, L"-fspv-target-env=vulkan1.1"},
				{ VK_API_VERSION_1_2, L"-fspv-target-env=vulkan1.1"},
				{ VK_API_VERSION_1_3, L"-fspv-target-env=vulkan1.1"},
			};

			arguments.push_back(target_version.at(shader_compiler_target_version));

			static const std::unordered_map<ShaderCompileOptions::OptimizationLevel, const wchar_t*> optimization_level = {
				{ ShaderCompileOptions::OptimizationLevel::O0, L"-O0" },
				{ ShaderCompileOptions::OptimizationLevel::O1, L"-O1" },
				{ ShaderCompileOptions::OptimizationLevel::O2, L"-O2" },
				{ ShaderCompileOptions::OptimizationLevel::O3, L"-O3" },
			};

			arguments.push_back(optimization_level.at(cinfo.compile_options.optimization_level));

			for (auto& arg : cinfo.compile_options.dxc_extra_arguments) {
				arguments.push_back(arg.data());
			}

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
			DXC_HR(compiler->Compile(&source_buf, arguments.data(), (UINT32)arguments.size(), &*include_handler, __uuidof(IDxcResult), (void**)&result),
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
		std::vector<std::string> entry_point_names;

		// accumulate descriptors from all stages
		Program accumulated_reflection;
		std::string pipe_name = "Pipeline:";
		for (auto i = 0; i < cinfo.shaders.size(); i++) {
			auto& contents = cinfo.shaders[i];
			if (contents.data_ptr == nullptr) {
				continue;
			}
			auto& sm = impl->shader_modules.acquire({ contents, cinfo.shader_paths[i], cinfo.defines, cinfo.compile_options });
			VkPipelineShaderStageCreateInfo shader_stage{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
			shader_stage.pSpecializationInfo = nullptr;
			shader_stage.stage = sm.stage;
			shader_stage.module = sm.shader_module;
			entry_point_names.push_back(contents.entry_point);
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
		pbi.entry_point_names = std::move(entry_point_names);
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
		pbi.patchControlPoints = cinfo.patchControlPoints;
		return pbi;
	}

	bool Context::load_pipeline_cache(std::span<std::byte> data) {
		VkPipelineCacheCreateInfo pcci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = data.size_bytes(), .pInitialData = data.data() };
		this->vkDestroyPipelineCache(device, vk_pipeline_cache, nullptr);
		this->vkCreatePipelineCache(device, &pcci, nullptr, &vk_pipeline_cache);
		return true;
	}

	std::vector<std::byte> Context::save_pipeline_cache() {
		size_t size;
		std::vector<std::byte> data;
		this->vkGetPipelineCacheData(device, vk_pipeline_cache, &size, nullptr);
		data.resize(size);
		this->vkGetPipelineCacheData(device, vk_pipeline_cache, &size, data.data());
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
		auto pbi = &impl->pipelinebase_cache.acquire(std::move(ci));
		std::lock_guard _(impl->named_pipelines_lock);
		impl->named_pipelines.insert_or_assign(name, pbi);
	}

	PipelineBaseInfo* Context::get_named_pipeline(Name name) {
		std::lock_guard _(impl->named_pipelines_lock);
		return impl->named_pipelines.at(name);
	}

	bool Context::is_pipeline_available(Name name) const {
		std::lock_guard _(impl->named_pipelines_lock);
		return impl->named_pipelines.contains(name);
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

	void Context::set_shader_target_version(const uint32_t target_version) {
		assert((target_version >= VK_API_VERSION_1_0 && target_version <= VK_API_VERSION_1_3) && "Invalid target version was passed.");
		shader_compiler_target_version = target_version;
	}

	Texture Context::allocate_texture(Allocator& allocator, ImageCreateInfo ici, SourceLocationAtFrame loc) {
		ici.imageType = ici.extent.depth > 1 ? ImageType::e3D : ici.extent.height > 1 ? ImageType::e2D : ImageType::e1D;
		VkImageFormatListCreateInfo listci = { VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO };
		auto unorm_fmt = srgb_to_unorm(ici.format);
		auto srgb_fmt = unorm_to_srgb(ici.format);
		VkFormat formats[2] = { (VkFormat)ici.format, unorm_fmt == vuk::Format::eUndefined ? (VkFormat)srgb_fmt : (VkFormat)unorm_fmt };
		listci.pViewFormats = formats;
		listci.viewFormatCount = formats[1] == VK_FORMAT_UNDEFINED ? 1 : 2;
		if (listci.viewFormatCount > 1) {
			ici.flags = vuk::ImageCreateFlagBits::eMutableFormat;
			ici.pNext = &listci;
		}
		Unique<Image> dst = allocate_image(allocator, ici).value(); // TODO: dropping error
		ImageViewCreateInfo ivci;
		ivci.format = ici.format;
		ivci.image = dst->image;
		ivci.subresourceRange.aspectMask = format_to_aspect(ici.format);
		ivci.subresourceRange.baseArrayLayer = 0;
		ivci.subresourceRange.baseMipLevel = 0;
		ivci.subresourceRange.layerCount = ici.arrayLayers;
		ivci.subresourceRange.levelCount = ici.mipLevels;

		ImageViewType view_type = ici.imageType == ImageType::e3D
			                          ? ImageViewType::e3D
			                          : ici.imageType == ImageType::e2D
			                          ? ImageViewType::e2D
			                          : ImageViewType::e1D;

		if (ici.arrayLayers > 1 && ici.imageType == ImageType::e2D)
			if (ici.flags & ImageCreateFlagBits::eCubeCompatible)
				view_type = ici.arrayLayers > 6 ? ImageViewType::eCubeArray : ImageViewType::eCube;
			else
				view_type = ImageViewType::e2DArray;
		else if (view_type == ImageViewType::e1D && ici.arrayLayers > 1)
			view_type = ImageViewType::e1DArray;

		ivci.viewType = view_type;

		return {
			.image = std::move(dst),
			.view = allocate_image_view(allocator, ivci, loc).value(), // TODO: dropping error
			.extent = ici.extent,
			.format = ici.format,
			.sample_count = ici.samples,
			.level_count = ici.mipLevels,
			.layer_count = ici.arrayLayers
		};
	}

	void Context::destroy(const DescriptorPool& dp) {
		dp.destroy(*this, device);
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

	void Context::destroy(const DescriptorSet&) {
		// no-op, we destroy the pools
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

			this->vkDestroyPipelineCache(device, vk_pipeline_cache, nullptr);

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

	Result<void> Context::wait_idle() {
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

		VkResult result = this->vkDeviceWaitIdle(device);
		if (result < 0) {
			return { expected_error, VkException{ result } };
		}
		return { expected_value };
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
		return create_persistent_descriptorset(allocator, { dslai, dslci, num_descriptors });
	}

	Unique<PersistentDescriptorSet> Context::create_persistent_descriptorset(Allocator& allocator, const PersistentDescriptorSetCreateInfo& ci) {
		Unique<PersistentDescriptorSet> pds(allocator);
		allocator.allocate_persistent_descriptor_sets(std::span{ &*pds, 1 }, std::span{ &ci, 1 });
		return pds;
	}

	Unique<PersistentDescriptorSet>
	Context::create_persistent_descriptorset(Allocator& allocator, const PipelineBaseInfo& base, unsigned set, unsigned num_descriptors) {
		return create_persistent_descriptorset(allocator, { base.layout_info[set], base.dslcis[set], num_descriptors });
	}

	Sampler Context::create(const create_info_t<Sampler>& cinfo) {
		VkSampler s;
		this->vkCreateSampler(device, (VkSamplerCreateInfo*)&cinfo, nullptr, &s);
		return wrap(s);
	}

	DescriptorPool Context::create(const create_info_t<DescriptorPool>& cinfo) {
		return DescriptorPool{};
	}

	Sampler Context::acquire_sampler(const SamplerCreateInfo& sci, uint64_t absolute_frame) {
		return impl->sampler_cache.acquire(sci, absolute_frame);
	}

	DescriptorPool& Context::acquire_descriptor_pool(const DescriptorSetLayoutAllocInfo& dslai, uint64_t absolute_frame) {
		return impl->pool_cache.acquire(dslai, absolute_frame);
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