#if VUK_USE_SHADERC
#include <shaderc/shaderc.hpp>
#endif
#if VUK_USE_DXC
#ifdef _WIN32
// dxcapi.h expects the COM API to be present on Windows.
// On other platforms, the Vulkan SDK will have WinAdapter.h alongside dxcapi.h that is automatically included to stand in for the COM API.
#include <atlbase.h>
#endif
#include <dxc/dxcapi.h>
#define DXC_HR(hr, msg) if (FAILED(hr)) { throw ShaderCompilationException{msg}; }
#endif
#include <algorithm>
#include <fstream>
#include <sstream>
#include <spirv_cross.hpp>

#include "vuk/Context.hpp"
#include "../src/ContextImpl.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/Program.hpp"
#include "vuk/Exception.hpp"
#include "vuk/Allocator.hpp"
#include "vuk/AllocatorHelpers.hpp"

namespace vuk {
	Context::Context(ContextCreateParameters params) :
		instance(params.instance),
		device(params.device),
		physical_device(params.physical_device),
		graphics_queue_family_index(params.graphics_queue_family_index),
		compute_queue_family_index(params.compute_queue_family_index),
		transfer_queue_family_index(params.transfer_queue_family_index),
		debug(*this) {
		impl = new ContextImpl(*this);
		auto queueSubmit2KHR = (PFN_vkQueueSubmit2KHR)vkGetDeviceProcAddr(device, "vkQueueSubmit2KHR");
		assert(queueSubmit2KHR != nullptr);
		if (params.graphics_queue != VK_NULL_HANDLE && params.graphics_queue_family_index != VK_QUEUE_FAMILY_IGNORED) {
			TimelineSemaphore ts;
			impl->device_vk_resource.allocate_timeline_semaphores(std::span{ &ts, 1 }, {});
			dedicated_graphics_queue.emplace(queueSubmit2KHR, params.graphics_queue, params.graphics_queue_family_index, ts);
			graphics_queue = &dedicated_graphics_queue.value();
		}
		if (params.compute_queue != VK_NULL_HANDLE && params.compute_queue_family_index != VK_QUEUE_FAMILY_IGNORED) {
			TimelineSemaphore ts;
			impl->device_vk_resource.allocate_timeline_semaphores(std::span{ &ts, 1 }, {});
			dedicated_compute_queue.emplace(queueSubmit2KHR, params.compute_queue, params.compute_queue_family_index, ts);
			compute_queue = &dedicated_compute_queue.value();
		} else {
			compute_queue = graphics_queue;
		}

		if (params.transfer_queue != VK_NULL_HANDLE && params.transfer_queue_family_index != VK_QUEUE_FAMILY_IGNORED) {
			TimelineSemaphore ts;
			impl->device_vk_resource.allocate_timeline_semaphores(std::span{ &ts, 1 }, {});
			dedicated_transfer_queue.emplace(queueSubmit2KHR, params.transfer_queue, params.transfer_queue_family_index, ts);
			transfer_queue = &dedicated_transfer_queue.value();
		} else {
			transfer_queue = compute_queue ? compute_queue : graphics_queue;
			transfer_queue_family_index = compute_queue ? params.compute_queue_family_index : params.graphics_queue_family_index;
		}
	}

	Queue::Queue(PFN_vkQueueSubmit2KHR fn, VkQueue queue, uint32_t queue_family_index, TimelineSemaphore ts) : queueSubmit2KHR(fn), queue(queue), family_index(queue_family_index), submit_sync(ts) {
	}

	Result<void> Queue::submit(std::span<VkSubmitInfo2KHR> sis, VkFence fence) {
		VkResult result = queueSubmit2KHR(queue, (uint32_t)sis.size(), sis.data(), fence);
		if (result != VK_SUCCESS) {
			return { expected_error, VkException{result} };
		}
		return { expected_value };
	}

	Result<void> Queue::submit(std::span<VkSubmitInfo> sis, VkFence fence) {
		std::lock_guard _(queue_lock);
		VkResult result = vkQueueSubmit(queue, (uint32_t)sis.size(), sis.data(), fence);
		if (result != VK_SUCCESS) {
			return { expected_error, VkException{result} };
		}
		return { expected_value };
	}
	
	Result<void> Context::wait_for_domains(std::span<std::pair<DomainFlags, uint64_t>> queue_waits) {
		std::array<uint32_t, 3> domain_to_sema_index = {~0u, ~0u, ~0u};
		std::array<VkSemaphore, 3> queue_timeline_semaphores;
		std::array<uint64_t, 3> values = {};

		uint32_t count = 0;
		for (auto [domain, v] : queue_waits) {
			auto idx = domain_to_queue_index(domain);
			auto& mapping = domain_to_sema_index[idx];
			if (mapping == -1) {
				mapping = count++;
			}
			auto& q = domain_to_queue(domain);
			queue_timeline_semaphores[mapping] = q.submit_sync.semaphore;
			values[mapping] = values[mapping] > v ? values[mapping] : v;
		}

		VkSemaphoreWaitInfo swi{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
		swi.pSemaphores = queue_timeline_semaphores.data();
		swi.pValues = values.data();
		swi.semaphoreCount = count;
		VkResult result = vkWaitSemaphores(device, &swi, UINT64_MAX);
		for (auto [domain, v] : queue_waits) {
			auto& q = domain_to_queue(domain);
			q.last_host_wait.store(v);
		}
		if (result != VK_SUCCESS) {
			return { expected_error, VkException{result} };
		}
		return { expected_value };
	}

	bool Context::DebugUtils::enabled() {
		return setDebugUtilsObjectNameEXT != nullptr;
	}

	Context::DebugUtils::DebugUtils(Context& ctx) : ctx(ctx) {
		setDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(ctx.device, "vkSetDebugUtilsObjectNameEXT");
		cmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(ctx.device, "vkCmdBeginDebugUtilsLabelEXT");
		cmdEndDebugUtilsLabelEXT = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(ctx.device, "vkCmdEndDebugUtilsLabelEXT");
	}

	void Context::DebugUtils::set_name(const Texture& tex, Name name) {
		if (!enabled()) return;
		set_name(tex.image.get(), name);
		set_name(tex.view.get().payload, name);
	}

	void Context::DebugUtils::begin_region(const VkCommandBuffer& cb, Name name, std::array<float, 4> color) {
		if (!enabled()) return;
		VkDebugUtilsLabelEXT label = { .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
		label.pLabelName = name.c_str();
		::memcpy(label.color, color.data(), sizeof(float) * 4);
		cmdBeginDebugUtilsLabelEXT(cb, &label);
	}

	void Context::DebugUtils::end_region(const VkCommandBuffer& cb) {
		if (!enabled()) return;
		cmdEndDebugUtilsLabelEXT(cb);
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

	LegacyGPUAllocator& Context::get_legacy_gpu_allocator() {
		return impl->legacy_gpu_allocator;
	}

	void PersistentDescriptorSet::update_combined_image_sampler(Context& ctx, unsigned binding, unsigned array_index, ImageView iv, SamplerCreateInfo sci, ImageLayout layout) {
		descriptor_bindings[binding][array_index].image = DescriptorImageInfo(ctx.acquire_sampler(sci, ctx.frame_counter), iv, layout);
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

		switch (cinfo.source.language) {
#if VUK_USE_SHADERC
            case ShaderSourceLanguage::eGlsl: {

				shaderc::Compiler compiler;
				shaderc::CompileOptions options;
				options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);

				const auto result = compiler.CompileGlslToSpv(cinfo.source.as_c_str(), shaderc_glsl_infer_from_source, cinfo.filename.c_str(), options);

				if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
					std::string message = result.GetErrorMessage().c_str();
					throw ShaderCompilationException{ message };
				}

				spirv = std::vector<uint32_t>{result.cbegin(), result.cend()};

				break;
            }
#endif
#if VUK_USE_DXC
            case ShaderSourceLanguage::eHlsl: {
				std::vector<LPCWSTR> arguments;
                arguments.push_back(L"-E");
                arguments.push_back(L"main");
                arguments.push_back(L"-spirv");
				arguments.push_back(L"-fspv-reflect");
                arguments.push_back(L"-fspv-target-env=vulkan1.1");
                arguments.push_back(L"-fvk-use-gl-layout");
                arguments.push_back(L"-no-warnings");
				
				static const std::pair<const char*, HlslShaderStage> inferred[] = {
					{ ".vert.", HlslShaderStage::eVertex },
					{ ".frag.", HlslShaderStage::ePixel },
					{ ".comp.", HlslShaderStage::eCompute },
					{ ".geom.", HlslShaderStage::eGeometry },
					{ ".mesh.", HlslShaderStage::eMesh },
					{ ".hull.", HlslShaderStage::eHull },
					{ ".dom.", HlslShaderStage::eDomain },
                    { ".amp.", HlslShaderStage::eAmplification }
				};

				static const std::unordered_map<HlslShaderStage, LPCWSTR> stage_mappings{
					{ HlslShaderStage::eVertex, L"vs_6_7" },
					{ HlslShaderStage::ePixel, L"ps_6_7" },
					{ HlslShaderStage::eCompute, L"cs_6_7" },
					{ HlslShaderStage::eGeometry, L"gs_6_7" },
					{ HlslShaderStage::eMesh, L"ms_6_7" },
					{ HlslShaderStage::eHull, L"hs_6_7" },
					{ HlslShaderStage::eDomain, L"ds_6_7" },
					{ HlslShaderStage::eAmplification, L"as_6_7" }
				};

				HlslShaderStage shader_stage = cinfo.source.hlsl_stage;
				if (shader_stage == HlslShaderStage::eInferred) {
                    for (const auto& [ext, stage] : inferred) {
                        if (cinfo.filename.contains(ext)) {
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
				
				CComPtr<IDxcResult> result = nullptr;
                DXC_HR(compiler->Compile(&source_buf, arguments.data(), arguments.size(), nullptr, __uuidof(IDxcResult), (void**)&result), "Failed to compile with DXC");
				
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

                break;
			}
#endif
            case ShaderSourceLanguage::eSpirv: {
                spirv = cinfo.source.data;
				break;
			}
		}

		spirv_cross::Compiler refl(spirv.data(), spirv.size());
		Program p;
		auto stage = p.introspect(refl);

		VkShaderModuleCreateInfo moduleCreateInfo{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
		moduleCreateInfo.codeSize = spirv.size() * sizeof(uint32_t);
		moduleCreateInfo.pCode = spirv.data();
		VkShaderModule sm;
		vkCreateShaderModule(device, &moduleCreateInfo, nullptr, &sm);
		std::string name = "ShaderModule: " + cinfo.filename;
		debug.set_name(sm, Name(name));
		return { sm, p, stage };
	}

	PipelineBaseInfo Context::create(const create_info_t<PipelineBaseInfo>& cinfo) {
		fixed_vector<VkPipelineShaderStageCreateInfo, graphics_stage_count> psscis;

		// accumulate descriptors from all stages
		Program accumulated_reflection;
		std::string pipe_name = "Pipeline:";
		for (auto i = 0; i < cinfo.shaders.size(); i++) {
			auto contents = cinfo.shaders[i];
			if (contents.data.empty())
				continue;
			auto& sm = impl->shader_modules.acquire({ contents, cinfo.shader_paths[i] });
			VkPipelineShaderStageCreateInfo shader_stage{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
			shader_stage.pSpecializationInfo = nullptr;
			shader_stage.stage = sm.stage;
			shader_stage.module = sm.shader_module;
			shader_stage.pName = "main"; //TODO: make param
			psscis.push_back(shader_stage);
			accumulated_reflection.append(sm.reflection_info);
			pipe_name += cinfo.shader_paths[i] + "+";
		}
		pipe_name = pipe_name.substr(0, pipe_name.size() - 1); //trim off last "+"

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
		std::array<DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> dslai;
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
		pbi.pipeline_name = Name(pipe_name);
		pbi.reflection_info = accumulated_reflection;
		pbi.binding_flags = cinfo.binding_flags;
		pbi.variable_count_max = cinfo.variable_count_max;
		return pbi;
	}

	ComputePipelineBaseInfo Context::create(const create_info_t<ComputePipelineBaseInfo>& cinfo) {
		VkPipelineShaderStageCreateInfo shader_stage{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
		std::string pipe_name = "Compute:";
		auto& sm = impl->shader_modules.acquire({ cinfo.shader, cinfo.shader_path });
		shader_stage.pSpecializationInfo = nullptr;
		shader_stage.stage = sm.stage;
		shader_stage.module = sm.shader_module;
		shader_stage.pName = "main"; //TODO: make param
		pipe_name += cinfo.shader_path;

		PipelineLayoutCreateInfo plci;
		plci.dslcis = PipelineBaseCreateInfo::build_descriptor_layouts(sm.reflection_info, cinfo);
		// use explicit descriptor layouts if there are any
		for (auto& l : cinfo.explicit_set_layouts) {
			plci.dslcis[l.index] = l;
		}
		plci.pcrs.insert(plci.pcrs.begin(), sm.reflection_info.push_constant_ranges.begin(), sm.reflection_info.push_constant_ranges.end());
		plci.plci.pushConstantRangeCount = (uint32_t)sm.reflection_info.push_constant_ranges.size();
		plci.plci.pPushConstantRanges = sm.reflection_info.push_constant_ranges.data();
		std::array<DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> dslai;
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

		ComputePipelineBaseInfo cpbi;
		cpbi.pssci = shader_stage;
		cpbi.layout_info = dslai;
		cpbi.pipeline_layout = impl->pipeline_layouts.acquire(plci);
		cpbi.pipeline_name = Name(pipe_name);
		cpbi.reflection_info = sm.reflection_info;
		cpbi.binding_flags = cinfo.binding_flags;
		cpbi.variable_count_max = cinfo.variable_count_max;

		return cpbi;
	}

	bool Context::load_pipeline_cache(std::span<std::byte> data) {
		VkPipelineCacheCreateInfo pcci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = data.size_bytes(), .pInitialData = data.data() };
		vkDestroyPipelineCache(device, impl->vk_pipeline_cache, nullptr);
		vkCreatePipelineCache(device, &pcci, nullptr, &impl->vk_pipeline_cache);
		return true;
	}

	std::vector<std::byte> Context::save_pipeline_cache() {
		size_t size;
		std::vector<std::byte> data;
		vkGetPipelineCacheData(device, impl->vk_pipeline_cache, &size, nullptr);
		data.resize(size);
		vkGetPipelineCacheData(device, impl->vk_pipeline_cache, &size, data.data());
		return data;
	}

	Queue& Context::domain_to_queue(DomainFlags domain) {
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

	uint32_t Context::domain_to_queue_index(DomainFlags domain) {
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

	uint32_t Context::domain_to_queue_family_index(DomainFlags domain) {
		return domain_to_queue_index(domain);
	}

	Query Context::create_timestamp_query() {
		return { impl->query_id_counter++ };
	}

	DeviceVkResource& Context::get_vk_resource() {
		return impl->device_vk_resource;
	}

	DescriptorSetLayoutAllocInfo Context::create(const create_info_t<DescriptorSetLayoutAllocInfo>& cinfo) {
		DescriptorSetLayoutAllocInfo ret;
		vkCreateDescriptorSetLayout(device, &cinfo.dslci, nullptr, &ret.layout);
		for (size_t i = 0; i < cinfo.bindings.size(); i++) {
			auto& b = cinfo.bindings[i];
			// if this is not a variable count binding, add it to the descriptor count
			if (cinfo.flags.size() <= i || !(cinfo.flags[i] & to_integral(DescriptorBindingFlagBits::eVariableDescriptorCount))) {
				ret.descriptor_counts[to_integral(b.descriptorType)] += b.descriptorCount;
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
		vkCreatePipelineLayout(device, &cinfo.plci, nullptr, &pl);
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

	void Context::create_named_pipeline(Name name, PipelineBaseCreateInfo ci) {
		std::lock_guard _(impl->named_pipelines_lock);
		impl->named_pipelines.insert_or_assign(name, &impl->pipelinebase_cache.acquire(std::move(ci)));
	}

	void Context::create_named_pipeline(Name name, ComputePipelineBaseCreateInfo ci) {
		std::lock_guard _(impl->named_pipelines_lock);
		impl->named_compute_pipelines.insert_or_assign(name, &impl->compute_pipelinebase_cache.acquire(std::move(ci)));
	}

	PipelineBaseInfo* Context::get_named_pipeline(Name name) {
		std::lock_guard _(impl->named_pipelines_lock);
		return impl->named_pipelines.at(name);
	}

	ComputePipelineBaseInfo* Context::get_named_compute_pipeline(Name name) {
		std::lock_guard _(impl->named_pipelines_lock);
		return impl->named_compute_pipelines.at(name);
	}

	PipelineBaseInfo* Context::get_pipeline(const PipelineBaseCreateInfo& pbci) {
		return &impl->pipelinebase_cache.acquire(pbci);
	}

	ComputePipelineBaseInfo* Context::get_pipeline(const ComputePipelineBaseCreateInfo& pbci) {
		return &impl->compute_pipelinebase_cache.acquire(pbci);
	}

	Program Context::get_pipeline_reflection_info(const PipelineBaseCreateInfo& pci) {
		auto& res = impl->pipelinebase_cache.acquire(pci);
		return res.reflection_info;
	}

	Program Context::get_pipeline_reflection_info(const ComputePipelineBaseCreateInfo& pci) {
		auto& res = impl->compute_pipelinebase_cache.acquire(pci);
		return res.reflection_info;
	}

	ShaderModule Context::compile_shader(ShaderSource source, std::string path) {
		ShaderModuleCreateInfo sci;
		sci.filename = std::move(path);
		sci.source = std::move(source);
		auto sm = impl->shader_modules.remove(sci);
		if (sm) {
			vkDestroyShaderModule(device, sm->shader_module, nullptr);
		}
		return impl->shader_modules.acquire(sci);
	}

	Context::TransientSubmitStub Context::fenced_upload(std::span<UploadItem> uploads, uint32_t dst_queue_family) {
		TransientSubmitBundle* bundle = impl->get_transient_bundle(transfer_queue_family_index);
		TransientSubmitBundle* head_bundle = bundle;
		VkCommandBuffer xfercbuf = impl->get_command_buffer(bundle);
		VkCommandBufferBeginInfo cbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
		vkBeginCommandBuffer(xfercbuf, &cbi);

		bool any_image_transfers = false;

		// size a staging buffer to fit all the uploads, with proper alignment
		size_t size = 0;
		size_t biggest_align = 1;
		for (auto& upload : uploads) {
			if (!upload.is_buffer) {
				size = align_up(size, (size_t)format_to_texel_block_size(upload.image.format));
				size += upload.image.data.size();
				biggest_align = std::max(biggest_align, (size_t)format_to_texel_block_size(upload.image.format));
				any_image_transfers = true;
			} else {
				size += upload.buffer.data.size();
			}
		}

		VkCommandBuffer dstcbuf;
		TransientSubmitBundle* dst_bundle = nullptr;
		// image transfers will finish on the dst queue, get a bundle for them and hook it up to our transfer bundle
		if (any_image_transfers) {
			dst_bundle = impl->get_transient_bundle(dst_queue_family);
			dst_bundle->next = bundle;
			dstcbuf = impl->get_command_buffer(dst_bundle);
			VkCommandBufferBeginInfo cbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
			vkBeginCommandBuffer(dstcbuf, &cbi);
			head_bundle = dst_bundle;
		}

		// create 1 big staging buffer
		// we need to have this aligned for the first upload
		// as that corresponds to offset = 0
		bundle->buffer = impl->legacy_gpu_allocator.allocate_buffer(MemoryUsage::eCPUonly, BufferUsageFlagBits::eTransferRead, size, biggest_align, true);
		auto staging = bundle->buffer;

		for (auto& upload : uploads) {
			// realign offset
			auto aligned = upload.is_buffer ? staging.offset : align_up(staging.offset, (size_t)format_to_texel_block_size(upload.image.format));
			auto delta = aligned - staging.offset;
			staging.offset = aligned;
			staging.mapped_ptr = staging.mapped_ptr + delta;

			// copy to staging
			auto& data = upload.is_buffer ? upload.buffer.data : upload.image.data;
			::memcpy(staging.mapped_ptr, data.data(), data.size());

			if (upload.is_buffer) {
				VkBufferCopy bc;
				bc.dstOffset = upload.buffer.dst.offset;
				bc.srcOffset = staging.offset;
				bc.size = upload.buffer.data.size();
				vkCmdCopyBuffer(xfercbuf, staging.buffer, upload.buffer.dst.buffer, 1, &bc);
			} else {
				// perform buffer->image copy on the xfer queue
				VkBufferImageCopy bc;
				bc.bufferOffset = staging.offset;
				bc.imageOffset = VkOffset3D{ 0, 0, 0 };
				bc.bufferRowLength = 0;
				bc.bufferImageHeight = 0;
				bc.imageExtent = upload.image.extent;
				bc.imageSubresource.aspectMask = (VkImageAspectFlags)format_to_aspect(upload.image.format);
				bc.imageSubresource.baseArrayLayer = upload.image.base_array_layer;
				bc.imageSubresource.mipLevel = upload.image.mip_level;
				bc.imageSubresource.layerCount = 1;

				VkImageMemoryBarrier copy_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
				copy_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				copy_barrier.oldLayout = (VkImageLayout)ImageLayout::eUndefined;
				copy_barrier.newLayout = (VkImageLayout)ImageLayout::eTransferDstOptimal;
				copy_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				copy_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				copy_barrier.image = upload.image.dst;
				copy_barrier.subresourceRange.aspectMask = bc.imageSubresource.aspectMask;
				copy_barrier.subresourceRange.layerCount = bc.imageSubresource.layerCount;
				copy_barrier.subresourceRange.baseArrayLayer = bc.imageSubresource.baseArrayLayer;
				copy_barrier.subresourceRange.baseMipLevel = bc.imageSubresource.mipLevel;
				copy_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;

				vkCmdPipelineBarrier(xfercbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &copy_barrier);
				vkCmdCopyBufferToImage(xfercbuf, staging.buffer, upload.image.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);

				if (!upload.image.generate_mips) {
					// transition the mips to SROO on xfer & release to dst_queue_family
					VkImageMemoryBarrier release_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };;
					release_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
					release_barrier.dstAccessMask = 0; // ignored
					release_barrier.oldLayout = (VkImageLayout)ImageLayout::eTransferDstOptimal;
					release_barrier.newLayout = (VkImageLayout)ImageLayout::eShaderReadOnlyOptimal;
					release_barrier.dstQueueFamilyIndex = transfer_queue_family_index;
					release_barrier.srcQueueFamilyIndex = dst_queue_family;
					release_barrier.image = upload.image.dst;
					release_barrier.subresourceRange = copy_barrier.subresourceRange;
					vkCmdPipelineBarrier(xfercbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &release_barrier);

					// acquire on dst_queue_family
					VkImageMemoryBarrier acq_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };;
					acq_barrier.srcAccessMask = 0; // ignored
					acq_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; // TODO: maybe memory read?
					acq_barrier.oldLayout = (VkImageLayout)ImageLayout::eTransferDstOptimal;
					acq_barrier.newLayout = (VkImageLayout)ImageLayout::eShaderReadOnlyOptimal;
					acq_barrier.dstQueueFamilyIndex = transfer_queue_family_index;
					acq_barrier.srcQueueFamilyIndex = dst_queue_family;
					acq_barrier.image = upload.image.dst;
					acq_barrier.subresourceRange = copy_barrier.subresourceRange;
					// no wait, no delay, we wait on host
					vkCmdPipelineBarrier(dstcbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &acq_barrier);
				} else {
					// for now, we blit, which requires the gfx queue
					assert(dst_queue_family == graphics_queue_family_index);
					// release to dst_queue_family
					VkImageMemoryBarrier release_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };;
					release_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
					release_barrier.dstAccessMask = 0; // ignored
					release_barrier.oldLayout = (VkImageLayout)ImageLayout::eTransferDstOptimal;
					release_barrier.newLayout = (VkImageLayout)ImageLayout::eTransferDstOptimal;
					release_barrier.srcQueueFamilyIndex = transfer_queue_family_index;
					release_barrier.dstQueueFamilyIndex = dst_queue_family;
					release_barrier.image = upload.image.dst;
					release_barrier.subresourceRange = copy_barrier.subresourceRange;
					vkCmdPipelineBarrier(xfercbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &release_barrier);

					// acquire on dst_queue_family
					VkImageMemoryBarrier acq_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };;
					acq_barrier.srcAccessMask = 0; // ignored
					acq_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
					acq_barrier.oldLayout = (VkImageLayout)ImageLayout::eTransferDstOptimal;
					acq_barrier.newLayout = (VkImageLayout)ImageLayout::eTransferDstOptimal;
					acq_barrier.srcQueueFamilyIndex = transfer_queue_family_index;
					acq_barrier.dstQueueFamilyIndex = dst_queue_family;
					acq_barrier.image = upload.image.dst;
					acq_barrier.subresourceRange = copy_barrier.subresourceRange;

					// no wait, no delay, sync'd by the sema
					vkCmdPipelineBarrier(dstcbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acq_barrier);
					MipGenerateCommand task;
					task.dst = upload.image.dst;
					task.extent = upload.image.extent;
					task.base_array_layer = upload.image.base_array_layer;
					task.base_mip_level = upload.image.mip_level;
					task.layer_count = 1;
					task.format = upload.image.format;
					record_mip_gen(dstcbuf, task, ImageLayout::eTransferDstOptimal);
				}
			}

			staging.offset += data.size();
			staging.mapped_ptr = staging.mapped_ptr + data.size();
		}
		vkEndCommandBuffer(xfercbuf);

		// get a fence, submit command buffer
		VkFence fence = impl->get_unpooled_fence();
		VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
		si.commandBufferCount = 1;
		si.pCommandBuffers = &xfercbuf;
		if (!any_image_transfers) {
			// only buffers, single submit to transfer, fence waits on single cbuf
			submit_transfer(std::span{ &si, 1 }, fence);
		} else {
			vkEndCommandBuffer(dstcbuf);
			// buffers and images, submit to transfer, signal sema
			si.signalSemaphoreCount = 1;
			auto sema = impl->get_unpooled_sema();
			si.pSignalSemaphores = &sema;
			submit_transfer(std::span{ &si, 1 }, VkFence{ VK_NULL_HANDLE });
			// second submit, to dst queue ideally, but for now to graphics
			si.signalSemaphoreCount = 0;
			si.waitSemaphoreCount = 1;
			si.pWaitSemaphores = &sema;
			// mipping happens in STAGE_TRANSFER for now
			VkPipelineStageFlags wait = VK_PIPELINE_STAGE_TRANSFER_BIT;
			si.pWaitDstStageMask = &wait;
			si.pCommandBuffers = &dstcbuf;
			// stash semaphore
			head_bundle->sema = sema;
			// submit with fence
			submit_graphics(std::span{ &si, 1 }, fence);
		}

		head_bundle->fence = fence;
		return head_bundle;
	}

	bool Context::poll_upload(TransientSubmitBundle* ur) {
		if (vkGetFenceStatus(device, ur->fence) == VK_SUCCESS) {
			std::lock_guard _(impl->transient_submit_lock);
			impl->cleanup_transient_bundle_recursively(ur);
			return true;
		} else {
			return false;
		}
	}

	Texture Context::allocate_texture(Allocator& allocator, ImageCreateInfo ici) {
		ici.imageType = ici.extent.depth > 1 ? ImageType::e3D :
			ici.extent.height > 1 ? ImageType::e2D :
			ImageType::e1D;
		Unique<Image> dst = allocate_image(allocator, ici).value(); // TODO: dropping error
		ImageViewCreateInfo ivci;
		ivci.format = ici.format;
		ivci.image = *dst;
		ivci.subresourceRange.aspectMask = format_to_aspect(ici.format);
		ivci.subresourceRange.baseArrayLayer = 0;
		ivci.subresourceRange.baseMipLevel = 0;
		ivci.subresourceRange.layerCount = 1;
		ivci.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
		ivci.viewType = ici.imageType == ImageType::e3D ? ImageViewType::e3D :
			ici.imageType == ImageType::e2D ? ImageViewType::e2D :
			ImageViewType::e1D;
		Texture tex{ std::move(dst), allocate_image_view(allocator, ivci).value() }; // TODO: dropping error
		tex.extent = ici.extent;
		tex.format = ici.format;
		tex.sample_count = ici.samples;
		return tex;
	}

	void Context::destroy(const RGImage& image) {
		vkDestroyImageView(device, image.image_view.payload, nullptr);
		impl->legacy_gpu_allocator.destroy_image(image.image);
	}

	void Context::destroy(const LegacyPoolAllocator& v) {
		impl->legacy_gpu_allocator.destroy(v);
	}

	void Context::destroy(const LegacyLinearAllocator& v) {
		impl->legacy_gpu_allocator.destroy(v);
	}

	void Context::destroy(const DescriptorPool& dp) {
		for (auto& p : dp.pools) {
			vkDestroyDescriptorPool(device, p, nullptr);
		}
	}

	void Context::destroy(const PipelineInfo& pi) {
		vkDestroyPipeline(device, pi.pipeline, nullptr);
	}

	void Context::destroy(const ComputePipelineInfo& pi) {
		vkDestroyPipeline(device, pi.pipeline, nullptr);
	}

	void Context::destroy(const ShaderModule& sm) {
		vkDestroyShaderModule(device, sm.shader_module, nullptr);
	}

	void Context::destroy(const DescriptorSetLayoutAllocInfo& ds) {
		vkDestroyDescriptorSetLayout(device, ds.layout, nullptr);
	}

	void Context::destroy(const VkPipelineLayout& pl) {
		vkDestroyPipelineLayout(device, pl, nullptr);
	}

	void Context::destroy(const VkRenderPass& rp) {
		vkDestroyRenderPass(device, rp, nullptr);
	}

	void Context::destroy(const DescriptorSet&) {
		// no-op, we destroy the pools
	}

	void Context::destroy(const VkFramebuffer& fb) {
		vkDestroyFramebuffer(device, fb, nullptr);
	}

	void Context::destroy(const Sampler& sa) {
		vkDestroySampler(device, sa.payload, nullptr);
	}

	void Context::destroy(const PipelineBaseInfo& pbi) {
		// no-op, we don't own device objects
	}

	void Context::destroy(const ComputePipelineBaseInfo& pbi) {
		// no-op, we don't own device objects
	}

	Context::~Context() {
		vkDeviceWaitIdle(device);

		for (auto& s : impl->swapchains) {
			for (auto& swiv : s.image_views) {
				vkDestroyImageView(device, swiv.payload, nullptr);
			}
			vkDestroySwapchainKHR(device, s.swapchain, nullptr);
		}
		for (auto& cp : impl->transient_submit_bundles) {
			if (cp.cpool != VK_NULL_HANDLE) {
				vkDestroyCommandPool(device, cp.cpool, nullptr);
			}
		}
		vkDestroyPipelineCache(device, impl->vk_pipeline_cache, nullptr);

		if (dedicated_graphics_queue) {
			impl->device_vk_resource.deallocate_timeline_semaphores(std::span{ &dedicated_graphics_queue->submit_sync, 1 });
		}

		if (dedicated_compute_queue) {
			impl->device_vk_resource.deallocate_timeline_semaphores(std::span{ &dedicated_compute_queue->submit_sync, 1 });
		}

		if (dedicated_transfer_queue) {
			impl->device_vk_resource.deallocate_timeline_semaphores(std::span{ &dedicated_transfer_queue->submit_sync, 1 });
		}

		delete impl;
	}

	void Context::next_frame() {
		frame_counter++;
		collect(frame_counter);
	}

	void Context::wait_idle() {
		std::unique_lock<std::mutex> graphics_lock;
		if (dedicated_graphics_queue) {
			graphics_lock = std::unique_lock{ graphics_queue->queue_lock };
		}
		std::unique_lock<std::mutex> compute_lock;
		if (dedicated_compute_queue) {
			compute_lock = std::unique_lock{ compute_queue->queue_lock };
		}
		std::unique_lock<std::mutex> transfer_lock;
		if (dedicated_transfer_queue) {
			transfer_lock = std::unique_lock{ transfer_queue->queue_lock };
		}

		vkDeviceWaitIdle(device);
	}

	ImageView Context::wrap(VkImageView iv, ImageViewCreateInfo ivci) {
		ImageView viv{ .payload = iv };
		viv.base_layer = ivci.subresourceRange.baseArrayLayer;
		viv.layer_count = ivci.subresourceRange.layerCount;
		viv.base_level = ivci.subresourceRange.baseMipLevel;
		viv.level_count = ivci.subresourceRange.levelCount;
		viv.format = ivci.format;
		viv.type = ivci.viewType;
		viv.image = ivci.image;
		viv.components = ivci.components;
		viv.id = unique_handle_id_counter++;
		return viv;
	}

	Unique<ImageView> Unique<ImageView>::SubrangeBuilder::apply() {
		ImageViewCreateInfo ivci;
		ivci.viewType = type == ImageViewType(0xdeadbeef) ? iv.type : type;
		ivci.subresourceRange.baseMipLevel = base_level == 0xdeadbeef ? iv.base_level : base_level;
		ivci.subresourceRange.levelCount = level_count == 0xdeadbeef ? iv.level_count : level_count;
		ivci.subresourceRange.baseArrayLayer = base_layer == 0xdeadbeef ? iv.base_layer : base_layer;
		ivci.subresourceRange.layerCount = layer_count == 0xdeadbeef ? iv.layer_count : layer_count;
		ivci.subresourceRange.aspectMask = format_to_aspect(iv.format);
		ivci.image = iv.image;
		ivci.format = iv.format;
		ivci.components = iv.components;
		return allocate_image_view(*allocator, ivci).value(); // TODO: dropping error
	}

	void Context::collect(uint64_t frame) {
		impl->collect(frame);
	}

	TransferStub Context::enqueue_transfer(Buffer src, Buffer dst) {
		std::lock_guard _(impl->transfer_mutex);
		TransferStub stub{ transfer_id++ };
		impl->buffer_transfer_commands.push({ src, dst, stub });
		return stub;
	}

	TransferStub Context::enqueue_transfer(Buffer src, Image dst, Extent3D extent, uint32_t base_layer, bool generate_mips) {
		std::lock_guard _(impl->transfer_mutex);
		TransferStub stub{ transfer_id++ };
		// TODO: expose extra transfer knobs
		impl->bufferimage_transfer_commands.push({ src, dst, extent, base_layer, 1, 0, generate_mips, stub });
		return stub;
	}

	Unique<PersistentDescriptorSet> Context::create_persistent_descriptorset(Allocator& allocator, DescriptorSetLayoutCreateInfo dslci,
		unsigned num_descriptors) {
		dslci.dslci.bindingCount = (uint32_t)dslci.bindings.size();
		dslci.dslci.pBindings = dslci.bindings.data();
		VkDescriptorSetLayoutBindingFlagsCreateInfo dslbfci{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
		if (dslci.flags.size() > 0) {
			dslbfci.bindingCount = (uint32_t)dslci.bindings.size();
			dslbfci.pBindingFlags = dslci.flags.data();
			dslci.dslci.pNext = &dslbfci;
		}
		auto& dslai = impl->descriptor_set_layouts.acquire(dslci, frame_counter);
		return create_persistent_descriptorset(allocator, { dslai, num_descriptors });
	}

	Unique<PersistentDescriptorSet> Context::create_persistent_descriptorset(Allocator& allocator, const PersistentDescriptorSetCreateInfo& ci) {
		Unique<PersistentDescriptorSet> pds(allocator);
		allocator.allocate_persistent_descriptor_sets(std::span{ &*pds, 1 }, std::span{ &ci, 1 });
		return pds;
	}

	Unique<PersistentDescriptorSet> Context::create_persistent_descriptorset(Allocator& allocator, const PipelineBaseInfo& base, unsigned set, unsigned num_descriptors) {
		return create_persistent_descriptorset(allocator, { base.layout_info[set], num_descriptors });
	}

	Unique<PersistentDescriptorSet> Context::create_persistent_descriptorset(Allocator& allocator, const ComputePipelineBaseInfo& base, unsigned set, unsigned num_descriptors) {
		return create_persistent_descriptorset(allocator, { base.layout_info[set], num_descriptors });
	}

	void Context::commit_persistent_descriptorset(PersistentDescriptorSet& array) {
		vkUpdateDescriptorSets(device, (uint32_t)array.pending_writes.size(), array.pending_writes.data(), 0, nullptr);
		array.pending_writes.clear();
	}

	size_t Context::get_allocation_size(Buffer buf) {
		return impl->legacy_gpu_allocator.get_allocation_size(buf);
	}

	// TODO: replace transfer machinery
	void Context::dma_task(Allocator& allocator) {
		std::lock_guard _(impl->transfer_mutex);
		while (!impl->pending_transfers.empty() && vkGetFenceStatus(device, impl->pending_transfers.front().fence) == VK_SUCCESS) {
			auto last = impl->pending_transfers.front();
			last_transfer_complete = last.last_transfer_id;
			impl->pending_transfers.pop();
		}

		if (impl->buffer_transfer_commands.empty() && impl->bufferimage_transfer_commands.empty()) return;
		auto cpool = *allocate_command_pool(allocator, { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, .queueFamilyIndex = graphics_queue_family_index });
		auto cbuf = *allocate_command_buffer(allocator, { .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .command_pool = *cpool});
		VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		vkBeginCommandBuffer(*cbuf, &cbi);
		size_t last = 0;
		while (!impl->buffer_transfer_commands.empty()) {
			auto task = impl->buffer_transfer_commands.front();
			impl->buffer_transfer_commands.pop();
			VkBufferCopy bc;
			bc.dstOffset = task.dst.offset;
			bc.srcOffset = task.src.offset;
			bc.size = task.src.size;
			vkCmdCopyBuffer(*cbuf, task.src.buffer, task.dst.buffer, 1, &bc);
			last = std::max(last, task.stub.id);
		}
		while (!impl->bufferimage_transfer_commands.empty()) {
			auto task = impl->bufferimage_transfer_commands.front();
			impl->bufferimage_transfer_commands.pop();
			record_buffer_image_copy(cbuf->command_buffer, task);
			last = std::max(last, task.stub.id);
		}
		vkEndCommandBuffer(*cbuf);
		auto fence = *allocate_fence(allocator);
		VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
		si.commandBufferCount = 1;
		si.pCommandBuffers = &cbuf->command_buffer;
		submit_graphics(std::span{ &si, 1 }, *fence);
		impl->pending_transfers.emplace(PendingTransfer{ last, *fence });
	}

	void Context::wait_all_transfers(Allocator& allocator) {
		// wait all transfers, immediately
		dma_task(allocator);
		while (!impl->pending_transfers.empty()) {
			vkWaitForFences(device, 1, &impl->pending_transfers.front().fence, true, UINT64_MAX);
			auto last = impl->pending_transfers.front();
			last_transfer_complete = last.last_transfer_id;
			impl->pending_transfers.pop();
		}
	}

	RGImage Context::create(const create_info_t<RGImage>& cinfo) {
		RGImage res{};
		res.image = impl->legacy_gpu_allocator.create_image_for_rendertarget(cinfo.ici);
		auto ivci = cinfo.ivci;
		ivci.image = res.image;
		std::string name = std::string("Image: RenderTarget ") + std::string(cinfo.name.to_sv());
		debug.set_name(res.image, Name(name));
		name = std::string("ImageView: RenderTarget ") + std::string(cinfo.name.to_sv());
		// skip creating image views for images that can't be viewed
		if (cinfo.ici.usage & (ImageUsageFlagBits::eColorAttachment | ImageUsageFlagBits::eDepthStencilAttachment | ImageUsageFlagBits::eInputAttachment | ImageUsageFlagBits::eSampled | ImageUsageFlagBits::eStorage)) {
			VkImageView iv;
			vkCreateImageView(device, (VkImageViewCreateInfo*)&ivci, nullptr, &iv);
			res.image_view = wrap(iv, ivci);
			debug.set_name(res.image_view.payload, Name(name));
		}
		return res;
	}

	VkRenderPass Context::create(const create_info_t<VkRenderPass>& cinfo) {
		VkRenderPass rp;
		vkCreateRenderPass(device, &cinfo, nullptr, &rp);
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
		VkPipelineInputAssemblyStateCreateInfo input_assembly_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = cinfo.topology, .primitiveRestartEnable = cinfo.primitive_restart_enable };
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
		VkPipelineColorBlendStateCreateInfo color_blend_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.attachmentCount = cinfo.attachmentCount
		};
		auto default_writemask = ColorComponentFlagBits::eR | ColorComponentFlagBits::eG | ColorComponentFlagBits::eB | ColorComponentFlagBits::eA;
		std::vector<VkPipelineColorBlendAttachmentState> pcbas(cinfo.attachmentCount,
			VkPipelineColorBlendAttachmentState{
				.blendEnable = false,
				.colorWriteMask = (VkColorComponentFlags)default_writemask
			}
		);
		if (cinfo.records.color_blend_attachments) {
			if (!cinfo.records.broadcast_color_blend_attachment_0) {
				for (auto& pcba : pcbas) {
					auto compressed = read<PipelineInstanceCreateInfo::PipelineColorBlendAttachmentState>(data_ptr);
					pcba = {
						compressed.blendEnable,
						(VkBlendFactor)compressed.srcColorBlendFactor,
						(VkBlendFactor)compressed.dstColorBlendFactor,
						(VkBlendOp)compressed.colorBlendOp,
						(VkBlendFactor)compressed.srcAlphaBlendFactor,
						(VkBlendFactor)compressed.dstAlphaBlendFactor,
						(VkBlendOp)compressed.alphaBlendOp,
						compressed.colorWriteMask
					};
				}
			} else { // handle broadcast
				auto compressed = read<PipelineInstanceCreateInfo::PipelineColorBlendAttachmentState>(data_ptr);
				for (auto& pcba : pcbas) {
					pcba = {
						compressed.blendEnable,
						(VkBlendFactor)compressed.srcColorBlendFactor,
						(VkBlendFactor)compressed.dstColorBlendFactor,
						(VkBlendOp)compressed.colorBlendOp,
						(VkBlendFactor)compressed.srcAlphaBlendFactor,
						(VkBlendFactor)compressed.dstAlphaBlendFactor,
						(VkBlendOp)compressed.alphaBlendOp,
						compressed.colorWriteMask
					};
				}
			}
		}
		if (cinfo.records.logic_op) {
			auto compressed = read<PipelineInstanceCreateInfo::BlendStateLogicOp>(data_ptr);
			color_blend_state.logicOpEnable = true;
			color_blend_state.logicOp = compressed.logic_op;
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
						specialization_map_entries.emplace_back(VkSpecializationMapEntry{
							sc.binding,
							data_offset,
							size
							});
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
		VkPipelineRasterizationStateCreateInfo rasterization_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = cinfo.cullMode,
			.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
			.lineWidth = 1.f
		};
		if (cinfo.records.non_trivial_raster_state) {
			auto rs = read<PipelineInstanceCreateInfo::RasterizationState>(data_ptr);
			rasterization_state = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.depthClampEnable = rs.depthClampEnable,
			.rasterizerDiscardEnable = rs.rasterizerDiscardEnable,
			.polygonMode = (VkPolygonMode)rs.polygonMode,
			.cullMode = cinfo.cullMode,
			.frontFace = (VkFrontFace)rs.frontFace,
			.lineWidth = 1.f
			};
		}
		if (cinfo.records.depth_bias) {
			auto db = read<PipelineInstanceCreateInfo::DepthBias>(data_ptr);
			rasterization_state.depthBiasEnable = true;
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
		VkPipelineMultisampleStateCreateInfo multisample_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
		if (cinfo.records.more_than_one_sample) {
			auto ms = read<PipelineInstanceCreateInfo::Multisample>(data_ptr);
			multisample_state.rasterizationSamples = ms.rasterization_samples;
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
			viewports = reinterpret_cast<const VkViewport*>(data_ptr);
			data_ptr += num_viewports * sizeof(VkViewport);
		}

		// SCISSORS
		const VkRect2D* scissors = nullptr;
		uint8_t num_scissors = 1;
		if (cinfo.records.scissors) {
			num_scissors = read<uint8_t>(data_ptr);
			scissors = reinterpret_cast<const VkRect2D*>(data_ptr);
			data_ptr += num_scissors * sizeof(VkRect2D);
		}

		VkPipelineViewportStateCreateInfo viewport_state{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
		viewport_state.pViewports = viewports;
		viewport_state.viewportCount = num_viewports;
		viewport_state.pScissors = scissors;
		viewport_state.scissorCount = num_scissors;
		gpci.pViewportState = &viewport_state;

		VkPipelineDynamicStateCreateInfo dynamic_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
		dynamic_state.dynamicStateCount = std::popcount(cinfo.dynamic_state_flags.m_mask);
		fixed_vector<VkDynamicState, VkDynamicState::VK_DYNAMIC_STATE_DEPTH_BOUNDS> dyn_states;
		uint64_t dyn_state_cnt = 0;
		uint64_t mask = cinfo.dynamic_state_flags.m_mask;
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
		VkResult res = vkCreateGraphicsPipelines(device, impl->vk_pipeline_cache, 1, &gpci, nullptr, &pipeline);
		assert(res == VK_SUCCESS);
		debug.set_name(pipeline, cinfo.base->pipeline_name);
		return { pipeline, gpci.layout, cinfo.base->layout_info };
	}

	ComputePipelineInfo Context::create(const create_info_t<ComputePipelineInfo>& cinfo) {
		// create compute pipeline
		VkComputePipelineCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
		cpci.layout = cinfo.base->pipeline_layout;
		cpci.stage = cinfo.base->pssci;

		VkPipeline pipeline;
		VkResult res = vkCreateComputePipelines(device, impl->vk_pipeline_cache, 1, &cpci, nullptr, &pipeline);
		assert(res == VK_SUCCESS);
		debug.set_name(pipeline, cinfo.base->pipeline_name);
		return { pipeline, cpci.layout, cinfo.base->layout_info, cinfo.base->reflection_info.local_size };
	}


	Sampler Context::create(const create_info_t<Sampler>& cinfo) {
		VkSampler s;
		vkCreateSampler(device, (VkSamplerCreateInfo*)&cinfo, nullptr, &s);
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
			auto result = vkGetQueryPoolResults(device, pool.pool, 0, pool.count, sizeof(uint64_t) * pool.count, host_values.data(), sizeof(uint64_t), VkQueryResultFlagBits::VK_QUERY_RESULT_64_BIT | VkQueryResultFlagBits::VK_QUERY_RESULT_WAIT_BIT);
			if (result != VK_SUCCESS) {
				return { expected_error, AllocateException{result} };
			}

			for (uint64_t i = 0; i < pool.count; i++) {
				impl->timestamp_result_map.emplace(pool.queries[i], host_values[i]);
			}
		}

		return { expected_value };
	}
}

