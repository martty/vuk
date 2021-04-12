#include <vuk/Context.hpp>
#include <vuk/GlobalAllocator.hpp>
#include <vuk/Partials.hpp>
#include <Cache.hpp>
#include <Allocator.hpp>

#if VUK_USE_SHADERC
#include <shaderc/shaderc.hpp>
#endif
#include <vuk/Program.hpp>
#include <vuk/Exception.hpp>
#include <RGImage.hpp>
#include <Allocator.hpp>
#include <fstream>
#include <sstream>
#include <spirv_cross.hpp>

namespace vuk {
	struct GlobalAllocatorImpl {
		GlobalAllocator& ga;
		VkDevice device;

		GlobalAllocatorImpl(GlobalAllocator& ga) :
			ga(ga),
			device(ga.device),
			pipelinebase_cache(ga),
			pipeline_cache(ga),
			compute_pipeline_cache(ga),
			renderpass_cache(ga),
			framebuffer_cache(ga),
			transient_images(ga),
			pool_cache(ga),
			sampler_cache(ga),
			shader_modules(ga),
			descriptor_set_layouts(ga),
			pipeline_layouts(ga) {

		}

		// caches serving allocations
		Cache<PipelineBaseInfo> pipelinebase_cache;
		Cache<PipelineInfo> pipeline_cache;
		Cache<ComputePipelineInfo> compute_pipeline_cache;
		Cache<VkRenderPass> renderpass_cache;
		Cache<VkFramebuffer> framebuffer_cache;
		Cache<RGImage> transient_images;
		Cache<vuk::DescriptorPool> pool_cache;
		Cache<vuk::Sampler> sampler_cache;
		Cache<vuk::ShaderModule> shader_modules;
		Cache<vuk::DescriptorSetLayoutAllocInfo> descriptor_set_layouts;
		Cache<VkPipelineLayout> pipeline_layouts;
	};

	GlobalAllocator::GlobalAllocator(Context& ctx) : Allocator(ctx),
		device_memory_allocator(new DeviceMemoryAllocator(ctx.instance, ctx.device, ctx.physical_device, ctx.graphics_queue_family_index, ctx.transfer_queue_family_index)), device(ctx.device), debug_utils(new DebugUtils(device)) {
		impl = new GlobalAllocatorImpl(*this);
		VkPipelineCacheCreateInfo pcci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
		vkCreatePipelineCache(ctx.device, &pcci, nullptr, &vk_pipeline_cache);
		vkGetPhysicalDeviceProperties(ctx.physical_device, &ctx.physical_device_properties);
	}

	VkFence GlobalAllocator::allocate_fence(uint64_t absolute_frame, SourceLocation) {
		return create(FenceCreateInfo{}, absolute_frame);
	}

	VkCommandBuffer GlobalAllocator::allocate_command_buffer(const CommandBufferCreateInfo& cinfo, uint64_t absolute_frame, SourceLocation) {
		return create(cinfo, absolute_frame);
	}

	VkSemaphore GlobalAllocator::allocate_semaphore(uint64_t absolute_frame, SourceLocation) {
		return create(SemaphoreCreateInfo{}, absolute_frame);
	}

	VkSemaphore GlobalAllocator::allocate_timeline_semaphore(uint64_t initial_value, uint64_t absolute_frame, SourceLocation) {
		return create(SemaphoreCreateInfo{ .timeline = true, .initial_value = initial_value }, absolute_frame);
	}

	VkFramebuffer GlobalAllocator::allocate_framebuffer(const vuk::FramebufferCreateInfo& fbci, uint64_t absolute_frame, SourceLocation) {
		return impl->framebuffer_cache.acquire(fbci, absolute_frame);
	}

	VkRenderPass GlobalAllocator::allocate_renderpass(const vuk::RenderPassCreateInfo& rpci, uint64_t absolute_frame, SourceLocation) {
		return impl->renderpass_cache.acquire(rpci, absolute_frame);
	}

	vuk::RGImage GlobalAllocator::allocate_rendertarget(const vuk::RGCI& rgci, uint64_t absolute_frame, SourceLocation) {
		return impl->transient_images.acquire(rgci, absolute_frame);
	}

	vuk::Sampler GlobalAllocator::allocate_sampler(const vuk::SamplerCreateInfo& sci, uint64_t absolute_frame, SourceLocation) {
		return impl->sampler_cache.acquire(sci);
	}

	vuk::DescriptorSet GlobalAllocator::allocate_descriptorset(const vuk::SetBinding& sb, uint64_t absolute_frame, SourceLocation) {
		return create(sb, absolute_frame);
	}

	vuk::PipelineInfo GlobalAllocator::allocate_pipeline(const vuk::PipelineInstanceCreateInfo& pici, uint64_t absolute_frame, SourceLocation) {
		return impl->pipeline_cache.acquire(pici, absolute_frame);
	}

	PipelineBaseInfo& GlobalAllocator::allocate_pipeline_base(const PipelineBaseCreateInfo& pbci, uint64_t absolute_frame, SourceLocation) {
		return impl->pipelinebase_cache.acquire(pbci, absolute_frame);
	}

	RGImage GlobalAllocator::create(const create_info_t<RGImage>& cinfo, uint64_t absolute_frame) {
		RGImage res{};
		res.image = device_memory_allocator->create_image_for_rendertarget(cinfo.ici);
		auto ivci = cinfo.ivci;
		ivci.image = res.image;
		std::string name = std::string("Image: RenderTarget ") + std::string(cinfo.name.to_sv());
		debug_utils->set_name(res.image, Name(name));
		name = std::string("ImageView: RenderTarget ") + std::string(cinfo.name.to_sv());
		// skip creating image views for images that can't be viewed
		if (cinfo.ici.usage & (ImageUsageFlagBits::eColorAttachment | ImageUsageFlagBits::eDepthStencilAttachment | ImageUsageFlagBits::eInputAttachment | ImageUsageFlagBits::eSampled | ImageUsageFlagBits::eStorage)) {
			VkImageView iv;
			vkCreateImageView(device, (VkImageViewCreateInfo*)&ivci, nullptr, &iv);
			res.image_view = wrap(iv);
			debug_utils->set_name(res.image_view.payload, Name(name));
		}
		return res;
	}

	void GlobalAllocator::destroy(const RGImage& image) {
		vkDestroyImageView(device, image.image_view.payload, nullptr);
		device_memory_allocator->destroy_image(image.image);
	}

	void GlobalAllocator::destroy(const PoolAllocator& v) {
		device_memory_allocator->destroy(v);
	}

	void GlobalAllocator::destroy(const LinearAllocator& v) {
		device_memory_allocator->destroy(v);
	}

	void GlobalAllocator::destroy(const Buffer& buffer) {
		// TODO: zombie list
	}

	VkFence GlobalAllocator::create(const FenceCreateInfo& cinfo, uint64_t absolute_frame) {
		VkFence fence;
		VkFenceCreateInfo fci{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
		vkCreateFence(device, &fci, nullptr, &fence);
		return fence;
	}

	VkSemaphore GlobalAllocator::create(const SemaphoreCreateInfo& cinfo, uint64_t absolute_frame) {
		VkSemaphore sema;
		VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
		if (cinfo.timeline) {
			VkSemaphoreTypeCreateInfo stci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
			stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
			stci.initialValue = cinfo.initial_value;
			sci.pNext = &stci;
		}
		vkCreateSemaphore(device, &sci, nullptr, &sema);
		return sema;
	}

	VkCommandBuffer GlobalAllocator::create(const CommandBufferCreateInfo& cinfo, uint64_t absolute_frame) {
		// TODO:
		assert(0);
		return VK_NULL_HANDLE;
	}

	ShaderModule GlobalAllocator::create(const create_info_t<ShaderModule>& cinfo, uint64_t absolute_frame) {
#if VUK_USE_SHADERC
		shaderc::SpvCompilationResult result;
		if (!cinfo.source.is_spirv) {
			// given source is GLSL, compile it via shaderc
			shaderc::Compiler compiler;
			shaderc::CompileOptions options;
			options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);

			result = compiler.CompileGlslToSpv(cinfo.source.as_glsl(), shaderc_glsl_infer_from_source, cinfo.filename.c_str(), options);

			if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
				std::string message = result.GetErrorMessage().c_str();
				throw ShaderCompilationException{ message };
			}
		}

		const std::vector<uint32_t>& spirv = cinfo.source.is_spirv ? cinfo.source.data : std::vector<uint32_t>(result.cbegin(), result.cend());
#else
		assert(cinfo.source.is_spirv && "Shaderc not enabled (VUK_USE_SHADERC == OFF), no runtime compilation possible.");
		const std::vector<uint32_t>& spirv = cinfo.source.data;
#endif
		spirv_cross::Compiler refl(spirv.data(), spirv.size());
		Program p;
		auto stage = p.introspect(refl);

		VkShaderModuleCreateInfo moduleCreateInfo{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
		moduleCreateInfo.codeSize = spirv.size() * sizeof(uint32_t);
		moduleCreateInfo.pCode = spirv.data();
		VkShaderModule sm;
		vkCreateShaderModule(device, &moduleCreateInfo, nullptr, &sm);
		std::string name = "ShaderModule: " + cinfo.filename;
		debug_utils->set_name(sm, Name(name));
		return { sm, p, stage };
	}

	PipelineBaseInfo GlobalAllocator::create(const create_info_t<PipelineBaseInfo>& cinfo, uint64_t absolute_frame) {
		std::vector<VkPipelineShaderStageCreateInfo> psscis;

		// accumulate descriptors from all stages
		Program accumulated_reflection;
		std::string pipe_name = "Pipeline:";
		for (auto i = 0; i < cinfo.shaders.size(); i++) {
			auto contents = cinfo.shaders[i];
			if (contents.data.empty())
				continue;
			auto sm = create(ShaderModuleCreateInfo{ contents, cinfo.shader_paths[i] }, absolute_frame);
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
			auto descset_layout_alloc_info = create(dsl, absolute_frame);
			dslai[dsl.index] = descset_layout_alloc_info;
			dsls.push_back(dslai[dsl.index].layout);
		}
		plci.plci.pSetLayouts = dsls.data();
		plci.plci.setLayoutCount = (uint32_t)dsls.size();

		PipelineBaseInfo pbi;
		pbi.psscis = std::move(psscis);
		pbi.color_blend_attachments = cinfo.color_blend_attachments;
		pbi.color_blend_state = cinfo.color_blend_state;
		pbi.depth_stencil_state = cinfo.depth_stencil_state;
		pbi.layout_info = dslai;
		pbi.pipeline_layout = create(plci, absolute_frame);
		pbi.rasterization_state = cinfo.rasterization_state;
		pbi.pipeline_name = Name(pipe_name);
		pbi.reflection_info = accumulated_reflection;
		pbi.binding_flags = cinfo.binding_flags;
		pbi.variable_count_max = cinfo.variable_count_max;
		return pbi;
	}

	ComputePipelineInfo GlobalAllocator::create(const create_info_t<ComputePipelineInfo>& cinfo, uint64_t absolute_frame) {
		VkPipelineShaderStageCreateInfo shader_stage{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
		std::string pipe_name = "Compute:";
		auto sm = create(ShaderModuleCreateInfo{ cinfo.shader, cinfo.shader_path }, absolute_frame);
		shader_stage.pSpecializationInfo = nullptr;
		shader_stage.stage = sm.stage;
		shader_stage.module = sm.shader_module;
		shader_stage.pName = "main"; //TODO: make param
		pipe_name += cinfo.shader_path;

		PipelineLayoutCreateInfo plci;
		plci.dslcis = PipelineBaseCreateInfo::build_descriptor_layouts(sm.reflection_info, cinfo);
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
			auto descset_layout_alloc_info = create(dsl, absolute_frame);
			dslai[dsl.index] = descset_layout_alloc_info;
			dsls.push_back(dslai[dsl.index].layout);
		}
		plci.plci.pSetLayouts = dsls.data();
		plci.plci.setLayoutCount = (uint32_t)dsls.size();

		VkComputePipelineCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
		cpci.stage = shader_stage;
		cpci.layout = create(plci, absolute_frame);
		VkPipeline pipeline;
		vkCreateComputePipelines(device, vk_pipeline_cache, 1, &cpci, nullptr, &pipeline);
		debug_utils->set_name(pipeline, Name(pipe_name));
		return { { pipeline, cpci.layout, dslai }, sm.reflection_info.local_size };
	}

	PipelineInfo GlobalAllocator::create(const create_info_t<PipelineInfo>& cinfo, uint64_t absolute_frame) {
		// create gfx pipeline
		VkGraphicsPipelineCreateInfo gpci = cinfo.to_vk();
		gpci.layout = cinfo.base->pipeline_layout;
		gpci.pStages = cinfo.base->psscis.data();
		gpci.stageCount = (uint32_t)cinfo.base->psscis.size();

		VkPipeline pipeline;
		vkCreateGraphicsPipelines(device, vk_pipeline_cache, 1, &gpci, nullptr, &pipeline);
		debug_utils->set_name(pipeline, cinfo.base->pipeline_name);
		return { pipeline, gpci.layout, cinfo.base->layout_info };
	}

	VkRenderPass GlobalAllocator::create(const create_info_t<VkRenderPass>& cinfo, uint64_t absolute_frame) {
		VkRenderPass rp;
		vkCreateRenderPass(device, &cinfo, nullptr, &rp);
		return rp;
	}

	VkFramebuffer GlobalAllocator::create(const create_info_t<VkFramebuffer>& cinfo, uint64_t absolute_frame) {
		VkFramebuffer fb;
		vkCreateFramebuffer(device, &cinfo, nullptr, &fb);
		return fb;
	}

	Sampler GlobalAllocator::create(const create_info_t<Sampler>& cinfo, uint64_t absolute_frame) {
		VkSampler s;
		vkCreateSampler(device, (VkSamplerCreateInfo*)&cinfo, nullptr, &s);
		return wrap(s);
	}

	DescriptorSetLayoutAllocInfo GlobalAllocator::create(const create_info_t<DescriptorSetLayoutAllocInfo>& cinfo, uint64_t absolute_frame) {
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

	VkPipelineLayout GlobalAllocator::create(const create_info_t<VkPipelineLayout>& cinfo, uint64_t absolute_frame) {
		VkPipelineLayout pl;
		vkCreatePipelineLayout(device, &cinfo.plci, nullptr, &pl);
		return pl;
	}

	DescriptorPool GlobalAllocator::create(const create_info_t<DescriptorPool>& cinfo, uint64_t absolute_frame) {
		return DescriptorPool{};
	}

	vuk::DescriptorSet GlobalAllocator::create(const create_info_t<vuk::DescriptorSet>& cinfo, uint64_t absolute_frame) {
		auto& pool = impl->pool_cache.acquire(cinfo.layout_info, absolute_frame);
		auto ds = pool.acquire(*this, cinfo.layout_info);
		auto mask = cinfo.used.to_ulong();
		unsigned long leading_ones = num_leading_ones(mask);
		std::array<VkWriteDescriptorSet, VUK_MAX_BINDINGS> writes = {};
		int j = 0;
		for (int i = 0; i < leading_ones; i++, j++) {
			if (!cinfo.used.test(i)) {
				j--;
				continue;
			}
			auto& write = writes[j];
			write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
			auto& binding = cinfo.bindings[i];
			write.descriptorType = (VkDescriptorType)binding.type;
			write.dstArrayElement = 0;
			write.descriptorCount = 1;
			write.dstBinding = i;
			write.dstSet = ds;
			switch (binding.type) {
			case vuk::DescriptorType::eUniformBuffer:
			case vuk::DescriptorType::eStorageBuffer:
				write.pBufferInfo = &binding.buffer;
				break;
			case vuk::DescriptorType::eSampledImage:
			case vuk::DescriptorType::eSampler:
			case vuk::DescriptorType::eCombinedImageSampler:
			case vuk::DescriptorType::eStorageImage:
				write.pImageInfo = &binding.image.dii;
				break;
			default:
				assert(0);
			}
		}
		vkUpdateDescriptorSets(device, j, writes.data(), 0, nullptr);
		return { ds, cinfo.layout_info };
	}

	ImageView GlobalAllocator::create(const ImageViewCreateInfo& ivci, uint64_t absolute_frame) {
		VkImageView iv;
		vkCreateImageView(device, (VkImageViewCreateInfo*)&ivci, nullptr, &iv);
		return wrap(iv);
	}

	LinearAllocator GlobalAllocator::create(const create_info_t<struct LinearAllocator>& cinfo, uint64_t absolute_frame) {
		return device_memory_allocator->allocate_linear(cinfo.mem_usage, cinfo.buffer_usage);
	}

	void GlobalAllocator::destroy(const DescriptorPool& dp) {
		for (auto& p : dp.pools) {
			vkDestroyDescriptorPool(device, p, nullptr);
		}
	}

	void GlobalAllocator::destroy(const PipelineInfo& pi) {
		vkDestroyPipeline(device, pi.pipeline, nullptr);
	}

	void GlobalAllocator::destroy(const ShaderModule& sm) {
		vkDestroyShaderModule(device, sm.shader_module, nullptr);
	}

	void GlobalAllocator::destroy(const DescriptorSetLayoutAllocInfo& ds) {
		vkDestroyDescriptorSetLayout(device, ds.layout, nullptr);
	}

	void GlobalAllocator::destroy(const VkPipelineLayout& pl) {
		vkDestroyPipelineLayout(device, pl, nullptr);
	}

	void GlobalAllocator::destroy(const VkRenderPass& rp) {
		vkDestroyRenderPass(device, rp, nullptr);
	}

	void GlobalAllocator::destroy(const DescriptorSet&) {
		// no-op, we destroy the pools
	}

	void GlobalAllocator::destroy(const VkFramebuffer& fb) {
		vkDestroyFramebuffer(device, fb, nullptr);
	}

	void GlobalAllocator::destroy(const Sampler& sa) {
		vkDestroySampler(device, sa.payload, nullptr);
	}

	void GlobalAllocator::destroy(const PipelineBaseInfo& pbi) {
		// no-op, we don't own device objects
	}

	void GlobalAllocator::destroy(Image image) {
		device_memory_allocator->destroy_image(image);
	}

	void GlobalAllocator::destroy(ImageView image) {
		vkDestroyImageView(device, image.payload, nullptr);
	}

	void GlobalAllocator::load_pipeline_cache(std::span<std::byte> data) {
		VkPipelineCacheCreateInfo pcci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = data.size_bytes(), .pInitialData = data.data() };
		vkDestroyPipelineCache(device, vk_pipeline_cache, nullptr);
		vkCreatePipelineCache(device, &pcci, nullptr, &vk_pipeline_cache);
	}

	std::vector<std::byte> GlobalAllocator::save_pipeline_cache() {
		size_t size;
		std::vector<std::byte> data;
		vkGetPipelineCacheData(device, vk_pipeline_cache, &size, nullptr);
		data.resize(size);
		vkGetPipelineCacheData(device, vk_pipeline_cache, &size, data.data());
		return data;
	}
	/*
	vuk::ShaderModule vuk::Context::compile_shader(ShaderSource source, std::string path) {
		vuk::ShaderModuleCreateInfo sci;
		sci.filename = std::move(path);
		sci.source = std::move(source);
		auto sm = impl->shader_modules.remove(sci);
		if (sm) {
			vkDestroyShaderModule(device, sm->shader_module, nullptr);
		}
		return impl->shader_modules.acquire(sci);
	}*/

	Unique<Buffer> GlobalAllocator::allocate_buffer(const BufferAllocationCreateInfo& baci, uint64_t absolute_frame, SourceLocation) {
		bool create_mapped = baci.mem_usage == MemoryUsage::eCPUonly || baci.mem_usage == MemoryUsage::eCPUtoGPU || baci.mem_usage == MemoryUsage::eGPUtoCPU;
		return Unique{ *this, device_memory_allocator->allocate_buffer(baci.mem_usage, baci.buffer_usage, baci.size, baci.alignment, create_mapped) };
	}

	Unique<Image> GlobalAllocator::allocate_image(const struct ImageCreateInfo& ici, uint64_t absolute_frame, SourceLocation) {
		return Unique{ *this, device_memory_allocator->create_image(ici) };
	}
	Unique<ImageView> GlobalAllocator::allocate_image_view(const struct ImageViewCreateInfo& ivci, uint64_t absolute_frame, SourceLocation) {
		return Unique{ *this, create(ivci, absolute_frame) };
	}

	Token GlobalAllocator::allocate_token() {
		return ctx.create_token();
	}

	std::pair<Texture, Token> GlobalAllocator::create_texture(vuk::Format format, vuk::Extent3D extent, void* data, bool generate_mips, uint64_t absolute_frame, SourceLocation) {
		vuk::ImageCreateInfo ici;
		ici.format = format;
		ici.extent = extent;
		ici.samples = vuk::Samples::e1;
		ici.imageType = vuk::ImageType::e2D;
		ici.initialLayout = vuk::ImageLayout::eUndefined;
		ici.tiling = vuk::ImageTiling::eOptimal;
		ici.usage = vuk::ImageUsageFlagBits::eTransferSrc | vuk::ImageUsageFlagBits::eTransferDst | vuk::ImageUsageFlagBits::eSampled;
		ici.mipLevels = generate_mips ? (uint32_t)log2f((float)std::max(extent.width, extent.height)) + 1 : 1;
		ici.arrayLayers = 1;
		auto tex = allocate_texture(ici, absolute_frame, VUK_HERE());
		auto tok = vuk::copy_to_image(*this, *tex.image, format, extent, 0, data, compute_image_size(format, extent));
		ctx.submit(*this, tok.token, vuk::Domain::eAny);
		return { std::move(tex), tok.token };
	}

	Texture GlobalAllocator::allocate_texture(const vuk::ImageCreateInfo& ici, uint64_t absolute_frame, SourceLocation) {
		auto dst = allocate_image(ici, absolute_frame, VUK_HERE()).release();
		vuk::ImageViewCreateInfo ivci;
		ivci.format = ici.format;
		ivci.image = dst;
		ivci.subresourceRange.aspectMask = format_to_aspect(ici.format);
		ivci.subresourceRange.baseArrayLayer = 0;
		ivci.subresourceRange.baseMipLevel = 0;
		ivci.subresourceRange.layerCount = 1;
		ivci.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
		ivci.viewType = vuk::ImageViewType::e2D;
		vuk::Texture tex{ Unique<Image>(*this, dst), allocate_image_view(ivci, absolute_frame, VUK_HERE()) };
		tex.extent = ici.extent;
		tex.format = ici.format;
		return tex;
	}

	struct TokenData& GlobalAllocator::get_token_data(Token t) {
		return ctx.get_token_data(t);
	}

	void GlobalAllocator::destroy(Token token) {
		ctx.destroy_token(token);
	}

	void GlobalAllocator::deallocate(VkPipeline) {
		assert(0);
	}
	void GlobalAllocator::deallocate(vuk::Image) {
		assert(0);
	}
	void GlobalAllocator::deallocate(vuk::ImageView) {
		assert(0);
	}
	void GlobalAllocator::deallocate(VkImageView) {
		assert(0);
	}
	void GlobalAllocator::deallocate(vuk::Buffer) {
		assert(0);
	}
	void GlobalAllocator::deallocate(vuk::PersistentDescriptorSet) {
		assert(0);
	}
}