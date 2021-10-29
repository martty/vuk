#if VUK_USE_SHADERC
#include <shaderc/shaderc.hpp>
#endif
#include <algorithm>
#include <fstream>
#include <sstream>
#include <spirv_cross.hpp>

#include <vuk/Context.hpp>
#include <../src/ContextImpl.hpp>
#include <vuk/RenderGraph.hpp>
#include <vuk/Program.hpp>
#include <vuk/Exception.hpp>

vuk::Context::Context(ContextCreateParameters params) :
	instance(params.instance),
	device(params.device),
	physical_device(params.physical_device),
	graphics_queue(params.graphics_queue),
	graphics_queue_family_index(params.graphics_queue_family_index),
	transfer_queue(params.transfer_queue),
	transfer_queue_family_index(params.transfer_queue_family_index),
	debug(*this){
    if(transfer_queue == VK_NULL_HANDLE || transfer_queue_family_index == VK_QUEUE_FAMILY_IGNORED) {
        transfer_queue = graphics_queue;
        transfer_queue_family_index = graphics_queue_family_index;
	}
    impl = new ContextImpl(*this);
}

bool vuk::Context::DebugUtils::enabled() {
	return setDebugUtilsObjectNameEXT != nullptr;
}

vuk::Context::DebugUtils::DebugUtils(Context& ctx) : ctx(ctx) {
	setDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(ctx.device, "vkSetDebugUtilsObjectNameEXT");
	cmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(ctx.device, "vkCmdBeginDebugUtilsLabelEXT");
	cmdEndDebugUtilsLabelEXT = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(ctx.device, "vkCmdEndDebugUtilsLabelEXT");
}

void vuk::Context::DebugUtils::set_name(const vuk::Texture& tex, Name name) {
	if (!enabled()) return;
	set_name(tex.image.get(), name);
	set_name(tex.view.get().payload, name);
}

void vuk::Context::DebugUtils::begin_region(const VkCommandBuffer& cb, Name name, std::array<float, 4> color) {
	if (!enabled()) return;
	VkDebugUtilsLabelEXT label = { .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
	label.pLabelName = name.c_str();
	::memcpy(label.color, color.data(), sizeof(float) * 4);
	cmdBeginDebugUtilsLabelEXT(cb, &label);
}

void vuk::Context::DebugUtils::end_region(const VkCommandBuffer& cb) {
	if (!enabled()) return;
	cmdEndDebugUtilsLabelEXT(cb);
}

void vuk::Context::submit_graphics(VkSubmitInfo si, VkFence fence) {
	std::lock_guard _(impl->gfx_queue_lock);
    VkResult res = vkQueueSubmit(graphics_queue, 1, &si, fence);
	assert(res == VK_SUCCESS);
}

void vuk::Context::submit_transfer(VkSubmitInfo si, VkFence fence) {
    if(transfer_queue == graphics_queue) {
        std::lock_guard _(impl->gfx_queue_lock);
        VkResult res = vkQueueSubmit(graphics_queue, 1, &si, fence);
        assert(res == VK_SUCCESS);
    } else {
        std::lock_guard _(impl->xfer_queue_lock);
        VkResult res = vkQueueSubmit(transfer_queue, 1, &si, fence);
        assert(res == VK_SUCCESS);
	}
}

void vuk::PersistentDescriptorSet::update_combined_image_sampler(PerThreadContext& ptc, unsigned binding, unsigned array_index, vuk::ImageView iv, vuk::SamplerCreateInfo sci, vuk::ImageLayout layout) {
    descriptor_bindings[binding][array_index].image = vuk::DescriptorImageInfo(ptc.acquire_sampler(sci), iv, layout);
    descriptor_bindings[binding][array_index].type = vuk::DescriptorType::eCombinedImageSampler;
	VkWriteDescriptorSet wds = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	wds.descriptorCount = 1;
	wds.descriptorType = (VkDescriptorType)vuk::DescriptorType::eCombinedImageSampler;
	wds.dstArrayElement = array_index;
	wds.dstBinding = binding;
    wds.pImageInfo = &descriptor_bindings[binding][array_index].image.dii;
	wds.dstSet = backing_set;
	pending_writes.push_back(wds);
}

void vuk::PersistentDescriptorSet::update_storage_image(PerThreadContext& ptc, unsigned binding, unsigned array_index, vuk::ImageView iv) {
	descriptor_bindings[binding][array_index].image = vuk::DescriptorImageInfo({}, iv, vuk::ImageLayout::eGeneral);
    descriptor_bindings[binding][array_index].type = vuk::DescriptorType::eStorageImage;
	VkWriteDescriptorSet wds = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	wds.descriptorCount = 1;
	wds.descriptorType = (VkDescriptorType)vuk::DescriptorType::eStorageImage;
	wds.dstArrayElement = array_index;
	wds.dstBinding = binding;
    wds.pImageInfo = &descriptor_bindings[binding][array_index].image.dii;
	wds.dstSet = backing_set;
	pending_writes.push_back(wds);
}

void vuk::PersistentDescriptorSet::update_uniform_buffer(PerThreadContext& ptc, unsigned binding, unsigned array_index, vuk::Buffer buffer) {
    descriptor_bindings[binding][array_index].buffer = VkDescriptorBufferInfo{buffer.buffer, buffer.offset, buffer.size};
    descriptor_bindings[binding][array_index].type = vuk::DescriptorType::eUniformBuffer;
    VkWriteDescriptorSet wds = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.descriptorCount = 1;
    wds.descriptorType = (VkDescriptorType)vuk::DescriptorType::eUniformBuffer;
    wds.dstArrayElement = 0;
    wds.dstBinding = binding;
    wds.pBufferInfo = &descriptor_bindings[binding][array_index].buffer;
    wds.dstSet = backing_set;
    pending_writes.push_back(wds);
}

void vuk::PersistentDescriptorSet::update_storage_buffer(PerThreadContext& ptc, unsigned binding, unsigned array_index, vuk::Buffer buffer) {
    descriptor_bindings[binding][array_index].buffer = VkDescriptorBufferInfo{buffer.buffer, buffer.offset, buffer.size};
    descriptor_bindings[binding][array_index].type = vuk::DescriptorType::eStorageBuffer;
    VkWriteDescriptorSet wds = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.descriptorCount = 1;
    wds.descriptorType = (VkDescriptorType)vuk::DescriptorType::eStorageBuffer;
    wds.dstArrayElement = 0;
    wds.dstBinding = binding;
    wds.pBufferInfo = &descriptor_bindings[binding][array_index].buffer;
    wds.dstSet = backing_set;
    pending_writes.push_back(wds);
}

vuk::ShaderModule vuk::Context::create(const create_info_t<vuk::ShaderModule>& cinfo) {
	// given source is GLSL, compile it via shaderc
#if VUK_USE_SHADERC
	shaderc::SpvCompilationResult result;
	if (!cinfo.source.is_spirv) {
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
	vuk::Program p;
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

vuk::PipelineBaseInfo vuk::Context::create(const create_info_t<PipelineBaseInfo>& cinfo) {
	std::vector<VkPipelineShaderStageCreateInfo> psscis;

	// accumulate descriptors from all stages
	vuk::Program accumulated_reflection;
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
	vuk::PipelineLayoutCreateInfo plci;
	plci.dslcis = vuk::PipelineBaseCreateInfo::build_descriptor_layouts(accumulated_reflection, cinfo);
    // use explicit descriptor layouts if there are any
    for(auto& l: cinfo.explicit_set_layouts) {
        plci.dslcis[l.index] = l;
    }
	plci.pcrs.insert(plci.pcrs.begin(), accumulated_reflection.push_constant_ranges.begin(), accumulated_reflection.push_constant_ranges.end());
	plci.plci.pushConstantRangeCount = (uint32_t)accumulated_reflection.push_constant_ranges.size();
	plci.plci.pPushConstantRanges = accumulated_reflection.push_constant_ranges.data();
	std::array<vuk::DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> dslai;
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
	pbi.color_blend_attachments = cinfo.color_blend_attachments;
	pbi.color_blend_state = cinfo.color_blend_state;
	pbi.depth_stencil_state = cinfo.depth_stencil_state;
	pbi.layout_info = dslai;
	pbi.pipeline_layout = impl->pipeline_layouts.acquire(plci);
	pbi.rasterization_state = cinfo.rasterization_state;
	pbi.pipeline_name = Name(pipe_name);
	pbi.reflection_info = accumulated_reflection;
	pbi.binding_flags = cinfo.binding_flags;
	pbi.variable_count_max = cinfo.variable_count_max;
	return pbi;
}

vuk::ComputePipelineBaseInfo vuk::Context::create(const create_info_t<vuk::ComputePipelineBaseInfo>& cinfo) {
	VkPipelineShaderStageCreateInfo shader_stage{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	std::string pipe_name = "Compute:";
	auto& sm = impl->shader_modules.acquire({ cinfo.shader, cinfo.shader_path });
	shader_stage.pSpecializationInfo = nullptr;
	shader_stage.stage = sm.stage;
	shader_stage.module = sm.shader_module;
	shader_stage.pName = "main"; //TODO: make param
	pipe_name += cinfo.shader_path;

	vuk::PipelineLayoutCreateInfo plci;
	plci.dslcis = vuk::PipelineBaseCreateInfo::build_descriptor_layouts(sm.reflection_info, cinfo);
    // use explicit descriptor layouts if there are any
    for(auto& l: cinfo.explicit_set_layouts) {
        plci.dslcis[l.index] = l;
    }
	plci.pcrs.insert(plci.pcrs.begin(), sm.reflection_info.push_constant_ranges.begin(), sm.reflection_info.push_constant_ranges.end());
	plci.plci.pushConstantRangeCount = (uint32_t)sm.reflection_info.push_constant_ranges.size();
	plci.plci.pPushConstantRanges = sm.reflection_info.push_constant_ranges.data();
	std::array<vuk::DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> dslai;
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

	vuk::ComputePipelineBaseInfo cpbi;
	cpbi.pssci = shader_stage;
	cpbi.layout_info = dslai;
	cpbi.pipeline_layout = impl->pipeline_layouts.acquire(plci);
	cpbi.pipeline_name = Name(pipe_name);
	cpbi.reflection_info = sm.reflection_info;
	cpbi.binding_flags = cinfo.binding_flags;
	cpbi.variable_count_max = cinfo.variable_count_max;

	return cpbi;
}

bool vuk::Context::load_pipeline_cache(std::span<uint8_t> data) {
	VkPipelineCacheCreateInfo pcci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = data.size_bytes(), .pInitialData = data.data() };
	vkDestroyPipelineCache(device, impl->vk_pipeline_cache, nullptr);
	vkCreatePipelineCache(device, &pcci, nullptr, &impl->vk_pipeline_cache);
	return true;
}

std::vector<uint8_t> vuk::Context::save_pipeline_cache() {
	size_t size;
	std::vector<uint8_t> data;
	vkGetPipelineCacheData(device, impl->vk_pipeline_cache, &size, nullptr);
	data.resize(size);
	vkGetPipelineCacheData(device, impl->vk_pipeline_cache, &size, data.data());
	return data;
}

vuk::Query vuk::Context::create_timestamp_query() {
	return { impl->query_id_counter++ };
}

vuk::DescriptorSetLayoutAllocInfo vuk::Context::create(const create_info_t<vuk::DescriptorSetLayoutAllocInfo>& cinfo) {
	vuk::DescriptorSetLayoutAllocInfo ret;
	vkCreateDescriptorSetLayout(device, &cinfo.dslci, nullptr, &ret.layout);
	for (size_t i = 0; i < cinfo.bindings.size(); i++) {
		auto& b = cinfo.bindings[i];
		// if this is not a variable count binding, add it to the descriptor count
		if (cinfo.flags.size() <= i || !(cinfo.flags[i] & to_integral(vuk::DescriptorBindingFlagBits::eVariableDescriptorCount))) {
			ret.descriptor_counts[to_integral(b.descriptorType)] += b.descriptorCount;
		} else { // a variable count binding
			ret.variable_count_binding = (uint32_t)i;
			ret.variable_count_binding_type = vuk::DescriptorType(b.descriptorType);
			ret.variable_count_binding_max_size = b.descriptorCount;
		}
	}
	return ret;
}

VkPipelineLayout vuk::Context::create(const create_info_t<VkPipelineLayout>& cinfo) {
	VkPipelineLayout pl;
	vkCreatePipelineLayout(device, &cinfo.plci, nullptr, &pl);
	return pl;
}

vuk::SwapchainRef vuk::Context::add_swapchain(Swapchain sw) {
	std::lock_guard _(impl->swapchains_lock);
	return &*impl->swapchains.emplace(sw);
}

void vuk::Context::remove_swapchain(SwapchainRef sw) {
	std::lock_guard _(impl->swapchains_lock);
	for (auto it = impl->swapchains.begin(); it != impl->swapchains.end(); it++) {
		if (&*it == sw) {
			impl->swapchains.erase(it);
			return;
		}
	}
}

void vuk::Context::create_named_pipeline(vuk::Name name, vuk::PipelineBaseCreateInfo ci) {
	std::lock_guard _(impl->named_pipelines_lock);
	impl->named_pipelines.insert_or_assign(name, &impl->pipelinebase_cache.acquire(std::move(ci)));
}

void vuk::Context::create_named_pipeline(vuk::Name name, vuk::ComputePipelineBaseCreateInfo ci) {
	std::lock_guard _(impl->named_pipelines_lock);
	impl->named_compute_pipelines.insert_or_assign(name, &impl->compute_pipelinebase_cache.acquire(std::move(ci)));
}

vuk::PipelineBaseInfo* vuk::Context::get_named_pipeline(vuk::Name name) {
	std::lock_guard _(impl->named_pipelines_lock);
	return impl->named_pipelines.at(name);
}

vuk::ComputePipelineBaseInfo* vuk::Context::get_named_compute_pipeline(vuk::Name name) {
	std::lock_guard _(impl->named_pipelines_lock);
	return impl->named_compute_pipelines.at(name);
}

vuk::PipelineBaseInfo* vuk::Context::get_pipeline(const vuk::PipelineBaseCreateInfo& pbci) {
	return &impl->pipelinebase_cache.acquire(pbci);
}

vuk::ComputePipelineBaseInfo* vuk::Context::get_pipeline(const vuk::ComputePipelineBaseCreateInfo& pbci) {
	return &impl->compute_pipelinebase_cache.acquire(pbci);
}

vuk::Program vuk::Context::get_pipeline_reflection_info(const vuk::PipelineBaseCreateInfo& pci) {
	auto& res = impl->pipelinebase_cache.acquire(pci);
	return res.reflection_info;
}

vuk::Program vuk::Context::get_pipeline_reflection_info(const vuk::ComputePipelineBaseCreateInfo& pci) {
	auto& res = impl->compute_pipelinebase_cache.acquire(pci);
	return res.reflection_info;
}

vuk::ShaderModule vuk::Context::compile_shader(ShaderSource source, std::string path) {
	vuk::ShaderModuleCreateInfo sci;
	sci.filename = std::move(path);
	sci.source = std::move(source);
	auto sm = impl->shader_modules.remove(sci);
	if (sm) {
		vkDestroyShaderModule(device, sm->shader_module, nullptr);
	}
	return impl->shader_modules.acquire(sci);
}

vuk::Context::TransientSubmitStub vuk::Context::fenced_upload(std::span<UploadItem> uploads, uint32_t dst_queue_family) {
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
	bundle->buffer = impl->allocator.allocate_buffer(vuk::MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, size, biggest_align, true);
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
			copy_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eUndefined;
			copy_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
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
				release_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
				release_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eShaderReadOnlyOptimal;
				release_barrier.dstQueueFamilyIndex = transfer_queue_family_index;
				release_barrier.srcQueueFamilyIndex = dst_queue_family;
				release_barrier.image = upload.image.dst;
				release_barrier.subresourceRange = copy_barrier.subresourceRange;
				vkCmdPipelineBarrier(xfercbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &release_barrier);

				// acquire on dst_queue_family
				VkImageMemoryBarrier acq_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };;
				acq_barrier.srcAccessMask = 0; // ignored
				acq_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; // TODO: maybe memory read?
				acq_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
				acq_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eShaderReadOnlyOptimal;
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
				release_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
				release_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
				release_barrier.srcQueueFamilyIndex = transfer_queue_family_index;
				release_barrier.dstQueueFamilyIndex = dst_queue_family;
				release_barrier.image = upload.image.dst;
				release_barrier.subresourceRange = copy_barrier.subresourceRange;
				vkCmdPipelineBarrier(xfercbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &release_barrier);

				// acquire on dst_queue_family
				VkImageMemoryBarrier acq_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };;
				acq_barrier.srcAccessMask = 0; // ignored
				acq_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				acq_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
				acq_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
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
				record_mip_gen(dstcbuf, task, vuk::ImageLayout::eTransferDstOptimal);
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
		submit_transfer(si, fence);
	} else {
		vkEndCommandBuffer(dstcbuf);
		// buffers and images, submit to transfer, signal sema
		si.signalSemaphoreCount = 1;
		auto sema = impl->get_unpooled_sema();
		si.pSignalSemaphores = &sema;
		submit_transfer(si, VkFence{ VK_NULL_HANDLE });
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
		submit_graphics(si, fence);
	}

	head_bundle->fence = fence;
	return head_bundle;
}

bool vuk::Context::poll_upload(TransientSubmitBundle* ur) {
	if (vkGetFenceStatus(device, ur->fence) == VK_SUCCESS) {
		std::lock_guard _(impl->transient_submit_lock);
		impl->cleanup_transient_bundle_recursively(ur);
		return true;
	} else {
		return false;
	}
}

vuk::Unique<vuk::Buffer> vuk::Context::allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment, bool create_mapped) {
	return Unique{ *this, impl->allocator.allocate_buffer(mem_usage, buffer_usage, size, alignment, create_mapped) };
}

vuk::Texture vuk::Context::allocate_texture(vuk::ImageCreateInfo ici) {
	ici.imageType = ici.extent.depth > 1?  vuk::ImageType::e3D :
	                ici.extent.height > 1? vuk::ImageType::e2D :
	                vuk::ImageType::e1D;
	auto dst = impl->allocator.create_image(ici);
	vuk::ImageViewCreateInfo ivci;
	ivci.format = ici.format;
	ivci.image = dst;
	ivci.subresourceRange.aspectMask = format_to_aspect(ici.format);
	ivci.subresourceRange.baseArrayLayer = 0;
	ivci.subresourceRange.baseMipLevel = 0;
	ivci.subresourceRange.layerCount = 1;
	ivci.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
	ivci.viewType = ici.imageType == vuk::ImageType::e3D? vuk::ImageViewType::e3D :
	                ici.imageType == vuk::ImageType::e2D? vuk::ImageViewType::e2D :
	                vuk::ImageViewType::e1D;
	vuk::Texture tex{ Unique<Image>(*this, dst), create_image_view(ivci) };
	tex.extent = ici.extent;
	tex.format = ici.format;
	tex.sample_count = ici.samples;
	return tex;
}


void vuk::Context::enqueue_destroy(vuk::Image i) {
	std::lock_guard _(impl->recycle_locks[frame_counter % FC]);
	impl->image_recycle[frame_counter % FC].push_back(i);
}

void vuk::Context::enqueue_destroy(vuk::ImageView iv) {
	std::lock_guard _(impl->recycle_locks[frame_counter % FC]);
	impl->image_view_recycle[frame_counter % FC].push_back(iv.payload);
}

void vuk::Context::enqueue_destroy(VkPipeline p) {
	std::lock_guard _(impl->recycle_locks[frame_counter % FC]);
	impl->pipeline_recycle[frame_counter % FC].push_back(p);
}

void vuk::Context::enqueue_destroy(vuk::Buffer b) {
	std::lock_guard _(impl->recycle_locks[frame_counter % FC]);
	impl->buffer_recycle[frame_counter % FC].push_back(b);
}

void vuk::Context::enqueue_destroy(vuk::PersistentDescriptorSet b) {
	std::lock_guard _(impl->recycle_locks[frame_counter % FC]);
	impl->pds_recycle[frame_counter % FC].push_back(std::move(b));
}

void vuk::Context::enqueue_destroy(VkFramebuffer fb) {
	std::lock_guard _(impl->recycle_locks[frame_counter % FC]);
	impl->fb_recycle[frame_counter % FC].push_back(fb);
}

void vuk::Context::destroy(const RGImage& image) {
	vkDestroyImageView(device, image.image_view.payload, nullptr);
	impl->allocator.destroy_image(image.image);
}

void vuk::Context::destroy(const PoolAllocator& v) {
	impl->allocator.destroy(v);
}

void vuk::Context::destroy(const LinearAllocator& v) {
	impl->allocator.destroy(v);
}

void vuk::Context::destroy(const vuk::DescriptorPool& dp) {
	for (auto& p : dp.pools) {
		vkDestroyDescriptorPool(device, p, nullptr);
	}
}

void vuk::Context::destroy(const vuk::PipelineInfo& pi) {
	vkDestroyPipeline(device, pi.pipeline, nullptr);
}

void vuk::Context::destroy(const vuk::ComputePipelineInfo& pi) {
	vkDestroyPipeline(device, pi.pipeline, nullptr);
}

void vuk::Context::destroy(const vuk::ShaderModule& sm) {
	vkDestroyShaderModule(device, sm.shader_module, nullptr);
}

void vuk::Context::destroy(const vuk::DescriptorSetLayoutAllocInfo& ds) {
	vkDestroyDescriptorSetLayout(device, ds.layout, nullptr);
}

void vuk::Context::destroy(const VkPipelineLayout& pl) {
	vkDestroyPipelineLayout(device, pl, nullptr);
}

void vuk::Context::destroy(const VkRenderPass& rp) {
	vkDestroyRenderPass(device, rp, nullptr);
}

void vuk::Context::destroy(const vuk::DescriptorSet&) {
	// no-op, we destroy the pools
}

void vuk::Context::destroy(const VkFramebuffer& fb) {
	vkDestroyFramebuffer(device, fb, nullptr);
}

void vuk::Context::destroy(const vuk::Sampler& sa) {
	vkDestroySampler(device, sa.payload, nullptr);
}

void vuk::Context::destroy(const vuk::PipelineBaseInfo& pbi) {
	// no-op, we don't own device objects
}

void vuk::Context::destroy(const vuk::ComputePipelineBaseInfo& pbi) {
	// no-op, we don't own device objects
}

vuk::Context::~Context() {
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
	delete impl;
}

vuk::InflightContext vuk::Context::begin() {
	std::lock_guard _(impl->begin_frame_lock);
	std::lock_guard recycle(impl->recycle_locks[_next(frame_counter.load(), FC)]);
	return InflightContext(*this, ++frame_counter, std::move(recycle));
}

void vuk::Context::wait_idle() {
	vkDeviceWaitIdle(device);
}

vuk::ImageView vuk::Context::wrap(VkImageView iv, vuk::ImageViewCreateInfo ivci) {
    vuk::ImageView viv{.payload = iv};
    viv.base_layer = ivci.subresourceRange.baseArrayLayer;
    viv.layer_count = ivci.subresourceRange.layerCount;
    viv.base_mip = ivci.subresourceRange.baseMipLevel;
    viv.mip_count = ivci.subresourceRange.levelCount;
    viv.format = ivci.format;
    viv.type = ivci.viewType;
    viv.image = ivci.image;
    viv.components = ivci.components;
    viv.id = unique_handle_id_counter++;
    return viv;
}

vuk::Unique<vuk::ImageView> vuk::Context::create_image_view(vuk::ImageViewCreateInfo ivci) {
	VkImageView iv;
	vkCreateImageView(device, (VkImageViewCreateInfo*)&ivci, nullptr, &iv);
	return vuk::Unique<vuk::ImageView>(*this, wrap(iv, ivci));
}

vuk::Unique<vuk::ImageView> vuk::Unique<vuk::ImageView>::SubrangeBuilder::apply(){
	ImageViewCreateInfo ivci;
	ivci.viewType = type == vuk::ImageViewType(0xdeadbeef) ? iv.type : type;
	ivci.subresourceRange.baseMipLevel = base_mip == 0xdeadbeef ? iv.base_mip : base_mip;
	ivci.subresourceRange.levelCount = mip_count == 0xdeadbeef ? iv.mip_count : mip_count;
	ivci.subresourceRange.baseArrayLayer = base_layer == 0xdeadbeef ? iv.base_layer : base_layer;
	ivci.subresourceRange.layerCount = layer_count == 0xdeadbeef ? iv.layer_count : layer_count;
	ivci.subresourceRange.aspectMask = format_to_aspect(iv.format);
	ivci.image = iv.image;
	ivci.format = iv.format;
	ivci.components = iv.components;
	return ctx->create_image_view(ivci);
}

vuk::Unique<vuk::ImageView>::~Unique() noexcept {
	if (context && payload != vuk::ImageView{})
		context->enqueue_destroy(std::move(payload));
}
void vuk::Unique<vuk::ImageView>::reset(vuk::ImageView value) noexcept {
	if (payload != value) {
		if (context && payload != vuk::ImageView{}) {
			context->enqueue_destroy(std::move(payload));
		}
		payload = std::move(value);
	}
}
