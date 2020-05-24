#include "Context.hpp"
#include "RenderGraph.hpp"
#include <shaderc/shaderc.hpp>
#include <algorithm>
#include "Program.hpp"
#include <fstream>
#include <sstream>
#include <spirv_cross.hpp>

std::string slurp(const std::string& path) {
	std::ostringstream buf;
	std::ifstream input(path.c_str());
    assert(input);
	buf << input.rdbuf();
	return buf.str();
}

void burp(const std::string& in, const std::string& path) {
	std::ofstream output(path.c_str(), std::ios::trunc);
	if (!output.is_open()) {
	}
	output << in;
	output.close();
}

vuk::InflightContext::InflightContext(Context& ctx, size_t absolute_frame, std::lock_guard<std::mutex>&& recycle_guard) :
	ctx(ctx),
	absolute_frame(absolute_frame),
	frame(absolute_frame% Context::FC),
	fence_pools(ctx.fence_pools.get_view(*this)), // must be first, so we wait for the fences
	commandbuffer_pools(ctx.cbuf_pools.get_view(*this)),
	semaphore_pools(ctx.semaphore_pools.get_view(*this)),
	pipeline_cache(*this, ctx.pipeline_cache),
    pipelinebase_cache(*this, ctx.pipelinebase_cache),
	renderpass_cache(*this, ctx.renderpass_cache),
	framebuffer_cache(*this, ctx.framebuffer_cache),
	transient_images(*this, ctx.transient_images),
	scratch_buffers(*this, ctx.scratch_buffers),
	descriptor_sets(*this, ctx.descriptor_sets),
	sampler_cache(*this, ctx.sampler_cache),
	sampled_images(ctx.sampled_images.get_view(*this)),
	pool_cache(*this, ctx.pool_cache),
	shader_modules(*this, ctx.shader_modules),
	descriptor_set_layouts(*this, ctx.descriptor_set_layouts),
	pipeline_layouts(*this, ctx.pipeline_layouts) {

	// image recycling
	for (auto& img : ctx.image_recycle[frame]) {
		ctx.allocator.destroy_image(img);
	}
	ctx.image_recycle[frame].clear();

	for (auto& iv : ctx.image_view_recycle[frame]) {
		ctx.device.destroy(iv);
	}
	ctx.image_view_recycle[frame].clear();

	for (auto& p : ctx.pipeline_recycle[frame]) {
		ctx.device.destroy(p);
	}
	ctx.pipeline_recycle[frame].clear();

	for (auto& b : ctx.buffer_recycle[frame]) {
		ctx.allocator.free_buffer(b);
	}
	ctx.buffer_recycle[frame].clear();


	for (auto& sb : scratch_buffers.cache.data[frame].pool) {
		ctx.allocator.reset_pool(sb);
	}

	auto ptc = begin();
	ptc.descriptor_sets.collect(Context::FC * 2);
}

vuk::TransferStub vuk::InflightContext::enqueue_transfer(Buffer src, Buffer dst) {
	std::lock_guard _(transfer_mutex);
	TransferStub stub{ transfer_id++ };
	buffer_transfer_commands.push({ src, dst, stub });
	return stub;
}

vuk::TransferStub vuk::InflightContext::enqueue_transfer(Buffer src, vk::Image dst, vk::Extent3D extent, bool generate_mips) {
	std::lock_guard _(transfer_mutex);
	TransferStub stub{ transfer_id++ };
    bufferimage_transfer_commands.push({src, dst, extent, generate_mips, stub});
    return stub;
}

void vuk::InflightContext::wait_all_transfers() {
	std::lock_guard _(transfer_mutex);

	while (!pending_transfers.empty()) {
		ctx.device.waitForFences(pending_transfers.front().fence, true, UINT64_MAX);
		auto last = pending_transfers.front();
		last_transfer_complete = last.last_transfer_id;
		pending_transfers.pop();
	}
}

void vuk::InflightContext::destroy(std::vector<vk::Image>&& images) {
	std::lock_guard _(recycle_lock);
	ctx.image_recycle[frame].insert(ctx.image_recycle[frame].end(), images.begin(), images.end());
}


void vuk::execute_submit_and_present_to_one(PerThreadContext& ptc, RenderGraph& rg, SwapchainRef swapchain, bool use_secondary_command_buffers) {
	auto render_complete = ptc.semaphore_pool.acquire(1)[0];
	auto present_rdy = ptc.semaphore_pool.acquire(1)[0];
	auto acq_result = ptc.ctx.device.acquireNextImageKHR(swapchain->swapchain, UINT64_MAX, present_rdy, vk::Fence{});
	auto index = acq_result.value;

	std::vector<std::pair<SwapChainRef, size_t>> swapchains_with_indexes = { { swapchain, index } };

	auto cb = rg.execute(ptc, swapchains_with_indexes, use_secondary_command_buffers);

	vk::SubmitInfo si;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cb;
	si.pSignalSemaphores = &render_complete;
	si.signalSemaphoreCount = 1;
	si.waitSemaphoreCount = 1;
	si.pWaitSemaphores = &present_rdy;
	vk::PipelineStageFlags flags = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	si.pWaitDstStageMask = &flags;
	auto fence = ptc.fence_pool.acquire(1)[0];
    {
        std::lock_guard _(ptc.ctx.gfx_queue_lock);
        ptc.ctx.graphics_queue.submit(si, fence);

        vk::PresentInfoKHR pi;
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain->swapchain;
        pi.pImageIndices = &acq_result.value;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &render_complete;
        ptc.ctx.graphics_queue.presentKHR(pi);
    }
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

void vuk::Context::DebugUtils::begin_region(const vk::CommandBuffer& cb, Name name, std::array<float, 4> color) {
	if (!enabled()) return;
	VkDebugUtilsLabelEXT label;
	label.pNext = nullptr;
	label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
	label.pLabelName = name.data();
	::memcpy(label.color, color.data(), sizeof(float) * 4);
	cmdBeginDebugUtilsLabelEXT(cb, &label);
}

void vuk::Context::DebugUtils::end_region(const vk::CommandBuffer& cb) {
	if (!enabled()) return;
	cmdEndDebugUtilsLabelEXT(cb);
}

vuk::PerThreadContext::PerThreadContext(InflightContext& ifc, unsigned tid) : ctx(ifc.ctx), ifc(ifc), tid(tid),
commandbuffer_pool(ifc.commandbuffer_pools.get_view(*this)),
semaphore_pool(ifc.semaphore_pools.get_view(*this)),
fence_pool(ifc.fence_pools.get_view(*this)),
pipeline_cache(*this, ifc.pipeline_cache),
pipelinebase_cache(*this, ifc.pipelinebase_cache),
renderpass_cache(*this, ifc.renderpass_cache),
framebuffer_cache(*this, ifc.framebuffer_cache),
transient_images(*this, ifc.transient_images),
scratch_buffers(*this, ifc.scratch_buffers),
descriptor_sets(*this, ifc.descriptor_sets),
sampler_cache(*this, ifc.sampler_cache),
sampled_images(ifc.sampled_images.get_view(*this)),
pool_cache(*this, ifc.pool_cache),
shader_modules(*this, ifc.shader_modules),
descriptor_set_layouts(*this, ifc.descriptor_set_layouts),
pipeline_layouts(*this, ifc.pipeline_layouts) {}

vuk::PerThreadContext::~PerThreadContext() {
	ifc.destroy(std::move(image_recycle));
	ifc.destroy(std::move(image_view_recycle));
}

void vuk::PerThreadContext::destroy(vk::Image image) {
	image_recycle.push_back(image);
}

void vuk::PerThreadContext::destroy(vuk::ImageView image) {
	image_view_recycle.push_back(image.payload);
}

void vuk::PerThreadContext::destroy(vuk::DescriptorSet ds) {
	// note that since we collect at integer times FC, we are releasing the DS back to the right pool
	pool_cache.acquire(ds.layout_info).free_sets.push_back(ds.descriptor_set);
}

vuk::Buffer vuk::PerThreadContext::_allocate_scratch_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, size_t size, bool create_mapped) {
    PoolSelect ps{mem_usage, buffer_usage};
	auto& pool = scratch_buffers.acquire(ps);
	return ifc.ctx.allocator.allocate_buffer(pool, size, 1, create_mapped);
}

vuk::Unique<vuk::Buffer> vuk::PerThreadContext::_allocate_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, size_t size, bool create_mapped) {
	return vuk::Unique<Buffer>(ifc.ctx, ifc.ctx.allocator.allocate_buffer(mem_usage, buffer_usage, size, 1, create_mapped));
}


bool vuk::PerThreadContext::is_ready(const TransferStub& stub) {
	return ifc.last_transfer_complete >= stub.id;
}

void vuk::PerThreadContext::wait_all_transfers() {
	// TODO: remove when we go MT
	dma_task(); // run one transfer so it is more easy to follow
	return ifc.wait_all_transfers();
}

std::pair<vuk::Texture, vuk::TransferStub> vuk::PerThreadContext::create_texture(vk::Format format, vk::Extent3D extents, void* data) {
    auto tex = ctx.allocate_texture(format, extents, 1);
	auto stub = upload(*tex.image, extents, std::span<std::byte>((std::byte*)data, extents.width * extents.height * extents.depth * 4), false);
	return { std::move(tex), stub };
}

void record_buffer_image_copy(vk::CommandBuffer& cbuf, vuk::InflightContext::BufferImageCopyCommand& task) {
    vk::BufferImageCopy bc;
    bc.bufferOffset = task.src.offset;
    bc.imageOffset = 0;
    bc.bufferRowLength = 0;
    bc.bufferImageHeight = 0;
    bc.imageExtent = task.extent;
    bc.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    bc.imageSubresource.baseArrayLayer = 0;
    bc.imageSubresource.mipLevel = 0;
    bc.imageSubresource.layerCount = 1;

    vk::ImageMemoryBarrier copy_barrier;
    copy_barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
    copy_barrier.oldLayout = vk::ImageLayout::eUndefined;
    copy_barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
    copy_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copy_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copy_barrier.image = task.dst;
    copy_barrier.subresourceRange.aspectMask = bc.imageSubresource.aspectMask;
    copy_barrier.subresourceRange.layerCount = bc.imageSubresource.layerCount;
    copy_barrier.subresourceRange.baseArrayLayer = bc.imageSubresource.baseArrayLayer;
    copy_barrier.subresourceRange.baseMipLevel = bc.imageSubresource.mipLevel;
    copy_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;

	// transition top mip to transfersrc
    vk::ImageMemoryBarrier top_mip_to_barrier;
    top_mip_to_barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    top_mip_to_barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
    top_mip_to_barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    top_mip_to_barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    top_mip_to_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    top_mip_to_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    top_mip_to_barrier.image = task.dst;
    top_mip_to_barrier.subresourceRange = copy_barrier.subresourceRange;
    top_mip_to_barrier.subresourceRange.levelCount = 1;

    // transition top mip to SROO
    vk::ImageMemoryBarrier top_mip_use_barrier;
    top_mip_use_barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
    top_mip_use_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    top_mip_use_barrier.oldLayout = task.generate_mips ? vk::ImageLayout::eTransferSrcOptimal : vk::ImageLayout::eTransferDstOptimal;
    top_mip_use_barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    top_mip_use_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    top_mip_use_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    top_mip_use_barrier.image = task.dst;
    top_mip_use_barrier.subresourceRange = copy_barrier.subresourceRange;
    top_mip_use_barrier.subresourceRange.levelCount = 1;

    // transition rest of the mips to SROO
    vk::ImageMemoryBarrier use_barrier;
    use_barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    use_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    use_barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    use_barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    use_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    use_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    use_barrier.image = task.dst;
    use_barrier.subresourceRange = copy_barrier.subresourceRange;
    use_barrier.subresourceRange.baseMipLevel = 1;

    cbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits(0), {}, {}, copy_barrier);
    cbuf.copyBufferToImage(task.src.buffer, task.dst, vk::ImageLayout::eTransferDstOptimal, bc);
    if(task.generate_mips) {
        cbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits(0), {}, {}, top_mip_to_barrier);
        auto mips = (uint32_t)std::min(std::log2f(task.extent.width), std::log2f(task.extent.height));

        for(auto miplevel = 1; miplevel < mips; miplevel++) {
            vk::ImageBlit blit;
            blit.srcSubresource.aspectMask = copy_barrier.subresourceRange.aspectMask;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.srcSubresource.mipLevel = 0;
            blit.srcOffsets[0] = vk::Offset3D{0};
            blit.srcOffsets[1] = vk::Offset3D(task.extent.width, task.extent.height, task.extent.depth);
            blit.dstSubresource = blit.srcSubresource;
            blit.dstSubresource.mipLevel = miplevel;
            blit.dstOffsets[0] = vk::Offset3D{0};
            blit.dstOffsets[1] = vk::Offset3D(task.extent.width >> miplevel, task.extent.height >> miplevel, task.extent.depth);
            cbuf.blitImage(task.dst, vk::ImageLayout::eTransferSrcOptimal, task.dst, vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);
        }

        cbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, vk::DependencyFlagBits(0), {}, {}, use_barrier);
    }
    cbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, vk::DependencyFlagBits(0), {}, {},
                         top_mip_use_barrier);
}

void vuk::PerThreadContext::dma_task() {
	std::lock_guard _(ifc.transfer_mutex);
	while (!ifc.pending_transfers.empty() && ctx.device.getFenceStatus(ifc.pending_transfers.front().fence) == vk::Result::eSuccess) {
		auto last = ifc.pending_transfers.front();
		ifc.last_transfer_complete = last.last_transfer_id;
		ifc.pending_transfers.pop();
	}

	if (ifc.buffer_transfer_commands.empty() && ifc.bufferimage_transfer_commands.empty()) return;
	auto cbuf = commandbuffer_pool.acquire(vk::CommandBufferLevel::ePrimary, 1)[0];
	cbuf.begin(vk::CommandBufferBeginInfo{});
	size_t last = 0;
	while (!ifc.buffer_transfer_commands.empty()) {
		auto task = ifc.buffer_transfer_commands.front();
		ifc.buffer_transfer_commands.pop();
		vk::BufferCopy bc;
		bc.dstOffset = task.dst.offset;
		bc.srcOffset = task.src.offset;
		bc.size = task.src.size;
		cbuf.copyBuffer(task.src.buffer, task.dst.buffer, bc);
		last = std::max(last, task.stub.id);
	}
	while (!ifc.bufferimage_transfer_commands.empty()) {
		auto task = ifc.bufferimage_transfer_commands.front();
		ifc.bufferimage_transfer_commands.pop();
        record_buffer_image_copy(cbuf, task);
        last = std::max(last, task.stub.id);
	}
	cbuf.end();
	auto fence = fence_pool.acquire(1)[0];
	vk::SubmitInfo si;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cbuf;
	ifc.ctx.graphics_queue.submit(si, fence);
	ifc.pending_transfers.emplace(InflightContext::PendingTransfer{ last, fence });
}

vuk::SampledImage& vuk::PerThreadContext::make_sampled_image(vuk::ImageView iv, vk::SamplerCreateInfo sci) {
	vuk::SampledImage si(vuk::SampledImage::Global{ iv, sci, vk::ImageLayout::eShaderReadOnlyOptimal });
	return sampled_images.acquire(si);
}

vuk::SampledImage& vuk::PerThreadContext::make_sampled_image(Name n, vk::SamplerCreateInfo sci) {
	vuk::SampledImage si(vuk::SampledImage::RenderGraphAttachment{ n, sci, {}, vk::ImageLayout::eShaderReadOnlyOptimal });
	return sampled_images.acquire(si);
}

vuk::SampledImage& vuk::PerThreadContext::make_sampled_image(Name n, vk::ImageViewCreateInfo ivci, vk::SamplerCreateInfo sci) {
	vuk::SampledImage si(vuk::SampledImage::RenderGraphAttachment{ n, sci, ivci, vk::ImageLayout::eShaderReadOnlyOptimal });
	return sampled_images.acquire(si);
}

vuk::DescriptorSet vuk::PerThreadContext::create(const create_info_t<vuk::DescriptorSet>& cinfo) {
	auto& pool = pool_cache.acquire(cinfo.layout_info);
	auto ds = pool.acquire(*this, cinfo.layout_info);
	auto mask = cinfo.used.to_ulong();
	unsigned long leading_ones = num_leading_ones(mask);
	std::array<vk::WriteDescriptorSet, VUK_MAX_BINDINGS> writes;
	std::array<vk::DescriptorImageInfo, VUK_MAX_BINDINGS> diis;
	for (unsigned i = 0; i < leading_ones; i++) {
		if (!cinfo.used.test(i)) continue;
		auto& write = writes[i];
		auto& binding = cinfo.bindings[i];
		write.descriptorType = binding.type;
		write.dstArrayElement = 0;
		write.descriptorCount = 1;
		write.dstBinding = i;
		write.dstSet = ds;
		switch (binding.type) {
		case vk::DescriptorType::eUniformBuffer:
		case vk::DescriptorType::eStorageBuffer:
			write.pBufferInfo = (vk::DescriptorBufferInfo*)&binding.buffer;
			break;
		case vk::DescriptorType::eSampledImage:
		case vk::DescriptorType::eSampler:
		case vk::DescriptorType::eCombinedImageSampler:
			diis[i] = binding.image;
			write.pImageInfo = &diis[i];
			break;
		default:
			assert(0);
		}
	}
	ctx.device.updateDescriptorSets(leading_ones, writes.data(), 0, nullptr);
	return { ds, cinfo.layout_info };
}

/*vuk::Allocator::Pool vuk::PerThreadContext::create(const create_info_t<vuk::Allocator::Pool>& cinfo) {
	return ctx.allocator.allocate_pool(cinfo.mem_usage, cinfo.buffer_usage);
}*/

vuk::Allocator::Linear vuk::PerThreadContext::create(const create_info_t<vuk::Allocator::Linear>& cinfo) {
	return ctx.allocator.allocate_linear(cinfo.mem_usage, cinfo.buffer_usage);
}


vuk::RGImage vuk::PerThreadContext::create(const create_info_t<vuk::RGImage>& cinfo) {
	RGImage res;
	res.image = ctx.allocator.create_image_for_rendertarget(cinfo.ici);
	auto ivci = cinfo.ivci;
	ivci.image = res.image;
	std::string name = std::string("Image: RenderTarget ") + std::string(cinfo.name);
	ctx.debug.set_name(res.image, name);
	name = std::string("ImageView: RenderTarget ") + std::string(cinfo.name);
	res.image_view = ctx.wrap(ctx.device.createImageView(ivci));
	ctx.debug.set_name(res.image_view.payload, name);
	return res;
}

vk::RenderPass vuk::PerThreadContext::create(const create_info_t<vk::RenderPass>& cinfo) {
	return ctx.device.createRenderPass(cinfo);
}

vuk::ShaderModule vuk::Context::create(const create_info_t<vuk::ShaderModule>& cinfo) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(cinfo.source, shaderc_glsl_infer_from_source, cinfo.filename.c_str(), options);

    if(result.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::string message = result.GetErrorMessage().c_str();
        throw ShaderCompilationException{message};
    } else {
        std::vector<uint32_t> spirv(result.cbegin(), result.cend());

        spirv_cross::Compiler refl(spirv.data(), spirv.size());
        vuk::Program p;
        auto stage = p.introspect(refl);

        vk::ShaderModuleCreateInfo moduleCreateInfo;
        moduleCreateInfo.codeSize = spirv.size() * sizeof(uint32_t);
        moduleCreateInfo.pCode = spirv.data();
        auto module = device.createShaderModule(moduleCreateInfo);
        std::string name = "ShaderModule: " + cinfo.filename;
        debug.set_name(module, name);
        return {module, p, stage};
    }
}

vuk::ShaderModule vuk::PerThreadContext::create(const create_info_t<vuk::ShaderModule>& cinfo) {
    return ctx.create(cinfo);
}

vuk::PipelineBaseInfo vuk::Context::create(const create_info_t<PipelineBaseInfo>& cinfo) {
	std::vector<vk::PipelineShaderStageCreateInfo> psscis;

	// accumulate descriptors from all stages
	vuk::Program accumulated_reflection;
	std::string pipe_name = "Pipeline:";
    for(auto i = 0; i < cinfo.shaders.size(); i++) {
		auto contents = cinfo.shaders[i];
        if(contents.empty())
            continue;
		auto& sm = shader_modules.acquire({ contents, cinfo.shader_paths[i] });
		vk::PipelineShaderStageCreateInfo shaderStage;
		shaderStage.pSpecializationInfo = nullptr;
		shaderStage.stage = sm.stage;
		shaderStage.module = sm.shader_module;
		shaderStage.pName = "main"; //TODO: make param
		psscis.push_back(shaderStage);
		accumulated_reflection.append(sm.reflection_info);
		pipe_name += cinfo.shader_paths[i] + "+";
	}
	pipe_name = pipe_name.substr(0, pipe_name.size() - 1); //trim off last "+"
														   // acquire descriptor set layouts (1 per set)
														   // acquire pipeline layout
	vuk::PipelineLayoutCreateInfo plci;
	plci.dslcis = vuk::PipelineBaseCreateInfo::build_descriptor_layouts(accumulated_reflection);
	plci.pcrs.insert(plci.pcrs.begin(), accumulated_reflection.push_constant_ranges.begin(), accumulated_reflection.push_constant_ranges.end());
	plci.plci.pushConstantRangeCount = (uint32_t)accumulated_reflection.push_constant_ranges.size();
	plci.plci.pPushConstantRanges = accumulated_reflection.push_constant_ranges.data();
	std::array<vuk::DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> dslai;
	std::vector<vk::DescriptorSetLayout> dsls;
	for (auto& dsl : plci.dslcis) {
		dsl.dslci.bindingCount = (uint32_t)dsl.bindings.size();
		dsl.dslci.pBindings = dsl.bindings.data();
		auto l = descriptor_set_layouts.acquire(dsl);
		dslai[dsl.index] = l;
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
	pbi.pipeline_layout = pipeline_layouts.acquire(plci);
    pbi.rasterization_state = cinfo.rasterization_state;
    pbi.pipeline_name = std::move(pipe_name);
    pbi.reflection_info = accumulated_reflection;
    return pbi;
}

vuk::PipelineBaseInfo vuk::PerThreadContext::create(const create_info_t<PipelineBaseInfo>& cinfo) {
    return ctx.create(cinfo);
}

vuk::PipelineInfo vuk::PerThreadContext::create(const create_info_t<PipelineInfo>& cinfo) {
	// create gfx pipeline
	vk::GraphicsPipelineCreateInfo gpci = cinfo.to_vk();
	gpci.layout = cinfo.base->pipeline_layout;
	gpci.pStages = cinfo.base->psscis.data();
	gpci.stageCount = (uint32_t)cinfo.base->psscis.size();

	auto pipeline = ctx.device.createGraphicsPipeline(*ctx.vk_pipeline_cache, gpci);
	ctx.debug.set_name(pipeline, cinfo.base->pipeline_name);
	return { pipeline, gpci.layout, cinfo.base->layout_info };
}

vk::Framebuffer vuk::PerThreadContext::create(const create_info_t<vk::Framebuffer>& cinfo) {
	return ctx.device.createFramebuffer(cinfo);
}

vk::Sampler vuk::PerThreadContext::create(const create_info_t<vk::Sampler>& cinfo) {
	return ctx.device.createSampler(cinfo);
}

vuk::DescriptorSetLayoutAllocInfo vuk::Context::create(const create_info_t<vuk::DescriptorSetLayoutAllocInfo>& cinfo) {
	vuk::DescriptorSetLayoutAllocInfo ret;
	ret.layout = device.createDescriptorSetLayout(cinfo.dslci);
	for (auto& b : cinfo.bindings) {
		ret.descriptor_counts[to_integral(b.descriptorType)] += b.descriptorCount;
	}
	return ret;
}

vuk::DescriptorSetLayoutAllocInfo vuk::PerThreadContext::create(const create_info_t<vuk::DescriptorSetLayoutAllocInfo>& cinfo) {
    return ctx.create(cinfo);
}

vk::PipelineLayout vuk::Context::create(const create_info_t<vk::PipelineLayout>& cinfo) {
	return device.createPipelineLayout(cinfo.plci);
}

vk::PipelineLayout vuk::PerThreadContext::create(const create_info_t<vk::PipelineLayout>& cinfo) {
    return ctx.create(cinfo);
}

vuk::DescriptorPool vuk::PerThreadContext::create(const create_info_t<vuk::DescriptorPool>& cinfo) {
	return vuk::DescriptorPool{};
}

vuk::SwapchainRef vuk::Context::add_swapchain(Swapchain sw) {
	std::lock_guard _(swapchains_lock);
	sw.image_views.reserve(sw._ivs.size());
	for (auto& v : sw._ivs) {
		sw.image_views.push_back(wrap(vk::ImageView{ v }));
	}

	return &*swapchains.emplace(sw);
}

vuk::Context::Context(vk::Instance instance, vk::Device device, vk::PhysicalDevice physical_device, vk::Queue graphics) :
	instance(instance),
	device(device),
	physical_device(physical_device),
	graphics_queue(graphics),
	allocator(instance, device, physical_device),
	cbuf_pools(*this),
	semaphore_pools(*this),
	fence_pools(*this),
	pipelinebase_cache(*this),
	pipeline_cache(*this),
	renderpass_cache(*this),
	framebuffer_cache(*this),
	transient_images(*this),
	scratch_buffers(*this),
	pool_cache(*this),
	descriptor_sets(*this),
	sampler_cache(*this),
	sampled_images(*this),
	shader_modules(*this),
	descriptor_set_layouts(*this),
	pipeline_layouts(*this),
	debug(*this) {
	vk_pipeline_cache = device.createPipelineCacheUnique({});
}

void vuk::Context::create_named_pipeline(const char* name, vuk::PipelineBaseCreateInfo ci) {
	std::lock_guard _(named_pipelines_lock);
	named_pipelines.insert_or_assign(name, &pipelinebase_cache.acquire(std::move(ci)));
}

vuk::PipelineBaseInfo* vuk::Context::get_named_pipeline(const char* name) {
	return named_pipelines.at(name);
}

vuk::PipelineBaseInfo* vuk::Context::get_pipeline(const vuk::PipelineBaseCreateInfo& pbci) {
    return &pipelinebase_cache.acquire(pbci);
}

vuk::Program vuk::PerThreadContext::get_pipeline_reflection_info(vuk::PipelineBaseCreateInfo pci) {
	auto& res = pipelinebase_cache.acquire(pci);
	return res.reflection_info;
}

vuk::Program vuk::Context::get_pipeline_reflection_info(vuk::PipelineBaseCreateInfo pci) {
	auto& res = pipelinebase_cache.acquire(pci);
	return res.reflection_info;
}

void vuk::Context::invalidate_shadermodule_and_pipelines(Name filename) {
    std::optional<vuk::ShaderModule> sm;
    do {
        sm = shader_modules.remove([&](auto& ci, auto& p) { return ci.filename == filename; });
        if(sm) {
            device.destroy(sm->shader_module);
            const PipelineBaseInfo* pipe_base;
            do {
                pipe_base = pipelinebase_cache.find([&](auto& ci, auto& p) {
                    if(std::find_if(ci.shader_paths.begin(), ci.shader_paths.end(), [=](auto& t) { return t == filename; }) != ci.shader_paths.end())
                        return true;
                    return false;
                });
                if(pipe_base) {
                    std::optional<PipelineInfo> pipe;
                    do {
                        pipe = pipeline_cache.remove([&](auto& ci, auto& p) { return ci.base == pipe_base; });
                        if(pipe)
                            enqueue_destroy(pipe->pipeline);
                    } while(pipe != std::nullopt);
                    pipelinebase_cache.remove_ptr(pipe_base);
                }
            } while(pipe_base != nullptr);
        }
    } while(sm != std::nullopt);
}

vuk::ShaderModule vuk::Context::compile_shader(std::string source, Name path) {
	vuk::ShaderModuleCreateInfo sci;
	sci.filename = path;
    sci.source = slurp(std::string(path));
	auto sm = shader_modules.remove(sci);
    if(sm)
		device.destroy(sm->shader_module);
    return shader_modules.acquire(sci);
}

vuk::Context::UploadResult vuk::Context::fenced_upload(std::span<BufferUpload> uploads) {
	// get a one time command buffer
    auto tid = get_thread_index ? get_thread_index() : 0;

	vk::CommandBuffer cbuf;
    {
        std::lock_guard _(one_time_pool_lock);
        if(xfer_one_time_pools.size() < (tid + 1)) {
            xfer_one_time_pools.resize(tid + 1, vk::CommandPool{});
        }

        auto& pool = xfer_one_time_pools[tid];
        if(pool == vk::CommandPool{}) {
            vk::CommandPoolCreateInfo cpci;
            cpci.queueFamilyIndex = transfer_queue_family_index;
            cpci.flags = vk::CommandPoolCreateFlagBits::eTransient;
            pool = device.createCommandPool(cpci);
        }

        vk::CommandBufferAllocateInfo cbai;
        cbai.commandPool = pool;
        cbai.commandBufferCount = 1;

        cbuf = device.allocateCommandBuffers(cbai)[0];
    }
    vk::CommandBufferBeginInfo cbi;
    cbi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    cbuf.begin(cbi);

	size_t size = 0;
	for (auto& upload : uploads) {
        size += upload.data.size();
	}

	// create 1 big staging buffer
	auto staging_alloc = allocator.allocate_buffer(vuk::MemoryUsage::eCPUonly, vk::BufferUsageFlagBits::eTransferSrc, size, 1, true);
    auto staging = staging_alloc;
    for(auto& upload: uploads) {
		// copy to staging
		::memcpy(staging.mapped_ptr, upload.data.data(), upload.data.size());

        vk::BufferCopy bc;
        bc.dstOffset = upload.dst.offset;
        bc.srcOffset = staging.offset;
        bc.size = upload.data.size();
        cbuf.copyBuffer(staging.buffer, upload.dst.buffer, bc);

		staging.offset += upload.data.size();
        staging.mapped_ptr = static_cast<unsigned char*>(staging.mapped_ptr) + upload.data.size();
    }
    cbuf.end();
	// get an unpooled fence
    auto fence = device.createFence({});
    vk::SubmitInfo si;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cbuf;
    {
        std::lock_guard _(xfer_queue_lock);
        transfer_queue.submit(si, fence);
    }
    return {fence, cbuf, staging_alloc, true, tid};
}

vuk::Context::UploadResult vuk::Context::fenced_upload(std::span<ImageUpload> uploads) {
    // get a one time command buffer
    auto tid = get_thread_index ? get_thread_index() : 0;
    vk::CommandBuffer cbuf;
    {
        std::lock_guard _(one_time_pool_lock);
        if(one_time_pools.size() < (tid + 1)) {
            one_time_pools.resize(tid + 1, vk::CommandPool{});
        }
        auto& pool = one_time_pools[tid];
        if(pool == vk::CommandPool{}) {
            vk::CommandPoolCreateInfo cpci;
            cpci.queueFamilyIndex = graphics_queue_family_index;
            cpci.flags = vk::CommandPoolCreateFlagBits::eTransient;
            pool = device.createCommandPool(cpci);
        }
        vk::CommandBufferAllocateInfo cbai;
        cbai.commandPool = pool;
        cbai.commandBufferCount = 1;

        cbuf = device.allocateCommandBuffers(cbai)[0];
    }
    vk::CommandBufferBeginInfo cbi;
    cbi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    cbuf.begin(cbi);

    size_t size = 0;
    for(auto& upload: uploads) {
        size += upload.data.size();
    }

    // create 1 big staging buffer
	auto staging_alloc = allocator.allocate_buffer(vuk::MemoryUsage::eCPUonly, vk::BufferUsageFlagBits::eTransferSrc, size, 1, true);
    auto staging = staging_alloc;
    for(auto& upload: uploads) {
        // copy to staging
        ::memcpy(staging.mapped_ptr, upload.data.data(), upload.data.size());

		InflightContext::BufferImageCopyCommand task;
        task.src = staging;
        task.dst = upload.dst;
        task.extent = upload.extent;
        task.generate_mips = true;
        record_buffer_image_copy(cbuf, task);
		staging.offset += upload.data.size();
        staging.mapped_ptr = static_cast<unsigned char*>(staging.mapped_ptr) + upload.data.size();
    }
    cbuf.end();
    // get an unpooled fence
    auto fence = device.createFence({});
    vk::SubmitInfo si;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cbuf;
    {
        std::lock_guard _(gfx_queue_lock);
        graphics_queue.submit(si, fence);
    }
    return {fence, cbuf, staging_alloc, false, tid};
}

void vuk::Context::free_upload_resources(const UploadResult& ur) {
    auto& pools = ur.is_buffer ? xfer_one_time_pools : one_time_pools;
    std::lock_guard _(one_time_pool_lock);
    device.freeCommandBuffers(pools[ur.thread_index], ur.command_buffer);
    allocator.free_buffer(ur.staging);
    device.destroyFence(ur.fence);
}

vuk::Buffer vuk::Context::allocate_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, size_t size, size_t alignment) {
    return allocator.allocate_buffer(mem_usage, buffer_usage, size, alignment, false);
}

vuk::Texture vuk::Context::allocate_texture(vk::Format format, vk::Extent3D extents, uint32_t miplevels) {
    vk::ImageCreateInfo ici;
	ici.format = format;
	ici.extent = extents;
	ici.arrayLayers = 1;
	ici.initialLayout = vk::ImageLayout::eUndefined;
	ici.mipLevels = miplevels;
	ici.imageType = vk::ImageType::e2D;
	ici.samples = vk::SampleCountFlagBits::e1;
	ici.tiling = vk::ImageTiling::eOptimal;
	ici.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc;
	auto dst = allocator.create_image(ici);
	vk::ImageViewCreateInfo ivci;
	ivci.format = format;
	ivci.image = dst;
	ivci.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	ivci.subresourceRange.baseArrayLayer = 0;
	ivci.subresourceRange.baseMipLevel = 0;
	ivci.subresourceRange.layerCount = 1;
	ivci.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
	ivci.viewType = vk::ImageViewType::e2D;
	auto iv = device.createImageView(ivci);
	vuk::Texture tex{ vuk::Unique<vk::Image>(*this, dst), vuk::Unique<vuk::ImageView>(*this, wrap(iv)) };
	tex.extent = extents;
	tex.format = format;
    return tex;
}


void vuk::Context::enqueue_destroy(vk::Image i) {
	std::lock_guard _(recycle_locks[frame_counter % FC]);
	image_recycle[frame_counter % FC].push_back(i);
}

void vuk::Context::enqueue_destroy(vuk::ImageView iv) {
	std::lock_guard _(recycle_locks[frame_counter % FC]);
	image_view_recycle[frame_counter % FC].push_back(iv.payload);
}

void vuk::Context::enqueue_destroy(vk::Pipeline p) {
	std::lock_guard _(recycle_locks[frame_counter % FC]);
	pipeline_recycle[frame_counter % FC].push_back(p);
}

void vuk::Context::enqueue_destroy(vuk::Buffer b) {
	std::lock_guard _(recycle_locks[frame_counter % FC]);
	buffer_recycle[frame_counter % FC].push_back(b);
}


void vuk::Context::destroy(const RGImage& image) {
	device.destroy(image.image_view.payload);
	allocator.destroy_image(image.image);
}

void vuk::Context::destroy(const Allocator::Pool& v) {
	allocator.destroy(v);
}

void vuk::Context::destroy(const Allocator::Linear& v) {
	allocator.destroy(v);
}


void vuk::Context::destroy(const vuk::DescriptorPool& dp) {
	for (auto& p : dp.pools) {
		device.destroy(p);
	}
}

void vuk::Context::destroy(const vuk::PipelineInfo& pi) {
	device.destroy(pi.pipeline);
}

void vuk::Context::destroy(const vuk::ShaderModule& sm) {
	device.destroy(sm.shader_module);
}

void vuk::Context::destroy(const vuk::DescriptorSetLayoutAllocInfo& ds) {
	device.destroy(ds.layout);
}

void vuk::Context::destroy(const vk::PipelineLayout& pl) {
	device.destroy(pl);
}

void vuk::Context::destroy(const vk::RenderPass& rp) {
	device.destroy(rp);
}

void vuk::Context::destroy(const vuk::DescriptorSet&) {
	// no-op, we destroy the pools
}

void vuk::Context::destroy(const vk::Framebuffer& fb) {
	device.destroy(fb);
}

void vuk::Context::destroy(const vk::Sampler& sa) {
	device.destroy(sa);
}

void vuk::Context::destroy(const vuk::PipelineBaseInfo& pbi) {
	// no-op, we don't own device objects
}

vuk::Context::~Context() {
	device.waitIdle();
	for (auto& s : swapchains) {
		for (auto& swiv : s.image_views) {
			device.destroy(swiv.payload);
		}
		device.destroy(s.swapchain);
    }
    for(auto& cp: one_time_pools) {
        if(cp != vk::CommandPool{}) {
            device.destroyCommandPool(cp);
        }
    }
    for(auto& cp: xfer_one_time_pools) {
        if(cp != vk::CommandPool{}) {
            device.destroyCommandPool(cp);
        }
    }
}

vuk::InflightContext vuk::Context::begin() {
	std::lock_guard _(begin_frame_lock);
	std::lock_guard recycle(recycle_locks[_next(frame_counter.load(), FC)]);
	return InflightContext(*this, ++frame_counter, std::move(recycle));
}

void vuk::Context::wait_idle() {
	device.waitIdle();
}

void vuk::InflightContext::destroy(std::vector<vk::ImageView>&& images) {
	std::lock_guard _(recycle_lock);
	ctx.image_view_recycle[frame].insert(ctx.image_view_recycle[frame].end(), images.begin(), images.end());
}

vuk::PerThreadContext vuk::InflightContext::begin() {
	return PerThreadContext{ *this, ctx.get_thread_index ? ctx.get_thread_index() : 0 };
}

