#include "vuk/Context.hpp"
#include "ContextImpl.hpp"

vuk::InflightContext::InflightContext(Context& ctx, size_t absolute_frame, std::lock_guard<std::mutex>&& recycle_guard) :
	ctx(ctx),
	absolute_frame(absolute_frame),
	frame(absolute_frame% Context::FC),
	fence_pools(ctx.impl->fence_pools.get_view(*this)), // must be first, so we wait for the fences
	commandbuffer_pools(ctx.impl->cbuf_pools.get_view(*this)),
	semaphore_pools(ctx.impl->semaphore_pools.get_view(*this)),
	pipeline_cache(*this, ctx.impl->pipeline_cache),
	compute_pipeline_cache(*this, ctx.impl->compute_pipeline_cache),
	pipelinebase_cache(*this, ctx.impl->pipelinebase_cache),
	renderpass_cache(*this, ctx.impl->renderpass_cache),
	framebuffer_cache(*this, ctx.impl->framebuffer_cache),
	transient_images(*this, ctx.impl->transient_images),
	scratch_buffers(*this, ctx.impl->scratch_buffers),
	descriptor_sets(*this, ctx.impl->descriptor_sets),
	sampler_cache(*this, ctx.impl->sampler_cache),
	sampled_images(ctx.impl->sampled_images.get_view(*this)),
	pool_cache(*this, ctx.impl->pool_cache),
	shader_modules(*this, ctx.impl->shader_modules),
	descriptor_set_layouts(*this, ctx.impl->descriptor_set_layouts),
	pipeline_layouts(*this, ctx.impl->pipeline_layouts) {

	// image recycling
	for (auto& img : ctx.impl->image_recycle[frame]) {
		ctx.impl->allocator.destroy_image(img);
	}
	ctx.impl->image_recycle[frame].clear();

	for (auto& iv : ctx.impl->image_view_recycle[frame]) {
		vkDestroyImageView(ctx.device, iv, nullptr);
	}
	ctx.impl->image_view_recycle[frame].clear();

	for (auto& p : ctx.impl->pipeline_recycle[frame]) {
		vkDestroyPipeline(ctx.device, p, nullptr);
	}
	ctx.impl->pipeline_recycle[frame].clear();

	for (auto& b : ctx.impl->buffer_recycle[frame]) {
		ctx.impl->allocator.free_buffer(b);
	}
	ctx.impl->buffer_recycle[frame].clear();

	for (auto& pds : ctx.impl->pds_recycle[frame]) {
		vkDestroyDescriptorPool(ctx.device, pds.backing_pool, nullptr);
	}
	ctx.impl->pds_recycle[frame].clear();

	for (auto& [k, v] : scratch_buffers.cache.data[frame].lru_map) {
		ctx.impl->allocator.reset_pool(v.value);
	}

	auto ptc = begin();
	ptc.descriptor_sets.collect(Context::FC * 2);
	ptc.transient_images.collect(Context::FC * 2);
	ptc.scratch_buffers.collect(Context::FC * 2);
}

vuk::PerThreadContext vuk::InflightContext::begin() {
	return PerThreadContext{ *this, ctx.get_thread_index ? ctx.get_thread_index() : 0 };
}

vuk::TransferStub vuk::InflightContext::enqueue_transfer(Buffer src, Buffer dst) {
	std::lock_guard _(transfer_mutex);
	TransferStub stub{ transfer_id++ };
	buffer_transfer_commands.push({ src, dst, stub });
	return stub;
}

vuk::TransferStub vuk::InflightContext::enqueue_transfer(Buffer src, vuk::Image dst, vuk::Extent3D extent, bool generate_mips) {
	std::lock_guard _(transfer_mutex);
	TransferStub stub{ transfer_id++ };
	bufferimage_transfer_commands.push({ src, dst, extent, generate_mips, stub });
	return stub;
}

void vuk::InflightContext::wait_all_transfers() {
	std::lock_guard _(transfer_mutex);

	while (!pending_transfers.empty()) {
		vkWaitForFences(ctx.device, 1, &pending_transfers.front().fence, true, UINT64_MAX);
		auto last = pending_transfers.front();
		last_transfer_complete = last.last_transfer_id;
		pending_transfers.pop();
	}
}

void vuk::InflightContext::destroy(std::vector<vuk::Image>&& images) {
	std::lock_guard _(recycle_lock);
	ctx.impl->image_recycle[frame].insert(ctx.impl->image_recycle[frame].end(), images.begin(), images.end());
}

void vuk::InflightContext::destroy(std::vector<VkImageView>&& images) {
	std::lock_guard _(recycle_lock);
	ctx.impl->image_view_recycle[frame].insert(ctx.impl->image_view_recycle[frame].end(), images.begin(), images.end());
}
