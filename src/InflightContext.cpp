#include "vuk/Context.hpp"
#include "ContextImpl.hpp"

vuk::InflightContext::InflightContext(Context& ctx, size_t absolute_frame, std::lock_guard<std::mutex>&& recycle_guard) :
	ctx(ctx),
	absolute_frame(absolute_frame),
	frame(absolute_frame% Context::FC),
	impl(new IFCImpl(ctx, *this)){

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

	for (auto& [k, v] : impl->scratch_buffers.cache.data[frame].lru_map) {
		ctx.impl->allocator.reset_pool(v.value);
	}

	auto ptc = begin();
	ptc.impl->descriptor_sets.collect(Context::FC * 2);
	ptc.impl->transient_images.collect(Context::FC * 2);
	ptc.impl->scratch_buffers.collect(Context::FC * 2);
}

vuk::InflightContext::~InflightContext() {
	delete impl;
}

vuk::PerThreadContext vuk::InflightContext::begin() {
	return PerThreadContext{ *this, ctx.get_thread_index ? ctx.get_thread_index() : 0 };
}

vuk::TransferStub vuk::InflightContext::enqueue_transfer(Buffer src, Buffer dst) {
	std::lock_guard _(impl->transfer_mutex);
	TransferStub stub{ transfer_id++ };
	impl->buffer_transfer_commands.push({ src, dst, stub });
	return stub;
}

vuk::TransferStub vuk::InflightContext::enqueue_transfer(Buffer src, vuk::Image dst, vuk::Extent3D extent, bool generate_mips) {
	std::lock_guard _(impl->transfer_mutex);
	TransferStub stub{ transfer_id++ };
	impl->bufferimage_transfer_commands.push({ src, dst, extent, generate_mips, stub });
	return stub;
}

void vuk::InflightContext::wait_all_transfers() {
	std::lock_guard _(impl->transfer_mutex);

	while (!impl->pending_transfers.empty()) {
		vkWaitForFences(ctx.device, 1, &impl->pending_transfers.front().fence, true, UINT64_MAX);
		auto last = impl->pending_transfers.front();
		last_transfer_complete = last.last_transfer_id;
		impl->pending_transfers.pop();
	}
}

void vuk::InflightContext::destroy(std::vector<vuk::Image>&& images) {
	std::lock_guard _(impl->recycle_lock);
	ctx.impl->image_recycle[frame].insert(ctx.impl->image_recycle[frame].end(), images.begin(), images.end());
}

void vuk::InflightContext::destroy(std::vector<VkImageView>&& images) {
	std::lock_guard _(impl->recycle_lock);
	ctx.impl->image_view_recycle[frame].insert(ctx.impl->image_view_recycle[frame].end(), images.begin(), images.end());
}