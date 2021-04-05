#include "vuk/Context.hpp"
#include "ContextImpl.hpp"
#include "Pool.hpp"

vuk::InflightContext::InflightContext(Context& ctx, size_t absolute_frame, std::lock_guard<std::mutex>&& recycle_guard) :
	ctx(ctx),
	absolute_frame(absolute_frame),
	frame(absolute_frame% Context::FC) {

	// extract query results before resetting
	std::unordered_map<uint64_t, uint64_t> query_results;
	for (auto& p : ctx.impl->tsquery_pools.per_frame_storage[frame]) {
		p.get_results(ctx);
		for (auto& [src, dst] : p.id_to_value_mapping) {
			query_results[src] = p.host_values[dst];
		}
	}

	impl = new IFCImpl(ctx, *this);

	impl->query_result_map = std::move(query_results);

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

	for (auto& lra : ctx.impl->lra_recycle[frame]) {
		ctx.impl->cleanup_transient_bundle_recursively(lra);
	}
	ctx.impl->lra_recycle[frame].clear();

	auto ptc = begin();
	ptc.impl->descriptor_sets.collect(Context::FC * 2);
	ctx.impl->transient_images.collect(absolute_frame, Context::FC * 2);
	ptc.impl->scratch_buffers.collect(Context::FC * 2);
}

vuk::InflightContext::~InflightContext() {
	delete impl;
}

vuk::PerThreadContext vuk::InflightContext::begin() {
	return PerThreadContext{ *this, ctx.get_thread_index ? ctx.get_thread_index() : 0 };
}

std::optional<uint64_t> vuk::InflightContext::get_timestamp_query_result(vuk::Query q) {
	auto it = impl->query_result_map.find(q.id);
	if (it != impl->query_result_map.end()) {
		return it->second;
	}
	return {};
}

std::optional<double> vuk::InflightContext::get_duration_query_result(vuk::Query q1, vuk::Query q2) {
	auto r1 = get_timestamp_query_result(q1);
	auto r2 = get_timestamp_query_result(q2);
	if (!r1 || !r2) {
		return {};
	}
	double period = ctx.impl->physical_device_properties.limits.timestampPeriod;
	auto ns = period * (r2.value() - r1.value());
	return ns * 1e-9;
}

vuk::TransferStub vuk::InflightContext::enqueue_transfer(Buffer src, Buffer dst) {
	std::lock_guard _(impl->transfer_mutex);
	TransferStub stub{ transfer_id++ };
	impl->buffer_transfer_commands.push({ src, dst, stub });
	return stub;
}

vuk::TransferStub vuk::InflightContext::enqueue_transfer(Buffer src, vuk::Image dst, vuk::Extent3D extent, uint32_t base_layer, bool generate_mips) {
	std::lock_guard _(impl->transfer_mutex);
	TransferStub stub{ transfer_id++ };
	// TODO: expose extra transfer knobs
	impl->bufferimage_transfer_commands.push({ src, dst, extent, base_layer, 1, 0, generate_mips, stub });
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

std::vector<vuk::SampledImage> vuk::InflightContext::get_sampled_images() {
	std::vector<vuk::SampledImage> sis;
	for (auto& p : impl->sampled_images.frame_values) {
		sis.insert(sis.end(), p.values.begin(), p.values.end());
	}
	return sis;
}

////// TEMP REGION

vuk::LinearResourceAllocator vuk::LinearResourceAllocator::clone() {
	assert(0);
	return *ctx->impl->get_linear_allocator(queue_family_index);
	/*next = alloc;
	return alloc;*/
}

VkFramebuffer vuk::LinearResourceAllocator::acquire_framebuffer(const vuk::FramebufferCreateInfo& fbci) {
	return ctx->impl->framebuffer_cache.acquire(fbci);
}

VkSemaphore vuk::LinearResourceAllocator::acquire_timeline_semaphore() {
	VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	VkSemaphoreTypeCreateInfo stci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
	stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	sci.pNext = &stci;
	VkSemaphore sema;
	vkCreateSemaphore(ctx->device, &sci, nullptr, &sema);
	return sema;
}

vuk::RGImage vuk::LinearResourceAllocator::acquire_rendertarget(const vuk::RGCI& rgci) {
	return ctx->impl->transient_images.acquire(rgci);
}

vuk::Sampler vuk::LinearResourceAllocator::acquire_sampler(const vuk::SamplerCreateInfo& sci) {
	return ctx->impl->sampler_cache.acquire(sci);
}

VkFence vuk::LinearResourceAllocator::acquire_fence() {
	return ctx->impl->get_unpooled_fence();
}

vuk::TimestampQuery vuk::LinearResourceAllocator::register_timestamp_query(vuk::Query handle) {
	assert(0);
	return { 0 };
}

vuk::PipelineInfo vuk::LinearResourceAllocator::acquire_pipeline(const vuk::PipelineInstanceCreateInfo& pici) {
	return ctx->impl->pipeline_cache.acquire(pici);
}

vuk::DescriptorSet vuk::LinearResourceAllocator::acquire_descriptorset(const vuk::SetBinding&) {
	assert(0);
	return {};
}

vuk::Buffer vuk::LinearResourceAllocator::allocate_scratch_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment) {
	assert(0);
	return {};
}