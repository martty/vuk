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

	for (auto& fb : ctx.impl->fb_recycle[frame]) {
		vkDestroyFramebuffer(ctx.device, fb, nullptr);
	}
	ctx.impl->fb_recycle[frame].clear();

	for (auto& [k, v] : impl->scratch_buffers.cache.data[frame].lru_map) {
		ctx.impl->allocator.reset_pool(v.value);
	}

	auto ptc = begin();
	ptc.impl->descriptor_sets.collect(Context::FC * 2);
	ptc.impl->transient_images.collect(Context::FC * 2);
	ptc.impl->scratch_buffers.collect(Context::FC * 2);
	// collect rarer resources
	static constexpr uint32_t cache_collection_frequency = 16;
	auto remainder = absolute_frame % cache_collection_frequency;
	switch (remainder) {
	case 0:
		ptc.impl->pipeline_cache.collect(cache_collection_frequency); break;
	case 1:
		ptc.impl->compute_pipeline_cache.collect(cache_collection_frequency); break;
	case 2:
		ptc.impl->renderpass_cache.collect(cache_collection_frequency); break;
	case 3:
		ptc.impl->sampler_cache.collect(cache_collection_frequency); break;
	case 4:
		ptc.impl->pipeline_layouts.collect(cache_collection_frequency); break;
	case 5:
		ptc.impl->pipelinebase_cache.collect(cache_collection_frequency); break;
	case 6:
		ptc.impl->compute_pipelinebase_cache.collect(cache_collection_frequency); break;
	}
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

void vuk::InflightContext::destroy(std::vector<vuk::Image>&& images) {
	std::lock_guard _(impl->recycle_lock);
	ctx.impl->image_recycle[frame].insert(ctx.impl->image_recycle[frame].end(), images.begin(), images.end());
}

void vuk::InflightContext::destroy(std::vector<VkImageView>&& images) {
	std::lock_guard _(impl->recycle_lock);
	ctx.impl->image_view_recycle[frame].insert(ctx.impl->image_view_recycle[frame].end(), images.begin(), images.end());
}

std::vector<vuk::SampledImage> vuk::InflightContext::get_sampled_images() {
	std::vector<vuk::SampledImage> sis;
	for (auto& p : impl->sampled_images.frame_values) {
		sis.insert(sis.end(), p.values.begin(), p.values.end());
	}
	return sis;
}