#include <vuk/FrameAllocator.hpp>
#include <ResourceBundle.hpp>
#include <array>
#include <vector>
#include <mutex>
#include <vuk/Types.hpp>
#include <vuk/Image.hpp>
#include <vuk/Buffer.hpp>

namespace vuk {
	struct FrameAllocatorImpl {
		std::mutex recycle_lock;
		std::vector<vuk::Image> image_recycle;
		std::vector<VkImageView> image_view_recycle;
		std::vector<VkPipeline> pipeline_recycle;
		std::vector<vuk::Buffer> buffer_recycle;
		std::vector<vuk::PersistentDescriptorSet> pds_recycle;
		std::vector<LinearResourceAllocator*> lra_recycle;

		PoolView<VkFence> fence_pools; // must be first, so we wait for the fences
		PoolView<VkCommandBuffer> commandbuffer_pools;
		PoolView<TimestampQuery> tsquery_pools;
		PoolView<VkSemaphore> semaphore_pools;
		PoolView<vuk::SampledImage> sampled_images;
		PerFrameCacheView<LinearAllocator> scratch_buffers;
		PerFrameCacheView<vuk::DescriptorSet> descriptor_sets;
	};

	FrameAllocator::~FrameAllocator() {
	}

	void FrameAllocator::reset() {
		// extract query results before resetting
		/*std::unordered_map<uint64_t, uint64_t> query_results;
		for (auto& p : impl->tsquery_pools.per_frame_storage) {
			p.get_results(ctx);
			for (auto& [src, dst] : p.id_to_value_mapping) {
				query_results[src] = p.host_values[dst];
			}
		}

		impl->query_result_map = std::move(query_results);*/

		impl->fence_pools.reset(); // must be first, so we wait for the fences
		impl->commandbuffer_pools.reset();
		impl->tsquery_pools.reset();
		impl->semaphore_pools.reset();
		impl->sampled_images.reset();

		// image recycling
		for (auto& img : impl->image_recycle) {
			parent.deallocate(img);
		}
		impl->image_recycle.clear();

		for (auto& iv : impl->image_view_recycle) {
			parent.deallocate(iv);
		}
		impl->image_view_recycle.clear();

		for (auto& p : impl->pipeline_recycle) {
			parent.deallocate(p);
		}
		impl->pipeline_recycle.clear();

		for (auto& b : impl->buffer_recycle) {
			parent.deallocate(b);
		}
		impl->buffer_recycle.clear();

		for (auto& pds : impl->pds_recycle) {
			parent.deallocate(pds);
		}
		impl->pds_recycle.clear();

		for (auto& [k, v] : impl->scratch_buffers.cache.data.lru_map) {
			impl->allocator.reset_pool(v.value);
		}

		/*
		for (auto& lra : impl->lra_recycle) {
			impl->cleanup_transient_bundle_recursively(lra);
		}
		impl->lra_recycle.clear();
		*/

		impl->descriptor_sets.collect(Context::FC * 2);
		impl->transient_images.collect(absolute_frame, Context::FC * 2);
		impl->scratch_buffers.collect(Context::FC * 2);
	}

	void FrameAllocator::deallocate(Image i) {
		std::lock_guard _(impl->recycle_lock);
		impl->image_recycle.push_back(i);
	}

	void FrameAllocator::deallocate(ImageView iv) {
		std::lock_guard _(impl->recycle_lock);
		impl->image_view_recycle.push_back(iv.payload);
	}

	void FrameAllocator::deallocate(VkPipeline p) {
		std::lock_guard _(impl->recycle_lock);
		impl->pipeline_recycle.push_back(p);
	}

	void FrameAllocator::deallocate(Buffer b) {
		std::lock_guard _(impl->recycle_lock);
		impl->buffer_recycle.push_back(b);
	}

	void FrameAllocator::deallocate(PersistentDescriptorSet b) {
		std::lock_guard _(impl->recycle_lock);
		impl->pds_recycle.push_back(std::move(b));
	}

	void FrameAllocator::deallocate(std::vector<vuk::Image>&& images) {
		std::lock_guard _(impl->recycle_lock);
		impl->image_recycle.insert(impl->image_recycle.end(), images.begin(), images.end());
	}

	void FrameAllocator::deallocate(std::vector<VkImageView>&& images) {
		std::lock_guard _(impl->recycle_lock);
		impl->image_view_recycle.insert(impl->image_view_recycle.end(), images.begin(), images.end());
	}

	void FrameAllocator::deallocate(std::vector<LinearResourceAllocator*>&& lras) {
		std::lock_guard _(impl->recycle_lock);
		impl->lra_recycle.insert(impl->lra_recycle.end(), lras.begin(), lras.end());
	}


	// -------------------- TL
	void ThreadLocalFrameAllocator::deallocate(vuk::DescriptorSet ds) {
		// note that since we collect at integer times FC, we are releasing the DS back to the right pool
		parent.impl->pool_cache.acquire(ds.layout_info, ifc->absolute_frame).free_sets.enqueue(ds.descriptor_set);
	}

	void ThreadLocalFrameAllocator::deallocate(vuk::Image image) {
		image_recycle.push_back(image);
	}

	void ThreadLocalFrameAllocator::deallocate(vuk::ImageView image) {
		image_view_recycle.push_back(image.payload);
	}
}