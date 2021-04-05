#pragma once

#include <vuk/vuk_fwd.hpp>
#include <vuk/GlobalAllocator.hpp>
#include <vuk/Descriptor.hpp>
#include <Pool.hpp>
#include <Cache.hpp>
#include <vuk/SampledImage.hpp>

namespace vuk {
	struct FrameAllocator;

	// thread-unsafe per-frame allocator
	struct ThreadLocalFrameAllocator {
		PTPoolView<VkCommandBuffer> commandbuffer_pool;
		PTPoolView<VkSemaphore> semaphore_pool;
		PTPoolView<VkFence> fence_pool;
		PTPoolView<TimestampQuery> tsquery_pool;
		PTPoolView<vuk::SampledImage> sampled_images;
		PerFrameCacheView<LinearAllocator> scratch_buffers;
		PerFrameCacheView<vuk::DescriptorSet> descriptor_sets;

		// recycling global objects
		std::vector<Buffer> buffer_recycle;
		std::vector<vuk::Image> image_recycle;
		std::vector<VkImageView> image_view_recycle;
		std::vector<LinearResourceAllocator*> linear_allocators;

		void deallocate(vuk::DescriptorSet ds);
		void deallocate(vuk::Image image);
		void deallocate(vuk::ImageView image);

		FrameAllocator& parent;
	};

	template<size_t FC> struct RingFrameAllocator;

	/// thread-safe per-frame allocator
	/// 
	struct FrameAllocator {
		template<size_t FC>
		FrameAllocator(RingFrameAllocator<FC>& rfa, unsigned frame);
		~FrameAllocator();

		void reset();

		void deallocate(Image);
		void deallocate(ImageView);
		void deallocate(VkPipeline);
		void deallocate(Buffer);
		void deallocate(PersistentDescriptorSet b);
		void deallocate(std::vector<vuk::Image>&& images);
		void deallocate(std::vector<VkImageView>&& images);
		void deallocate(std::vector<LinearResourceAllocator*>&& lras);

		GlobalAllocator& parent;
		struct FrameAllocatorImpl* impl;
		unsigned frame; // frame index into frames-in-flight (aka. #frame % #frames-in-flight)
		const unsigned frames_in_flight;
	};

	template<size_t FC>
	struct RingFrameAllocator {
		RingFrameAllocator(GlobalAllocator& ga) :
			cbuf_pools(ga),
			tsquery_pools(ga),
			semaphore_pools(ga),
			fence_pools(ga),
			scratch_buffers(ga),
			descriptor_sets(ga),
			sampled_images(ga),
			parent(ga) {

		}

		Pool<VkCommandBuffer, FC> cbuf_pools;
		Pool<TimestampQuery, FC> tsquery_pools;
		Pool<VkSemaphore, FC> semaphore_pools;
		Pool<VkFence, FC> fence_pools;
		Pool<vuk::SampledImage, FC> sampled_images;
		PerFrameCache<LinearAllocator, FC> scratch_buffers;
		PerFrameCache<vuk::DescriptorSet, FC> descriptor_sets;

		std::vector<FrameAllocatorImpl> allocators;
		GlobalAllocator& parent;
	};

	template<size_t FC>
	FrameAllocator::FrameAllocator(RingFrameAllocator<FC>& rfa, unsigned frame)
		: parent(rfa.parent), frame(frame), frames_in_flight(FC), impl(rfa.allocators[frame]) {
		reset();
	}
}