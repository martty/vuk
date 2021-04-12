#pragma once

#include <vuk/vuk_fwd.hpp>
#include <vuk/GlobalAllocator.hpp>
#include <vuk/Descriptor.hpp>
#include <vuk/Query.hpp>
#include <Pool.hpp>
#include <Allocator.hpp>
#include <Cache.hpp>
#include <vuk/SampledImage.hpp>
#include <vuk/Allocator.hpp>
#include <span>

namespace vuk {
	struct FrameAllocator;
	struct TokenData;

	// thread-unsafe per-frame allocator
	struct ThreadLocalFrameAllocator final : public Allocator {
		ThreadLocalFrameAllocator(FrameAllocator& fa, unsigned tid);

		Buffer allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment);
		Token allocate_token();
		VkCommandBuffer allocate_command_buffer(VkCommandBufferLevel, uint32_t queue_family_index);
		ImageView allocate_image_view(const ImageViewCreateInfo&);
		Sampler allocate_sampler(const SamplerCreateInfo&);
		DescriptorSet allocate_descriptorset(const SetBinding&);
		PipelineInfo allocate_pipeline(const PipelineInstanceCreateInfo&);
		VkFramebuffer allocate_framebuffer(const struct FramebufferCreateInfo&);
		VkRenderPass allocate_renderpass(const struct RenderPassCreateInfo&);
		VkSemaphore allocate_semaphore();
		VkSemaphore allocate_timeline_semaphore(uint64_t initial_value, uint64_t frame, SourceLocation) override;
		VkFence allocate_fence();
		struct RGImage allocate_rendertarget(const struct RGCI&);
		// ???
		TimestampQuery register_timestamp_query(Query);
		// sugars
		// Host visible buffer with data
		Buffer create_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, void* data, size_t size, size_t alignment);
		template<class T>
		Buffer create_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, std::span<T> data) {
			return create_buffer(mem_usage, buffer_usage, data.data(), data.size_bytes(), alignof(T));
		}

		void deallocate(vuk::DescriptorSet ds);
		void deallocate(vuk::Image image);
		void deallocate(vuk::ImageView image);
		
		void destroy(Token t) override;
		TokenData& get_token_data(Token t) override;

		unsigned tid = 0;
		struct ThreadLocalFrameAllocatorImpl* impl;
		FrameAllocator& parent;
	};

	struct RingFrameAllocator;
	struct FrameAllocatorImpl;

	/// thread-safe per-frame allocator
	/// 
	struct FrameAllocator final : public Allocator {
		FrameAllocator(RingFrameAllocator& rfa, uint64_t absolute_frame, unsigned frame);
		~FrameAllocator();

		VkSemaphore allocate_semaphore();
		VkFence allocate_fence();
		VkSemaphore allocate_timeline_semaphore(uint64_t initial_value, uint64_t frame, SourceLocation) override;

		void deallocate(Image);
		void deallocate(ImageView);
		void deallocate(VkPipeline);
		void deallocate(Buffer);
		void deallocate(PersistentDescriptorSet b);
		void deallocate(DescriptorSet ds);
		void deallocate(std::vector<vuk::Image>&& images);
		void deallocate(std::vector<VkImageView>&& images);
		//void deallocate(std::vector<LinearResourceAllocator*>&& lras);

		void destroy(Token) override;
		TokenData& get_token_data(Token) override;

		GlobalAllocator& parent;
		FrameAllocatorImpl* impl;
		uint64_t absolute_frame;
		unsigned frame; // frame index into frames-in-flight (aka. #frame % #frames-in-flight)
		const unsigned frames_in_flight;
	};

	struct RingFrameAllocator {
		RingFrameAllocator(GlobalAllocator& ga, size_t FC);
		~RingFrameAllocator();
		FrameAllocatorImpl* allocators;

		Context& ctx;
		GlobalAllocator& parent;

		size_t FC;
		Pool<VkCommandBuffer> commandbuffer_pools;
		Pool<TimestampQuery> tsquery_pools;
		Pool<VkSemaphore> semaphore_pools;
		Pool<VkFence> fence_pools;
		Pool<vuk::SampledImage> sampled_images;
		PerFrameCache<LinearAllocator> scratch_buffers;
		PerFrameCache<vuk::DescriptorSet> descriptor_sets;
		Cache<vuk::DescriptorPool> pool_cache;
	};
}