#pragma once

#include "../src/LegacyGPUAllocator.hpp"
#include "vuk/Allocator.hpp"
#include "vuk/Exception.hpp"
#include "vuk/Query.hpp"
#include "vuk/resources/DeviceNestedResource.hpp"
#include "vuk/resources/DeviceVkResource.hpp"

#include <atomic>

namespace vuk {
	/// @brief Represents "per-frame" resources - temporary allocations that persist through a frame. Handed out by DeviceSuperFrameResource, cannot be constructed directly.
	///
	/// Allocations from this resource are tied to the "frame" - all allocations recycled when a DeviceFrameResource is recycled.
	/// Furthermore all resources allocated are also deallocated at recycle time - it is not necessary (but not an error) to deallocate them.
	struct DeviceFrameResource : DeviceNestedResource {
		std::mutex sema_mutex;
		std::vector<VkSemaphore> semaphores;

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override;

		void deallocate_semaphores(std::span<const VkSemaphore> src) override; // noop

		std::mutex fence_mutex;
		std::vector<VkFence> fences;

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override;

		void deallocate_fences(std::span<const VkFence> src) override; // noop

		std::mutex cbuf_mutex;
		std::vector<CommandBufferAllocation> cmdbuffers_to_free;
		std::vector<CommandPool> cmdpools_to_free;

		Result<void, AllocateException> allocate_command_buffers(std::span<CommandBufferAllocation> dst,
		                                                         std::span<const CommandBufferAllocationCreateInfo> cis,
		                                                         SourceLocationAtFrame loc) override;

		void deallocate_command_buffers(std::span<const CommandBufferAllocation> src) override; // no-op, deallocated with pools

		Result<void, AllocateException>
		allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_command_pools(std::span<const CommandPool> dst) override; // no-op

		// buffers are lockless
		Result<void, AllocateException>
		allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const BufferCrossDevice> src) override; // no-op, linear

		Result<void, AllocateException> allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const BufferGPU> src) override; // no-op, linear

		std::mutex framebuffer_mutex;
		std::vector<VkFramebuffer> framebuffers;

		Result<void, AllocateException>
		allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_framebuffers(std::span<const VkFramebuffer> src) override; // noop

		std::mutex images_mutex;
		std::vector<Image> images;

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_images(std::span<const Image> src) override; // noop

		std::mutex image_views_mutex;
		std::vector<ImageView> image_views;

		Result<void, AllocateException>
		allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_image_views(std::span<const ImageView> src) override; // noop

		std::mutex pds_mutex;
		std::vector<PersistentDescriptorSet> persistent_descriptor_sets;

		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
		                                                                    std::span<const PersistentDescriptorSetCreateInfo> cis,
		                                                                    SourceLocationAtFrame loc) override;

		void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) override; // noop

		std::mutex ds_mutex;
		std::vector<DescriptorSet> descriptor_sets;

		Result<void, AllocateException> allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) override;

		void deallocate_descriptor_sets(std::span<const DescriptorSet> src) override; // noop

		// only for use via SuperframeAllocator
		std::mutex buffers_mutex;
		std::vector<BufferGPU> buffer_gpus;
		std::vector<BufferCrossDevice> buffer_cross_devices;

		std::vector<TimestampQueryPool> ts_query_pools;
		std::mutex query_pool_mutex;

		Result<void, AllocateException>
		allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) override; // noop

		std::mutex ts_query_mutex;
		uint64_t query_index = 0;
		uint64_t current_ts_pool;

		Result<void, AllocateException>
		allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_timestamp_queries(std::span<const TimestampQuery> src) override; // noop

		std::mutex tsema_mutex;
		std::vector<TimelineSemaphore> tsemas;

		Result<void, AllocateException> allocate_timeline_semaphores(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc) override;

		void deallocate_timeline_semaphores(std::span<const TimelineSemaphore> src) override; // noop

		std::mutex swapchain_mutex;
		std::vector<VkSwapchainKHR> swapchains;

		void deallocate_swapchains(std::span<const VkSwapchainKHR> src) override;

		/// @brief Wait for the fences / timeline semaphores referencing this frame to complete
		///
		/// Called automatically when recycled
		void wait();

		/// @brief Retrieve the parent Context
		/// @return the parent Context
		Context& get_context() override {
			return upstream->get_context();
		}

	private:
		VkDevice device;
		uint64_t current_frame = -1;
		LegacyLinearAllocator linear_cpu_only;
		LegacyLinearAllocator linear_cpu_gpu;
		LegacyLinearAllocator linear_gpu_cpu;
		LegacyLinearAllocator linear_gpu_only;

		friend struct DeviceSuperFrameResource;

		DeviceFrameResource(VkDevice device, DeviceSuperFrameResource& upstream);
	};

	/// @brief DeviceSuperFrameResource is an allocator that gives out DeviceFrameResource allocators, and manages their resources
	///
	/// DeviceSuperFrameResource models resource lifetimes that span multiple frames - these can be allocated directly from this resource
	/// Allocation of these resources are persistent, and they can be deallocated at any time - they will be recycled when the current frame is recycled
	/// This resource also hands out DeviceFrameResources in a round-robin fashion.
	/// The lifetime of resources allocated from those allocators is frames_in_flight number of frames (until the DeviceFrameResource is recycled).
	struct DeviceSuperFrameResource : DeviceResource {
		DeviceSuperFrameResource(Context& ctx, uint64_t frames_in_flight);

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override;

		void deallocate_semaphores(std::span<const VkSemaphore> src) override;

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override;

		void deallocate_fences(std::span<const VkFence> src) override;

		Result<void, AllocateException> allocate_command_buffers(std::span<CommandBufferAllocation> dst,
		                                                         std::span<const CommandBufferAllocationCreateInfo> cis,
		                                                         SourceLocationAtFrame loc) override;

		void deallocate_command_buffers(std::span<const CommandBufferAllocation> src) override;

		Result<void, AllocateException>
		allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_command_pools(std::span<const CommandPool> src) override;

		Result<void, AllocateException>
		allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const BufferCrossDevice> src) override;

		Result<void, AllocateException> allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const BufferGPU> src) override;

		Result<void, AllocateException>
		allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_framebuffers(std::span<const VkFramebuffer> src) override;

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_images(std::span<const Image> src) override;

		Result<void, AllocateException>
		allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_image_views(std::span<const ImageView> src) override;

		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
		                                                                    std::span<const PersistentDescriptorSetCreateInfo> cis,
		                                                                    SourceLocationAtFrame loc) override;

		void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) override;

		Result<void, AllocateException> allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) override;

		void deallocate_descriptor_sets(std::span<const DescriptorSet> src) override;

		Result<void, AllocateException>
		allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) override;

		Result<void, AllocateException>
		allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_timestamp_queries(std::span<const TimestampQuery> src) override; // noop

		Result<void, AllocateException> allocate_timeline_semaphores(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc) override;

		void deallocate_timeline_semaphores(std::span<const TimelineSemaphore> src) override;

		void deallocate_swapchains(std::span<const VkSwapchainKHR> src) override;

		/// @brief Recycle the least-recently-used frame and return it to be used again
		/// @return DeviceFrameResource for use
		DeviceFrameResource& get_next_frame();

		virtual ~DeviceSuperFrameResource();

		Context& get_context() override {
			return *direct.ctx;
		}

		DeviceVkResource direct;
		std::mutex new_frame_mutex;
		std::atomic<uint64_t> frame_counter;
		std::atomic<uint64_t> local_frame;
		const uint64_t frames_in_flight;

	private:
		DeviceFrameResource& get_last_frame();
		void deallocate_frame(DeviceFrameResource& f);

		std::unique_ptr<char[]> frames_storage;
		DeviceFrameResource* frames;

		std::mutex command_pool_mutex;
		std::array<std::vector<VkCommandPool>, 3> command_pools;
	};
} // namespace vuk