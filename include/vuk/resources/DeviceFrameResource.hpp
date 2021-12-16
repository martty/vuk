#pragma once

#include "vuk/Allocator.hpp"
#include "vuk/resources/DeviceNestedResource.hpp"
#include "vuk/resources/DeviceVkResource.hpp"
#include "../src/LegacyGPUAllocator.hpp"
#include "vuk/Exception.hpp"
#include "vuk/Query.hpp"

#include <atomic>

namespace vuk {
	struct DeviceSuperFrameResource;

	/// @brief Represents "per-frame" resources - temporary allocations that persist through a frame. Can only be used via the DeviceSuperFrameResource
	struct DeviceFrameResource : DeviceNestedResource {
		DeviceFrameResource(VkDevice device, DeviceSuperFrameResource& upstream);

		std::mutex sema_mutex;
		std::vector<VkSemaphore> semaphores;

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override;

		void deallocate_semaphores(std::span<const VkSemaphore> src) override; // noop

		std::mutex fence_mutex;
		std::vector<VkFence> fences;

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override;

		void deallocate_fences(std::span<const VkFence> src) override; // noop

		std::mutex cbuf_mutex;
		std::vector<VkCommandPool> cmdpools_to_free;

		Result<void, AllocateException> allocate_command_buffers(std::span<CommandBufferAllocation> dst, std::span<const CommandBufferAllocationCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_command_buffers(std::span<const CommandBufferAllocation> src) override; // no-op, deallocated with pools

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_commandpools(std::span<const VkCommandPool> dst) override; // no-op

		// buffers are lockless
		Result<void, AllocateException> allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const BufferCrossDevice> src) override; // no-op, linear

		Result<void, AllocateException> allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const BufferGPU> src) override; // no-op, linear

		std::mutex framebuffer_mutex;
		std::vector<VkFramebuffer> framebuffers;

		Result<void, AllocateException> allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_framebuffers(std::span<const VkFramebuffer> src) override; // noop

		std::mutex images_mutex;
		std::vector<Image> images;

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_images(std::span<const Image> src) override; // noop

		std::mutex image_views_mutex;
		std::vector<ImageView> image_views;

		Result<void, AllocateException> allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_image_views(std::span<const ImageView> src) override; // noop

		std::mutex pds_mutex;
		std::vector<PersistentDescriptorSet> persistent_descriptor_sets;

		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) override; // noop

		std::mutex ds_mutex;
		std::vector<DescriptorSet> descriptor_sets;

		Result<void, AllocateException> allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) override;

		void deallocate_descriptor_sets(std::span<const DescriptorSet> src) override; // noop

		// only for use via SuperframeAllocator
		std::mutex buffers_mutex;
		std::vector<BufferGPU> buffer_gpus;
		std::vector<BufferCrossDevice> buffer_cross_devices;

		template<class T>
		struct LRUEntry {
			T value;
			size_t last_use_frame;
		};

		template<class T>
		struct Cache {
			robin_hood::unordered_map<create_info_t<T>, LRUEntry<T>> lru_map;
			std::array<std::vector<T>, 32> per_thread_append_v;
			std::array<std::vector<create_info_t<T>>, 32> per_thread_append_k;

			std::mutex cache_mtx;

			T& acquire(uint64_t current_frame, const create_info_t<T>& ci);
			void collect(uint64_t current_frame, size_t threshold);
		};

		Cache<DescriptorSet> descriptor_set_cache;

		std::vector<TimestampQueryPool> ts_query_pools;
		std::mutex query_pool_mutex;

		Result<void, AllocateException> allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) override; // noop

		std::mutex ts_query_mutex;
		uint64_t query_index = 0;
		uint64_t current_ts_pool;

		Result<void, AllocateException> allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_timestamp_queries(std::span<const TimestampQuery> src) override; // noop

		std::mutex tsema_mutex;
		std::vector<TimelineSemaphore> tsemas;

		Result<void, AllocateException> allocate_timeline_semaphores(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc) override;

		void deallocate_timeline_semaphores(std::span<const TimelineSemaphore> src) override; // noop

		void wait();

		Context& get_context() override {
			return upstream->get_context();
		}

		VkDevice device;
		uint64_t current_frame = -1;
		LegacyLinearAllocator linear_cpu_only;
		LegacyLinearAllocator linear_cpu_gpu;
		LegacyLinearAllocator linear_gpu_cpu;
		LegacyLinearAllocator linear_gpu_only;
	};

	/// @brief DeviceSuperFrameResource is an allocator that gives out DeviceFrameResource allocators, and manages their resources
	struct DeviceSuperFrameResource : DeviceResource {
		DeviceSuperFrameResource(Context& ctx, uint64_t frames_in_flight);

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override;

		void deallocate_semaphores(std::span<const VkSemaphore> src) override;

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override;

		void deallocate_fences(std::span<const VkFence> src) override;

		Result<void, AllocateException> allocate_command_buffers(std::span<CommandBufferAllocation> dst, std::span<const CommandBufferAllocationCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_command_buffers(std::span<const CommandBufferAllocation> src) override;

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_commandpools(std::span<const VkCommandPool> src) override;

		Result<void, AllocateException> allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const BufferCrossDevice> src) override;

		Result<void, AllocateException> allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const BufferGPU> src) override;

		Result<void, AllocateException> allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_framebuffers(std::span<const VkFramebuffer> src) override;

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_images(std::span<const Image> src) override;

		Result<void, AllocateException> allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_image_views(std::span<const ImageView> src) override;

		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) override;

		Result<void, AllocateException> allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) override;
		void deallocate_descriptor_sets(std::span<const DescriptorSet> src) override;

		Result<void, AllocateException> allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) override;

		Result<void, AllocateException> allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_timestamp_queries(std::span<const TimestampQuery> src) override; // noop

		Result<void, AllocateException> allocate_timeline_semaphores(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc) override;

		void deallocate_timeline_semaphores(std::span<const TimelineSemaphore> src) override;

		DeviceFrameResource& get_next_frame();

		void deallocate_frame(DeviceFrameResource& f);

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
		std::unique_ptr<char[]> frames_storage;
		DeviceFrameResource* frames;
	};
}