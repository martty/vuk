#pragma once

#include "vuk/runtime/Allocator.hpp"
#include "vuk/runtime/DeviceNestedResource.hpp"
#include "vuk/runtime/vk/DeviceVkResource.hpp"

#include <memory>

namespace vuk {
	/// @brief Represents resources not tied to a frame, that are deallocated only when the resource is destroyed. Not thread-safe.
	///
	/// Allocations from this resource are deallocated into the upstream resource when the DeviceLinearResource is destroyed.
	/// All resources allocated are automatically deallocated at recycle time - it is not necessary (but not an error) to deallocate them.
	struct DeviceLinearResource : DeviceNestedResource {
		DeviceLinearResource(DeviceResource& upstream);
		~DeviceLinearResource();

		DeviceLinearResource(DeviceLinearResource&&);
		DeviceLinearResource& operator=(DeviceLinearResource&&);

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override;

		void deallocate_semaphores(std::span<const VkSemaphore> src) override; // noop

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override;

		void deallocate_fences(std::span<const VkFence> src) override; // noop

		Result<void, AllocateException> allocate_command_buffers(std::span<CommandBufferAllocation> dst,
		                                                         std::span<const CommandBufferAllocationCreateInfo> cis,
		                                                         SourceLocationAtFrame loc) override;

		void deallocate_command_buffers(std::span<const CommandBufferAllocation> src) override; // no-op, deallocated with pools

		Result<void, AllocateException>
		allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_command_pools(std::span<const CommandPool> dst) override; // no-op

		// buffers are lockless
		Result<void, AllocateException> allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const Buffer> src) override; // no-op, linear

		Result<void, AllocateException>
		allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_framebuffers(std::span<const VkFramebuffer> src) override; // noop

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_images(std::span<const Image> src) override; // noop

		Result<void, AllocateException>
		allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_image_views(std::span<const ImageView> src) override; // noop

		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
		                                                                    std::span<const PersistentDescriptorSetCreateInfo> cis,
		                                                                    SourceLocationAtFrame loc) override;

		void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) override; // noop

		Result<void, AllocateException>
		allocate_descriptor_sets_with_value(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) override;

		Result<void, AllocateException>
		allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const DescriptorSetLayoutAllocInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_descriptor_sets(std::span<const DescriptorSet> src) override;

		Result<void, AllocateException>
		allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) override; // noop

		Result<void, AllocateException>
		allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_timestamp_queries(std::span<const TimestampQuery> src) override; // noop

		void wait_sync_points(std::span<const SyncPoint> src) override;

		/// @brief Wait for the fences / timeline semaphores referencing this allocator
		void wait();

		/// @brief Release the resources of this resource into the upstream
		void free();

		/// @brief Retrieve the parent Runtime
		/// @return the parent Runtime
		Runtime& get_context() override {
			return upstream->get_context();
		}

	private:
		std::unique_ptr<struct DeviceLinearResourceImpl> impl;

		friend struct DeviceLinearResourceImpl;
	};
} // namespace vuk