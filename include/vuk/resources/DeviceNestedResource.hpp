#pragma once

#include "vuk/Allocator.hpp"
#include "vuk/Exception.hpp"
#include "vuk/vuk_fwd.hpp"

namespace vuk {
	/// @brief Helper base class for DeviceResources. Forwards all allocations and deallocations to the upstream DeviceResource.
	struct DeviceNestedResource : DeviceResource {
		DeviceNestedResource(DeviceResource* upstream) : upstream(upstream) {}

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override;

		void deallocate_semaphores(std::span<const VkSemaphore> sema) override;

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override;

		void deallocate_fences(std::span<const VkFence> dst) override;

		Result<void, AllocateException> allocate_command_buffers(std::span<CommandBufferAllocation> dst,
		                                                         std::span<const CommandBufferAllocationCreateInfo> cis,
		                                                         SourceLocationAtFrame loc) override;

		void deallocate_command_buffers(std::span<const CommandBufferAllocation> dst) override;

		Result<void, AllocateException>
		allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_command_pools(std::span<const CommandPool> dst) override;

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

		void deallocate_timestamp_queries(std::span<const TimestampQuery> src) override;

		Result<void, AllocateException> allocate_timeline_semaphores(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc) override;

		void deallocate_timeline_semaphores(std::span<const TimelineSemaphore> src) override;

		Result<void, AllocateException> allocate_acceleration_structures(std::span<VkAccelerationStructureKHR> dst,
		                                                                 std::span<const VkAccelerationStructureCreateInfoKHR> cis,
		                                                                 SourceLocationAtFrame loc) override;

		void deallocate_acceleration_structures(std::span<const VkAccelerationStructureKHR> src) override;

		void deallocate_swapchains(std::span<const VkSwapchainKHR> src) override;

		DeviceResource* upstream = nullptr;
	};
} // namespace vuk