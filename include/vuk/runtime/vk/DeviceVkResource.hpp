#pragma once

#include "vuk/Config.hpp"
#include "vuk/runtime/vk/Allocator.hpp"

namespace vuk {
	/// @brief Device resource that performs direct allocation from the resources from the Vulkan runtime.
	struct DeviceVkResource final : DeviceResource {
		DeviceVkResource(Runtime& ctx);
		~DeviceVkResource();

		DeviceVkResource(DeviceVkResource&) = delete;
		DeviceVkResource& operator=(DeviceVkResource&) = delete;

		DeviceVkResource(DeviceVkResource&&) = delete;
		DeviceVkResource& operator=(DeviceVkResource&&) = delete;

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override;

		void deallocate_semaphores(std::span<const VkSemaphore> src) override;

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override;

		void deallocate_fences(std::span<const VkFence> src) override;

		Result<void, AllocateException> allocate_command_buffers(std::span<CommandBufferAllocation> dst,
		                                                         std::span<const CommandBufferAllocationCreateInfo> cis,
		                                                         SourceLocationAtFrame loc) override;

		void deallocate_command_buffers(std::span<const CommandBufferAllocation> dst) override;

		Result<void, AllocateException>
		allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_command_pools(std::span<const CommandPool> src) override;

		Result<void, AllocateException> allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const Buffer> src) override;

		void set_buffer_allocation_name(Buffer& dst, Name name) override final;

		Result<void, AllocateException>
		allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_framebuffers(std::span<const VkFramebuffer> src) override;

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_images(std::span<const Image> src) override;

		void set_image_allocation_name(Image& dst, Name name) override final;

		Result<void, AllocateException>
		allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_image_views(std::span<const ImageView> src) override;

		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
		                                                                    std::span<const PersistentDescriptorSetCreateInfo> cis,
		                                                                    SourceLocationAtFrame loc) override;

		void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) override;

		Result<void, AllocateException>
		allocate_descriptor_sets_with_value(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) override;

		Result<void, AllocateException>
		allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const DescriptorSetLayoutAllocInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_descriptor_sets(std::span<const DescriptorSet> src) override;

		Result<void, AllocateException>
		allocate_descriptor_pools(std::span<VkDescriptorPool> dst, std::span<const VkDescriptorPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_descriptor_pools(std::span<const VkDescriptorPool> src) override;

		Result<void, AllocateException>
		allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) override;

		Result<void, AllocateException>
		allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_timestamp_queries(std::span<const TimestampQuery> src) override; // no-op, deallocate pools

		void wait_sync_points(std::span<const SyncPoint> src) override; // no-op

		Result<void, AllocateException> allocate_acceleration_structures(std::span<VkAccelerationStructureKHR> dst,
		                                                                 std::span<const VkAccelerationStructureCreateInfoKHR> cis,
		                                                                 SourceLocationAtFrame loc) override;

		void deallocate_acceleration_structures(std::span<const VkAccelerationStructureKHR> src) override;

		void deallocate_swapchains(std::span<const VkSwapchainKHR> src) override;

		Result<void, AllocateException> allocate_graphics_pipelines(std::span<GraphicsPipelineInfo> dst,
		                                                            std::span<const GraphicsPipelineInstanceCreateInfo> cis,
		                                                            SourceLocationAtFrame loc) override;
		void deallocate_graphics_pipelines(std::span<const GraphicsPipelineInfo> src) override;

		Result<void, AllocateException>
		allocate_compute_pipelines(std::span<ComputePipelineInfo> dst, std::span<const ComputePipelineInstanceCreateInfo> cis, SourceLocationAtFrame loc) override;
		void deallocate_compute_pipelines(std::span<const ComputePipelineInfo> src) override;

		Result<void, AllocateException> allocate_ray_tracing_pipelines(std::span<RayTracingPipelineInfo> dst,
		                                                               std::span<const RayTracingPipelineInstanceCreateInfo> cis,
		                                                               SourceLocationAtFrame loc) override;
		void deallocate_ray_tracing_pipelines(std::span<const RayTracingPipelineInfo> src) override;

		Result<void, AllocateException>
		allocate_render_passes(std::span<VkRenderPass> dst, std::span<const RenderPassCreateInfo> cis, SourceLocationAtFrame loc) override;
		void deallocate_render_passes(std::span<const VkRenderPass> src) override;

		Runtime& get_context() override {
			return *ctx;
		}

		Runtime* ctx;
		VkDevice device;

	private:
		struct DeviceVkResourceImpl* impl;
	};
} // namespace vuk
