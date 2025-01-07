#pragma once

#include "vuk/Exception.hpp"
#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/vuk_fwd.hpp"

namespace vuk {
	/// @brief Helper base class for DeviceResources. Forwards all allocations and deallocations to the upstream DeviceResource.
	struct DeviceNestedResource : DeviceResource {
		explicit DeviceNestedResource(DeviceResource& upstream) : upstream(&upstream) {}

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

		Result<void, AllocateException> allocate_memory(std::span<ptr_base> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;
		void deallocate_memory(std::span<const ptr_base> dst) override;

		Result<void, AllocateException> allocate_memory_views(std::span<generic_view_base> dst, std::span<const BVCI> cis, SourceLocationAtFrame loc) override;
		void deallocate_memory_views(std::span<const generic_view_base> dst) override;

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

		void deallocate_timestamp_queries(std::span<const TimestampQuery> src) override;

		void wait_sync_points(std::span<const SyncPoint> src) override;

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

		Result<void, AllocateException>
		allocate_virtual_address_spaces(std::span<VirtualAddressSpace> dst, std::span<const VirtualAddressSpaceCreateInfo> cis, SourceLocationAtFrame loc) override;
		void deallocate_virtual_address_spaces(std::span<const VirtualAddressSpace> src) override;

		Result<void, AllocateException>
		allocate_virtual_allocations(std::span<VirtualAllocation> dst, std::span<const VirtualAllocationCreateInfo> cis, SourceLocationAtFrame loc) override;
		void deallocate_virtual_allocations(std::span<const VirtualAllocation> src) override;

		Runtime& get_context() override {
			return upstream->get_context();
		}

		DeviceResource* upstream = nullptr;
	};
} // namespace vuk
