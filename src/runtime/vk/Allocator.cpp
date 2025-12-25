#include "vuk/Exception.hpp"
#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/runtime/vk/Address.hpp"
#include "vuk/runtime/vk/PipelineInstance.hpp"
#include "vuk/runtime/vk/Query.hpp"
#include "vuk/runtime/vk/RenderPass.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"

namespace vuk {
	BufferUsageFlags DeviceResource::get_all_buffer_usage_flags(Runtime& runtime) {
		auto all_buffer_usage_flags = BufferUsageFlagBits::eTransferRead | BufferUsageFlagBits::eTransferWrite | BufferUsageFlagBits::eUniformTexelBuffer |
		                              BufferUsageFlagBits::eStorageTexelBuffer | BufferUsageFlagBits::eUniformBuffer | BufferUsageFlagBits::eStorageBuffer |
		                              BufferUsageFlagBits::eIndexBuffer | BufferUsageFlagBits::eVertexBuffer | BufferUsageFlagBits::eIndirectBuffer |
		                              BufferUsageFlagBits::eShaderDeviceAddress;
		if (runtime.vkCmdBuildAccelerationStructuresKHR) {
			all_buffer_usage_flags |= BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | BufferUsageFlagBits::eAccelerationStructureStorageKHR |
			                          BufferUsageFlagBits::eShaderBindingTableKHR;
		}
		return all_buffer_usage_flags;
	}

	/****Allocator impls *****/

	Result<void, AllocateException> Allocator::allocate(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
		return device_resource->allocate_semaphores(dst, loc);
	}

	Result<void, AllocateException> Allocator::allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
		return device_resource->allocate_semaphores(dst, loc);
	}

	void Allocator::deallocate(std::span<const VkSemaphore> src) {
		device_resource->deallocate_semaphores(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<VkFence> dst, SourceLocationAtFrame loc) {
		return device_resource->allocate_fences(dst, loc);
	}

	Result<void, AllocateException> Allocator::allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) {
		return device_resource->allocate_fences(dst, loc);
	}

	void Allocator::deallocate(std::span<const VkFence> src) {
		device_resource->deallocate_fences(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_command_pools(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_command_pools(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const CommandPool> src) {
		device_resource->deallocate_command_pools(src);
	}

	Result<void, AllocateException>
	Allocator::allocate(std::span<CommandBufferAllocation> dst, std::span<const CommandBufferAllocationCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_command_buffers(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_command_buffers(std::span<CommandBufferAllocation> dst,
	                                                                    std::span<const CommandBufferAllocationCreateInfo> cis,
	                                                                    SourceLocationAtFrame loc) {
		return device_resource->allocate_command_buffers(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const CommandBufferAllocation> src) {
		device_resource->deallocate_command_buffers(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<ptr_base> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_memory(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_memory(std::span<ptr_base> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_memory(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const ptr_base> src) {
		device_resource->deallocate_memory(src);
	}

	/* void Allocator::set_allocation_name(Buffer& dst, Name name) {
	  device_resource->set_buffer_allocation_name(dst, name);
	}*/

	Result<void, AllocateException> Allocator::allocate(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_framebuffers(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_framebuffers(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const VkFramebuffer> src) {
		device_resource->deallocate_framebuffers(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<Image<>> dst, std::span<const ICI> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_images(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_images(std::span<Image<>> dst, std::span<const ICI> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_images(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const Image<>> src) {
		device_resource->deallocate_images(src);
	}

	void Allocator::set_allocation_name(Image<>& dst, Name name) {
		device_resource->set_image_allocation_name(dst, name);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<ImageView<>> dst, std::span<const IVCI> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_image_views(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_image_views(std::span<ImageView<>> dst, std::span<const IVCI> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_image_views(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const ImageView<>> src) {
		device_resource->deallocate_image_views(src);
	}

	Result<void, AllocateException>
	Allocator::allocate(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_persistent_descriptor_sets(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
	                                                                               std::span<const PersistentDescriptorSetCreateInfo> cis,
	                                                                               SourceLocationAtFrame loc) {
		return device_resource->allocate_persistent_descriptor_sets(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const PersistentDescriptorSet> src) {
		device_resource->deallocate_persistent_descriptor_sets(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_descriptor_sets_with_value(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_descriptor_sets_with_value(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_descriptor_sets_with_value(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate(std::span<DescriptorSet> dst, std::span<const DescriptorSetLayoutAllocInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_descriptor_sets(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const DescriptorSetLayoutAllocInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_descriptor_sets(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const DescriptorSet> src) {
		device_resource->deallocate_descriptor_sets(src);
	}

	Result<void, AllocateException>
	Allocator::allocate(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_timestamp_query_pools(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_timestamp_query_pools(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const TimestampQueryPool> src) {
		device_resource->deallocate_timestamp_query_pools(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_timestamp_queries(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_timestamp_queries(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const TimestampQuery> src) {
		device_resource->deallocate_timestamp_queries(src);
	}

	void Allocator::wait_sync_points(std::span<const SyncPoint> src) {
		device_resource->wait_sync_points(src);
	}

	Result<void, AllocateException>
	Allocator::allocate(std::span<VkAccelerationStructureKHR> dst, std::span<const VkAccelerationStructureCreateInfoKHR> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_acceleration_structures(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_acceleration_structures(std::span<VkAccelerationStructureKHR> dst,
	                                                                            std::span<const VkAccelerationStructureCreateInfoKHR> cis,
	                                                                            SourceLocationAtFrame loc) {
		return device_resource->allocate_acceleration_structures(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const VkAccelerationStructureKHR> src) {
		device_resource->deallocate_acceleration_structures(src);
	}

	void Allocator::deallocate(std::span<const VkSwapchainKHR> src) {
		device_resource->deallocate_swapchains(src);
	}

	Result<void, AllocateException>
	Allocator::allocate(std::span<GraphicsPipelineInfo> dst, std::span<const GraphicsPipelineInstanceCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_graphics_pipelines(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_graphics_pipelines(std::span<GraphicsPipelineInfo> dst,
	                                                                       std::span<const GraphicsPipelineInstanceCreateInfo> cis,
	                                                                       SourceLocationAtFrame loc) {
		return device_resource->allocate_graphics_pipelines(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const GraphicsPipelineInfo> src) {
		device_resource->deallocate_graphics_pipelines(src);
	}

	Result<void, AllocateException>
	Allocator::allocate(std::span<ComputePipelineInfo> dst, std::span<const ComputePipelineInstanceCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_compute_pipelines(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_compute_pipelines(std::span<ComputePipelineInfo> dst, std::span<const ComputePipelineInstanceCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_compute_pipelines(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const ComputePipelineInfo> src) {
		device_resource->deallocate_compute_pipelines(src);
	}

	Result<void, AllocateException>
	Allocator::allocate(std::span<RayTracingPipelineInfo> dst, std::span<const RayTracingPipelineInstanceCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_ray_tracing_pipelines(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_ray_tracing_pipelines(std::span<RayTracingPipelineInfo> dst,
	                                                                          std::span<const RayTracingPipelineInstanceCreateInfo> cis,
	                                                                          SourceLocationAtFrame loc) {
		return device_resource->allocate_ray_tracing_pipelines(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const RayTracingPipelineInfo> src) {
		device_resource->deallocate_ray_tracing_pipelines(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<VkRenderPass> dst, std::span<const RenderPassCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_render_passes(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_render_passes(std::span<VkRenderPass> dst, std::span<const RenderPassCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_render_passes(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const VkRenderPass> src) {
		device_resource->deallocate_render_passes(src);
	}

	Result<void, AllocateException>
	Allocator::allocate(std::span<VirtualAddressSpace> dst, std::span<const VirtualAddressSpaceCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_virtual_address_spaces(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_virtual_address_spaces(std::span<VirtualAddressSpace> dst, std::span<const VirtualAddressSpaceCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_virtual_address_spaces(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const VirtualAddressSpace> src) {
		device_resource->deallocate_virtual_address_spaces(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<VirtualAllocation> dst, std::span<const VirtualAllocationCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_virtual_allocations(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_virtual_allocations(std::span<VirtualAllocation> dst, std::span<const VirtualAllocationCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_virtual_allocations(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const VirtualAllocation> src) {
		device_resource->deallocate_virtual_allocations(src);
	}
} // namespace vuk
