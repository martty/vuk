#pragma once

#include "vuk/Config.hpp"
#include "vuk/Result.hpp"
#include "vuk/SourceLocation.hpp"
#include "vuk/SyncPoint.hpp"
#include "vuk/runtime/vk/Image.hpp"
#include "vuk/vuk_fwd.hpp"

#define VUK_DO_OR_RETURN(what)                                                                                                                                 \
	if (auto res = what; !res) {                                                                                                                                 \
		return std::move(res);                                                                                                                                     \
	}
/// @endcond

#include <span>

namespace vuk {
	struct FramebufferCreateInfo;
	struct TimestampQueryCreateInfo;
	struct RenderPassCreateInfo;
	struct BufferCreateInfo;
	struct PersistentDescriptorSet;
	struct PersistentDescriptorSetCreateInfo;
	struct DescriptorSet;
	struct SetBinding;
	struct TimestampQueryPool;
	struct TimestampQuery;
	struct GraphicsPipelineInfo;
	struct GraphicsPipelineInstanceCreateInfo;
	struct ComputePipelineInfo;
	struct ComputePipelineInstanceCreateInfo;
	struct RayTracingPipelineInfo;
	struct RayTracingPipelineInstanceCreateInfo;

	/// @brief DeviceResource is a polymorphic interface over allocation of GPU resources.
	/// A DeviceResource must prevent reuse of cross-device resources after deallocation until CPU-GPU timelines are synchronized. GPU-only resources may be
	/// reused immediately.
	struct DeviceResource {
		// missing here: Events (gpu only)

		// gpu only
		virtual Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_semaphores(std::span<const VkSemaphore> src) = 0;

		virtual Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_fences(std::span<const VkFence> dst) = 0;

		virtual Result<void, AllocateException>
		allocate_command_buffers(std::span<CommandBufferAllocation> dst, std::span<const CommandBufferAllocationCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_command_buffers(std::span<const CommandBufferAllocation> dst) = 0;

		virtual Result<void, AllocateException>
		allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_command_pools(std::span<const CommandPool> dst) = 0;

		virtual Result<void, AllocateException> allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_buffers(std::span<const Buffer> dst) = 0;
		virtual void set_buffer_allocation_name(Buffer& dst, Name name) = 0;

		virtual Result<void, AllocateException>
		allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_framebuffers(std::span<const VkFramebuffer> dst) = 0;

		// gpu only
		virtual Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_images(std::span<const Image> dst) = 0;
		virtual void set_image_allocation_name(Image& dst, Name name) = 0;

		virtual Result<void, AllocateException>
		allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_image_views(std::span<const ImageView> src) = 0;

		virtual Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
		                                                                            std::span<const PersistentDescriptorSetCreateInfo> cis,
		                                                                            SourceLocationAtFrame loc) = 0;
		virtual void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) = 0;

		virtual Result<void, AllocateException>
		allocate_descriptor_sets_with_value(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) = 0;
		virtual Result<void, AllocateException>
		allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const struct DescriptorSetLayoutAllocInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_descriptor_sets(std::span<const DescriptorSet> src) = 0;

		virtual Result<void, AllocateException>
		allocate_descriptor_pools(std::span<VkDescriptorPool> dst, std::span<const VkDescriptorPoolCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_descriptor_pools(std::span<const VkDescriptorPool> src) = 0;

		virtual Result<void, AllocateException>
		allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) = 0;

		virtual Result<void, AllocateException>
		allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_timestamp_queries(std::span<const TimestampQuery> src) = 0;

		virtual void wait_sync_points(std::span<const SyncPoint> src) = 0;

		virtual Result<void, AllocateException> allocate_acceleration_structures(std::span<VkAccelerationStructureKHR> dst,
		                                                                         std::span<const VkAccelerationStructureCreateInfoKHR> cis,
		                                                                         SourceLocationAtFrame loc) = 0;
		virtual void deallocate_acceleration_structures(std::span<const VkAccelerationStructureKHR> src) = 0;

		virtual void deallocate_swapchains(std::span<const VkSwapchainKHR> src) = 0;

		virtual Result<void, AllocateException>
		allocate_graphics_pipelines(std::span<GraphicsPipelineInfo> dst, std::span<const GraphicsPipelineInstanceCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_graphics_pipelines(std::span<const GraphicsPipelineInfo> src) = 0;

		virtual Result<void, AllocateException>
		allocate_compute_pipelines(std::span<ComputePipelineInfo> dst, std::span<const ComputePipelineInstanceCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_compute_pipelines(std::span<const ComputePipelineInfo> src) = 0;

		virtual Result<void, AllocateException> allocate_ray_tracing_pipelines(std::span<RayTracingPipelineInfo> dst,
		                                                                       std::span<const RayTracingPipelineInstanceCreateInfo> cis,
		                                                                       SourceLocationAtFrame loc) = 0;
		virtual void deallocate_ray_tracing_pipelines(std::span<const RayTracingPipelineInfo> src) = 0;

		virtual Result<void, AllocateException>
		allocate_render_passes(std::span<VkRenderPass> dst, std::span<const RenderPassCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_render_passes(std::span<const VkRenderPass> src) = 0;

		virtual Runtime& get_context() = 0;

		BufferUsageFlags get_all_buffer_usage_flags(Runtime& runtime);
	};

	struct DeviceVkResource;

	/// @brief Interface for allocating device resources
	///
	/// The Allocator is a concrete value type wrapping over a polymorphic DeviceResource, forwarding allocations and deallocations to it.
	/// The allocation functions take spans of creation parameters and output values, reporting error through the return value of Result<void, AllocateException>.
	/// The deallocation functions can't fail.
	class Allocator {
	public:
		/// @brief Create new Allocator that wraps a DeviceResource
		/// @param device_resource The DeviceResource to allocate from
		explicit Allocator(DeviceResource& device_resource) : ctx(&device_resource.get_context()), device_resource(&device_resource) {}

		/// @brief Allocate semaphores from this Allocator
		/// @param dst Destination span to place allocated semaphores into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate(std::span<VkSemaphore> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate semaphores from this Allocator
		/// @param dst Destination span to place allocated semaphores into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate semaphores previously allocated from this Allocator
		/// @param src Span of semaphores to be deallocated
		void deallocate(std::span<const VkSemaphore> src);

		/// @brief Allocate fences from this Allocator
		/// @param dst Destination span to place allocated fences into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate(std::span<VkFence> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate fences from this Allocator
		/// @param dst Destination span to place allocated fences into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate fences previously allocated from this Allocator
		/// @param src Span of fences to be deallocated
		void deallocate(std::span<const VkFence> src);

		/// @brief Allocate command pools from this Allocator
		/// @param dst Destination span to place allocated command pools into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate command pools from this Allocator
		/// @param dst Destination span to place allocated command pools into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate command pools previously allocated from this Allocator
		/// @param src Span of command pools to be deallocated
		void deallocate(std::span<const CommandPool> src);

		/// @brief Allocate command buffers from this Allocator
		/// @param dst Destination span to place allocated command buffers into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<CommandBufferAllocation> dst, std::span<const CommandBufferAllocationCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate command buffers from this Allocator
		/// @param dst Destination span to place allocated command buffers into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_command_buffers(std::span<CommandBufferAllocation> dst,
		                                                         std::span<const CommandBufferAllocationCreateInfo> cis,
		                                                         SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate command buffers previously allocated from this Allocator
		/// @param src Span of command buffers to be deallocated
		void deallocate(std::span<const CommandBufferAllocation> src);

		/// @brief Allocate buffers from this Allocator
		/// @param dst Destination span to place allocated buffers into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate buffers from this Allocator
		/// @param dst Destination span to place allocated buffers into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate buffers previously allocated from this Allocator
		/// @param src Span of buffers to be deallocated
		void deallocate(std::span<const Buffer> src);

		/// @brief Allocate framebuffers from this Allocator
		/// @param dst Destination span to place allocated framebuffers into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate framebuffers from this Allocator
		/// @param dst Destination span to place allocated framebuffers into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate framebuffers previously allocated from this Allocator
		/// @param src Span of framebuffers to be deallocated
		void deallocate(std::span<const VkFramebuffer> src);

		/// @brief Allocate images from this Allocator
		/// @param dst Destination span to place allocated images into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate images from this Allocator
		/// @param dst Destination span to place allocated images into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate images previously allocated from this Allocator
		/// @param src Span of images to be deallocated
		void deallocate(std::span<const Image> src);

		/// @brief Allocate image views from this Allocator
		/// @param dst Destination span to place allocated image views into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate image views from this Allocator
		/// @param dst Destination span to place allocated image views into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate image views previously allocated from this Allocator
		/// @param src Span of image views to be deallocated
		void deallocate(std::span<const ImageView> src);

		/// @brief Allocate persistent descriptor sets from this Allocator
		/// @param dst Destination span to place allocated persistent descriptor sets into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate persistent descriptor sets from this Allocator
		/// @param dst Destination span to place allocated persistent descriptor sets into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
		                                                                    std::span<const PersistentDescriptorSetCreateInfo> cis,
		                                                                    SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate persistent descriptor sets previously allocated from this Allocator
		/// @param src Span of persistent descriptor sets to be deallocated
		void deallocate(std::span<const PersistentDescriptorSet> src);

		/// @brief Allocate descriptor sets from this Allocator
		/// @param dst Destination span to place allocated descriptor sets into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate descriptor sets from this Allocator
		/// @param dst Destination span to place allocated descriptor sets into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_descriptor_sets_with_value(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate descriptor sets from this Allocator
		/// @param dst Destination span to place allocated descriptor sets into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<DescriptorSet> dst, std::span<const DescriptorSetLayoutAllocInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate descriptor sets from this Allocator
		/// @param dst Destination span to place allocated descriptor sets into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const DescriptorSetLayoutAllocInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate descriptor sets previously allocated from this Allocator
		/// @param src Span of descriptor sets to be deallocated
		void deallocate(std::span<const DescriptorSet> src);

		/// @brief Allocate timestamp query pools from this Allocator
		/// @param dst Destination span to place allocated timestamp query pools into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate timestamp query pools from this Allocator
		/// @param dst Destination span to place allocated timestamp query pools into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst,
		                                                               std::span<const VkQueryPoolCreateInfo> cis,
		                                                               SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate timestamp query pools previously allocated from this Allocator
		/// @param src Span of timestamp query pools to be deallocated
		void deallocate(std::span<const TimestampQueryPool> src);

		/// @brief Allocate timestamp queries from this Allocator
		/// @param dst Destination span to place allocated timestamp queries into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate timestamp queries from this Allocator
		/// @param dst Destination span to place allocated timestamp queries into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate timestamp queries previously allocated from this Allocator
		/// @param src Span of timestamp queries to be deallocated
		void deallocate(std::span<const TimestampQuery> src);

		/// @brief Make this allocator wait for the given SyncPoints before recycling
		/// If the underlying resource does not support this, then this call is ignored
		void wait_sync_points(std::span<const SyncPoint> src);

		/// @brief Allocate acceleration structures from this Allocator
		/// @param dst Destination span to place allocated acceleration structures into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate(std::span<VkAccelerationStructureKHR> dst,
		                                         std::span<const VkAccelerationStructureCreateInfoKHR> cis,
		                                         SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate acceleration structures from this Allocator
		/// @param dst Destination span to place allocated acceleration structures into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_acceleration_structures(std::span<VkAccelerationStructureKHR> dst,
		                                                                 std::span<const VkAccelerationStructureCreateInfoKHR> cis,
		                                                                 SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate acceleration structures previously allocated from this Allocator
		/// @param src Span of acceleration structures to be deallocated
		void deallocate(std::span<const VkAccelerationStructureKHR> src);

		/// @brief Deallocate swapchains previously allocated from this Allocator
		/// @param src Span of swapchains to be deallocated
		void deallocate(std::span<const VkSwapchainKHR> src);

		/// @brief Allocate graphics pipelines from this Allocator
		/// @param dst Destination span to place allocated pipelines into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<GraphicsPipelineInfo> dst, std::span<const GraphicsPipelineInstanceCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate graphics pipelines from this Allocator
		/// @param dst Destination span to place allocated pipelines into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_graphics_pipelines(std::span<GraphicsPipelineInfo> dst,
		                                                            std::span<const GraphicsPipelineInstanceCreateInfo> cis,
		                                                            SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate pipelines previously allocated from this Allocator
		/// @param src Span of pipelines to be deallocated
		void deallocate(std::span<const GraphicsPipelineInfo> src);

		/// @brief Allocate compute pipelines from this Allocator
		/// @param dst Destination span to place allocated pipelines into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<ComputePipelineInfo> dst, std::span<const ComputePipelineInstanceCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate compute pipelines from this Allocator
		/// @param dst Destination span to place allocated pipelines into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_compute_pipelines(std::span<ComputePipelineInfo> dst,
		                                                           std::span<const ComputePipelineInstanceCreateInfo> cis,
		                                                           SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate pipelines previously allocated from this Allocator
		/// @param src Span of pipelines to be deallocated
		void deallocate(std::span<const ComputePipelineInfo> src);

		/// @brief Allocate ray tracing pipelines from this Allocator
		/// @param dst Destination span to place allocated pipelines into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<RayTracingPipelineInfo> dst, std::span<const RayTracingPipelineInstanceCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate ray tracing pipelines from this Allocator
		/// @param dst Destination span to place allocated pipelines into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_ray_tracing_pipelines(std::span<RayTracingPipelineInfo> dst,
		                                                               std::span<const RayTracingPipelineInstanceCreateInfo> cis,
		                                                               SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate pipelines previously allocated from this Allocator
		/// @param src Span of pipelines to be deallocated
		void deallocate(std::span<const RayTracingPipelineInfo> src);

		/// @brief Allocate render passes from this Allocator
		/// @param dst Destination span to place allocated render passes into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<VkRenderPass> dst, std::span<const RenderPassCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate render passes from this Allocator
		/// @param dst Destination span to place allocated render passes into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_render_passes(std::span<VkRenderPass> dst, std::span<const RenderPassCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate render passes previously allocated from this Allocator
		/// @param src Span of render passes to be deallocated
		void deallocate(std::span<const VkRenderPass> src);

		/// @brief Get the underlying DeviceResource
		/// @return the underlying DeviceResource
		DeviceResource& get_device_resource() {
			return *device_resource;
		}

		/// @brief Get the parent Runtime
		/// @return the parent Runtime
		Runtime& get_context() {
			return *ctx;
		}

	private:
		Runtime* ctx;
		DeviceResource* device_resource;
	};

	template<class ContainerType>
	concept Container = requires(ContainerType a) {
		std::begin(a);
		std::end(a);
	};

	/// @brief Customization point for deallocation of user types
	/// @tparam T
	/// @param allocator
	/// @param src
	template<class T, size_t N>
	void deallocate(Allocator& allocator, T (&src)[N]) {
		allocator.deallocate(std::span<const T>{ src, N });
	}

	/// @brief Customization point for deallocation of user types
	/// @tparam T
	/// @param allocator
	/// @param src
	template<class T>
	void deallocate(Allocator& allocator, const T& src)
	  requires(!Container<T>)
	{
		allocator.deallocate(std::span<const T>{ &src, 1 });
	}

	/// @brief Customization point for deallocation of user types
	/// @tparam T
	/// @param allocator
	/// @param src
	template<class T>
	void deallocate(Allocator& allocator, const T& src)
	  requires(Container<T>)
	{
		allocator.deallocate(std::span(src));
	}
} // namespace vuk

namespace vuk {
	template<typename Type>
	Unique<Type>::~Unique() noexcept {
		if (allocator && payload != Type{}) {
			deallocate(*allocator, payload);
		}
	}

	template<typename Type>
	void Unique<Type>::reset(Type value) noexcept {
		if (payload != value) {
			if (allocator && payload != Type{}) {
				deallocate(*allocator, std::move(payload));
			}
			payload = std::move(value);
		}
	}
} // namespace vuk
