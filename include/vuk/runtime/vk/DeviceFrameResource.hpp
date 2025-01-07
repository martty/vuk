#pragma once

#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/runtime/vk/DeviceNestedResource.hpp"
#include "vuk/runtime/vk/DeviceVkResource.hpp"

#include <memory>

namespace vuk {
	struct DeviceSuperFrameResource;

	/// @brief Represents "per-frame" resources - temporary allocations that persist through a frame. Handed out by DeviceSuperFrameResource, cannot be
	/// constructed directly.
	///
	/// Allocations from this resource are tied to the "frame" - all allocations recycled when a DeviceFrameResource is recycled.
	/// Furthermore all resources allocated are also deallocated at recycle time - it is not necessary (but not an error) to deallocate them.
	struct DeviceFrameResource : DeviceNestedResource {
		DeviceFrameResource(const DeviceFrameResource&) = delete;
		DeviceFrameResource& operator=(const DeviceFrameResource&) = delete;

		DeviceFrameResource(DeviceFrameResource&&) = delete;
		DeviceFrameResource& operator=(DeviceFrameResource&&) = delete;

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
		Result<void, AllocateException> allocate_memory(std::span<ptr_base> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_memory(std::span<const ptr_base> src) override; // no-op, linear

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

		/// @brief Wait for the fences / timeline semaphores referencing this frame to complete
		///
		/// Called automatically when recycled
		void wait();

		/// @brief Retrieve the parent Runtime
		/// @return the parent Runtime
		Runtime& get_context() override {
			return upstream->get_context();
		}

	protected:
		VkDevice device;
		uint64_t construction_frame = -1;
		std::unique_ptr<struct DeviceFrameResourceImpl> impl;

		friend struct DeviceSuperFrameResource;
		friend struct DeviceSuperFrameResourceImpl;

		DeviceFrameResource(VkDevice device, DeviceSuperFrameResource& upstream);
	};

	/// @brief Represents temporary allocations that persist through multiple frames, eg. history buffers. Handed out by DeviceSuperFrameResource. Don't
	/// construct it directly.
	///
	/// Allocations from this resource are tied to the "multi-frame" - all allocations recycled when a DeviceMultiFrameResource is recycled.
	/// All resources allocated are also deallocated at recycle time - it is not necessary (but not an error) to deallocate them.
	struct DeviceMultiFrameResource final : DeviceFrameResource {
		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) override;

		DeviceMultiFrameResource(VkDevice device, DeviceSuperFrameResource& upstream, uint32_t frame_lifetime);

	private:
		uint32_t frame_lifetime;
		uint32_t remaining_lifetime;
		uint32_t multiframe_id;

		friend struct DeviceSuperFrameResource;
		friend struct DeviceSuperFrameResourceImpl;

		DeviceMultiFrameResource(const DeviceMultiFrameResource&) = delete;
		DeviceMultiFrameResource& operator=(const DeviceMultiFrameResource&) = delete;

		DeviceMultiFrameResource(DeviceMultiFrameResource&&) = delete;
		DeviceMultiFrameResource& operator=(DeviceMultiFrameResource&&) = delete;
	};

	/// @brief DeviceSuperFrameResource is an allocator that gives out DeviceFrameResource allocators, and manages their resources
	///
	/// DeviceSuperFrameResource models resource lifetimes that span multiple frames - these can be allocated directly from this resource
	/// Allocation of these resources are persistent, and they can be deallocated at any time - they will be recycled when the current frame is recycled
	/// This resource also hands out DeviceFrameResources in a round-robin fashion.
	/// The lifetime of resources allocated from those allocators is frames_in_flight number of frames (until the DeviceFrameResource is recycled).
	struct DeviceSuperFrameResource : DeviceNestedResource {
		DeviceSuperFrameResource(Runtime& ctx, uint64_t frames_in_flight);
		DeviceSuperFrameResource(DeviceResource& upstream, uint64_t frames_in_flight);

		DeviceSuperFrameResource(const DeviceSuperFrameResource&) = delete;
		DeviceSuperFrameResource& operator=(const DeviceSuperFrameResource&) = delete;

		DeviceSuperFrameResource(DeviceSuperFrameResource&&) = delete;
		DeviceSuperFrameResource& operator=(DeviceSuperFrameResource&&) = delete;

		void deallocate_semaphores(std::span<const VkSemaphore> src) override;

		void deallocate_fences(std::span<const VkFence> src) override;

		void deallocate_command_buffers(std::span<const CommandBufferAllocation> src) override;

		Result<void, AllocateException>
		allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_command_pools(std::span<const CommandPool> src) override;

		void deallocate_memory(std::span<const ptr_base> src) override;

		void deallocate_framebuffers(std::span<const VkFramebuffer> src) override;

		void deallocate_images(std::span<const Image> src) override;

		Result<void, AllocateException> allocate_cached_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc);

		Result<void, AllocateException> allocate_cached_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc);
		
		Result<void, AllocateException> allocate_memory(std::span<ptr_base> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		/*
		Result<void, AllocateException> allocate_views(std::span<view_base> dst, std::span<const VCI> cis, SourceLocationAtFrame loc) override;
		void deallocate_views(std::span<const view_base> dst) override;*/

		void deallocate_image_views(std::span<const ImageView> src) override;

		void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) override;

		void deallocate_descriptor_sets(std::span<const DescriptorSet> src) override;

		Result<void, AllocateException>
		allocate_descriptor_pools(std::span<VkDescriptorPool> dst, std::span<const VkDescriptorPoolCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_descriptor_pools(std::span<const VkDescriptorPool> src) override;

		void deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) override;

		void deallocate_timestamp_queries(std::span<const TimestampQuery> src) override; // noop

		void wait_sync_points(std::span<const SyncPoint> src) override;

		void deallocate_acceleration_structures(std::span<const VkAccelerationStructureKHR> src) override;

		void deallocate_swapchains(std::span<const VkSwapchainKHR> src) override;

		void deallocate_graphics_pipelines(std::span<const GraphicsPipelineInfo> src) override;

		void deallocate_compute_pipelines(std::span<const ComputePipelineInfo> src) override;

		void deallocate_ray_tracing_pipelines(std::span<const RayTracingPipelineInfo> src) override;

		void deallocate_render_passes(std::span<const VkRenderPass> src) override;

		void deallocate_virtual_address_spaces(std::span<const VirtualAddressSpace> src) override;

		void deallocate_virtual_allocations(std::span<const VirtualAllocation> src) override;

		/// @brief Recycle the least-recently-used frame and return it to be used again
		/// @return DeviceFrameResource for use
		DeviceFrameResource& get_next_frame();

		/// @brief Get a multiframe resource for the current frame with the specified frame lifetime count
		/// The returned resource ensures that any resource allocated from it will be usable for at least `frame_lifetime_count`
		DeviceMultiFrameResource& get_multiframe_allocator(uint32_t frame_lifetime_count);

		void force_collect();

		virtual ~DeviceSuperFrameResource();

		const uint64_t frames_in_flight;
		DeviceVkResource* direct = nullptr;

	private:
		DeviceFrameResource& get_last_frame();
		template<class T>
		void deallocate_frame(T& f);

		struct DeviceSuperFrameResourceImpl* impl;
		friend struct DeviceFrameResource;
	};
} // namespace vuk