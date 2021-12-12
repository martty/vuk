#pragma once

#include "vuk/Allocator.hpp"

namespace vuk {
	/// @brief Helper base class for DeviceResources. Forwards all allocations and deallocations to the upstream DeviceResource.
	struct DeviceNestedResource : DeviceResource {
		DeviceNestedResource(DeviceResource* upstream) : upstream(upstream) {}

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override {
			return upstream->allocate_semaphores(dst, loc);
		}

		void deallocate_semaphores(std::span<const VkSemaphore> sema) override {
			upstream->deallocate_semaphores(sema);
		}

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override { return upstream->allocate_fences(dst, loc); }
		void deallocate_fences(std::span<const VkFence> dst) override { upstream->deallocate_fences(dst); }

		Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<const VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_commandbuffers(dst, cis, loc);
		}

		void deallocate_commandbuffers(VkCommandPool pool, std::span<const VkCommandBuffer> dst) override {
			upstream->deallocate_commandbuffers(pool, dst);
		}

		Result<void, AllocateException> allocate_hl_commandbuffers(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_hl_commandbuffers(dst, cis, loc);
		}

		void deallocate_hl_commandbuffers(std::span<const HLCommandBuffer> dst) override {
			upstream->deallocate_hl_commandbuffers(dst);
		}

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_commandpools(dst, cis, loc);
		}

		void deallocate_commandpools(std::span<const VkCommandPool> dst) override {
			upstream->deallocate_commandpools(dst);
		}

		Result<void, AllocateException> allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_buffers(dst, cis, loc);
		}

		void deallocate_buffers(std::span<const BufferCrossDevice> src) override {
			upstream->deallocate_buffers(src);
		}

		Result<void, AllocateException> allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_buffers(dst, cis, loc);
		}

		void deallocate_buffers(std::span<const BufferGPU> src) override {
			upstream->deallocate_buffers(src);
		}

		Result<void, AllocateException> allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_framebuffers(dst, cis, loc);
		}

		void deallocate_framebuffers(std::span<const VkFramebuffer> src) override {
			upstream->deallocate_framebuffers(src);
		}

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_images(dst, cis, loc);
		}

		void deallocate_images(std::span<const Image> src) override {
			upstream->deallocate_images(src);
		}

		Result<void, AllocateException> allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_image_views(dst, cis, loc);
		}

		void deallocate_image_views(std::span<const ImageView> src) override {
			upstream->deallocate_image_views(src);
		}

		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_persistent_descriptor_sets(dst, cis, loc);
		}

		void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) override {
			upstream->deallocate_persistent_descriptor_sets(src);
		}

		Result<void, AllocateException> allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_descriptor_sets(dst, cis, loc);
		}

		void deallocate_descriptor_sets(std::span<const DescriptorSet> src) override {
			upstream->deallocate_descriptor_sets(src);
		}

		DeviceResource* upstream = nullptr;
	};
}