#pragma once

#include "vuk/Allocator.hpp"
#include "vuk/Exception.hpp"

namespace vuk {
	/// @brief Device resource that performs direct allocation from the resources from the Vulkan runtime.
	struct DeviceVkResource final : DeviceResource {
		DeviceVkResource(Context& ctx, LegacyGPUAllocator& alloc);

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override {
			VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
			for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
				VkResult res = vkCreateSemaphore(device, &sci, nullptr, &dst[i]);
				if (res != VK_SUCCESS) {
					deallocate_semaphores({ dst.data(), (uint64_t)i });
					return { expected_error, AllocateException{res} };
				}
			}
			return { expected_value };
		}

		void deallocate_semaphores(std::span<const VkSemaphore> src) override {
			for (auto& v : src) {
				if (v != VK_NULL_HANDLE) {
					vkDestroySemaphore(device, v, nullptr);
				}
			}
		}

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override {
			VkFenceCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
			for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
				VkResult res = vkCreateFence(device, &sci, nullptr, &dst[i]);
				if (res != VK_SUCCESS) {
					deallocate_fences({ dst.data(), (uint64_t)i });
					return { expected_error, AllocateException{res} };
				}
			}
			return { expected_value };
		}

		void deallocate_fences(std::span<const VkFence> src) override {
			for (auto& v : src) {
				if (v != VK_NULL_HANDLE) {
					vkDestroyFence(device, v, nullptr);
				}
			}
		}

		Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<const VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) override {
			assert(dst.size() == cis.size());
			VkResult res = vkAllocateCommandBuffers(device, cis.data(), dst.data());
			if (res != VK_SUCCESS) {
				return { expected_error, AllocateException{res} };
			}
			return { expected_value };
		}

		void deallocate_commandbuffers(VkCommandPool pool, std::span<const VkCommandBuffer> dst) override {
			vkFreeCommandBuffers(device, pool, (uint32_t)dst.size(), dst.data());
		}

		Result<void, AllocateException> allocate_hl_commandbuffers(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			assert(dst.size() == cis.size());

			for (uint64_t i = 0; i < dst.size(); i++) {
				auto& ci = cis[i];
				VkCommandPoolCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
				cpci.queueFamilyIndex = ci.queue_family_index;
				cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
				allocate_commandpools(std::span{ &dst[i].command_pool, 1 }, std::span{ &cpci, 1 }, loc);

				VkCommandBufferAllocateInfo cbai{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
				cbai.commandBufferCount = 1;
				cbai.commandPool = dst[i].command_pool;
				cbai.level = ci.level;
				allocate_commandbuffers(std::span{ &dst[i].command_buffer, 1 }, std::span{ &cbai, 1 }, loc);
			}

			return { expected_value };
		}

		void deallocate_hl_commandbuffers(std::span<const HLCommandBuffer> dst) override {
			for (auto& c : dst) {
				deallocate_commandpools(std::span{ &c.command_pool, 1 });
			}
		}

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override {
			assert(dst.size() == cis.size());
			for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
				VkResult res = vkCreateCommandPool(device, &cis[i], nullptr, &dst[i]);
				if (res != VK_SUCCESS) {
					deallocate_commandpools({ dst.data(), (uint64_t)i });
					return { expected_error, AllocateException{res} };
				}
			}
			return { expected_value };
		}

		void deallocate_commandpools(std::span<const VkCommandPool> src) override {
			for (auto& v : src) {
				if (v != VK_NULL_HANDLE) {
					vkDestroyCommandPool(device, v, nullptr);
				}
			}
		}

		Result<void, AllocateException> allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const BufferCrossDevice> src) override;

		Result<void, AllocateException> allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const BufferGPU> src) override;

		Result<void, AllocateException> allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			assert(dst.size() == cis.size());
			for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
				VkResult res = vkCreateFramebuffer(device, &cis[i], nullptr, &dst[i]);
				if (res != VK_SUCCESS) {
					deallocate_framebuffers({ dst.data(), (uint64_t)i });
					return { expected_error, AllocateException{res} };
				}
			}
			return { expected_value };
		}

		void deallocate_framebuffers(std::span<const VkFramebuffer> src) override {
			for (auto& v : src) {
				if (v != VK_NULL_HANDLE) {
					vkDestroyFramebuffer(device, v, nullptr);
				}
			}
		}

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_images(std::span<const Image> src) override;

		Result<void, AllocateException> allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_image_views(std::span<const ImageView> src) override {
			for (auto& v : src) {
				if (v.payload != VK_NULL_HANDLE) {
					vkDestroyImageView(device, v.payload, nullptr);
				}
			}
		}

		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) override;

		Result<void, AllocateException> allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) override;

		void deallocate_descriptor_sets(std::span<const DescriptorSet> src);

		Context& get_context() override {
			return *ctx;
		}

		Context* ctx;
		LegacyGPUAllocator* legacy_gpu_allocator;
		VkDevice device;
	};
}