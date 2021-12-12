#pragma once

#include "vuk/Allocator.hpp"
#include "vuk/resources/DeviceNestedResource.hpp"
#include "../src/LegacyGPUAllocator.hpp"

namespace vuk {
	struct DeviceLinearResource : DeviceNestedResource {
		enum class SyncScope { eInline, eScope };
		static constexpr SyncScope eInline = SyncScope::eInline;
		static constexpr SyncScope eScope = SyncScope::eScope;

		DeviceLinearResource(DeviceResource& upstream, SyncScope scope);

		bool should_subsume = false;
		std::vector<VkFence> fences;

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override {
			auto result = upstream->allocate_fences(dst, loc);
			fences.insert(fences.end(), dst.begin(), dst.end());
			return result;
		}

		void deallocate_fences(std::span<const VkFence>) override {} // linear allocator, noop

		std::vector<VkCommandPool> command_pools;

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override {
			auto result = upstream->allocate_commandpools(dst, cis, loc);
			command_pools.insert(command_pools.end(), dst.begin(), dst.end());
			return result;
		}

		void deallocate_commandpools(std::span<const VkCommandPool>) override {} // linear allocator, noop

		// do not record the command buffers - they come from the pools
		Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<const VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_commandbuffers(dst, cis, loc);
		}

		void deallocate_commandbuffers(VkCommandPool, std::span<const VkCommandBuffer>) override {} // noop, the pools own the command buffers

		std::vector<VkCommandPool> direct_command_pools;

		Result<void, AllocateException> allocate_hl_commandbuffers(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			for (uint64_t i = 0; i < dst.size(); i++) {
				auto& ci = cis[i];
				direct_command_pools.resize(direct_command_pools.size() < (ci.queue_family_index + 1) ? (ci.queue_family_index + 1) : direct_command_pools.size(), VK_NULL_HANDLE);
				auto& pool = direct_command_pools[ci.queue_family_index];
				if (pool == VK_NULL_HANDLE) {
					VkCommandPoolCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
					cpci.queueFamilyIndex = ci.queue_family_index;
					cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
					auto res = upstream->allocate_commandpools(std::span{ &pool, 1 }, std::span{ &cpci, 1 }, loc);
					if (!res) { // if we fail here, we don't need to free - other pools are freed during dtor
						return { expected_error, res.error() };
					}
				}

				dst[i].command_pool = pool;
				VkCommandBufferAllocateInfo cbai{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
				cbai.commandBufferCount = 1;
				cbai.commandPool = pool;
				cbai.level = ci.level;
				auto res = upstream->allocate_commandbuffers(std::span{ &dst[i].command_buffer, 1 }, std::span{ &cbai, 1 }, loc);
				if (!res) { // if we fail here, we don't need to free - pools are freed during dtor
					return { expected_error, res.error() };
				}
			}
			return { expected_value };
		}

		std::vector<VkFramebuffer> framebuffers;

		Result<void, AllocateException> allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			auto result = upstream->allocate_framebuffers(dst, cis, loc);
			framebuffers.insert(framebuffers.end(), dst.begin(), dst.end());
			return result;
		}

		void deallocate_framebuffers(std::span<const VkFramebuffer>) override {} // linear allocator, noop

		void wait() {
			if (fences.size() > 0) {
				vkWaitForFences(device, (uint32_t)fences.size(), fences.data(), true, UINT64_MAX);
			}
		}

		Context& get_context() override {
			return *ctx;
		}

		~DeviceLinearResource() {
			if (scope == SyncScope::eScope) {
				wait();
			}
			upstream->deallocate_fences(fences);
			upstream->deallocate_commandpools(command_pools);
			upstream->deallocate_commandpools(direct_command_pools);
			upstream->deallocate_framebuffers(framebuffers);
		}

		Context* ctx;
		VkDevice device;
		SyncScope scope;
		LegacyLinearAllocator linear_cpu_only;
		LegacyLinearAllocator linear_cpu_gpu;
		LegacyLinearAllocator linear_gpu_cpu;
		LegacyLinearAllocator linear_gpu_only;
	};
}