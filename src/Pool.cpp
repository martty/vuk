#include "Pool.hpp"
#include "Context.hpp"

namespace vuk {
	// pools

	gsl::span<vk::Semaphore> PooledType<vk::Semaphore>::acquire(PerThreadContext& ptc, size_t count) {
		if (values.size() < (needle + count)) {
			auto remaining = values.size() - needle;
			for (auto i = 0; i < (count - remaining); i++) {
				auto nalloc = ptc.ctx.device.createSemaphore({});
				values.push_back(nalloc);
			}
		}
		gsl::span<vk::Semaphore> ret{ &*values.begin() + needle, (ptrdiff_t)count };
		needle += count;
		return ret;
	}

	template<>
	gsl::span<vk::Fence> PooledType<vk::Fence>::acquire(PerThreadContext& ptc, size_t count) {
		if (values.size() < (needle + count)) {
			auto remaining = values.size() - needle;
			for (auto i = 0; i < (count - remaining); i++) {
				auto nalloc = ptc.ctx.device.createFence({});
				values.push_back(nalloc);
			}
		}
		gsl::span<vk::Fence> ret{ &*values.begin() + needle, (ptrdiff_t)count };
		needle += count;
		return ret;
	}

	template<class T>
	void PooledType<T>::free(Context& ctx) {
		for (auto& v : values) {
			ctx.device.destroy(v);
		}
	}

	template struct PooledType<vk::Semaphore>;
	template struct PooledType<vk::Fence>;

	void PooledType<vk::Fence>::reset(Context& ctx) {
		ctx.device.waitForFences(values, true, UINT64_MAX);
		ctx.device.resetFences(values);
		needle = 0;
	}

	// vk::CommandBuffer pool
	PooledType<vk::CommandBuffer>::PooledType(Context& ctx) {
		pool = ctx.device.createCommandPoolUnique({});
	}

	gsl::span<vk::CommandBuffer> PooledType<vk::CommandBuffer>::acquire(PerThreadContext& ptc, size_t count) {
		if (values.size() < (needle + count)) {
			auto remaining = values.size() - needle;
			vk::CommandBufferAllocateInfo cbai;
			cbai.commandBufferCount = (unsigned)(count - remaining);
			cbai.commandPool = *pool;
			cbai.level = vk::CommandBufferLevel::ePrimary;
			auto nalloc = ptc.ctx.device.allocateCommandBuffers(cbai);
			values.insert(values.end(), nalloc.begin(), nalloc.end());
		}
		gsl::span<vk::CommandBuffer> ret{ &*values.begin() + needle, (ptrdiff_t)count };
		needle += count;
		return ret;
	}
	void PooledType<vk::CommandBuffer>::reset(Context& ctx) {
		vk::CommandPoolResetFlags flags = {};
		ctx.device.resetCommandPool(*pool, flags);
		needle = 0;
	}

	void PooledType<vk::CommandBuffer>::free(Context& ctx) {
		ctx.device.freeCommandBuffers(*pool, values);
		pool.reset();
	}

	// vk::DescriptorSet pool
	PooledType<vk::DescriptorSet>::PooledType(Context& ctx) {}

	vk::DescriptorPool PooledType<vk::DescriptorSet>::get_pool(PerThreadContext& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info) {
		if (pools.size() < (pool_needle + 1)) {
			vk::DescriptorPoolCreateInfo dpci;
			dpci.maxSets = sets_allocated == 0 ? 1 : sets_allocated * 2;
			std::array<vk::DescriptorPoolSize, VkDescriptorType::VK_DESCRIPTOR_TYPE_END_RANGE> descriptor_counts = {};
			size_t used_idx = 0;
			for (auto i = 0; i < descriptor_counts.size(); i++) {
				if (layout_alloc_info.descriptor_counts[i] > 0) {
					auto& d = descriptor_counts[used_idx];
					d.type = vk::DescriptorType(i);
					d.descriptorCount = layout_alloc_info.descriptor_counts[i] * dpci.maxSets;
					used_idx++;
				}
			}
			dpci.pPoolSizes = descriptor_counts.data();
			dpci.poolSizeCount = used_idx;
			pools.emplace_back(ptc.ifc.ctx.device.createDescriptorPoolUnique(dpci));
			sets_allocated = dpci.maxSets;
		}
		return *pools[pool_needle];
	}

	vk::DescriptorSet PooledType<vk::DescriptorSet>::acquire(PerThreadContext& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info) {
		if (free_sets.size() > 0) {
			vk::DescriptorSet ret = free_sets.back();
			free_sets.pop_back();
			return ret;
		}
		// allocate new ds
		// we find a descriptorpool that can still allocate
		// or create a new pool
		// typically we don't loop
		VkDescriptorSet ds;
		while (true) {
			vk::DescriptorSetAllocateInfo dsai;
			dsai.descriptorPool = get_pool(ptc, layout_alloc_info);
			dsai.descriptorSetCount = 1;
			dsai.pSetLayouts = &layout_alloc_info.layout;
			auto result = vkAllocateDescriptorSets(ptc.ctx.device, &(VkDescriptorSetAllocateInfo)dsai, &ds);
			if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
				pool_needle++;
			} else {
				assert(result == VK_SUCCESS);
				return vk::DescriptorSet(ds);
			}
		}
	}

	void PooledType<vk::DescriptorSet>::reset(Context& ctx) {
	}

	void PooledType<vk::DescriptorSet>::free(Context& ctx) {
	}
}
