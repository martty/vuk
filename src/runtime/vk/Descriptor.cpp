#include "vuk/runtime/vk/Descriptor.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"

#include <concurrentqueue.h>
#include <mutex>
#include <robin_hood.h>

namespace vuk {
	struct DescriptorPoolImpl {
		std::mutex grow_mutex;
		std::vector<VkDescriptorPool> pools;
		uint32_t sets_allocated = 0;
		moodycamel::ConcurrentQueue<VkDescriptorSet> free_sets{ 1024 };
	};

	DescriptorPool::DescriptorPool() : impl(new DescriptorPoolImpl) {}
	DescriptorPool::~DescriptorPool() {
		delete impl;
	}

	DescriptorPool::DescriptorPool(DescriptorPool&& o) noexcept {
		if (impl) {
			delete impl;
		}
		impl = o.impl;
		o.impl = nullptr;
	}

	void DescriptorPool::grow(Context& ctx, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info) {
		if (!impl->grow_mutex.try_lock())
			return;
		VkDescriptorPoolCreateInfo dpci{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		dpci.maxSets = impl->sets_allocated == 0 ? 1 : impl->sets_allocated * 2;
		std::array<VkDescriptorPoolSize, 12> descriptor_counts = {};
		size_t count = ctx.vkCmdBuildAccelerationStructuresKHR ? descriptor_counts.size() : descriptor_counts.size() - 1;
		uint32_t used_idx = 0;
		for (size_t i = 0; i < count; i++) {
			if (layout_alloc_info.descriptor_counts[i] > 0) {
				auto& d = descriptor_counts[used_idx];
				d.type = i == 11 ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR : VkDescriptorType(i);
				d.descriptorCount = layout_alloc_info.descriptor_counts[i] * dpci.maxSets;
				used_idx++;
			}
		}
		dpci.pPoolSizes = descriptor_counts.data();
		dpci.poolSizeCount = used_idx;
		VkDescriptorPool pool;
		ctx.vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &pool);
		impl->pools.emplace_back(pool);

		VkDescriptorSetAllocateInfo dsai{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		dsai.descriptorPool = impl->pools.back();
		dsai.descriptorSetCount = dpci.maxSets;
		std::vector<VkDescriptorSetLayout> layouts(dpci.maxSets, layout_alloc_info.layout);
		dsai.pSetLayouts = layouts.data();
		// allocate all the descriptorsets
		std::vector<VkDescriptorSet> sets(dsai.descriptorSetCount);
		ctx.vkAllocateDescriptorSets(ctx.device, &dsai, sets.data());
		impl->free_sets.enqueue_bulk(sets.data(), sets.size());
		impl->sets_allocated = dpci.maxSets;

		impl->grow_mutex.unlock();
	}

	VkDescriptorSet DescriptorPool::acquire(Context& ctx, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info) {
		VkDescriptorSet ds;
		while (!impl->free_sets.try_dequeue(ds)) {
			grow(ctx, layout_alloc_info);
		}
		return ds;
	}

	void DescriptorPool::release(VkDescriptorSet ds) {
		impl->free_sets.enqueue(ds);
	}

	void DescriptorPool::destroy(Context& ctx, VkDevice device) const {
		for (auto& p : impl->pools) {
			ctx.vkDestroyDescriptorPool(device, p, nullptr);
		}
	}

	SetBinding SetBinding::finalize(Bitset<VUK_MAX_BINDINGS> used_mask) {
		SetBinding final;
		final.used = used_mask;
		final.layout_info = layout_info;
		uint32_t mask = (uint32_t)used_mask.to_ulong();
		for (size_t i = 0; i < VUK_MAX_BINDINGS; i++) {
			if ((mask & (1 << i)) == 0) {
				continue;
			} else {
				final.bindings[i] = bindings[i];
			}
		}
		return final;
	}
} // namespace vuk