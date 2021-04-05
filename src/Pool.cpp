#include "Pool.hpp"
#include "vuk/Context.hpp"

namespace vuk {
	// pools
	template<>
	std::span<VkSemaphore> PooledType<VkSemaphore>::acquire(GlobalAllocator& ga, size_t count) {
		if (values.size() < (needle + count)) {
			auto remaining = values.size() - needle;
			for (auto i = 0; i < (count - remaining); i++) {
				VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
				VkSemaphore sema;
				vkCreateSemaphore(ga.device, &sci, nullptr, &sema);
				values.push_back(sema);
			}
		}
		std::span<VkSemaphore> ret{ &*values.begin() + needle, count };
		needle += count;
		return ret;
	}

	template<>
	std::span<VkFence> PooledType<VkFence>::acquire(GlobalAllocator& ga, size_t count) {
		if (values.size() < (needle + count)) {
			auto remaining = values.size() - needle;
			for (auto i = 0; i < (count - remaining); i++) {
				VkFenceCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
				VkFence fence;
				vkCreateFence(ga.device, &sci, nullptr, &fence);
				values.push_back(fence);
			}
		}
		std::span<VkFence> ret{ &*values.begin() + needle, count };
		needle += count;
		return ret;
	}

	template<>
	void PooledType<VkSemaphore>::free(GlobalAllocator& ga) {
		for (auto& v : values) {
			vkDestroySemaphore(ga.device, v, nullptr);
		}
	}

	template<>
	void PooledType<VkFence>::free(GlobalAllocator& ga) {
		for (auto& v : values) {
			vkDestroyFence(ga.device, v, nullptr);
		}
	}

	template struct PooledType<VkSemaphore>;
	template struct PooledType<VkFence>;

	template<>
	void PooledType<VkFence>::reset(GlobalAllocator& ga) {
		if (needle > 0) {
			vkWaitForFences(ga.device, (uint32_t)needle, values.data(), true, UINT64_MAX);
			vkResetFences(ga.device, (uint32_t)needle, values.data());
		}
		needle = 0;
	}

	// vk::CommandBuffer pool
	PooledType<VkCommandBuffer>::PooledType(GlobalAllocator& ga) {
		VkCommandPoolCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
		vkCreateCommandPool(ga.device, &cpci, nullptr, &pool);
	}

	std::span<VkCommandBuffer> PooledType<VkCommandBuffer>::acquire(GlobalAllocator& ga, VkCommandBufferLevel level, size_t count) {
		auto& values = level == VK_COMMAND_BUFFER_LEVEL_PRIMARY ? p_values : s_values;
		auto& needle = level == VK_COMMAND_BUFFER_LEVEL_PRIMARY ? p_needle : s_needle;
		if (values.size() < (needle + count)) {
			auto remaining = values.size() - needle;
			VkCommandBufferAllocateInfo cbai{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
			cbai.commandBufferCount = (unsigned)(count - remaining);
			cbai.commandPool = pool;
			cbai.level = level;
			auto ori_end = values.size();
			values.resize(needle + count);
			vkAllocateCommandBuffers(ga.device, &cbai, values.data() + ori_end);
		}
		std::span<VkCommandBuffer> ret{ &*values.begin() + needle, count };
		needle += count;
		return ret;
	}

	void PooledType<VkCommandBuffer>::reset(GlobalAllocator& ga) {
		VkCommandPoolResetFlags flags = {};
		vkResetCommandPool(ga.device, pool, flags);
		p_needle = s_needle = 0;
	}

	void PooledType<VkCommandBuffer>::free(GlobalAllocator& ga) {
		vkDestroyCommandPool(ga.device, pool, nullptr);
	}

	// TimestampQuery pool
	PooledType<TimestampQuery>::PooledType(GlobalAllocator& ga) {
		VkQueryPoolCreateInfo qpci{ .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 128 };
		vkCreateQueryPool(ga.device, &qpci, nullptr, &pool);
		vkResetQueryPool(ga.device, pool, 0, 128);
		values.resize(128);
		host_values.resize(128);
		for (uint32_t i = 0; i < 128; i++) {
			values[i] = { pool, i };
		}
	}

	std::span<TimestampQuery> PooledType<TimestampQuery>::acquire(GlobalAllocator&, size_t count) {
		if (values.size() < (needle + count)) {
			//auto remaining = values.size() - needle;

			assert(false && "Unimpl.");
		}
		std::span<TimestampQuery> ret{ &*values.begin() + needle, count };
		needle += count;
		return ret;
	}

	void PooledType<TimestampQuery>::get_results(GlobalAllocator& ga) {
		// harvest query results
		vkGetQueryPoolResults(ga.device, pool, 0, 128, sizeof(uint64_t) * 128, host_values.data(), sizeof(uint64_t), VkQueryResultFlagBits::VK_QUERY_RESULT_64_BIT);
	}

	void PooledType<TimestampQuery>::reset(GlobalAllocator& ga) {
		vkResetQueryPool(ga.device, pool, 0, 128);
		needle = 0;
	}

	void PooledType<TimestampQuery>::free(GlobalAllocator& ga) {
		vkDestroyQueryPool(ga.device, pool, nullptr);
	}
}
