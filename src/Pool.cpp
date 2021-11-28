#include "Pool.hpp"
#include "vuk/Context.hpp"

namespace vuk {
	// TimestampQuery pool
	PooledType<TimestampQuery>::PooledType(Context& ctx) {
		VkQueryPoolCreateInfo qpci{ .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 128 };
		vkCreateQueryPool(ctx.device, &qpci, nullptr, &pool);
		vkResetQueryPool(ctx.device, pool, 0, 128);
		values.resize(128);
		host_values.resize(128);
		for (uint32_t i = 0; i < 128; i++) {
			values[i] = { pool, i };
		}
	}

	std::span<TimestampQuery> PooledType<TimestampQuery>::acquire(PerThreadContext& ptc, size_t count) {
		if (values.size() < (needle + count)) {
			//auto remaining = values.size() - needle;

			assert(false && "Unimpl.");
		}
		std::span<TimestampQuery> ret{ &*values.begin() + needle, count };
		needle += count;
		return ret;
	}

	void PooledType<TimestampQuery>::get_results(Context& ctx) {
		// harvest query results
		vkGetQueryPoolResults(ctx.device, pool, 0, 128, sizeof(uint64_t) * 128, host_values.data(), sizeof(uint64_t), VkQueryResultFlagBits::VK_QUERY_RESULT_64_BIT);
	}

	void PooledType<TimestampQuery>::reset(Context& ctx) {
		vkResetQueryPool(ctx.device, pool, 0, 128);
		needle = 0;
	}

	void PooledType<TimestampQuery>::free(Context& ctx) {
		vkDestroyQueryPool(ctx.device, pool, nullptr);
	}
}
