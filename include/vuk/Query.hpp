#pragma once

#include <stdint.h>

namespace vuk {
	/// @brief Handle to a query result
	struct Query {
		uint64_t id;

		constexpr bool operator==(const Query& other) const noexcept {
			return id == other.id;
		}
	};

	struct TimestampQueryPool {
		static constexpr uint32_t num_queries = 32;

		VkQueryPool pool;
		Query queries[num_queries];
		uint8_t count = 0;
	};

	struct TimestampQuery {
		VkQueryPool pool;
		uint32_t id;
	};

	struct TimestampQueryCreateInfo {
		TimestampQueryPool* pool = nullptr;
		Query query;
	};
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::Query> {
		size_t operator()(vuk::Query const& s) const {
			return hash<uint64_t>()(s.id);
		}
	};
} // namespace std