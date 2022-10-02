#pragma once

#include "vuk/Future.hpp"
#include "vuk/partials/CountWithIndirect.hpp"
#include "vuk/partials/Scan.hpp"
#include "vuk/partials/Scatter.hpp"
#include "vuk/partials/StaticComputePBI.hpp"

namespace vuk {
	// stream compaction: we take src, perform a prefix scan with bool predicate into a temporary buffer (containing the indices for the compacted array)
	// then perform a gather to build the compacted array and produce the compacted count

	struct CompactionResult {
		Future result;
		Future count;
	};

	template<class T, class F>
	inline Future compact(Context& ctx, Future src, Future dst, Future count, uint32_t max_size, const F& fn) {
		auto [scan_result, count_p] = scan<T, F>(ctx, src, {}, std::move(count), max_size, fn);
		auto result = scatter<T>(ctx, src, dst, std::move(scan_result), std::move(count_p));
		return result;
	}
} // namespace vuk