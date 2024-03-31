#pragma once

#include "vuk/Types.hpp"

namespace vuk {
	struct SyncPoint {
		struct Executor* executor;
		uint64_t visibility; // results are available if waiting for {executor, visibility}
	};
} // namespace vuk