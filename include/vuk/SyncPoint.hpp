#pragma once

#include "vuk/Types.hpp"
#include <vector>

namespace vuk {
	struct SyncPoint {
		struct Executor* executor = nullptr;
		uint64_t visibility; // results are available if waiting for {executor, visibility}
	};

	/// @brief Encapsulates a SyncPoint that can be synchronized against in the future
	struct Signal {
	public:
		enum class Status {
			eDisarmed,       // the Signal is in the initial state - it must be armed before it can be sync'ed against
			eSynchronizable, // this syncpoint has been submitted (result is available on device with appropriate sync)
			eHostAvailable   // the result is available on host, available on device without sync
		};

		Status status = Status::eDisarmed;
		SyncPoint source;
	};

	struct ResourceUse;

	struct AcquireRelease : Signal {
		std::vector<ResourceUse> last_use; // last access performed on resource before signalling
	};
} // namespace vuk