#pragma once

#include "vuk/SyncPoint.hpp"
#include "vuk/Types.hpp"
#include "vuk/runtime/vk/VkTypes.hpp" // TODO: leaking vk

namespace vuk {
	struct ResourceUse {
		PipelineStageFlags stages;
		AccessFlags access;
		ImageLayout layout; // ignored for buffers

		bool operator==(const ResourceUse&) const = default;
	};

	struct StreamResourceUse : ResourceUse {
		struct Stream* stream;
	};
} // namespace vuk