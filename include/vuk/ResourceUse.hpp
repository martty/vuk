#pragma once

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

	struct Acquire {
		ResourceUse src_use;
		DomainFlagBits initial_domain = DomainFlagBits::eAny;
		uint64_t initial_visibility;
		bool unsynchronized = false;
	};
} // namespace vuk