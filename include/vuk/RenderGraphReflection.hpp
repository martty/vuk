#pragma once

#include "vuk/RelSpan.hpp"
#include "vuk/RenderGraph.hpp"
#include <optional>

// struct describing use chains
namespace vuk {
	struct ChainAccess {
		int32_t pass;
		int32_t resource = -1;
	};

	struct ChainLink {
		ChainLink* source = nullptr; // in subchains, this denotes the end of the undiverged chain
		ChainLink* prev = nullptr;   // if this came from a previous undef, we link them together
		std::optional<ChainAccess> def;
		RelSpan<ChainAccess> reads;
		Resource::Type type;
		std::optional<ChainAccess> undef;
		ChainLink* next = nullptr;        // if this links to a def, we link them together
		ChainLink* destination = nullptr; // in subchains, this denotes the start of the converged chain
	};

} // namespace vuk