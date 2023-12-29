#pragma once

#include "vuk/RelSpan.hpp"
#include "vuk/RenderGraph.hpp"
#include <optional>

// struct describing use chains
namespace vuk {
	struct ChainLink {
		Ref urdef;                   // the first def
		ChainLink* source = nullptr; // in subchains, this denotes the end of the undiverged chain
		ChainLink* prev = nullptr;   // if this came from a previous undef, we link them together
		Ref def;
		RelSpan<Ref> reads;
		Type* type;
		Node* undef = nullptr;
		ChainLink* next = nullptr;        // if this links to a def, we link them together
		ChainLink* destination = nullptr; // in subchains, this denotes the start of the converged chain
		RelSpan<ChainLink*> child_chains;
	};

} // namespace vuk