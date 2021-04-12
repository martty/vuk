#pragma once

#include <vuk/vuk_fwd.hpp>

namespace vuk {
	struct Allocator {
		Allocator(class Context& ctx) : ctx(ctx) {}
		virtual VkSemaphore allocate_timeline_semaphore(uint64_t initial_value, uint64_t absolute_frame, SourceLocation) = 0;

		virtual struct TokenData& get_token_data(struct Token) = 0;
		virtual void destroy(Token) = 0;

		class Context& ctx;
	};
}