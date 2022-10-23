#pragma once

#include <source_location>

namespace vuk {
/// @cond INTERNAL
#ifndef __cpp_consteval
	struct source_location {
		uint_least32_t _Line{};
		uint_least32_t _Column{};
		const char* _File = "";
		const char* _Function = "";

		[[nodiscard]] constexpr source_location() noexcept = default;

		[[nodiscard]] static source_location current(const uint_least32_t _Line_ = __builtin_LINE(),
		                                             const uint_least32_t _Column_ = __builtin_COLUMN(),
		                                             const char* const _File_ = __builtin_FILE(),
		                                             const char* const _Function_ = __builtin_FUNCTION()) noexcept {
			source_location _Result;
			_Result._Line = _Line_;
			_Result._Column = _Column_;
			_Result._File = _File_;
			_Result._Function = _Function_;
			return _Result;
		}
	};

	struct SourceLocationAtFrame {
		source_location location;
		uint64_t absolute_frame;
	};
#else
	struct SourceLocationAtFrame {
		std::source_location location;
		uint64_t absolute_frame;
	};

	using source_location = std::source_location;
#endif
#define VUK_HERE_AND_NOW()                                                                                                                                     \
	SourceLocationAtFrame { source_location::current(), (uint64_t)-1LL }

} // namespace vuk