#pragma once

#include <source_location>

namespace vuk {
/// @cond INTERNAL
#ifndef __cpp_consteval
	struct source_location {
		uint_least32_t line_{};
		uint_least32_t column_{};
		const char* file = "";
		const char* function = "";

		[[nodiscard]] constexpr source_location() noexcept = default;

		[[nodiscard]] static source_location current(const uint_least32_t line_ = __builtin_LINE(),
		                                             const uint_least32_t column_ = __builtin_COLUMN(),
		                                             const char* const file_ = __builtin_FILE(),
		                                             const char* const function_ = __builtin_FUNCTION()) noexcept {
			source_location result;
			result.line_ = line_;
			result.column_ = column_;
			result.file = file_;
			result.function = function_;
			return result;
		}

		[[nodiscard]] constexpr uint_least32_t line() const noexcept {
			return line_;
		}

		[[nodiscard]] constexpr uint_least32_t column() const noexcept {
			return line_;
		}

		[[nodiscard]] constexpr const char* file_name() const noexcept {
			return file;
		}

		[[nodiscard]] constexpr const char* function_name() const noexcept {
			return function;
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
} // namespace vuk

/// @cond INTERNAL
#define VUK_HERE_AND_NOW()                                                                                                                                     \
	SourceLocationAtFrame {                                                                                                                                      \
		vuk::source_location::current(), (uint64_t)-1LL                                                                                                            \
	}
/// @endcond