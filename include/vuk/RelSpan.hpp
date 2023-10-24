#pragma once

#include <span>
#include <vector>

namespace vuk {
	template<class T>
	struct RelSpan {
		size_t offset0 = 0;
		size_t offset1 = 0;

		constexpr size_t size() const noexcept {
			return offset1 - offset0;
		}

		constexpr std::span<T> to_span(T* base) const noexcept {
			return std::span{ base + offset0, base + offset1 };
		}

		constexpr std::span<T> to_span(std::vector<T>& base) const noexcept {
			return std::span{ base.data() + offset0, base.data() + offset1 };
		}

		constexpr std::span<const T> to_span(const std::vector<T>& base) const noexcept {
			return std::span{ base.data() + offset0, base.data() + offset1 };
		}

		void append(std::vector<T>& base, T value) {
			// easy case: we have space at the end of the vector
			if (offset1 == base.size()) {
				base.push_back(std::move(value));
				offset1++;
				return;
			}
			// non-easy case: copy the span to the end and extend
			auto new_offset0 = base.size();
			base.resize(base.size() + offset1 - offset0 + 1);
			std::copy(base.begin() + offset0, base.begin() + offset1, base.begin() + new_offset0);
			base.back() = std::move(value);
			offset0 = new_offset0;
			offset1 = base.size();
		}
	};
} // namespace vuk