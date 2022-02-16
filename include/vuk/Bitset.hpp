#pragma once

#include "vuk_fwd.hpp"

#include <bit>
#include <cstdint>

namespace vuk {
	template<uint64_t Count>
	struct Bitset {
		static constexpr uint64_t bitmask(uint64_t const onecount) {
			return static_cast<uint64_t>(-(onecount != 0)) & (static_cast<uint64_t>(-1) >> ((sizeof(uint64_t)) - onecount));
		}

		static constexpr uint64_t n_bits = sizeof(uint64_t) * 8;
		static constexpr uint64_t n_words = idivceil(Count, n_bits);
		static constexpr uint64_t remainder = Count - n_bits * (Count / n_bits);
		static constexpr uint64_t last_word_mask = bitmask(remainder);
		uint64_t words[n_words];

		Bitset& set(uint64_t pos, bool value = true) noexcept {
			auto word = pos / n_bits;
			if (value) {
				words[word] |= 1ULL << (pos - n_bits * word);
			} else {
				words[word] &= ~(1ULL << (pos - n_bits * word));
			}
			return *this;
		}

		uint64_t count() const noexcept {
			uint64_t accum = 0;
			for (uint64_t i = 0; i < (Count / n_bits); i++) {
				accum += std::popcount(words[i]);
			}
			if constexpr (remainder > 0) {
				accum += std::popcount(words[n_words - 1] & last_word_mask);
			}
			return accum;
		}

		bool test(uint64_t pos) const noexcept {
			auto word = pos / n_bits;
			return words[word] & 1ULL << (pos - n_bits * word);
		}

		void clear() noexcept {
			for (uint64_t i = 0; i < (Count / n_bits); i++) {
				words[i] = 0;
			}
		}

		bool operator==(const Bitset& other) const noexcept {
			for (uint64_t i = 0; i < (Count / n_bits); i++) {
				if (words[i] != other.words[i])
					return false;
			}
			if constexpr (remainder > 0) {
				return (words[n_words - 1] & last_word_mask) == (other.words[n_words - 1] & last_word_mask);
			}
			return true;
		}
	};
} // namespace vuk