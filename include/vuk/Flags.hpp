#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace vuk {
	template<typename BitType>
	class Flags {
	public:
		using MaskType = typename std::underlying_type_t<BitType>;

		// constructors
		constexpr Flags() noexcept = default;

		constexpr Flags(BitType bit) noexcept : m_mask(static_cast<MaskType>(bit)) {}

		constexpr explicit Flags(MaskType flags) noexcept : m_mask(flags) {}

		constexpr bool operator<(Flags<BitType> const& rhs) const noexcept {
			return m_mask < rhs.m_mask;
		}

		constexpr bool operator<=(Flags<BitType> const& rhs) const noexcept {
			return m_mask <= rhs.m_mask;
		}

		constexpr bool operator>(Flags<BitType> const& rhs) const noexcept {
			return m_mask > rhs.m_mask;
		}

		constexpr bool operator>=(Flags<BitType> const& rhs) const noexcept {
			return m_mask >= rhs.m_mask;
		}

		constexpr bool operator==(Flags<BitType> const& rhs) const noexcept {
			return m_mask == rhs.m_mask;
		}

		constexpr bool operator!=(Flags<BitType> const& rhs) const noexcept {
			return m_mask != rhs.m_mask;
		}

		// logical operator
		constexpr bool operator!() const noexcept {
			return !m_mask;
		}

		// assignment operators
		constexpr Flags<BitType>& operator|=(Flags<BitType> const& rhs) noexcept {
			m_mask |= rhs.m_mask;
			return *this;
		}

		constexpr Flags<BitType>& operator&=(Flags<BitType> const& rhs) noexcept {
			m_mask &= rhs.m_mask;
			return *this;
		}

		constexpr Flags<BitType>& operator^=(Flags<BitType> const& rhs) noexcept {
			m_mask ^= rhs.m_mask;
			return *this;
		}

		// cast operators
		explicit constexpr operator bool() const noexcept {
			return !!m_mask;
		}

		explicit constexpr operator MaskType() const noexcept {
			return m_mask;
		}

		// bitwise operators
		friend constexpr Flags<BitType> operator&(Flags<BitType> const& lhs, Flags<BitType> const& rhs) noexcept {
			return Flags<BitType>(lhs.m_mask & rhs.m_mask);
		}

		friend constexpr Flags<BitType> operator|(Flags<BitType> const& lhs, Flags<BitType> const& rhs) noexcept {
			return Flags<BitType>(lhs.m_mask | rhs.m_mask);
		}

		friend constexpr Flags<BitType> operator^(Flags<BitType> const& lhs, Flags<BitType> const& rhs) noexcept {
			return Flags<BitType>(lhs.m_mask ^ rhs.m_mask);
		}

		friend constexpr Flags<BitType> operator&(Flags<BitType> const& lhs, BitType const& rhs) noexcept {
			return Flags<BitType>(lhs.m_mask & (std::underlying_type_t<BitType>)rhs);
		}

		friend constexpr Flags<BitType> operator|(Flags<BitType> const& lhs, BitType const& rhs) noexcept {
			return Flags<BitType>(lhs.m_mask | (std::underlying_type_t<BitType>)rhs);
		}

		friend constexpr Flags<BitType> operator^(Flags<BitType> const& lhs, BitType const& rhs) noexcept {
			return Flags<BitType>(lhs.m_mask ^ (std::underlying_type_t<BitType>)rhs);
		}

		MaskType m_mask{ 0 };
	};

	template<typename BitType>
	std::string format_as(const Flags<BitType>& flags) {
		using MaskType = typename Flags<BitType>::MaskType;

		if (flags.m_mask == 0) {
			return "None";
		}

		std::string result;
		bool first = true;

		// Iterate through all possible bit positions
		for (size_t i = 0; i < sizeof(MaskType) * 8; ++i) {
			MaskType bit_value = MaskType(1) << i;
			if (flags.m_mask & bit_value) {
				if (!first) {
					result += "|";
				}
				first = false;

				// Convert the bit back to the enum type and format it
				BitType bit_enum = static_cast<BitType>(bit_value);
				result += format_as(bit_enum);
			}
		}

		return result;
	}
} // namespace vuk

namespace std {
	template<class BitType>
	struct hash<vuk::Flags<BitType>> {
		size_t operator()(vuk::Flags<BitType> const& x) const {
			return std::hash<typename vuk::Flags<BitType>::MaskType>()((typename vuk::Flags<BitType>::MaskType)x);
		}
	};
}; // namespace std
