#pragma once

#include <type_traits>
#include <memory>
#include <utility>

namespace vuk {
	template<typename BitType>
	class Flags {
	public:
		using MaskType = typename std::underlying_type_t<BitType>;

		// constructors
		constexpr Flags() noexcept : m_mask(0) {}

		constexpr Flags(BitType bit) noexcept : m_mask(static_cast<MaskType>(bit)) {}

		constexpr Flags(Flags<BitType> const& rhs) noexcept : m_mask(rhs.m_mask) {}

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
		constexpr Flags<BitType>& operator=(Flags<BitType> const& rhs) noexcept {
			m_mask = rhs.m_mask;
			return *this;
		}

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

		MaskType m_mask;
	};
} // namespace vuk

namespace std {
	template<class BitType>
	struct hash<vuk::Flags<BitType>> {
		size_t operator()(vuk::Flags<BitType> const& x) const {
			return std::hash<typename vuk::Flags<BitType>::MaskType>()((typename vuk::Flags<BitType>::MaskType)x);
		}
	};
}; // namespace std
