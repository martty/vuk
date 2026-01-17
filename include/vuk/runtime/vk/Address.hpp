#pragma once

#include "vuk/Types.hpp"
#include <cstddef>
#include <cstdint>

namespace vuk {
	struct VirtualAddressSpace;

	/// @brief A continuous range of addresses in a virtual address space
	struct VirtualAllocation {
		void* allocation = nullptr;
		uint64_t offset = 0;
		VirtualAddressSpace* address_space = nullptr;

		constexpr bool operator==(const VirtualAllocation& o) const noexcept = default;

		constexpr explicit operator bool() const noexcept {
			return allocation != nullptr;
		}

		/// @brief Get the virtual address at the given offset
		/// @param byte_offset Offset in bytes from the start of this allocation
		/// @return Virtual address (offset from the start of the address space)
		constexpr uint64_t operator[](uint64_t byte_offset) const noexcept {
			return offset + byte_offset;
		}
	};

	/// @brief A block of virtual address space
	struct VirtualAddressSpace {
		void* block = nullptr;
		size_t size = 0;

		constexpr bool operator==(const VirtualAddressSpace& o) const noexcept = default;

		constexpr explicit operator bool() const noexcept {
			return block != nullptr;
		}
	};

	/// @brief VirtualAllocation creation parameters
	struct VirtualAllocationCreateInfo {
		/// @brief Size of the allocation in bytes
		size_t size;
		/// @brief Alignment of the allocation in bytes
		size_t alignment = 1;
		/// @brief Address space to allocate from
		VirtualAddressSpace* address_space = nullptr;
	};

	/// @brief VirtualAddressSpace creation parameters
	struct VirtualAddressSpaceCreateInfo {
		/// @brief Size of the address space in bytes
		size_t size;
	};
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::VirtualAllocation> {
		size_t operator()(vuk::VirtualAllocation const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.allocation, x.offset, x.address_space);
			return h;
		}
	};

	template<>
	struct hash<vuk::VirtualAddressSpace> {
		size_t operator()(vuk::VirtualAddressSpace const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.block, x.size);
			return h;
		}
	};
} // namespace std
