#pragma once

#include "vuk/runtime/vk/Image.hpp"
#include <span>

namespace vuk {
	struct AllocationEntry;
	struct ViewEntry;

	struct Resolver {
		inline static thread_local Resolver* per_thread;

		void commit(uint64_t base, size_t size, AllocationEntry& ae);
		void decommit(uint64_t base, size_t size);

		void add_generic_view(uint64_t key, ViewEntry& ve);
		void remove_generic_view(uint64_t key);

		struct BufferWithOffset {
			VkBuffer buffer;
			size_t offset;
		};

		struct BufferWithOffsetAndSize : BufferWithOffset {
			size_t size;
		};

		AllocationEntry& resolve_ptr(ptr_base ptr);
		BufferWithOffset ptr_to_buffer_offset(ptr_base ptr);
		ViewEntry& resolve_view(generic_view_base view);
		void install_as_thread_resolver();

		RadixTree<AllocationEntry> allocations;
		RadixTree<ViewEntry> memory_views;
	};

	template<Format f>
	struct FormatT {};

	template<class Type = byte>
	struct ptr : ptr_base {
		using pointed_T = Type;
		using UnwrappedT = detail::unwrap<Type>::T;

		UnwrappedT* operator->()
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			auto& ae = Resolver::per_thread->resolve_ptr(*this);
			auto offset = device_address - ae.buffer.base_address;
			return reinterpret_cast<UnwrappedT*>(ae.host_ptr + offset);
		}

		auto& operator*()
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			auto& ae = Resolver::per_thread->resolve_ptr(*this);
			auto offset = device_address - ae.buffer.base_address;
			return *reinterpret_cast<UnwrappedT*>(ae.host_ptr + offset);
		}

		const auto& operator*() const
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			auto& ae = Resolver::per_thread->resolve_ptr(*this);
			auto offset = device_address - ae.buffer.base_address;
			return *reinterpret_cast<const UnwrappedT*>(ae.host_ptr + offset);
		}

		auto& operator[](size_t index)
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			auto& ae = Resolver::per_thread->resolve_ptr(*this);
			auto offset = device_address - ae.buffer.base_address;
			return *(reinterpret_cast<std::remove_extent_t<UnwrappedT>*>(ae.host_ptr + offset) + index);
		}

		ptr operator+(size_t offset) const
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			return { device_address + offset * sizeof(UnwrappedT) };
		}

		void operator+=(size_t offset)
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			device_address += offset * sizeof(UnwrappedT);
		}
	};

	template<class T>
	using Unique_ptr = Unique<ptr<T>>;

	template<class Type = void, size_t Extent = dynamic_extent>
	struct view : generic_view_base {
		auto& operator[](size_t index)
		  requires(!std::is_same_v<Type, void>)
		{
			if ((key & 0x3) == 0) { // generic memory view
				auto& ve = Resolver::per_thread->resolve_view(*this);
				assert(index < ve.buffer.count);
				return static_cast<ptr<Type>>(ve.ptr)[index];
			} else if ((key & 0x3) == 0x2) { // specific memory view
				return reinterpret_cast<view<BufferLike<Type>>*>(key & ~0x3)->operator[](index);
			}
		}

		const auto& operator[](size_t index) const
		  requires(!std::is_same_v<Type, void>)
		{
			if ((key & 0x3) == 0) { // generic memory view
				auto& ve = Resolver::per_thread->resolve_view(*this);
				assert(index < ve.buffer.count);
				return static_cast<ptr<Type>>(ve.ptr)[index];
			} else if ((key & 0x3) == 0x2) { // specific memory view
				return reinterpret_cast<const view<BufferLike<Type>>*>(key & ~0x3)->operator[](index);
			}
		}

		size_t size_bytes() const {
			if ((key & 0x3) == 0) { // generic memory view
				auto& ve = Resolver::per_thread->resolve_view(*this);
				return ve.buffer.count * ve.buffer.elem_size;
			} else if ((key & 0x3) == 0x2) { // specific memory view
				return reinterpret_cast<view<BufferLike<Type>>*>(key & ~0x3)->size_bytes();
			}
		}

		size_t count() const noexcept {
			return size_bytes() / sizeof(Type);
		}

		auto& data() {
			if ((key & 0x3) == 0) { // generic memory view
				auto& ve = Resolver::per_thread->resolve_view(*this);
				return static_cast<ptr<Type>&>(ve.ptr);
			} else if ((key & 0x3) == 0x2) { // specific memory view
				return reinterpret_cast<view<BufferLike<Type>>*>(key & ~0x3)->ptr;
			}
		}
	};

	/// @brief Buffer creation parameters
	struct BufferCreateInfo {
		/// @brief Memory usage to determine which heap to allocate the memory from
		MemoryUsage memory_usage = MemoryUsage::eUnset;
		/// @brief Size of the allocation in bytes
		VkDeviceSize size = ~(0u);
		/// @brief Alignment of the allocation in bytes
		VkDeviceSize alignment = 1;

		std::strong_ordering operator<=>(const BufferCreateInfo&) const noexcept = default;
	};

	/// @brief A contiguous portion of GPU-visible memory
	// fixed extent
	template<class Type, size_t Extent>
	struct view<BufferLike<Type>, Extent> {
		ptr<BufferLike<Type>> ptr;
		static constexpr size_t sz_bytes = Extent * sizeof(Type);

		view() = default;

		view(vuk::ptr<BufferLike<Type>> ptr)
		  requires(!std::is_array_v<Type>)
		    : ptr(ptr) {}

		auto& operator[](size_t index)
		  requires(!std::is_same_v<Type, void>)
		{
			assert(index < (sz_bytes / sizeof(Type)));
			return ptr[index];
		}

		const auto& operator[](size_t index) const
		  requires(!std::is_same_v<Type, void>)
		{
			assert(index < (sz_bytes / sizeof(Type)));
			return ptr[index];
		}

		explicit operator bool() const noexcept {
			return !!ptr;
		}

		[[nodiscard]] auto& data() noexcept {
			return ptr;
		}

		[[nodiscard]] size_t size_bytes() const noexcept {
			return sz_bytes;
		}

		[[nodiscard]] size_t count() const noexcept {
			return sz_bytes / sizeof(Type);
		}

		[[nodiscard]] view<BufferLike<byte>, sz_bytes> to_byte_view() const noexcept {
			return view<BufferLike<byte>, sz_bytes>{ vuk::ptr<BufferLike<byte>>{ ptr.device_address } };
		}

		template<class new_T>
		[[nodiscard]] view<BufferLike<new_T>, Extent * sizeof(Type) / sizeof(new_T)> cast() const noexcept {
			return view<BufferLike<new_T>, Extent * sizeof(Type) / sizeof(new_T)>{ vuk::ptr<BufferLike<new_T>>{ ptr.device_address } };
		}

		/// @brief Create a new view that is a subset of the original
		[[nodiscard]] view<BufferLike<Type>> subview(VkDeviceSize offset, VkDeviceSize new_count = ~(0ULL)) const {
			if (new_count == ~0ULL) {
				new_count = count() - offset;
			} else {
				assert(offset + new_count <= count());
			}
			return { ptr + offset, new_count };
		}

		[[nodiscard]] std::span<Type> to_span() noexcept {
			return std::span{ &*ptr, count() };
		}

		[[nodiscard]] std::span<const Type> to_span() const noexcept {
			return std::span{ &*ptr, count() };
		}

		operator view<BufferLike<Type>, dynamic_extent>() const noexcept {
			return { ptr, sz_bytes };
		}
		std::strong_ordering operator<=>(const view<BufferLike<Type>, Extent>&) const noexcept = default;
	};

	template<size_t FixedExtent>
	static view<BufferLike<Type>, FixedExtent> fixed_view(vuk::ptr<BufferLike<Type>> ptr) {
		return { ptr };
	}

	template<class Type>
	struct view<BufferLike<Type>, dynamic_extent> {
		ptr<BufferLike<Type>> ptr;
		size_t sz_bytes;

		view() = default;
		view(vuk::ptr<BufferLike<Type>> ptr, size_t count)
		  requires(!std::is_array_v<Type>)
		    : ptr(ptr), sz_bytes(count * sizeof(Type)) {}

		auto& operator[](size_t index)
		  requires(!std::is_same_v<Type, void>)
		{
			assert(index < (sz_bytes / sizeof(Type)));
			return ptr[index];
		}

		const auto& operator[](size_t index) const
		  requires(!std::is_same_v<Type, void>)
		{
			assert(index < (sz_bytes / sizeof(Type)));
			return ptr[index];
		}

		explicit operator bool() const noexcept {
			return !!ptr;
		}

		[[nodiscard]] auto& data() noexcept {
			return ptr;
		}

		[[nodiscard]] size_t size_bytes() const noexcept {
			return sz_bytes;
		}

		[[nodiscard]] size_t count() const noexcept {
			return sz_bytes / sizeof(Type);
		}

		[[nodiscard]] view<BufferLike<byte>> to_byte_view() const noexcept {
			return { vuk::ptr<BufferLike<byte>>{ ptr.device_address }, sz_bytes };
		}

		template<class new_T>
		[[nodiscard]] view<BufferLike<new_T>> cast() const noexcept {
			return { vuk::ptr<BufferLike<new_T>>{ ptr.device_address }, sz_bytes };
		}

		/// @brief Create a new view that is a subset of the original
		[[nodiscard]] view<BufferLike<Type>> subview(VkDeviceSize offset, VkDeviceSize new_count = ~(0ULL)) const {
			if (new_count == ~0ULL) {
				new_count = count() - offset;
			} else {
				assert(offset + new_count <= count());
			}
			return { ptr + offset, new_count };
		}

		[[nodiscard]] std::span<Type> to_span() noexcept {
			return std::span{ &*ptr, count() };
		}

		[[nodiscard]] std::span<const Type> to_span() const noexcept {
			return std::span{ &*ptr, count() };
		}
		std::strong_ordering operator<=>(const view<BufferLike<Type>, dynamic_extent>&) const noexcept = default;
	};

	template<class T, size_t Extent>
	struct is_view_type<view<T, Extent>> {
		static constexpr bool value = true;
	};

	template<class T>
	struct is_bufferlike_view_type<view<BufferLike<T>>> {
		static constexpr bool value = true;
	};

	struct BufferViewCreateInfo {
		size_t elem_size;
		size_t count;
		Format format = Format::eUndefined;
	};

	struct AllocationEntry {
		byte* host_ptr;
		union {
			struct : BufferCreateInfo {
				VkBuffer buffer;
				size_t offset;
				uint64_t base_address;
			} buffer = {};
			struct : ImageCreateInfo {
				VkImage image;
			} image;
		};
		struct ViewEntry* default_view;
		VkDeviceMemory device_memory;
		void* allocation;
		// enum class PTEFlags {} flags;
	};

	struct BVCI {
		ptr_base ptr;
		BufferViewCreateInfo vci;
	};

	struct IVCI2 {
		ptr_base ptr;
		IVCI vci;
	};

	struct ViewEntry {
		ptr_base ptr;
		Offset size;
		union {
			struct : BufferViewCreateInfo {
			} buffer;
			struct : IVCI {
				VkImageView view;
			} image;
		};
		// enum class ViewEntryFlags {} flags;
	};
} // namespace vuk

namespace std {
	template<class T, size_t Extent>
	struct hash<vuk::view<vuk::BufferLike<T>, Extent>> {
		size_t operator()(vuk::view<vuk::BufferLike<T>, Extent> const& x) const {
			uint32_t v = std::hash<uint32_t>()(x.ptr);
			hash_combine_direct(v, std::hash<uint32_t>(x.sz_bytes));
			return v;
		}
	};
}; // namespace std