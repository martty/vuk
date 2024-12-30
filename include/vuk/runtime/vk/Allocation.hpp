#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/runtime/vk/Image.hpp"

namespace vuk {
	struct AllocationEntry;
	struct ViewEntry;

	struct Resolver {
		inline static thread_local Resolver* per_thread;

		void commit(uint64_t base, size_t size, AllocationEntry& ae);
		void decommit(uint64_t base, size_t size);

		void add_generic_view(uint64_t key, ViewEntry& ve);
		void remove_generic_view(uint64_t key);

		AllocationEntry& resolve_ptr(ptr_base ptr);
		ViewEntry& resolve_view(generic_view_base view);
		void install_as_thread_resolver();

		RadixTree<AllocationEntry> allocations;
		RadixTree<ViewEntry> memory_views;
	};

	template<Format f>
	struct FormatT {};

	template<class Type = void, class... Constraints>
	struct ptr : ptr_base {
		using pointed_T = Type;
		using UnwrappedT = detail::unwrap<Type>::T;

		UnwrappedT* operator->()
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			return reinterpret_cast<UnwrappedT*>(Resolver::per_thread->resolve_ptr(*this).host_ptr);
		}

		auto& operator*()
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			return *reinterpret_cast<UnwrappedT*>(Resolver::per_thread->resolve_ptr(*this).host_ptr);
		}

		auto& operator[](size_t index)
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			return *(reinterpret_cast<std::remove_extent_t<UnwrappedT>*>(Resolver::per_thread->resolve_ptr(*this).host_ptr) + index);
		}

		ptr operator+(size_t offset)
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			return { device_address + offset * sizeof(UnwrappedT) };
		}
	};

	template<class T>
	using Unique_ptr = Unique<ptr<T>>;

	template<class Type = void, class... Constraints>
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

	template<class Type, class... Constraints>
	struct view_base<view<Type, Constraints...>> {
		static constexpr bool value = true;
	};

	using byte = std::byte;

	template<class Type, class... Constraints>
	struct view<BufferLike<Type>, Constraints...> {
		ptr<Type> ptr;
		size_t sz_bytes;

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

		operator view<Type>() const noexcept {
			view<Type> v;
			v.key = reinterpret_cast<uintptr_t>(this) | 0x2;
			return v;
		}

		explicit operator bool() const noexcept {
			return !!ptr;
		}

		auto& data() noexcept {
			return ptr;
		}

		size_t size_bytes() const noexcept {
			return sz_bytes;
		}

		size_t count() const noexcept {
			return sz_bytes / sizeof(Type);
		}

		view<BufferLike<byte>> to_byte_view() const noexcept {
			return { vuk::ptr<BufferLike<byte>>{ ptr.device_address }, sz_bytes };
		}

		std::strong_ordering operator<=>(const view<BufferLike<Type>, Constraints...>&) const noexcept = default;
	};

	template<class... Constraints>
	struct view<BufferLike<FormatT<Format::eUndefined>>, Constraints...> {
		ptr<BufferLike<FormatT<Format::eUndefined>>> ptr;
		size_t size_bytes;
		Format format;
	};

	struct AllocationEntry {
		void* host_ptr;
		union {
			struct : BufferCreateInfo {
				VkBuffer buffer;
				size_t offset;
				uint64_t base_address;
			} buffer;
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