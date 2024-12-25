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

		AllocationEntry& resolve_ptr(ptr_base ptr);
		ViewEntry& resolve_view(view_base view);
		void install_as_thread_resolver();

		RadixTree<AllocationEntry> allocations;
		RadixTree<ViewEntry> views;
	};

	template<class Type = void, class... Constraints>
	struct ptr : ptr_base {
		using T = Type;

		Type* operator->()
		  requires (!std::is_same_v<Type, void>)
		{
			return reinterpret_cast<Type*>(Resolver::per_thread->resolve_ptr(*this).host_ptr);
		}

		Type& operator*()
		  requires (!std::is_same_v<Type, void>)
		{
			return *reinterpret_cast<Type*>(Resolver::per_thread->resolve_ptr(*this).host_ptr);
		}

		std::remove_extent_t<Type>& operator[](size_t index)
		  requires(!std::is_same_v<Type, void>)
		{
			return *(reinterpret_cast<std::remove_extent_t<Type>*>(Resolver::per_thread->resolve_ptr(*this).host_ptr) + index);
		}

		ptr operator+(size_t offset)
		  requires (!std::is_same_v<Type, void>)
		{
			return { device_address + offset * sizeof(Type) };
		}
	};

	template<class T>
	using Unique_ptr = Unique<ptr<T>>;

	template<class Type = void, class... Constraints>
	struct view : view_base {};

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
		//enum class PTEFlags {} flags;
	};

	template<Format f>
	struct FormatT {};

	struct VCI {
		ptr_base allocation;
		BufferViewCreateInfo buffer;
	};

	struct ViewEntry {
		ptr_base allocation;
		union {
			struct : BufferViewCreateInfo {
				size_t offset;
			} buffer;
			struct : IVCI {
				VkImageView view;
			} image;
		};
		enum class ViewEntryFlags {} flags;
	};
} // namespace vuk