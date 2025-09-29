#pragma once

#include <vuk/Value.hpp>

// Allocator GlobalAllocator;
// thread_local Compiler GlobalCompiler;

namespace vuk {
	enum Grow { eNone, eHost, eDeviceConcurrent };
	enum Deletes { eBack, eUnordered, eRandomAccess };

	template<class T>
	struct linear_container {
		ptr<T> ptr;
		size_t size_capacity_layout; /* 31:31:2 */
	};

	/// <summary>
	/// Containers
	/// </summary>
	template<class T>
	struct Value<linear_container<T>> {
		val_ptr<T> ptr;
		Value<size_t> size_capacity_layout;
		DomainFlagBits domain;

		template<Grow grow = Grow::eNone, Deletes deletes = Deletes::eBack>
		Array(size_t size, DomainFlagBits initial_domain, Allocator allocator /* = GlobalAllocator*/);

		Value<T> operator[](Value<size_t> i) {
			// IR_BOUNDS_CHECK?
			return ptr[i];
		}

		val_ptr<T> begin() {
			return ptr;
		}
		val_ptr<T> end() {
			return ptr + size;
		}

		Value<size_t> size() const {
			return size;
		}

		operator val_view<BufferLike<T>>() {
			return { ptr, size };
		}

		void to_device();
		void to_host();

	private:
	};

} // namespace vuk