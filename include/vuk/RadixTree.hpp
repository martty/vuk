#pragma once

#include "vuk/Types.hpp"
#include <utility>

namespace vuk {
	template<class T>
	class RadixTree {
	public:
		RadixTree();

		static constexpr uint64_t first_bit = 0x8000'0000'0000'0000;

		T* find(uint64_t base);

		bool insert(uint64_t base, size_t size, T value);

		template<class F, F f, class... Args>
		bool handle_unaligned(size_t base, size_t size, Args... values);

		bool insert_unaligned(size_t base, size_t size, T value);

		bool erase(uint64_t base, size_t size = -1);

		void erase_unaligned(uint64_t base, size_t size);

	private:
		void* root;
	};

	extern template class RadixTree<int>;
	extern template class RadixTree<std::pair<size_t, size_t>>;
	extern template class RadixTree<void*>;
} // namespace vuk