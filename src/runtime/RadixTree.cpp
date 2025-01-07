#include "vuk/RadixTree.hpp"
#include <atomic>
#include <bit>
#include <fmt/format.h>

#include "vuk/runtime/vk/Allocation.hpp"

namespace vuk {
	template<class T>
	struct RadixTreeNode {
		std::atomic<RadixTreeNode*> right = nullptr;
		std::atomic<RadixTreeNode*> left = nullptr;
		std::atomic_bool present = false;
		T value = {};
	};

	template<class T>
	RadixTree<T>::RadixTree() {
		root = new RadixTreeNode<T>;
	}

	template<class T>
	T* RadixTree<T>::find(uint64_t key) {
		uint64_t bit = first_bit;
		RadixTreeNode<T>* node = reinterpret_cast<RadixTreeNode<T>*>(root);

		while (node) {
			if (node->present) {
				return &node->value;
			}

			if (key & bit) {
				node = node->right;
			} else {
				node = node->left;
			}

			bit >>= 1;
		}

		return nullptr;
	}

	template<class T>
	bool RadixTree<T>::insert(uint64_t key, size_t size, T value) {
		bool already_exists = false;
		uint64_t bit = first_bit;
		auto width = std::bit_width(size);
		uint64_t mask = ~((1 << (width - 1)) - 1);
		RadixTreeNode<T>* node = reinterpret_cast<RadixTreeNode<T>*>(root);
		RadixTreeNode<T>* next = reinterpret_cast<RadixTreeNode<T>*>(root);

		while (bit & mask) {
			if (node->present) {
				already_exists = true;
			}
			node->present = false;

			if (key & bit) {
				next = node->right;
			} else {
				next = node->left;
			}

			if (!next) {
				break;
			}

			bit >>= 1;
			node = next;
		}

		if (next) { // the tree nodes exist to the depth needed, set value and done
			node->value = value;
			node->present = true;
			return already_exists;
		}

		// tree nodes need insertion
		// there might be contention on node insertion with another insertion happening
		while (bit & mask) {
			next = new RadixTreeNode<T>;

			bool insert_right = key & bit;
			RadixTreeNode<T>* expected = nullptr;
			if (!std::atomic_compare_exchange_strong(insert_right ? &node->right : &node->left, &expected, next)) {
				// other thread won -> lets just retry one level deeper
				delete next;
				next = insert_right ? node->right : node->left;
			}

			bit >>= 1;
			if (node->present) {
				already_exists = true;
			}
			node->present = false;
			node = next;
		}
		// no contention possible here
		node->value = value;
		node->present = true;
		// at this point there is a value present here, so readers cannot descend below this node
		// TODO: clean up any nodes below
		return already_exists;
	}

	template<class T>
	template<class F, F f, class... Args>
	bool RadixTree<T>::handle_unaligned(size_t base, size_t size, Args... values) {
		bool value = false;
		// fmt::println("unaligned: {}->{} {}", base, base + size - 1, size);
		auto p2size = previous_pow2(size);
		// move the beginning of the allocation up
		auto start_up = align_up(base, p2size);

		if (start_up > base) {
			bool r = handle_unaligned<F, f>(base, start_up - base, values...);
			value |= r;
		}
		size -= (start_up - base);
		p2size = previous_pow2(size);
		// move the end of the allocation down
		auto size_sliver = size - p2size;
		// if size is unaligned, align it and the unaligned bit send separately
		if (size_sliver > 0) {
			bool r = handle_unaligned<F, f>(start_up + size - size_sliver, size_sliver, values...);
			value |= r;
		}
		size = p2size;
		// middle part is now aligned
		if (size > 0) {
			bool r = (this->*f)(start_up, size, values...);
			value |= r;
		}

		return value;
	}

	template<class T>
	bool RadixTree<T>::insert_unaligned(size_t base, size_t size, T value) {
		return handle_unaligned<decltype(&RadixTree::insert), &RadixTree::insert>(base, size, value);
	}

	template<class T>
	bool RadixTree<T>::erase(uint64_t base, size_t size) {
		uint64_t bit = first_bit;
		RadixTreeNode<T>* node = reinterpret_cast<RadixTreeNode<T>*>(root);

		while (node) {
			if (node->present) {
				node->present = false;
				return false;
			}

			if (base & bit) {
				node = node->right;
			} else {
				node = node->left;
			}

			bit >>= 1;
		}

		return false;
	}

	template<class T>
	void RadixTree<T>::erase_unaligned(uint64_t base, size_t size) {
		handle_unaligned<decltype(&RadixTree::erase), &RadixTree::erase>(base, size);
	}

	template class RadixTree<int>;
	template class RadixTree<std::pair<size_t, size_t>>;
	template class RadixTree<bool>;
	template class RadixTree<AllocationEntry>;
	template class RadixTree<ViewEntry>;
} // namespace vuk