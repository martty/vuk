#pragma once

namespace vuk {
	// generic iterator for both const_iterator and iterator.
	template<class Key, class Value>
	class ConstMapIterator {
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = std::pair<const Key&, Value>;
		using reference = value_type;
		using iterator_category = std::forward_iterator_tag;

		ConstMapIterator(void* iter) : _iter(iter) {}
		ConstMapIterator(ConstMapIterator const&) noexcept;
		ConstMapIterator& operator=(ConstMapIterator const& other) noexcept {
			ConstMapIterator tmp(other);
			swap(tmp);
			return *this;
		}

		~ConstMapIterator();

		void swap(ConstMapIterator& other) noexcept {
			using std::swap;
			swap(_iter, other._iter);
		}

		ConstMapIterator& operator++() noexcept;

		ConstMapIterator operator++(int) const noexcept {
			ConstMapIterator tmp = *this;
			++(*this);
			return tmp;
		}

		reference operator*() noexcept;

		bool operator==(ConstMapIterator<Key, Value> const& o) const noexcept;
		bool operator!=(ConstMapIterator<Key, Value> const& o) const noexcept {
			return !(*this == o);
		}

	private:
		void* _iter;
	};

	template<class Key, class Value>
	struct MapProxy {
		using const_iterator = ConstMapIterator<Key, Value>;

		MapProxy(void* map) : _map(map) {}

		const_iterator begin() const noexcept {
			return cbegin();
		}
		const_iterator end() const noexcept {
			return cend();
		}

		const_iterator cbegin() const noexcept;
		const_iterator cend() const noexcept;

		size_t size() const noexcept;
		const_iterator find(Key) const noexcept;

	private:
		void* _map;
	};
} // namespace vuk