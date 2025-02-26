#pragma once

#include <algorithm>
#include <iterator>
#include <memory>
#include <string.h>
#include <type_traits>
#include <utility>

namespace vuk {
	// https://gist.github.com/ThePhD/8153067
	template<typename T, std::size_t n, std::size_t a = std::alignment_of<T>::value>
	class fixed_vector {
	public:
		typedef T value_type;
		typedef T& reference;
		typedef const T& const_reference;
		typedef T* pointer_type;
		typedef std::size_t size_type;
		typedef std::ptrdiff_t difference_type;
		typedef pointer_type iterator;
		typedef const pointer_type const_iterator;

	private:
		alignas(a) std::byte items[sizeof(T) * n];
		std::size_t len;

		constexpr T* ptrat(std::size_t idx) {
			return static_cast<T*>(static_cast<void*>(&items)) + idx;
		}

		constexpr const T* ptrat(std::size_t idx) const {
			return static_cast<const T*>(static_cast<const void*>(&items)) + idx;
		}

		constexpr T& refat(std::size_t idx) {
			return *ptrat(idx);
		}

		constexpr const T& refat(std::size_t idx) const {
			return *ptrat(idx);
		}

	public:
		constexpr static std::size_t max_size() {
			return n;
		}

		constexpr fixed_vector() : len(0) {
			memset(&items, 0, sizeof(T) * n);
		}

		constexpr fixed_vector(std::size_t capacity) : len(std::min(n, capacity)) {
			memset(&items, 0, sizeof(T) * n);
		}

		template<std::size_t c>
		constexpr fixed_vector(const T (&arr)[c]) : len(c) {
			memset(&items, 0, sizeof(T) * n);
			static_assert(c < n, "Array too large to initialize fixed_vector");
			std::copy(std::addressof(arr[0]), std::addressof(arr[c]), data());
		}

		constexpr fixed_vector(std::initializer_list<T> initializer) : len(std::min(n, initializer.size())) {
			memset(&items, 0, sizeof(T) * n);
			std::copy(initializer.begin(), initializer.begin() + len, data());
		}

		constexpr fixed_vector(const fixed_vector& o) {
			memset(&items, 0, sizeof(T) * n);
			std::uninitialized_copy(o.begin(), o.end(), begin());
			len = o.len;
		}

		constexpr fixed_vector& operator=(const fixed_vector& o) {
			auto existing = std::min(len, o.len);
			std::copy_n(o.begin(), existing, begin());
			std::uninitialized_copy(o.begin() + existing, o.end(), begin() + existing);
			resize(o.len);
			return *this;
		}

		constexpr fixed_vector(fixed_vector&& o) {
			memset(&items, 0, sizeof(T) * n);
			std::uninitialized_move(o.begin(), o.end(), begin());
			len = o.len;
			o.resize(0);
		}

		constexpr fixed_vector& operator=(fixed_vector&& o) {
			auto existing = std::min(len, o.len);
			std::copy_n(std::make_move_iterator(o.begin()), existing, begin());
			std::uninitialized_move(o.begin() + existing, o.end(), begin() + existing);
			resize(o.len);
			o.resize(0);
			return *this;
		}

		constexpr ~fixed_vector() {
			for (std::size_t i = 0; i < len; i++) {
				ptrat(i)->~T();
			}
		}

		constexpr bool empty() const {
			return len < 1;
		}

		constexpr bool not_empty() const {
			return len > 0;
		}

		constexpr bool full() const {
			return len >= n;
		}

		constexpr void push_back(const T& item) {
			new (ptrat(len++)) T(item);
		}

		constexpr void push_back(T&& item) {
			new (ptrat(len++)) T(std::move(item));
		}

		template<typename... Tn>
		constexpr T& emplace_back(Tn&&... argn) {
			return *(new (ptrat(len++)) T(std::forward<Tn>(argn)...));
		}

		constexpr void pop_back() {
			T& addr = refat(--len);
			addr.~T();
		}

		constexpr void clear() {
			for (; len > 0;) {
				pop_back();
			}
		}

		constexpr std::size_t size() const {
			return len;
		}

		constexpr std::size_t capacity() const {
			return n;
		}

		constexpr void resize(std::size_t sz) {
			auto old_len = len;
			while (len > sz)
				pop_back();
			if (old_len > len) {
				memset(reinterpret_cast<char*>(&items) + len * sizeof(T), 0, sizeof(T) * (old_len - len));
			}
			len = std::min(sz, n);
		}

		constexpr void resize(std::size_t sz, const value_type& value) {
			auto old_len = len;
			while (len > sz)
				pop_back();
			if (old_len > len) {
				memset(reinterpret_cast<char*>(&items) + len * sizeof(T), 0, sizeof(T) * (old_len - len));
			}

			len = std::min(sz, n);
			if (len > old_len) {
				std::uninitialized_fill(begin() + old_len, begin() + len, value);
			}
		}

		constexpr T* data() {
			return ptrat(0);
		}

		constexpr const T* data() const {
			return ptrat(0);
		}

		constexpr T& operator[](std::size_t idx) {
			return refat(idx);
		}

		constexpr const T& operator[](std::size_t idx) const {
			return refat(idx);
		}

		constexpr T& front() {
			return refat(0);
		}

		constexpr T& back() {
			return refat(len - 1);
		}

		constexpr const T& front() const {
			return refat(0);
		}

		constexpr const T& back() const {
			return refat(len - 1);
		}

		constexpr T* begin() {
			return data();
		}

		constexpr const T* cbegin() {
			return data();
		}

		constexpr const T* begin() const {
			return data();
		}

		constexpr const T* cbegin() const {
			return data();
		}

		constexpr T* end() {
			return data() + len;
		}

		constexpr const T* cend() {
			return data() + len;
		}

		constexpr const T* end() const {
			return data() + len;
		}

		constexpr const T* cend() const {
			return data() + len;
		}

		/*
		iterator insert(const_iterator pos, const T& value);
		iterator insert(const_iterator pos, T&& value);
		iterator insert( const_iterator pos, std::initializer_list<T> ilist );
		*/

		template<class InputIt>
		constexpr iterator insert(const_iterator pos, InputIt first, InputIt last) {
			auto clen = std::distance(first, last);
			if (clen == 0)
				return pos;
			std::copy(pos, pos + (len - std::distance(begin(), pos)), pos + clen);
			std::copy(first, last, pos);
			len += clen;
			return pos;
		}

		constexpr bool operator==(const fixed_vector& o) const noexcept {
			return std::equal(begin(), end(), o.begin(), o.end());
		}
	};
} // namespace vuk
