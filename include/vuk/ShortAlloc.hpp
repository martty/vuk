#pragma once
//http://howardhinnant.github.io/stack_alloc.html
//https://codereview.stackexchange.com/a/31575
// but modified to use a heap arena
#include <cassert>
#include <cstddef>
#include <new>

class arena {
	static const std::size_t alignment = 16;
	std::size_t size_;
	char* buf_;
	char* ptr_;

	std::size_t align_up(std::size_t n) noexcept {
		return (n + (alignment - 1)) & ~(alignment - 1);
	}

	bool pointer_in_buffer(char* p) noexcept {
		return buf_ <= p && p <= buf_ + size_;
	}

public:
	arena(std::size_t N) noexcept {
		buf_ = (char*) operator new[](N, (std::align_val_t{ alignment }));
		ptr_ = buf_;
		size_ = N;
	}
	~arena() {
		::operator delete[](buf_, std::align_val_t{ alignment });
		ptr_ = nullptr;
	}
	arena(const arena& o) {
		size_ = o.size_;
		buf_ = (char*) operator new[](size_, (std::align_val_t{ alignment }));
		ptr_ = buf_;
	}
	arena& operator=(const arena& o) {
		::operator delete[](buf_, std::align_val_t{ alignment });
		size_ = o.size_;
		buf_ = (char*) operator new[](size_, (std::align_val_t{ alignment }));
		ptr_ = buf_;
		return *this;
	};

	char* allocate(std::size_t n);
	void deallocate(char* p, std::size_t n) noexcept;

	std::size_t size() {
		return size_;
	}
	std::size_t used() const {
		return static_cast<std::size_t>(ptr_ - buf_);
	}
	void reset() {
		ptr_ = buf_;
	}
};

inline char* arena::allocate(std::size_t n) {
	assert(pointer_in_buffer(ptr_) && "short_alloc has outlived arena");
	n = align_up(n);
	if (buf_ + size_ - ptr_ >= (int64_t)n) {
		char* r = ptr_;
		ptr_ += n;
		return r;
	}
	return static_cast<char*>(::operator new(n));
}

inline void arena::deallocate(char* p, std::size_t n) noexcept {
	assert(pointer_in_buffer(ptr_) && "short_alloc has outlived arena");
	if (pointer_in_buffer(p)) {
		n = align_up(n);
		if (p + n == ptr_)
			ptr_ = p;
	} else
		::operator delete(p);
}

template<class T, std::size_t N>
class short_alloc {
	arena& a_;
public:
	typedef T value_type;

public:
	template<class _Up>
	struct rebind {
		typedef short_alloc<_Up, N> other;
	};


	short_alloc(arena& a) : a_(a) {}
	template<class U>
	short_alloc(const short_alloc<U, N>& a) noexcept : a_(a.a_) {}
	short_alloc(const short_alloc&) = default;
	short_alloc& operator=(const short_alloc&) = delete;

	T* allocate(std::size_t n) {
		return reinterpret_cast<T*>(a_.allocate(n * sizeof(T)));
	}
	void deallocate(T* p, std::size_t n) noexcept {
		a_.deallocate(reinterpret_cast<char*>(p), n * sizeof(T));
	}

	template<class T1, std::size_t N1, class U, std::size_t M>
	friend bool operator==(const short_alloc<T1, N1>& x, const short_alloc<U, M>& y) noexcept;

	template<class U, std::size_t M>
	friend class short_alloc;
};

template<class T, std::size_t N, class U, std::size_t M>
inline bool operator==(const short_alloc<T, N>& x, const short_alloc<U, M>& y) noexcept {
	return N == M && &x.a_ == &y.a_;
}

template<class T, std::size_t N, class U, std::size_t M>
inline bool operator!=(const short_alloc<T, N>& x, const short_alloc<U, M>& y) noexcept {
	return !(x == y);
}
