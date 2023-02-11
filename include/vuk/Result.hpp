#pragma once

// based on https://github.com/kociap/anton_core/blob/master/public/anton/expected.hpp

#include "vuk/Config.hpp"
#include "vuk/Exception.hpp"
#include "vuk/vuk_fwd.hpp"

#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

#define FWD(x)               (static_cast<decltype(x)&&>(x))
#define MOV(x)               (static_cast<std::remove_reference_t<decltype(x)>&&>(x))

namespace vuk {
	namespace detail {
		template<class U>
		void destroy_at(U* p) {
			if constexpr (std::is_array_v<U>) {
				for (auto& elem : *p) {
					(&elem)->~U();
				}
			} else {
				p->~U();
			}
		}
	} // namespace detail

	struct Exception;
	struct ResultErrorTag {
		explicit ResultErrorTag() = default;
	};

	constexpr ResultErrorTag expected_error;

	struct ResultValueTag {
		explicit ResultValueTag() = default;
	};

	constexpr ResultValueTag expected_value;

	template<typename T, typename E>
	struct Result {
	public:
		using value_type = T;
		using error_type = E;

		template<typename... Args>
		Result(ResultValueTag, Args&&... args) : _value{ FWD(args)... }, _holds_value(true) {}

		template<class U>
		Result(ResultErrorTag, U&& err_t) : _holds_value(false) {
			using V = std::remove_reference_t<U>;
			static_assert(std::is_base_of_v<E, V>, "Concrete error must derive from E");
#if VUK_FAIL_FAST
			fprintf(stderr, "%s", err_t.what());
			assert(0);
#endif
			_error = new V(MOV(err_t));
		}

		Result(Result const& other) = delete;

		template<class U, class F>
		Result(Result<U, F>&& other) : _null_state(), _holds_value(other._holds_value) {
			static_assert(std::is_convertible_v<F*, E*>, "error must be convertible");
			if constexpr (!std::is_same_v<U, T>) {
				assert(!other._holds_value);
			} else {
				if (other._holds_value) {
					std::construct_at(&_value, MOV(other._value));
				}
			}

			if (!other._holds_value) {
				_error = MOV(other._error);
				other._error = nullptr;
			}

			_extracted = other._extracted;
		}

		Result& operator=(Result const& other) = delete;

		template<class U, class F>
		Result& operator=(Result<U, F>&& other) {
			static_assert(std::is_convertible_v<F*, E*>, "error must be convertible");
			if constexpr (!std::is_same_v<U, T>) {
				assert(!other.holds_value);
			}
			if (_holds_value) {
				if (other._holds_value) {
					_value = MOV(other._value);
				} else {
					detail::destroy_at(&_value);
					std::construct_at(&_error, MOV(other._error));
					other._error = nullptr;
					_holds_value = false;
				}
			} else {
				delete _error;
				if (other._holds_value) {
					std::construct_at(&_value, MOV(other._value));
					_holds_value = true;
				} else {
					_error = MOV(other._error);
					other._error = nullptr;
				}
			}
			_extracted = other._extracted;
			return *this;
		}

		~Result() noexcept(false) {
			if (_holds_value) {
				_value.~T();
			} else {
				if (!_extracted && _error) {
#if VUK_USE_EXCEPTIONS
					_error->throw_this();
#else
					std::abort();
#endif
				}
				delete _error;
			}
		}

		[[nodiscard]] operator bool() const {
			return _holds_value;
		}

		[[nodiscard]] bool holds_value() const {
			return _holds_value;
		}

		[[nodiscard]] T* operator->() {
			assert(_holds_value && "cannot call operator-> on Result that does not hold a value");
			return &_value;
		}

		[[nodiscard]] T const* operator->() const {
			assert(_holds_value && "cannot call operator-> on Result that does not hold a value");
			return &_value;
		}

		[[nodiscard]] T& operator*() & {
			assert(_holds_value && "cannot call operator* on Result that does not hold a value");
			return _value;
		}

		[[nodiscard]] T const& operator*() const& {
			assert(_holds_value && "cannot call operator* on Result that does not hold a value");
			return _value;
		}

		[[nodiscard]] T&& operator*() && {
			assert(_holds_value && "cannot call operator* on Result that does not hold a value");
			return MOV(_value);
		}

		[[nodiscard]] T const&& operator*() const&& {
			assert(_holds_value && "cannot call operator* on Result that does not hold a value");
			return _value;
		}

		[[nodiscard]] T& value() & {
			assert(_holds_value && "cannot call value() on Result that does not hold a value");
			return _value;
		}

		[[nodiscard]] T const& value() const& {
			assert(_holds_value && "cannot call value() on Result that does not hold a value");
			return _value;
		}

		[[nodiscard]] T&& value() && {
			assert(_holds_value && "cannot call value() on Result that does not hold a value");
			return MOV(_value);
		}

		[[nodiscard]] E& error() & {
			assert(!_holds_value && "cannot call error() on Result that does not hold an error");
			_extracted = true;
			return *_error;
		}

		[[nodiscard]] E const& error() const& {
			assert(!_holds_value && "cannot call error() on Result that does not hold an error");
			_extracted = true;
			return *_error;
		}

		friend void swap(Result& lhs, Result& rhs) {
			using std::swap;
			if (lhs._holds_value) {
				if (rhs._holds_value) {
					swap(lhs._value, rhs._value);
				} else {
					std::construct_at(&rhs._value, MOV(lhs._value));
					detail::destroy_at(&lhs._value);
					std::construct_at(&lhs._error, MOV(rhs._error));
					swap(lhs._holds_value, rhs._holds_value);
				}
			} else {
				if (rhs._holds_value) {
					std::construct_at(&rhs._error, MOV(lhs._error));
					std::construct_at(&lhs._value, MOV(rhs._value));
					detail::destroy_at(&rhs._value);
					swap(lhs._holds_value, rhs._holds_value);
				} else {
					swap(lhs._error, rhs._error);
				}
			}
			swap(lhs._extracted, rhs._extracted);
		}

	private:
		union {
			T _value;
			E* _error;
			bool _null_state;
		};
		bool _holds_value;
		mutable bool _extracted = false;

		template<class U, class B>
		friend struct Result;
	};

	template<typename E>
	struct Result<void, E> {
	public:
		using value_type = void;
		using error_type = E;

		Result(ResultValueTag) : _holds_value(true) {}

		template<class U>
		Result(ResultErrorTag, U&& err_t) : _holds_value(false) {
			using V = std::remove_reference_t<U>;
			static_assert(std::is_base_of_v<E, V>, "Concrete error must derive from E");
#if VUK_FAIL_FAST
			fprintf(stderr, "%s", err_t.what());
			assert(0);
#endif
			_error = new V(MOV(err_t));
		}

		Result(Result const& other) = delete;

		template<class U, class F>
		Result(Result<U, F>&& other) : _null_state(), _holds_value(other._holds_value) {
			static_assert(std::is_convertible_v<F*, E*>, "error must be convertible");
			if (!other._holds_value) {
				_error = other._error;
				other._error = nullptr;
				_extracted = other._extracted;
			}
		}

		Result& operator=(Result const& other) = delete;

		template<class U, class F>
		Result& operator=(Result<U, F>&& other) {
			static_assert(std::is_convertible_v<F*, E*>, "error must be convertible");
			if constexpr (!std::is_same_v<U, value_type>) {
				assert(!other._holds_value);
			}
			if (_holds_value) {
				if (!other._holds_value) {
					_error = MOV(other._error);
					other._error = nullptr;
					_holds_value = false;
				}
			} else {
				delete _error;
				if (other._holds_value) {
					_holds_value = true;
				} else {
					_error = MOV(other._error);
					other._error = nullptr;
				}
			}
			_extracted = other._extracted;

			return *this;
		}

		~Result() noexcept(false) {
			if (!_holds_value) {
				if (!_extracted && _error) {
#if VUK_USE_EXCEPTIONS
					_error->throw_this();
#else
					std::abort();
#endif
				}
				delete _error;
			}
		}

		[[nodiscard]] operator bool() const {
			return _holds_value;
		}

		[[nodiscard]] bool holds_value() const {
			return _holds_value;
		}

		[[nodiscard]] E& error() & {
			assert(!_holds_value && "cannot call error() on Result that does not hold an error");
			_extracted = true;
			return *_error;
		}

		[[nodiscard]] E const& error() const& {
			assert(!_holds_value && "cannot call error() on Result that does not hold an error");
			_extracted = true;
			return *_error;
		}

		friend void swap(Result& lhs, Result& rhs) {
			using std::swap;
			if (lhs._holds_value) {
				if (!rhs._holds_value) {
					std::construct_at(&lhs._error, MOV(rhs._error));
					delete rhs._error;
					swap(lhs._holds_value, rhs._holds_value);
				}
			} else {
				if (rhs._holds_value) {
					std::construct_at(&rhs._error, MOV(lhs._error));
					swap(lhs._holds_value, rhs._holds_value);
				}
			}
		}

	private:
		union {
			E* _error;
			bool _null_state;
		};
		bool _holds_value;
		mutable bool _extracted = false;

		template<class U, class B>
		friend struct Result;
	};
} // namespace vuk

#undef FWD
#undef MOV