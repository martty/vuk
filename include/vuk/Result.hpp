#pragma once

// based on https://github.com/kociap/anton_core/blob/master/public/anton/expected.hpp

#include "vuk/Config.hpp"
#include "vuk/Exception.hpp"
#include "vuk/vuk_fwd.hpp"

#include <cassert>
#include <type_traits>
#include <utility>
#include <memory>

#define FWD(x)               (static_cast<decltype(x)&&>(x))
#define MOV(x)               (static_cast<std::remove_reference_t<decltype(x)>&&>(x))
#define CONSTRUCT_AT(p, ...) ::new (const_cast<void*>(static_cast<const volatile void*>(p))) decltype (*p)(__VA_ARGS__)

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
	}

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
		Result(ResultValueTag, Args&&... args) : _holds_value(true), _value{ FWD(args)... } {}

		template<class U>
		Result(ResultErrorTag, U&& err_t) : _holds_value(false) {
			using V = std::remove_reference_t<U>;
			static_assert(std::is_base_of_v<E, V>, "Concrete error must derive from E");
			_error = new V(MOV(err_t));
		}

		Result(Result const& other) = delete;

		template<class U, class F>
		Result(Result<U, F>&& other) : _holds_value(other._holds_value), _null_state() {
			static_assert(std::is_convertible_v<F*, E*>, "error must be convertible");
			if constexpr (!std::is_same_v<U, T>) {
				assert(!other._holds_value);
			} else {
				if (other._holds_value) {
					CONSTRUCT_AT(&_value, MOV(other._value));
				}
			}
			
			if (!other._holds_value) {
				_error = MOV(other._error);
				other._error = nullptr;
			}
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
					CONSTRUCT_AT(&_error, MOV(other._error));
					_holds_value = false;
				}
			} else {
				if (other._holds_value) {
					delete _error;
					CONSTRUCT_AT(&_value, MOV(other._value));
					_holds_value = true;
				} else {
					_error = MOV(other._error);
					other._error = nullptr;
				}
			}

			return *this;
		}

		~Result() noexcept(false) {
			if (_holds_value) {
				_value.~T();
			} else {
				if (!_extracted) {
					if constexpr (vuk::use_exceptions) {
						_error->throw_this();
					} else {
						std::abort();
					}
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
					CONSTRUCT_AT(&rhs._value, MOV(lhs._value));
					detail::destroy_at(&lhs._value);
					CONSTRUCT_AT(&lhs._error, MOV(rhs._error));
					swap(lhs._holds_value, rhs._holds_value);
				}
			} else {
				if (rhs._holds_value) {
					CONSTRUCT_AT(&rhs._error, MOV(lhs._error));
					CONSTRUCT_AT(&lhs._value, MOV(rhs._value));
					detail::destroy_at(&rhs._value);
					swap(lhs._holds_value, rhs._holds_value);
				} else {
					swap(lhs._error, rhs._error);
				}
			}
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
			_error = new V(MOV(err_t));
		}

		Result(Result const& other) = delete;

		template<class U, class F>
		Result(Result<U, F>&& other) : _holds_value(other._holds_value), _null_state() {
			static_assert(std::is_convertible_v<F*, E*>, "error must be convertible");
			if (!other._holds_value) {
				_error = other._error;
				other._error = nullptr;
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
					delete other._error;
					_holds_value = false;
				}
			} else {
				if (other._holds_value) {
					delete _error;
					_holds_value = true;
				} else {
					_error = MOV(other._error);
				}
			}

			return *this;
		}

		~Result() noexcept(false) {
			if (!_holds_value) {
				if (!_extracted) {
					if constexpr (vuk::use_exceptions) {
						_error->throw_this();
					} else {
						std::abort();
					}
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
					CONSTRUCT_AT(&lhs._error, MOV(rhs._error));
					delete rhs._error;
					swap(lhs._holds_value, rhs._holds_value);
				}
			} else {
				if (rhs._holds_value) {
					CONSTRUCT_AT(&rhs._error, MOV(lhs._error));
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

#undef CONSTRUCT_AT
#undef FWD
#undef MOV