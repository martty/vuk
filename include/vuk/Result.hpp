#pragma once

// based on https://github.com/kociap/anton_core/blob/master/public/anton/expected.hpp

#include "vuk/Config.hpp"
#include "vuk/Exception.hpp"
#include "vuk/vuk_fwd.hpp"

#include <cassert>
#include <type_traits>
#include <utility>

#define FWD(x)               (static_cast<decltype(x)&&>(x))
#define MOV(x)               (static_cast<std::remove_reference_t<decltype(x)>&&>(x))
#define CONSTRUCT_AT(p, ...) ::new (const_cast<void*>(static_cast<const volatile void*>(p))) decltype (*p)(__VA_ARGS__)
#define DESTROY_AT(p)                                                                                                                                          \
	do {                                                                                                                                                         \
		using U = decltype(*p);                                                                                                                                    \
		if constexpr (std::is_array_v<U>) {                                                                                                                        \
			for (auto& elem : *p) {                                                                                                                                  \
				(&elem)->~U();                                                                                                                                         \
			}                                                                                                                                                        \
		} else {                                                                                                                                                   \
			p->~U();                                                                                                                                                 \
		}                                                                                                                                                          \
	} while (0)

namespace vuk {
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

		template<typename... Args>
		Result(ResultErrorTag, Args&&... args) : _holds_value(false), _error{ FWD(args)... } {}

		Result(Result const& other) : _holds_value(other._holds_value), _null_state() {
			if (other._holds_value) {
				CONSTRUCT_AT(&_value, other._value);
			} else {
				CONSTRUCT_AT(&_error, other._error);
			}
		}

		Result(Result&& other) : _holds_value(other._holds_value), _null_state() {
			if (other._holds_value) {
				CONSTRUCT_AT(&_value, MOV(other._value));
			} else {
				CONSTRUCT_AT(&_error, MOV(other._error));
			}
		}

		Result& operator=(Result const& other) {
			if (_holds_value) {
				if (other._holds_value) {
					_value = other._value;
				} else {
					DESTROY_AT(&_value);
					CONSTRUCT_AT(&_error, other._error);
					_holds_value = false;
				}
			} else {
				if (other._holds_value) {
					DESTROY_AT(&_error);
					CONSTRUCT_AT(&_value, other._value);
					_holds_value = true;
				} else {
					_error = other._error;
				}
			}

			return *this;
		}

		Result& operator=(Result&& other) {
			if (_holds_value) {
				if (other._holds_value) {
					_value = MOV(other._value);
				} else {
					DESTROY_AT(&_value);
					CONSTRUCT_AT(&_error, MOV(other._error));
					_holds_value = false;
				}
			} else {
				if (other._holds_value) {
					DESTROY_AT(&_error);
					CONSTRUCT_AT(&_value, MOV(other._value));
					_holds_value = true;
				} else {
					_error = MOV(other._error);
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
						throw _error;
					} else {
						std::abort();
					}
				}
				_error.~E();
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
			_extracted = true;
			assert(!_holds_value && "cannot call error() on Result that does not hold an error");
			return _error;
		}

		[[nodiscard]] E const& error() const& {
			_extracted = true;
			assert(!_holds_value && "cannot call error() on Result that does not hold an error");
			return _error;
		}

		[[nodiscard]] E&& error() && {
			_extracted = true;
			assert(!_holds_value && "cannot call error() on Result that does not hold an error");
			return MOV(_error);
		}

		friend void swap(Result& lhs, Result& rhs) {
			using std::swap;
			if (lhs._holds_value) {
				if (rhs._holds_value) {
					swap(lhs._value, rhs._value);
				} else {
					CONSTRUCT_AT(&rhs._value, MOV(lhs._value));
					DESTROY_AT(&lhs._value);
					CONSTRUCT_AT(&lhs._error, MOV(rhs._error));
					DESTROY_AT(&rhs._error);
					swap(lhs._holds_value, rhs._holds_value);
				}
			} else {
				if (rhs._holds_value) {
					CONSTRUCT_AT(&rhs._error, MOV(lhs._error));
					DESTROY_AT(&lhs._error);
					CONSTRUCT_AT(&lhs._value, MOV(rhs._value));
					DESTROY_AT(&rhs._value);
					swap(lhs._holds_value, rhs._holds_value);
				} else {
					swap(lhs._error, rhs._error);
				}
			}
		}

	private:
		bool _holds_value;
		mutable bool _extracted = false;
		union {
			T _value;
			E _error;
			bool _null_state;
		};
	};

	template<typename E>
	struct Result<void, E> {
	public:
		using value_type = void;
		using error_type = E;

		Result(ResultValueTag) : _holds_value(true) {}

		template<typename... Args>
		Result(ResultErrorTag, Args&&... args) : _holds_value(false), _error{ FWD(args)... } {}

		Result(Result const& other) : _holds_value(other._holds_value), _null_state() {
			if (!other._holds_value) {
				CONSTRUCT_AT(&_error, other._error);
			}
		}

		Result(Result&& other) : _holds_value(other._holds_value), _null_state() {
			if (!other._holds_value) {
				CONSTRUCT_AT(&_error, MOV(other._error));
			}
		}

		Result& operator=(Result const& other) {
			if (_holds_value) {
				if (!other._holds_value) {
					CONSTRUCT_AT(&_error, other._error);
					DESTROY_AT(&other._error);
					_holds_value = false;
				}
			} else {
				if (other._holds_value) {
					DESTROY_AT(&_error);
					_holds_value = true;
				} else {
					_error = other._error;
				}
			}

			return *this;
		}

		Result& operator=(Result&& other) {
			if (_holds_value) {
				if (!other._holds_value) {
					CONSTRUCT_AT(&_error, MOV(other._error));
					DESTROY_AT(&other._error);
					_holds_value = false;
				}
			} else {
				if (other._holds_value) {
					DESTROY_AT(&_error);
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
						throw MOV(_error);
					} else {
						std::abort();
					}
				}
				_error.~E();
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
			return _error;
		}

		[[nodiscard]] E const& error() const& {
			assert(!_holds_value && "cannot call error() on Result that does not hold an error");
			_extracted = true;
			return _error;
		}

		[[nodiscard]] E&& error() && {
			assert(!_holds_value && "cannot call error() on Result that does not hold an error");
			_extracted = true;
			return MOV(_error);
		}

		friend void swap(Result& lhs, Result& rhs) {
			using std::swap;
			if (lhs._holds_value) {
				if (!rhs._holds_value) {
					CONSTRUCT_AT(&lhs._error, MOV(rhs._error));
					DESTROY_AT(&rhs._error);
					swap(lhs._holds_value, rhs._holds_value);
				}
			} else {
				if (rhs._holds_value) {
					CONSTRUCT_AT(&rhs._error, MOV(lhs._error));
					DESTROY_AT(&lhs._error);
					swap(lhs._holds_value, rhs._holds_value);
				}
			}
		}

	private:
		bool _holds_value;
		mutable bool _extracted = false;
		union {
			E _error;
			bool _null_state;
		};
	};
} // namespace vuk

#undef DESTROY_AT
#undef CONSTRUCT_AT
#undef FWD
#undef MOV