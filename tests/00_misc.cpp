#include <doctest/doctest.h>
#include <iostream>
#include <source_location>

#include "vuk/SourceLocation.hpp"

struct S {
	std::source_location location;

	S(const std::source_location& loc) : location(loc) {}

	constexpr int line() const noexcept {
		return location.line();
	}
	constexpr const char* function_name() const noexcept {
		return location.function_name();
	}
	constexpr const char* file_name() const noexcept {
		return location.file_name();
	}
};

void f(S loc = S{ std::source_location::current() }) {
	std::cout << loc.file_name() << std::endl;
	std::cout << loc.line() << std::endl;
	std::cout << loc.function_name() << std::endl;
}

void g(std::source_location loc = std::source_location::current()) {
	std::cout << loc.file_name() << std::endl;
	std::cout << loc.line() << std::endl;
	std::cout << loc.function_name() << std::endl;
}

void h(vuk::SourceLocationAtFrame s = VUK_HERE_AND_NOW()) {
	std::cout << s.location.file_name() << std::endl;
	std::cout << s.location.line() << std::endl;
	std::cout << s.location.function_name() << std::endl;
}

TEST_CASE("source location") {
	std::cout << "Custom type, With default parameter:" << std::endl;
	f();

	std::cout << "Custom type, with explicit parameter:" << std::endl;
	f(std::source_location::current());

	std::cout << "std::source_location, default parameter:" << std::endl;
	g();

	std::cout << "std::source_location, explicit parameter:" << std::endl;
	g(std::source_location::current());

	h();
}