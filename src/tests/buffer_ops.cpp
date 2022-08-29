#include "TestContext.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Partials.hpp"
#include <doctest/doctest.h>

using namespace vuk;

TEST_CASE("test text_context preparation") {
	REQUIRE(test_context.prepare());
}

constexpr bool operator==(const std::span<uint32_t>& lhs, const std::span<uint32_t>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<uint32_t>& lhs, const std::span<const uint32_t>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<const uint32_t>& lhs, const std::span<const uint32_t>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

TEST_CASE("test buffer harness") {
	REQUIRE(test_context.prepare());
	auto data = { 1u, 2u, 3u };
	auto [buf, fut] = create_buffer_cross_device(*test_context.allocator, MemoryUsage::eCPUtoGPU, std::span(data));
	auto res = fut.get<Buffer>(*test_context.allocator, test_context.compiler);
	CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(data));
}

TEST_CASE("test buffer upload/download") {
	REQUIRE(test_context.prepare());
	{
		auto data = { 1u, 2u, 3u };
		auto [buf, fut] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));

		auto res = download_buffer(fut).get<Buffer>(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(data));
	}
	{
		auto data = { 1u, 2u, 3u, 4u, 5u };
		auto [buf, fut] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));

		auto res = download_buffer(fut).get<Buffer>(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 5) == std::span(data));
	}
}