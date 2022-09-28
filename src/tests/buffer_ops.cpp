#include "TestContext.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Partials.hpp"
#include "vuk/partials/Map.hpp"
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

constexpr bool operator==(const std::span<float>& lhs, const std::span<float>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<float>& lhs, const std::span<const float>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<const float>& lhs, const std::span<const float>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

struct CountWithIndirect {
	CountWithIndirect(uint32_t count, uint32_t wg_size) : workgroup_count((uint32_t)idivceil(count, wg_size)), count(count) {}

	uint32_t workgroup_count;
	uint32_t yz[2] = { 1, 1 };
	uint32_t count;
};

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

TEST_CASE("test unary_map") {
	REQUIRE(test_context.prepare());
	{
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector data = { 1u, 2u, 3u };
		// function to apply
		auto func = [](auto A) {
			return A + 3u + 33u;
		};
		std::vector<uint32_t> expected;
		// cpu result
		std::transform(data.begin(), data.end(), std::back_inserter(expected), func);

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 64 };
		auto [_2, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = unary_map<uint32_t>(*test_context.context, src, {}, cnt, func);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((uint32_t*)res->mapped_ptr, data.size());
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
	{
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector data = { 1u, 2u, 3u };
		// function to apply
		auto func = [](auto A) {
			return spirv::select(A > 1u, 1u, 2u);
		};
		std::vector<uint32_t> expected;
		// cpu result
		std::transform(data.begin(), data.end(), std::back_inserter(expected), func);

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 64 };
		auto [_2, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = unary_map<uint32_t>(*test_context.context, src, {}, cnt, func);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((uint32_t*)res->mapped_ptr, data.size());
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
	{
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector data = { 1.f, 2.f, 3.f };
		// function to apply
		auto func = [](auto A) {
			return spirv::select(A > 1.f, 3.f + A, 4.f) * spirv::select(A >= 1.f, 3.f + A, -A);
		};
		std::vector<float> expected;
		// cpu result
		std::transform(data.begin(), data.end(), std::back_inserter(expected), func);

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 64 };
		auto [_2, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = unary_map<float>(*test_context.context, src, {}, cnt, func);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((float*)res->mapped_ptr, data.size());
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
}

TEST_CASE("test binary_map") {
	REQUIRE(test_context.prepare());
	{
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector data0 = { 1.f, 2.f, 3.f };
		std::vector data1 = { 1.f, 2.f, 3.f };
		// function to apply
		auto func = [](auto A, auto B) {
			return spirv::select(A > 1.f, 3.f + A, 4.f) * spirv::select(B >= 1.f, 3.f + B, -B);
		};
		std::vector<float> expected;
		// cpu result
		std::transform(data0.begin(), data0.end(), data1.begin(), std::back_inserter(expected), func);

		// put data on gpu
		auto [_1, src_a] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data0));
		auto [_2, src_b] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data1));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data0.size(), 64 };
		auto [_3, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = binary_map<float>(*test_context.context, src_a, src_b, {}, cnt, func);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((float*)res->mapped_ptr, data0.size());
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
}

struct POD {
	unsigned foo;
	float bar;

	bool operator==(const POD&) const = default;
};

constexpr bool operator==(const std::span<POD>& lhs, const std::span<POD>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<POD>& lhs, const std::span<const POD>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<const POD>& lhs, const std::span<const POD>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

template<>
struct spirv::Type<POD> : spirv::TypeStruct<spirv::Member<spirv::Type<uint32_t>, 0>, spirv::Member<spirv::Type<float>, sizeof(uint32_t)>> {
	using type = POD;
};

template<class Ctx>
struct spirv::TypeContext<Ctx, spirv::Type<POD>> {
	SpvExpression<CompositeExtract<Type<uint32_t>, Ctx, Id>> foo = { static_cast<Ctx&>(*this), 0u };
	SpvExpression<CompositeExtract<Type<float>, Ctx, Id>> bar = { static_cast<Ctx&>(*this), 1u };
};

TEST_CASE("test unary_map, custom type") {
	REQUIRE(test_context.prepare());
	{
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data

		std::vector data = { POD{ 1, 2.f }, POD{ 1, 3.f }, POD{ 1, 4.f } };
		// function to apply
		auto func = [](auto A) {
			return spirv::make<POD>(A.foo * 2u + spirv::cast<uint32_t>(A.bar), spirv::cast<float>(A.foo) + A.bar * 2.f);
		};
		std::vector<POD> expected;
		// cpu result
		std::transform(data.begin(), data.end(), std::back_inserter(expected), func);

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 64 };
		auto [_2, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = unary_map<POD>(*test_context.context, src, {}, cnt, func);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((POD*)res->mapped_ptr, data.size());
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
}