#include "vuk/partials/Compact.hpp"
#include "TestContext.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Partials.hpp"
#include <doctest/doctest.h>
#include <numeric>

using namespace vuk;

TEST_CASE("test compact") {
	SUBCASE("uint compact") {
		REQUIRE(test_context.prepare());
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector<uint32_t> data(1024);
		std::iota(data.begin(), data.end(), 1u);

		auto func = [](auto A) {
			return spirv::select(A < 30u, 1u, 0u); // TODO: implement cast for this
		};

		std::vector<uint32_t> expected = data;
		// cpu result
		expected.erase(std::remove_if(expected.begin(), expected.end(), [&](auto& p) { return !func(p); }),
		               expected.end()); // remove_if requires the opposite predicate

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 512 };
		auto [_2, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = compact<uint32_t>(*test_context.context, src, {}, cnt, (uint32_t)data.size(), func);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((uint32_t*)res->mapped_ptr, expected.size());
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
struct spirv::Type<POD> : spirv::TypeStruct<spirv::Member<Type<uint32_t>>, spirv::Member<Type<float>>> {
	using type = POD;
};

template<class Ctx>
struct spirv::TypeContext<Ctx, spirv::Type<POD>> {
	uint32_t cnt = 0;
	SpvExpression<CompositeExtract<Type<uint32_t>, Ctx, Id>> foo = { static_cast<Ctx&>(*this), cnt++ };
	SpvExpression<CompositeExtract<Type<float>, Ctx, Id>> bar = { static_cast<Ctx&>(*this), cnt++ };
};

TEST_CASE("struct compact") {
	REQUIRE(test_context.prepare());
	if (test_context.rdoc_api)
		test_context.rdoc_api->StartFrameCapture(NULL, NULL);
	// src data
	std::vector<POD> data = { POD{ 20, 2.f }, POD{ 30, 3.f }, POD{ 40, 4.f } };

	auto func = [](auto A) {
		return spirv::select(A.foo < 30u, 1u, 0u); // TODO: implement cast for this
	};

	std::vector<POD> expected = data;
	// cpu result
	expected.erase(std::remove_if(expected.begin(), expected.end(), [&](auto& p) { return !func(p); }),
	               expected.end()); // remove_if requires the opposite predicate

	// put data on gpu
	auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
	// put count on gpu
	CountWithIndirect count_data{ (uint32_t)data.size(), 512 };
	auto [_2, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

	// apply function on gpu
	auto calc = compact<POD>(*test_context.context, src, {}, cnt, (uint32_t)data.size(), func);
	// bring data back to cpu
	auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
	auto out = std::span((POD*)res->mapped_ptr, expected.size());
	if (test_context.rdoc_api)
		test_context.rdoc_api->EndFrameCapture(NULL, NULL);
	CHECK(out == std::span(expected));
}