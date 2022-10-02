#include "vuk/partials/Compact.hpp"
#include "TestContext.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Partials.hpp"
#include <doctest/doctest.h>
#include <numeric>

using namespace vuk;

TEST_CASE("test compact") {
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
	auto calc = compact<uint32_t>(*test_context.context, src, {}, cnt, data.size(), func);
	// bring data back to cpu
	auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
	auto out = std::span((uint32_t*)res->mapped_ptr, expected.size());
	if (test_context.rdoc_api)
		test_context.rdoc_api->EndFrameCapture(NULL, NULL);
	CHECK(out == std::span(expected));
}