#include "vuk/partials/Scatter.hpp"
#include "TestContext.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Partials.hpp"
#include <doctest/doctest.h>
#include <numeric>

using namespace vuk;

TEST_CASE("test scatter") {
	REQUIRE(test_context.prepare());
	if (test_context.rdoc_api)
		test_context.rdoc_api->StartFrameCapture(NULL, NULL);
	// src data
	std::vector<uint32_t> data(1024);
	std::iota(data.begin(), data.end(), 1u);
	std::vector<uint32_t> indirection(data.size());
	std::iota(indirection.begin(), indirection.end(), 0u);
	std::reverse(indirection.begin(), indirection.end());

	std::vector<uint32_t> expected(data.size());
	// cpu result
	for (size_t i = 0; i < data.size(); i++) {
		expected[i] = data[indirection[i]];
	}

	// put data on gpu
	auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
	auto [_2, indir] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(indirection));
	// put count on gpu
	CountWithIndirect count_data{ (uint32_t)data.size(), 512 };
	auto [_3, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

	// apply function on gpu
	auto calc = scatter<uint32_t>(*test_context.context, src, {}, indir, cnt);
	// bring data back to cpu
	auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
	auto out = std::span((uint32_t*)res->mapped_ptr, data.size());
	if (test_context.rdoc_api)
		test_context.rdoc_api->EndFrameCapture(NULL, NULL);
	CHECK(out == std::span(expected));
}