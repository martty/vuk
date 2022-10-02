#include "vuk/partials/Scan.hpp"
#include "TestContext.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Partials.hpp"
#include <doctest/doctest.h>
#include <numeric>

using namespace vuk;

TEST_CASE("test scan") {
	REQUIRE(test_context.prepare());
	SUBCASE("smaller than 1 WG scan") {
		// fill a buffer with 25 elements, but then only scan 15
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector<uint32_t> data(25);
		std::fill(data.begin(), data.end(), 1u);
		// function to apply
		auto func = [](auto A) {
			return spirv::select(A > 10u, A, 1u);
		};
		std::vector<uint32_t> expected, temp;
		std::transform(data.begin(), data.begin() + 15, std::back_inserter(temp), func);
		// cpu result
		std::exclusive_scan(temp.begin(), temp.end(), std::back_inserter(expected), 0);

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)15, 512 };
		auto [_2, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto [calc, cnt_p] = scan<uint32_t>(*test_context.context, src, {}, std::move(cnt), 15u, func);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((uint32_t*)res->mapped_ptr, 15);
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
	SUBCASE("2-level scan") {
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector<unsigned> data(512 * 512);
		std::fill(data.begin(), data.end(), 1u);
		// function to apply
		auto func = [](auto A) {
			return spirv::select(A > 513u, A, 1u);
		};
		std::vector<uint32_t> expected, temp;
		std::transform(data.begin(), data.end(), std::back_inserter(temp), func);
		// cpu result
		std::exclusive_scan(data.begin(), data.end(), std::back_inserter(expected), 0);

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 512 };
		auto [_2, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto [calc, cnt_p] = scan<uint32_t>(*test_context.context, src, {}, std::move(cnt), (uint32_t)data.size(), func);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((uint32_t*)res->mapped_ptr, data.size());
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
	SUBCASE("3-level scan") {
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector<unsigned> data(512 * 512 * 2);
		std::fill(data.begin(), data.end(), 1u);
		// function to apply
		auto func = [](auto A) {
			return spirv::select(A > 513u, A, 1u);
		};
		std::vector<uint32_t> expected, temp;
		std::transform(data.begin(), data.end(), std::back_inserter(temp), func);
		// cpu result
		std::exclusive_scan(data.begin(), data.end(), std::back_inserter(expected), 0);

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 512 };
		auto [_2, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto [calc, cnt_p] = scan<uint32_t>(*test_context.context, src, {}, std::move(cnt), (uint32_t)data.size(), func);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((uint32_t*)res->mapped_ptr, data.size());
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
	SUBCASE("3-level scan, float") {
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector<float> data(512 * 512 * 2);
		std::fill(data.begin(), data.end(), 1.f);
		// function to apply
		auto func = [](auto A) {
			return spirv::select(A > 513.f, A, 1.f);
		};
		std::vector<float> expected, temp;
		std::transform(data.begin(), data.end(), std::back_inserter(temp), func);
		// cpu result
		std::exclusive_scan(data.begin(), data.end(), std::back_inserter(expected), 0.f);

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 512 };
		auto [_2, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto [calc, cnt_p] = scan<float>(*test_context.context, src, {}, std::move(cnt), (uint32_t)data.size(), func);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((float*)res->mapped_ptr, data.size());
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
}