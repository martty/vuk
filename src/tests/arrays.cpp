#include "TestContext.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Partials.hpp"
#include <doctest/doctest.h>

using namespace vuk;

TEST_CASE("arrayed buffers") {
	{
		auto data = { 0xfeu, 0xfeu, 0xfeu, 0xfeu };
		auto data2 = { 0xfdu, 0xfdu, 0xfdu, 0xfdu };
		auto buf = allocate_buffer(*test_context.allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(uint32_t) * 4 });
		auto buf2 = allocate_buffer(*test_context.allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(uint32_t) * 4 });

		auto fill = make_pass("fill two", [](CommandBuffer& cbuf, VUK_ARG(Buffer[], Access::eTransferWrite) dst) {
			cbuf.fill_buffer(dst[0], 0xfe);
			cbuf.fill_buffer(dst[1], 0xfd);
			return dst;
		});

		auto arr = declare_array("buffers", declare_buf("src", **buf), declare_buf("src2", **buf2));
		TypedFuture<Buffer[]> filled_bufs = fill(arr);
		auto res = download_buffer(filled_bufs[0]).get(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(data));
		res = download_buffer(filled_bufs[1]).get(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(data2));
	}
}


TEST_CASE("arrayed buffers, internal loop") {
	{
		auto data = { 0xfeu, 0xfeu, 0xfeu, 0xfeu };
		auto data2 = { 0xfdu, 0xfdu, 0xfdu, 0xfdu };
		auto buf = allocate_buffer(*test_context.allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(uint32_t) * 4 });
		auto buf2 = allocate_buffer(*test_context.allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(uint32_t) * 4 });

		auto fill = make_pass("fill two", [](CommandBuffer& cbuf, VUK_ARG(Buffer[], Access::eTransferWrite) dst) {
			for (size_t i = 0; i < dst.size(); i++) {
				cbuf.fill_buffer(dst[i], 0xfe - i);
			}
			return dst;
		});

		auto arr = declare_array("buffers", declare_buf("src", **buf), declare_buf("src2", **buf2));
		TypedFuture<Buffer[]> filled_bufs = fill(arr);
		auto res = download_buffer(filled_bufs[0]).get(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(data));
		res = download_buffer(filled_bufs[1]).get(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(data2));
	}
}