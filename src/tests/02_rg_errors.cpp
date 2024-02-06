#include "TestContext.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Partials.hpp"
#include <doctest/doctest.h>

using namespace vuk;

TEST_CASE("error: can't construct incomplete") {
	auto data = { 1u, 2u, 3u };
	auto [b0, buf0] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));
	auto buf1 = declare_buf("b1");
	buf1->memory_usage = MemoryUsage::eGPUonly;
	buf1.same_size(buf0);
	auto buf2 = declare_buf("b2");
	buf2->memory_usage = MemoryUsage::eGPUonly;
	buf2.same_size(buf1);
	auto buf3 = declare_buf("b3");
	buf3->memory_usage = MemoryUsage::eGPUonly;

	auto copy = make_pass("cpy", [](CommandBuffer& cbuf, VUK_BA(Access::eTransferRead) src, VUK_BA(Access::eTransferWrite) dst) {
		cbuf.copy_buffer(src, dst);
		return dst;
	});

	REQUIRE_THROWS(download_buffer(copy(std::move(buf0), std::move(buf3))).get(*test_context.allocator, test_context.compiler));
}