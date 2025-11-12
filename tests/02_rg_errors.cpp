#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <doctest/doctest.h>

using namespace vuk;

#if VUK_FAIL_FAST
#error "can't run these on FAIL_FAST"
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-result"
#pragma warning(push)
#pragma warning(disable : 4834)

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

	auto copy = make_pass("cpy", [](CommandBuffer& cbuf, VUK_BA(Access::eTransferWrite) src, VUK_BA(Access::eTransferWrite) dst) {
		cbuf.copy_buffer(src, dst);
		return dst;
	});

	REQUIRE_THROWS(download_buffer(copy(std::move(buf0), std::move(buf3))).get(*test_context.allocator, test_context.compiler));
}

static auto image2buf = make_pass("copy image to buffer", [](CommandBuffer& cbuf, VUK_IA(Access::eTransferRead) src, VUK_BA(Access::eTransferWrite) dst) {
	BufferImageCopy bc;
	bc.imageOffset = { 0, 0, 0 };
	bc.bufferRowLength = 0;
	bc.bufferImageHeight = 0;
	bc.imageExtent = src->base_mip_extent();
	bc.imageSubresource.aspectMask = format_to_aspect(src->format);
	bc.imageSubresource.mipLevel = src->base_level;
	bc.imageSubresource.baseArrayLayer = src->base_layer;
	assert(src->layer_count == 1); // unsupported yet
	bc.imageSubresource.layerCount = src->layer_count;
	bc.bufferOffset = dst->offset;
	cbuf.copy_image_to_buffer(src, dst, bc);
	return dst;
});

static auto blit_down = make_pass("blit down", [](CommandBuffer& cbuf, VUK_IA(Access::eTransferRead | Access::eTransferWrite) img) {
	ImageBlit region = {};
	region.srcOffsets[0] = Offset3D{};
	region.srcOffsets[1] = Offset3D{ 2, 2, 1 };
	region.dstOffsets[0] = Offset3D{};
	region.dstOffsets[1] = Offset3D{ 1, 1, 1 };
	region.srcSubresource.aspectMask = format_to_aspect(img->format);
	region.srcSubresource.baseArrayLayer = 0;
	region.srcSubresource.layerCount = 1;
	region.srcSubresource.mipLevel = 0;

	region.dstSubresource.baseArrayLayer = 0;
	region.dstSubresource.layerCount = 1;
	region.dstSubresource.mipLevel = 1;

	region.dstSubresource.aspectMask = format_to_aspect(img->format);

	cbuf.blit_image(img, img, region, vuk::Filter::eNearest);
	return img;
});

TEST_CASE("error: read without write") {
	{
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, 100, 1 });
		auto buf = vuk::discard_buf("a", *dst);

		auto rd_buf = vuk::make_pass("rd", [](CommandBuffer&, VUK_BA(vuk::eTransferRead) buf) { return buf; });

		REQUIRE_THROWS(rd_buf(buf).get(*test_context.allocator, test_context.compiler)); // report an error also if the splice remains
		REQUIRE_THROWS(rd_buf(std::move(buf)).get(*test_context.allocator, test_context.compiler));
	}
}

TEST_CASE("error: attaching something twice decl/decl") {
	{
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, 100, 1 });
		auto buf_a = vuk::discard_buf("a", *dst);
		auto buf_b = vuk::discard_buf("a again", *dst);

		auto wr_buf = vuk::make_pass("wr", [](CommandBuffer&, VUK_BA(vuk::eTransferWrite) buf, VUK_BA(vuk::eTransferWrite) bufb) { return buf; });

		REQUIRE_THROWS(wr_buf(buf_a, buf_b).get(*test_context.allocator, test_context.compiler));
	}
}
/*
TEST_CASE("not an error: attaching something twice acq/acq") {
  {
    auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, 100, 1 });
    auto buf_a = vuk::acquire_buf("a", *dst, vuk::Access::eNone);
    auto buf_b = vuk::acquire_buf("a again", *dst, vuk::Access::eNone);

    auto wr_buf = vuk::make_pass("wr", [](CommandBuffer&, VUK_BA(vuk::eTransferWrite) buf, VUK_BA(vuk::eTransferWrite) bufb) { return buf; });

    REQUIRE_NOTHROW(wr_buf(buf_a, buf_b).get(*test_context.allocator, test_context.compiler));
  }
}*/

TEST_CASE("error: attaching something twice decl/acq") {
	{
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, 100, 1 });
		auto buf_a = vuk::discard_buf("a", *dst);
		auto buf_b = vuk::acquire_buf("a again", *dst, vuk::Access::eNone);

		auto wr_buf = vuk::make_pass("wr", [](CommandBuffer&, VUK_BA(vuk::eTransferWrite) buf, VUK_BA(vuk::eTransferWrite) bufb) { return buf; });

		REQUIRE_THROWS(wr_buf(buf_a, buf_b).get(*test_context.allocator, test_context.compiler));
	}
}

TEST_CASE("error: passing same things with different access") {
	{
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, 100, 1 });
		auto buf_a = vuk::acquire_buf("a", *dst, vuk::Access::eNone);

		auto wr_buf = vuk::make_pass("wr", [](CommandBuffer&, VUK_BA(vuk::eTransferWrite) buf, VUK_BA(vuk::eTransferRead) bufb) { return buf; });

		REQUIRE_THROWS(wr_buf(buf_a, buf_a).get(*test_context.allocator, test_context.compiler));
	}
}

#pragma clang diagnostic pop
#pragma warning(pop)
