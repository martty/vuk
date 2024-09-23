#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <doctest/doctest.h>

using namespace vuk;

#if VUK_FAIL_FAST
#error "can't run these on FAIL_FAST"
#endif

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

auto image2buf = make_pass("copy image to buffer", [](CommandBuffer& cbuf, VUK_IA(Access::eTransferRead) src, VUK_BA(Access::eTransferWrite) dst) {
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

auto blit_down = make_pass("blit down", [](CommandBuffer& cbuf, VUK_IA(Access::eTransferRead | Access::eTransferWrite) img) {
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
/*
TEST_CASE("error: reconvergence, time-travel forbidden") {
	{
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		ia.level_count = 2;
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

		size_t alignment = format_to_texel_block_size(fut->format);
		size_t size = compute_image_size(fut->format, fut->extent);
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });

		{
			auto m1 = clear_image(fut.mip(0), vuk::ClearColor(5u, 5u, 5u, 5u));
			auto futp = blit_down(fut);
			auto m2 = clear_image(
			    fut.mip(1),
			    vuk::ClearColor(6u, 6u, 6u, 6u)); // m2 would need to travel before the reconvergence, which was implict on the previous line - this is not allowed
			auto dst_buf = declare_buf("dst", *dst);
			REQUIRE_THROWS(download_buffer(image2buf(futp.mip(1), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler));
		}
	}
}*/