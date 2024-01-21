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
		Value<Buffer[]> filled_bufs = fill(arr);
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
				cbuf.fill_buffer(dst[i], (uint32_t)(0xfe - i));
			}
			return dst;
		});

		auto arr = declare_array("buffers", declare_buf("src", **buf), declare_buf("src2", **buf2));
		Value<Buffer[]> filled_bufs = fill(arr);
		auto res = download_buffer(filled_bufs[0]).get(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(data));
		res = download_buffer(filled_bufs[1]).get(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(data2));
	}
}

auto image2buf = make_pass("copy image to buffer", [](CommandBuffer& cbuf, VUK_IA(Access::eTransferRead) src, VUK_BA(Access::eTransferWrite) dst) {
	BufferImageCopy bc;
	bc.imageOffset = { 0, 0, 0 };
	bc.bufferRowLength = 0;
	bc.bufferImageHeight = 0;
	bc.imageExtent = static_cast<Extent3D>(src->extent.extent);
	bc.imageSubresource.aspectMask = format_to_aspect(src->format);
	bc.imageSubresource.mipLevel = src->base_level;
	bc.imageSubresource.baseArrayLayer = src->base_layer;
	assert(src->layer_count == 1); // unsupported yet
	bc.imageSubresource.layerCount = src->layer_count;
	bc.bufferOffset = dst->offset;
	cbuf.copy_image_to_buffer(src, dst, bc);
	return dst;
});

TEST_CASE("arrayed images, commands") {
	{
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));
		auto [img2, fut2] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

		size_t alignment = format_to_texel_block_size(fut->format);
		assert(fut->extent.sizing == Sizing::eAbsolute);
		size_t size = compute_image_size(fut->format, static_cast<Extent3D>(fut->extent.extent));
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });

		auto arr = declare_array("images", std::move(fut), std::move(fut2));
		{
			auto futc = clear_image(std::move(arr[0]), vuk::ClearColor(5u, 5u, 5u, 5u));
			auto dst_buf = declare_buf("dst", *dst);
			auto res = download_buffer(image2buf(std::move(futc), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
		}
		{
			auto futc2 = clear_image(std::move(arr[1]), vuk::ClearColor(6u, 6u, 6u, 6u));
			auto dst_buf = declare_buf("dst", *dst);
			auto res = download_buffer(image2buf(std::move(futc2), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 6; }));
		}
	}
}