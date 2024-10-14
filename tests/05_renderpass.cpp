#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <doctest/doctest.h>

using namespace vuk;

auto image2buf = make_pass("copy image to buffer", [](CommandBuffer& cbuf, VUK_IA(Access::eTransferRead) src, VUK_BA(Access::eTransferWrite) dst) {
	BufferImageCopy bc;
	bc.imageOffset = { 0, 0, 0 };
	bc.bufferRowLength = 0;
	bc.bufferImageHeight = 0;
	bc.imageExtent = src->extent;
	bc.imageSubresource.aspectMask = format_to_aspect(src->format);
	bc.imageSubresource.mipLevel = src->base_level;
	bc.imageSubresource.baseArrayLayer = src->base_layer;
	assert(src->layer_count == 1); // unsupported yet
	bc.imageSubresource.layerCount = src->layer_count;
	bc.bufferOffset = dst->offset;
	cbuf.copy_image_to_buffer(src, dst, bc);
	return dst;
});

TEST_CASE("renderpass clear") {
	{
		auto rpclear = make_pass("rp clear", [](CommandBuffer& cbuf, VUK_IA(Access::eColorWrite) dst) {
			cbuf.clear_image(dst, ClearColor(5u, 5u, 5u, 5u));
			return dst;
		});
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		ia.level_count = 1;
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

		size_t alignment = format_to_texel_block_size(fut->format);
		size_t size = compute_image_size(fut->format, fut->extent);
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
		auto fut2 = rpclear(fut);
		auto dst_buf = discard_buf("dst", *dst);
		auto res = download_buffer(image2buf(fut2, dst_buf)).get(*test_context.allocator, test_context.compiler);
		auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
		CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
	}
}

TEST_CASE("renderpass framebuffer inference") {
	{
		auto rpclear = make_pass("rp clear", [](CommandBuffer& cbuf, VUK_IA(Access::eColorWrite) dst, VUK_IA(Access::eDepthStencilRW)) {
			cbuf.clear_image(dst, ClearColor(5u, 5u, 5u, 5u));
			return dst;
		});
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		ia.level_count = 1;
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

		size_t alignment = format_to_texel_block_size(fut->format);
		size_t size = compute_image_size(fut->format, fut->extent);
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });

		auto depth_img = declare_ia("depth");
		depth_img->format = vuk::Format::eD32Sfloat;

		auto fut2 = rpclear(fut, std::move(depth_img));
		auto dst_buf = discard_buf("dst", *dst);
		auto res = download_buffer(image2buf(fut2, dst_buf)).get(*test_context.allocator, test_context.compiler);
		auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
		CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
	}
}

TEST_CASE("buffer size inference") {
	auto data = { 1u, 2u, 3u };
	auto [b0, buf0] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));
	auto buf1 = declare_buf("b1");
	buf1->memory_usage = MemoryUsage::eGPUonly;
	buf1.same_size(buf0);
	auto buf2 = declare_buf("b2");
	buf2->memory_usage = MemoryUsage::eGPUonly;
	buf2.same_size(buf1);
	auto buf3 = declare_buf("b3");
	buf3.same_size(buf2);
	buf3->memory_usage = MemoryUsage::eGPUonly;

	auto copy = make_pass("cpy", [](CommandBuffer& cbuf, VUK_BA(Access::eTransferRead) src, VUK_BA(Access::eTransferWrite) dst) {
		cbuf.copy_buffer(src, dst);
		return dst;
	});

	auto res = download_buffer(copy(std::move(buf0), std::move(buf3))).get(*test_context.allocator, test_context.compiler);
	CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(data));
}

TEST_CASE("buffer size with inference with math") {
	auto data = { 1u, 2u, 3u };
	auto [b0, buf0] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));
	auto buf1 = declare_buf("b1");
	buf1->memory_usage = MemoryUsage::eGPUonly;
	buf1.same_size(buf0);
	auto buf2 = declare_buf("b2");
	buf2->memory_usage = MemoryUsage::eGPUonly;
	buf2.same_size(buf1);
	auto buf3 = declare_buf("b3");
	buf3.set_size(buf2.get_size() * 2);
	buf3->memory_usage = MemoryUsage::eGPUonly;

	auto data2 = { 1u, 2u, 3u, 4u, 5u, 6u };
	auto [b4, buf4] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data2));

	auto copy = make_pass("cpy", [](CommandBuffer& cbuf, VUK_BA(Access::eTransferRead) src, VUK_BA(Access::eTransferWrite) dst) {
		cbuf.copy_buffer(src, dst);
		return dst;
	});

	auto res = download_buffer(copy(std::move(buf4), std::move(buf3))).get(*test_context.allocator, test_context.compiler);
	CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(data));
}