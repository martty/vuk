#include "TestContext.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Partials.hpp"
#include <doctest/doctest.h>

using namespace vuk;

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
		assert(fut->extent.sizing == Sizing::eAbsolute);
		size_t size = compute_image_size(fut->format, static_cast<Extent3D>(fut->extent.extent));
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
		auto fut2 = rpclear(fut);
		auto dst_buf = declare_buf("dst", *dst);
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
		assert(fut->extent.sizing == Sizing::eAbsolute);
		size_t size = compute_image_size(fut->format, static_cast<Extent3D>(fut->extent.extent));
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });

		auto depth_img = declare_ia("depth");
		depth_img->format = vuk::Format::eD32Sfloat;

		auto fut2 = rpclear(fut, depth_img);
		auto dst_buf = declare_buf("dst", *dst);
		auto res = download_buffer(image2buf(fut2, dst_buf)).get(*test_context.allocator, test_context.compiler);
		auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
		CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
	}
}
