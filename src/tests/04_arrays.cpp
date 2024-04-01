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

TEST_CASE("arrayed images, commands") {
	{
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));
		auto [img2, fut2] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

		size_t alignment = format_to_texel_block_size(fut->format);
		size_t size = compute_image_size(fut->format, fut->extent);
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

TEST_CASE("image slicing, mips") {
	{
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		ia.level_count = 2;
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

		size_t alignment = format_to_texel_block_size(fut->format);
		size_t size = compute_image_size(fut->format, fut->extent);
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });

		{
			auto futc = clear_image(fut.mip(0), vuk::ClearColor(5u, 5u, 5u, 5u));
			auto dst_buf = declare_buf("dst", *dst);
			auto res = download_buffer(image2buf(std::move(futc), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
		}
		{
			auto futc2 = clear_image(fut.mip(1), vuk::ClearColor(6u, 6u, 6u, 6u));
			auto dst_buf = declare_buf("dst", *dst);
			auto res = download_buffer(image2buf(std::move(futc2), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 1);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 6; }));
		}
	}
}

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

TEST_CASE("image slicing, reconvergence") {
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
			auto m2 = clear_image(fut.mip(1), vuk::ClearColor(6u, 6u, 6u, 6u));
			auto futp = blit_down(std::move(fut));

			auto dst_buf = declare_buf("dst", *dst);
			auto res = download_buffer(image2buf(futp.mip(1), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 1);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
		}
	}
}

auto layout = make_pass("layout", [](CommandBuffer& cbuf, VUK_IA(Access::eTransferWrite) img) { return img; });

TEST_CASE("image slicing, reconvergence 2") {
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
			auto m2 = layout(clear_image(fut.mip(1), vuk::ClearColor(6u, 6u, 6u, 6u)));
			auto futp = blit_down(std::move(fut));

			auto dst_buf = declare_buf("dst", *dst);
			auto res = download_buffer(image2buf(futp.mip(1), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 1);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
		}
	}
}

TEST_CASE("image slicing, reconvergence 3") {
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
			auto dst_buf = declare_buf("dst", *dst);
			auto res = download_buffer(image2buf(futp.mip(1), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 1);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
		}
	}
}

inline void void_clear_image(Value<ImageAttachment> in, Clear clear_value) {
	static auto clear = make_pass("void clear image", [=](CommandBuffer& cbuf, VUK_IA(Access::eClear) dst) {
		cbuf.clear_image(dst, clear_value);
	});

	clear(std::move(in));
}

TEST_CASE("image slicing, reconvergence with undef") {
	{
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		ia.level_count = 2;
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

		size_t alignment = format_to_texel_block_size(fut->format);
		size_t size = compute_image_size(fut->format, fut->extent);
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });

		{
			void_clear_image(fut.mip(0), vuk::ClearColor(7u, 7u, 7u, 7u));
			auto futp = blit_down(fut);
			auto dst_buf = declare_buf("dst", *dst);
			auto res = download_buffer(image2buf(futp.mip(1), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 1);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 7; }));
		}
	}
}