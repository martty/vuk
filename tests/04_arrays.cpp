#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
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

		auto arr = declare_array("buffers", discard_buf("src", **buf), discard_buf("src2", **buf2));
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

		auto arr = declare_array("buffers", discard_buf("src", **buf), discard_buf("src2", **buf2));
		Value<Buffer[]> filled_bufs = fill(arr);
		auto res = download_buffer(filled_bufs[0]).get(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(data));
		res = download_buffer(filled_bufs[1]).get(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(data2));
	}
}

TEST_CASE("zero len arrayed buffers") {
	{
		auto buf = allocate_buffer(*test_context.allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(uint32_t) * 4 });
		auto buf2 = allocate_buffer(*test_context.allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(uint32_t) * 4 });

		std::string trace = "";

		auto fill = make_pass("fill two", [&](CommandBuffer& cbuf, VUK_ARG(Buffer[], Access::eTransferWrite) dst) {
			for (size_t i = 0; i < dst.size(); i++) {
				cbuf.fill_buffer(dst[i], (uint32_t)(0xfe - i));
				trace += "+";
			}
			return dst;
		});

		auto arr = declare_array("buffers", std::span<vuk::Value<vuk::Buffer>>{});
		Value<Buffer[]> filled_bufs = fill(arr);
		auto res = filled_bufs.wait(*test_context.allocator, test_context.compiler);
		CHECK(trace == "");
	}
}

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
			auto futc = clear_image(arr[0], vuk::ClearColor(5u, 5u, 5u, 5u));
			auto dst_buf = discard_buf("dst", *dst);
			auto res = download_buffer(copy(std::move(futc), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
		}
		{
			auto futc2 = clear_image(arr[1], vuk::ClearColor(6u, 6u, 6u, 6u));
			auto dst_buf = discard_buf("dst", *dst);
			auto res = download_buffer(copy(std::move(futc2), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 6; }));
		}
	}
}

// clang barfs on image_use<Acc>
#if VUK_COMPILER_MSVC
TEST_CASE("arrayed images, divergent source sync") {
	{
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));
		auto [img2, fut2] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

		size_t alignment = format_to_texel_block_size(fut->format);
		size_t size = compute_image_size(fut->format, fut->extent);
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });

		fut = image_use<eFragmentRead>(clear_image(std::move(fut), vuk::ClearColor(5u, 5u, 5u, 5u)));
		fut2 = image_use<eTransferRead>(clear_image(std::move(fut2), vuk::ClearColor(6u, 6u, 6u, 6u)));
		auto arr = declare_array("images", std::move(fut), std::move(fut2));

		auto array_use = make_pass("array_use", [](CommandBuffer& cbuf, VUK_ARG(ImageAttachment[], Access::eTransferWrite) img) {
			auto& first = img[0];
			auto& second = img[1];
			return img;
		});
		arr = array_use(std::move(arr));

		{
			auto dst_buf = discard_buf("dst", *dst);
			auto res = download_buffer(copy(std::move(arr[0]), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
		}
		{
			auto dst_buf = discard_buf("dst", *dst);
			auto res = download_buffer(copy(std::move(arr[1]), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 6; }));
		}
	}
}
#endif
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
			auto dst_buf = discard_buf("dst", *dst);
			auto res = download_buffer(copy(std::move(futc), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
		}
		{
			auto futc2 = clear_image(fut.mip(1), vuk::ClearColor(6u, 6u, 6u, 6u));
			auto dst_buf = discard_buf("dst", *dst);
			auto res = download_buffer(copy(std::move(futc2), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
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

		auto dst_buf = discard_buf("dst", *dst);
		auto res = download_buffer(copy(futp.mip(1), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
		auto updata = std::span((uint32_t*)res->mapped_ptr, 1);
		CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
	}
}
#if VUK_COMPILER_MSVC
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
			auto m2 = image_use<Access::eTransferWrite>(clear_image(fut.mip(1), vuk::ClearColor(6u, 6u, 6u, 6u)));
			auto futp = blit_down(std::move(fut));

			auto dst_buf = discard_buf("dst", *dst);
			auto res = download_buffer(copy(futp.mip(1), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 1);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
		}
	}
}
#endif
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
			auto dst_buf = discard_buf("dst", *dst);
			auto res = download_buffer(copy(futp.mip(1), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 1);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
		}
	}
}

inline void void_clear_image(Value<ImageAttachment> in, Clear clear_value) {
	static auto clear = make_pass("void clear image", [=](CommandBuffer& cbuf, VUK_IA(Access::eClear) dst) { cbuf.clear_image(dst, clear_value); });

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
			auto dst_buf = discard_buf("dst", *dst);
			auto res = download_buffer(copy(futp.mip(1), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 1);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 7; }));
		}
	}
}

TEST_CASE("image slicing, forced reconvergence") {
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
			auto dst_buf = discard_buf("dst", *dst);
			auto res = download_buffer(copy(std::move(futc), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
		}
		{
			auto futp = blit_down(fut);
			auto dst_buf = discard_buf("dst", *dst);
			auto res = download_buffer(copy(std::move(fut.mip(1)), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
			auto updata = std::span((uint32_t*)res->mapped_ptr, 1);
			CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
		}
	}
}

vuk::Value<vuk::ImageAttachment> generate_mips(std::string& trace, vuk::Value<vuk::ImageAttachment> image, uint32_t mip_count) {
	auto ia = image.mip(0);

	for (uint32_t mip_level = 1; mip_level < mip_count; mip_level++) {
		auto pass = vuk::make_pass(fmt::format("mip_{}", mip_level).c_str(),
		                           [&trace, mip_level](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eTransferRead) src, VUK_IA(vuk::eTransferWrite) dst) {
			                           ImageBlit blit;
			                           const auto extent = src->extent;

			                           blit.srcSubresource.aspectMask = format_to_aspect(src->format);
			                           blit.srcSubresource.baseArrayLayer = src->base_layer;
			                           blit.srcSubresource.layerCount = src->layer_count;
			                           blit.srcSubresource.mipLevel = mip_level - 1;
			                           blit.srcOffsets[0] = Offset3D{ 0 };
			                           blit.srcOffsets[1] = Offset3D{ std::max(static_cast<int32_t>(extent.width) >> (mip_level - 1), 1),
				                                                        std::max(static_cast<int32_t>(extent.height) >> (mip_level - 1), 1),
				                                                        std::max(static_cast<int32_t>(extent.depth) >> (mip_level - 1), 1) };
			                           blit.dstSubresource = blit.srcSubresource;
			                           blit.dstSubresource.mipLevel = mip_level;
			                           blit.dstOffsets[0] = Offset3D{ 0 };
			                           blit.dstOffsets[1] = Offset3D{ std::max(static_cast<int32_t>(extent.width) >> (mip_level), 1),
				                                                        std::max(static_cast<int32_t>(extent.height) >> (mip_level), 1),
				                                                        std::max(static_cast<int32_t>(extent.depth) >> (mip_level), 1) };
			                           command_buffer.blit_image(src, dst, blit, Filter::eLinear);

			                           trace += fmt::format("{}", mip_level);

			                           return dst;
		                           });

		ia = pass(std::move(ia), image.mip(mip_level));
	}
	return image;
}

TEST_CASE("mip generation") {
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 64, 64, 1 }, Samples::e1);
	auto img = vuk::clear_image(vuk::declare_ia("src", ia), vuk::ClearColor(0.1f, 0.1f, 0.1f, 0.1f));
	std::string trace = "";
	generate_mips(trace, std::move(img), 5).wait(*test_context.allocator, test_context.compiler);
	CHECK(trace == "1234");
}

TEST_CASE("read convergence") {
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 64, 64, 1 }, Samples::e1);
	auto img = vuk::clear_image(vuk::declare_ia("src", ia), vuk::ClearColor(0.1f, 0.1f, 0.1f, 0.1f));
	std::string trace = "";
	auto mipped = generate_mips(trace, std::move(img), 5);
	auto passr = vuk::make_pass("rd", [&trace](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eTransferRead) src) {
		trace += "r";
		return src;
	});
	auto passw = vuk::make_pass("wr", [&trace](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eTransferWrite) src) {
		trace += "w";
		return src;
	});
	passw(passr(mipped)).wait(*test_context.allocator, test_context.compiler);
	CHECK(trace == "1234rw");
}

TEST_CASE("read convergence 2") {
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 64, 64, 1 }, Samples::e1);
	auto img = vuk::clear_image(vuk::declare_ia("src", ia), vuk::ClearColor(0.1f, 0.1f, 0.1f, 0.1f));
	auto img2 = vuk::clear_image(vuk::declare_ia("src2", ia), vuk::ClearColor(0.1f, 0.1f, 0.1f, 0.1f));
	std::string trace = "";
	auto mipped = generate_mips(trace, img, 5);
	auto pass = vuk::make_pass("rd", [&trace](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eTransferRead) src, VUK_IA(vuk::eTransferWrite) src2) {
		trace += "r";
		return src;
	});
	pass(std::move(mipped), img2).wait(*test_context.allocator, test_context.compiler);
	CHECK(trace == "1234r");
}

TEST_CASE("mip generation 2") {
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 64, 64, 1 }, Samples::e1);
	auto img = vuk::declare_ia("src", ia);
	vuk::clear_image(img.mip(0), vuk::ClearColor(0.1f, 0.1f, 0.1f, 0.1f));
	vuk::clear_image(img.mip(4), vuk::ClearColor(0.6f, 0.1f, 0.1f, 0.1f));
	std::string trace = "";

	auto mipped = generate_mips(trace, std::move(img), 5);
	size_t alignment = format_to_texel_block_size(mipped->format);
	size_t size = compute_image_size(mipped->format, { 1, 1, 1 });
	auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto res = download_buffer(copy(mipped.mip(4), discard_buf("dst", *dst))).get(*test_context.allocator, test_context.compiler);
	auto updata = std::span((float*)res->mapped_ptr, 1);
	CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem - 0.1f < 0.001f; }));
	CHECK(trace == "1234");
}

void generate_mips_2(std::string& trace, vuk::Value<vuk::ImageAttachment> image, uint32_t mip_count) {
	auto blit_mip = vuk::make_pass("blit_mip", [&trace](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eTransferRead) src, VUK_IA(vuk::eTransferWrite) dst) {
		ImageBlit blit;
		const auto extent = src->extent;

		blit.srcSubresource.aspectMask = format_to_aspect(src->format);
		blit.srcSubresource.baseArrayLayer = src->base_layer;
		blit.srcSubresource.layerCount = src->layer_count;
		blit.srcSubresource.mipLevel = src->base_level;
		blit.srcOffsets[0] = Offset3D{ 0 };
		blit.srcOffsets[1] = Offset3D{ std::max(static_cast<int32_t>(extent.width) >> (src->base_level), 1),
			                             std::max(static_cast<int32_t>(extent.height) >> (src->base_level), 1),
			                             std::max(static_cast<int32_t>(extent.depth) >> (src->base_level), 1) };
		blit.dstSubresource = blit.srcSubresource;
		blit.dstSubresource.mipLevel = dst->base_level;
		blit.dstOffsets[0] = Offset3D{ 0 };
		blit.dstOffsets[1] = Offset3D{ std::max(static_cast<int32_t>(extent.width) >> (dst->base_level), 1),
			                             std::max(static_cast<int32_t>(extent.height) >> (dst->base_level), 1),
			                             std::max(static_cast<int32_t>(extent.depth) >> (dst->base_level), 1) };
		command_buffer.blit_image(src, dst, blit, Filter::eLinear);

		trace += fmt::format("{}", dst->base_level);
	});

	for (uint32_t mip_level = 1; mip_level < mip_count; mip_level++) {
		blit_mip(image.mip(mip_level - 1), image.mip(mip_level));
	}
}

TEST_CASE("mip generation 3") {
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 64, 64, 1 }, Samples::e1);
	auto img = vuk::declare_ia("src", ia);
	vuk::clear_image(img.mip(0), vuk::ClearColor(0.1f, 0.1f, 0.1f, 0.1f));
	vuk::clear_image(img.mip(4), vuk::ClearColor(0.6f, 0.1f, 0.1f, 0.1f));
	std::string trace = "";

	generate_mips_2(trace, img, 5);
	size_t alignment = format_to_texel_block_size(img->format);
	size_t size = compute_image_size(img->format, { 1, 1, 1 });
	auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto res = download_buffer(copy(img.mip(4), discard_buf("dst", *dst))).get(*test_context.allocator, test_context.compiler);
	auto updata = std::span((float*)res->mapped_ptr, 1);
	CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem - 0.1f < 0.001f; }));
	CHECK(trace == "1234");
}

TEST_CASE("mip generation 4") {
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 64, 64, 1 }, Samples::e1);
	auto img = vuk::declare_ia("src", ia);
	vuk::clear_image(img.mip(0), vuk::ClearColor(0.1f, 0.1f, 0.1f, 0.1f));
	vuk::clear_image(img.mip(4), vuk::ClearColor(0.6f, 0.1f, 0.1f, 0.1f));
	std::string trace = "";

	generate_mips_2(trace, img, 5);
	size_t alignment = format_to_texel_block_size(img->format);
	size_t size = compute_image_size(img->format, { 1, 1, 1 });
	auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto res = download_buffer(copy(img.mip(4), discard_buf("dst", *dst))).get(*test_context.allocator, test_context.compiler);
	auto updata = std::span((float*)res->mapped_ptr, 1);
	CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem - 0.1f < 0.001f; }));
	CHECK(trace == "1234");
}

TEST_CASE("mip generation 5") {
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 64, 64, 1 }, Samples::e1);
	auto img = vuk::declare_ia("src", ia);
	vuk::clear_image(img, vuk::ClearColor(0.1f, 0.1f, 0.1f, 0.1f));
	std::string trace = "";

	generate_mips_2(trace, img, 5);
	size_t alignment = format_to_texel_block_size(img->format);
	size_t size = compute_image_size(img->format, { 1, 1, 1 });
	auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto res = download_buffer(copy(img.mip(4), discard_buf("dst", *dst))).get(*test_context.allocator, test_context.compiler);
	auto updata = std::span((float*)res->mapped_ptr, 1);
	CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem - 0.1f < 0.001f; }));
	CHECK(trace == "1234");
}

TEST_CASE("mip2mip dep") {
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 64, 64, 1 }, Samples::e1);
	auto img = vuk::declare_ia("src", ia);
	vuk::clear_image(img.mip(0), vuk::ClearColor(0.1f, 0.1f, 0.1f, 0.1f));
	vuk::clear_image(img.mip(4), vuk::ClearColor(0.6f, 0.1f, 0.1f, 0.1f));

	auto a = img.mip(0);
	blit_image(a, img.mip(4), Filter::eLinear);
	size_t alignment = format_to_texel_block_size(img->format);
	size_t size = compute_image_size(img->format, { 1, 1, 1 });
	auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto res = download_buffer(copy(img.mip(4), discard_buf("dst", *dst))).get(*test_context.allocator, test_context.compiler);
	auto updata = std::span((float*)res->mapped_ptr, 1);
	CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem - 0.1f < 0.001f; }));
}

vuk::Value<vuk::ImageAttachment> bloom_pass(std::string& trace,
                                            vuk::Value<vuk::ImageAttachment> downsample_image,
                                            vuk::Value<vuk::ImageAttachment> upsample_image,
                                            vuk::Value<vuk::ImageAttachment> input) {
	auto bloom_mip_count = downsample_image->level_count;
	auto prefilter = vuk::make_pass(
	    "bloom_prefilter", [&trace](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eComputeRW) target, VUK_IA(vuk::eComputeSampled) input) { trace += "p"; });

	prefilter(downsample_image.mip(0), input);

	for (uint32_t i = 1; i < bloom_mip_count; i++) {
		auto pass = vuk::make_pass(fmt::format("bloom_downsample_{}", i).c_str(),
		                           [i, &trace](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eComputeRW) target, VUK_IA(vuk::eComputeSampled) input) {
			                           trace += fmt::format("d{}", i);
		                           });
		pass(downsample_image.mip(i), downsample_image.mip(i - 1));
	}

	// Upsampling
	// https://www.froyok.fr/blog/2021-12-ue4-custom-bloom/resources/code/bloom_down_up_demo.jpg

	auto upsample_src_mip = downsample_image.mip(bloom_mip_count - 1);

	for (int32_t i = (int32_t)bloom_mip_count - 2; i >= 0; i--) {
		auto pass = vuk::make_pass(
		    fmt::format("bloom_upsample_{}", i).c_str(),
		    [i, &trace](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eComputeRW) output, VUK_IA(vuk::eComputeSampled) src1, VUK_IA(vuk::eComputeSampled) src2) {
			    trace += fmt::format("u{}", i);
		    });

		pass(upsample_image.mip(i), upsample_src_mip, downsample_image.mip(i));
		upsample_src_mip = upsample_image.mip(i);
	}

	return upsample_image;
}

TEST_CASE("mip down-up") {
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 64, 64, 1 }, Samples::e1);
	auto src = vuk::clear_image(vuk::declare_ia("src", ia), vuk::ClearColor(0.1f, 0.1f, 0.1f, 0.1f));
	auto downsample = vuk::declare_ia("down", ia);
	auto upsample = vuk::declare_ia("up", ia);
	std::string trace = "";
	bloom_pass(trace, std::move(downsample), std::move(upsample), std::move(src)).wait(*test_context.allocator, test_context.compiler);
	CHECK(trace == "pd1d2d3d4d5d6u5u4u3u2u1u0");
}

TEST_CASE("buffer slicing") {
	auto data = { 0xfeu, 0xfeu, 0xfeu, 0xfeu, 0xfeu, 0xfeu, 0xfeu };
	auto [alloc, buf] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));

	auto fill1 = make_pass("fill some 1", [](CommandBuffer& cbuf, VUK_ARG(Buffer, Access::eTransferWrite) dst) {
		cbuf.fill_buffer(dst, 0xfd);
		return dst;
	});

	fill1(buf.subrange(1 * sizeof(uint32_t), sizeof(uint32_t)));

	auto fill2 = make_pass("fill some 2", [](CommandBuffer& cbuf, VUK_ARG(Buffer, Access::eTransferWrite) dst) {
		cbuf.fill_buffer(dst, 0xfc);
		return dst;
	});

	fill2(buf.subrange(3 * sizeof(uint32_t), sizeof(uint32_t)));
	// equal reslice
	fill1(buf.subrange(4 * sizeof(uint32_t), sizeof(uint32_t)).subrange(0, sizeof(uint32_t)));
	// shrinking reslice
	fill2(buf.subrange(5 * sizeof(uint32_t), 2 * sizeof(uint32_t)).subrange(sizeof(uint32_t), sizeof(uint32_t)));

	auto check = { 0xfeu, 0xfdu, 0xfeu, 0xfcu, 0xfdu, 0xfeu, 0xfcu };
	auto res = download_buffer(buf).get(*test_context.allocator, test_context.compiler);
	auto schpen = std::span((uint32_t*)res->mapped_ptr, check.size());
	CHECK(schpen == std::span(check));
}

TEST_CASE("buffer slice-conv-slice") {
	auto data = { 0xfeu, 0xfeu, 0xfeu, 0xfeu };
	auto [alloc, buf] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));

	auto fill1 = make_pass("fill some 1", [](CommandBuffer& cbuf, VUK_ARG(Buffer, Access::eTransferWrite) dst) {
		cbuf.fill_buffer(dst, 0xfd);
		return dst;
	});

	fill1(buf.subrange(1 * sizeof(uint32_t), sizeof(uint32_t)));

	auto fill2 = make_pass("fill some 2", [](CommandBuffer& cbuf, VUK_ARG(Buffer, Access::eTransferWrite) dst) {
		cbuf.fill_buffer(dst, 0xfc);
		return dst;
	});

	fill2(buf);
	// reslice
	fill1(buf.subrange(1 * sizeof(uint32_t), sizeof(uint32_t)));

	auto check = { 0xfcu, 0xfdu, 0xfcu, 0xfcu };
	auto res = download_buffer(buf).get(*test_context.allocator, test_context.compiler);
	auto schpen = std::span((uint32_t*)res->mapped_ptr, check.size());
	CHECK(schpen == std::span(check));
}

TEST_CASE("buffer two range") {
	auto data = { 0xfeu, 0xfeu, 0xfeu, 0xfeu };
	auto [alloc, buf] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));

	auto range1 = buf.subrange(1 * sizeof(uint32_t), sizeof(uint32_t));
	auto range2 = buf.subrange(2 * sizeof(uint32_t), sizeof(uint32_t));
	auto fill1 = make_pass("fill some 1", [](CommandBuffer& cbuf, VUK_ARG(Buffer, Access::eTransferWrite) dst) {
		cbuf.fill_buffer(dst, 0xfd);
		return dst;
	});
	auto fill2 = make_pass("fill some 2", [](CommandBuffer& cbuf, VUK_ARG(Buffer, Access::eTransferWrite) dst) {
		cbuf.fill_buffer(dst, 0xfc);
		return dst;
	});
	auto res = download_buffer(buf).get(*test_context.allocator, test_context.compiler);
	fill1(range1);
	fill2(range2);

	auto check = { 0xfeu, 0xfdu, 0xfcu, 0xfeu };
	res = download_buffer(buf).get(*test_context.allocator, test_context.compiler);
	auto schpen = std::span((uint32_t*)res->mapped_ptr, check.size());
	CHECK(schpen == std::span(check));
}
/*
auto frw_pass = make_pass("frw", [](CommandBuffer& cbuf, VUK_IA(Access::eTransferWrite) img) {
});

TEST_CASE("alienated subresource") {
	auto data = { 1u, 2u, 3u, 4u };
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
	ia.level_count = 2;
	auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

	size_t alignment = format_to_texel_block_size(fut->format);
	size_t size = compute_image_size(fut->format, fut->extent);
	auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });

	auto mip0 = fut.mip(0);
	auto mip1 = fut.mip(1);
	auto dst_buf = discard_buf("dst", *dst);
	auto res = download_buffer(copy(fut, std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);

	{
		auto futc = clear_image(mip0, vuk::ClearColor(5u, 5u, 5u, 5u));
		auto futc2 = clear_image(mip1, vuk::ClearColor(6u, 6u, 6u, 6u));
		frw_pass(mip1);
		auto dst_buf = discard_buf("dst", *dst);
		auto res = download_buffer(copy(mip1, std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
		auto updata = std::span((uint32_t*)res->mapped_ptr, 1);
		CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 6; }));
	}

	{
		frw_pass(fut);
		auto dst_buf = discard_buf("dst", *dst);
		auto res = download_buffer(copy(fut.mip(0), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
		auto updata = std::span((uint32_t*)res->mapped_ptr, 1);
		CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
	}
}

*/