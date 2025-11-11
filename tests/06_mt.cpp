#include <thread>

#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <doctest/doctest.h>

using namespace vuk;

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

inline void void_clear_image(Value<ImageAttachment> in, Clear clear_value) {
	static auto clear = make_pass("void clear image", [=](CommandBuffer& cbuf, VUK_IA(Access::eClear) dst) { cbuf.clear_image(dst, clear_value); });

	clear(std::move(in));
}

TEST_CASE("MT") {
	auto data = { 1u, 2u, 3u, 4u };
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
	auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

	size_t alignment = format_to_texel_block_size(*fut->format);
	size_t size = compute_image_size(*fut->format, *fut->extent);

	auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto dst_buf = discard("dst", *dst);
	std::jthread worker([&]() { dst_buf = copy(fut, std::move(dst_buf)); });
	worker.join();
	auto res = download_buffer(dst_buf).get(*test_context.allocator, test_context.compiler);
	auto updata = res->to_span<uint32_t>();
	CHECK(updata == std::span(data));
}

TEST_CASE("MT reconvergence") {
	for (int i = 0; i < 2; i++) {
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		ia.level_count = 2;
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

		size_t alignment = format_to_texel_block_size(*fut->format);
		size_t size = compute_image_size(*fut->format, *fut->extent);
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });

		vuk::Value<ImageAttachment> futp;

		std::jthread worker([&]() {
			void_clear_image(fut.mip(0), vuk::ClearColor(7u, 7u, 7u, 7u));
			futp = blit_down(fut);
		});
		worker.join();

		auto dst_buf = discard("dst", *dst);
		auto res = download_buffer(copy(futp.mip(1), std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
		auto updata = res->to_span<uint32_t>();
		CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 7; }));
	}
}