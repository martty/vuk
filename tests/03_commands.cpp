#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <doctest/doctest.h>
#include <thread>

using namespace vuk;

TEST_CASE("buffer harness") {
	auto data = { 1u, 2u, 3u };
	auto [buf, fut] = create_buffer(*test_context.allocator, MemoryUsage::eCPUtoGPU, vuk::DomainFlagBits::eTransferOnTransfer, std::span(data));
	auto res = fut.get(*test_context.allocator, test_context.compiler);
	CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(data));
}

TEST_CASE("buffer upload/download") {
	{
		auto data = { 1u, 2u, 3u };
		auto [buf, fut] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));

		auto res = download_buffer(fut).get(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(data));
	}
	{
		auto data = { 1u, 2u, 3u, 4u, 5u };
		auto [buf, fut] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));

		auto res = download_buffer(fut).get(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 5) == std::span(data));
	}
}

TEST_CASE("buffer fill & update") {
	{
		auto data = { 0xfeu, 0xfeu, 0xfeu, 0xfeu };
		auto buf = allocate_buffer(*test_context.allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(uint32_t) * 4 });

		auto fill = make_pass("fill", [](CommandBuffer& cbuf, VUK_BA(Access::eTransferWrite) dst) {
			cbuf.fill_buffer(dst, 0xfe);
			return dst;
		});

		auto res = download_buffer(fill(discard_buf("src", **buf))).get(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(data));
	}
	{
		std::array<const uint32_t, 4> data = { 0xfeu, 0xfeu, 0xfeu, 0xfeu };
		auto buf = allocate_buffer(*test_context.allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(uint32_t) * 4 });

		auto fill = make_pass("update", [data](CommandBuffer& cbuf, VUK_BA(Access::eTransferWrite) dst) {
			cbuf.update_buffer(dst, &data[0]);
			return dst;
		});

		auto res = download_buffer(fill(discard_buf("src", **buf))).get(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(data));
	}
}

TEST_CASE("image upload/download") {
	{
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

		size_t alignment = format_to_texel_block_size(fut->format);
		size_t size = compute_image_size(fut->format, fut->extent);
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
		auto dst_buf = discard_buf("dst", *dst);
		auto res = download_buffer(copy(fut, std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
		auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
		CHECK(updata == std::span(data));
	}
}

TEST_CASE("image clear") {
	{
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

		size_t alignment = format_to_texel_block_size(fut->format);
		size_t size = compute_image_size(fut->format, fut->extent);
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
		auto fut2 = clear_image(fut, vuk::ClearColor(5u, 5u, 5u, 5u));
		auto dst_buf = discard_buf("dst", *dst);
		auto res = download_buffer(copy(fut2, dst_buf)).get(*test_context.allocator, test_context.compiler);
		auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
		CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
	}
}

TEST_CASE("image blit") {
	{
		auto data = { 1.f, 0.f, 0.f, 1.f };
		auto ia_src = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 2, 2, 1 }, Samples::e1);
		ia_src.level_count = 1;
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia_src, std::span(data));
		auto ia_dst = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 1, 1, 1 }, Samples::e1);
		ia_dst.level_count = 1;
		auto img2 = allocate_image(*test_context.allocator, ia_dst);
		size_t alignment = format_to_texel_block_size(fut->format);
		size_t size = compute_image_size(fut->format, fut->extent);
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
		auto fut2 = blit_image(fut, declare_ia("dst_i", ia_dst), Filter::eLinear);
		auto dst_buf = discard_buf("dst", *dst);
		auto res = download_buffer(copy(fut2, dst_buf)).get(*test_context.allocator, test_context.compiler);
		auto updata = std::span((float*)res->mapped_ptr, 1);
		CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 0.5f; }));
	}
	{
		auto data = { 1.f, 0.f, 0.f, 1.f };
		auto ia_src = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 2, 2, 1 }, Samples::e1);
		ia_src.level_count = 1;
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia_src, std::span(data));
		auto ia_dst = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 1, 1, 1 }, Samples::e1);
		ia_dst.level_count = 1;
		auto img2 = allocate_image(*test_context.allocator, ia_dst);
		size_t alignment = format_to_texel_block_size(fut->format);
		size_t size = compute_image_size(fut->format, fut->extent);
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
		auto fut2 = blit_image(fut, declare_ia("dst_i", ia_dst), Filter::eNearest);
		auto dst_buf = discard_buf("dst", *dst);
		auto res = download_buffer(copy(fut2, dst_buf)).get(*test_context.allocator, test_context.compiler);
		auto updata = std::span((float*)res->mapped_ptr, 1);
		CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 1.f; }));
	}
}

TEST_CASE("poll wait") {
	{
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

		size_t alignment = format_to_texel_block_size(fut->format);
		size_t size = compute_image_size(fut->format, fut->extent);
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
		auto fut2 = clear_image(fut, vuk::ClearColor(5u, 5u, 5u, 5u));
		auto dst_buf = discard_buf("dst", *dst);
		download_buffer(copy(fut2, dst_buf)).submit(*test_context.allocator, test_context.compiler);
		while (*dst_buf.poll() != Signal::Status::eHostAvailable) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		auto updata = std::span((uint32_t*)dst_buf->mapped_ptr, 4);
		CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
	}
}

TEST_CASE("buffers in same allocation") {
	auto data = { 5u, 5u, 5u, 5u };
	auto buf = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, 4 * sizeof(uint32_t), 1 });
	auto bufa = buf->subrange(0, 2 * sizeof(uint32_t));
	auto bufb = buf->subrange(2 * sizeof(uint32_t), 2 * sizeof(uint32_t));
	auto fut_a = discard_buf("a", bufa);
	auto fut_b = discard_buf("b", bufb);
	fill(fut_a, 5u);
	copy(fut_a, fut_b);
	auto res = download_buffer(fut_b).get(*test_context.allocator, test_context.compiler);
	CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(data));
}

// TEST TODOS: image2image copy, resolve
