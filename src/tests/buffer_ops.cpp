#include "TestContext.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Partials.hpp"
#include <doctest/doctest.h>

using namespace vuk;

TEST_CASE("test text_context preparation") {
	REQUIRE(test_context.prepare());
}

/*
TEST_CASE("test buffer harness") {
	REQUIRE(test_context.prepare());
	auto data = { 1u, 2u, 3u };
	auto [buf, fut] = create_buffer(*test_context.allocator, MemoryUsage::eCPUtoGPU, vuk::DomainFlagBits::eTransferOnTransfer, std::span(data));
	auto res = fut.get(*test_context.allocator, test_context.compiler);
	CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(data));
}

TEST_CASE("test buffer upload/download") {
	REQUIRE(test_context.prepare());
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

TEST_CASE("test buffer fill") {
	REQUIRE(test_context.prepare());
	{
		auto data = { 0xfeu, 0xfeu, 0xfeu, 0xfeu };
		auto buf = allocate_buffer(*test_context.allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(uint32_t) * 4 });

		auto fill = make_pass("fill", [](CommandBuffer& cbuf, VUK_BA(Access::eTransferWrite) dst) {
			cbuf.fill_buffer(dst, 0xfe);
			return dst;
		});

		auto res = download_buffer(fill(declare_buf("src", **buf))).get(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(data));
	}
}

TEST_CASE("test buffer update") {
	REQUIRE(test_context.prepare());
	{
		std::array<const uint32_t, 4> data = { 0xfeu, 0xfeu, 0xfeu, 0xfeu };
		auto buf = allocate_buffer(*test_context.allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(uint32_t) * 4 });

		auto fill = make_pass("update", [data](CommandBuffer& cbuf, VUK_BA(Access::eTransferWrite) dst) {
			cbuf.update_buffer(dst, &data[0]);
			return dst;
		});

		auto res = download_buffer(fill(declare_buf("src", **buf))).get(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(data));
	}
}

std::pair<Unique<Image>, TypedFuture<ImageAttachment>>
create_image(Allocator& allocator, DomainFlagBits copy_domain, ImageAttachment ia, void* data, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
	auto image = allocate_image(allocator, ia, loc);
	ia.image = **image;
	return { std::move(*image), host_data_to_image(allocator, copy_domain, ia, data) };
}

template<class T>
std::pair<Unique<Image>, TypedFuture<ImageAttachment>>
create_image(Allocator& allocator, DomainFlagBits copy_domain, ImageAttachment ia, std::span<T> data, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
	return create_image(allocator, copy_domain, ia, data.data(), loc);
}

auto image2buf = make_pass("copy image to buffer", [](CommandBuffer& cbuf, VUK_IA(Access::eTransferRead) src, VUK_BA(Access::eTransferWrite) dst) {
	cbuf.copy_image_to_buffer(src, dst, );
	return dst;
});

TEST_CASE("test image upload/download") {
	REQUIRE(test_context.prepare());
	{
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eMap2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		auto [img, fut] = create_image(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

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