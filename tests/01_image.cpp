#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <doctest/doctest.h>
#include <string_view>

using namespace vuk;

// ============================================================================
// IR Integration Tests
// ============================================================================

TEST_CASE("constant_image_view_metadata") {
	auto data = { 11u, 22u, 33u, 44u };
	auto ici = from_preset(Preset::eMap2D, Format::eR32Uint, Extent3D{ 2, 2, 1 }, Samples::e1);

	auto [view, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ici, std::span(data));

	// Make the image view a constant in the IR
	auto view_const = make_constant("test_img_view", *view);

	// Get metadata from the constant view and compare to creation parameters
	auto meta = *(ImageViewEntry*)(*eval(view_const.get_meta().get_head()));

	// Verify the metadata matches our creation parameters
	CHECK(meta.format == Format::eR32Uint);
	CHECK(meta.extent.width == 2);
	CHECK(meta.extent.height == 2);
	CHECK(meta.extent.depth == 1);
	CHECK(meta.sample_count == Samples::e1);
	CHECK(meta.base_level == 0);
	CHECK(meta.level_count == 1);
	CHECK(meta.base_layer == 0);
	CHECK(meta.layer_count == 1);
}

TEST_CASE("image_as_ir_constant") {
	auto data = { 1u, 2u, 3u, 4u };
	auto ici = from_preset(Preset::eMap2D, Format::eR32Uint, Extent3D{ 2, 2, 1 }, Samples::e1);

	auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ici, std::span(data));

	// Make the image a constant in the IR
	auto img_const = make_constant("test_img", *img);

	// Download and verify the data
	fut.format.dump_ir();
	eval(fut.format.get_head());
	size_t alignment = format_to_texel_block_size(*fut.format);
	size_t size = compute_image_size(*fut.format, *fut.extent);
	auto dst = *allocate_buffer<uint32_t>(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto dst_buf = discard("dst", *dst);
	auto res = download_buffer(copy(fut, std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
	auto updata = res->to_span();
	CHECK(updata == std::span(data));
}

TEST_CASE("image_acquire_external") {
	auto data = { 5u, 6u, 7u, 8u };
	auto ici = from_preset(Preset::eMap2D, Format::eR32Uint, Extent3D{ 2, 2, 1 }, Samples::e1);

	auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ici, std::span(data));

	// Acquire as an external resource with fragment sampling access
	auto acquired = acquire("external_img", *img, Access::eFragmentSampled);

	// Use the acquired image in a pass that just passes it through
	auto pass = make_pass("passthrough", [](CommandBuffer& cb, VUK_IA(Access::eFragmentSampled) input) {
		// Just pass through
		return input;
	});

	auto result = pass(acquired);

	// Download and verify the data
	size_t alignment = format_to_texel_block_size(*fut.format);
	size_t size = compute_image_size(*fut.format, *fut.extent);
	auto dst = *allocate_buffer<uint32_t>(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto dst_buf = discard("dst", *dst);
	auto res = download_buffer(copy(fut, std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
	auto updata = res->to_span();
	CHECK(updata == std::span(data));
}

TEST_CASE("image_view_in_pass") {
	auto data = { 10u, 20u, 30u, 40u };
	auto ici = from_preset(Preset::eMap2D, Format::eR32Uint, Extent3D{ 2, 2, 1 }, Samples::e1);

	auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ici, std::span(data));

	// Get the view and acquire it
	auto acquired_view = acquire("img_view", *img, Access::eFragmentSampled);

	// Pass the image view through a render pass
	auto pass = make_pass("test_pass", [](CommandBuffer& cb, VUK_IA(Access::eFragmentSampled) input) {
		// Just test that we can pass an image view through
		return input;
	});

	auto result = pass(acquired_view);

	// Download and verify the data
	size_t alignment = format_to_texel_block_size(*fut.format);
	size_t size = compute_image_size(*fut.format, *fut.extent);
	auto dst = *allocate_buffer<uint32_t>(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto dst_buf = discard("dst", *dst);
	auto res = download_buffer(copy(fut, std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
	auto updata = res->to_span();
	CHECK(updata == std::span(data));
}

// ============================================================================
// Sampler Integration
// ============================================================================

TEST_CASE("combine_image_sampler") {
	auto data = { 100u, 200u, 300u, 400u };
	auto ici = from_preset(Preset::eMap2D, Format::eR32Uint, Extent3D{ 2, 2, 1 }, Samples::e1);

	auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ici, std::span(data));

	// Get the view and acquire it
	auto acquired_view = acquire("img_view", *img, Access::eFragmentSampled);

	// Create a sampler
	SamplerCreateInfo sci{};
	sci.magFilter = Filter::eLinear;
	sci.minFilter = Filter::eLinear;

	auto sampler = acquire_sampler("test_sampler", sci);

	// Combine the image view with the sampler
	auto sampled_image = combine_image_sampler("combined", acquired_view, sampler);

	// Pass the combined sampled image through a pass
	auto pass = make_pass("use_sampled", [](CommandBuffer& cb, VUK_ARG(SampledImage, Access::eFragmentSampled) input) {
		// Just pass through to verify the combined image sampler works
		return input;
	});

	auto result = pass(sampled_image);

	// Download and verify the original data is still intact
	size_t alignment = format_to_texel_block_size(*fut.format);
	size_t size = compute_image_size(*fut.format, *fut.extent);
	auto dst = *allocate_buffer<uint32_t>(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto dst_buf = discard("dst", *dst);
	auto res = download_buffer(copy(fut, std::move(dst_buf))).get(*test_context.allocator, test_context.compiler);
	auto updata = res->to_span();
	CHECK(updata == std::span(data));
}