#include "TestContext.hpp"
#include "vuk/ir/IRPass.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <algorithm>
#include <doctest/doctest.h>
#include <string_view>

using namespace vuk;

// =================================================
// IR-based Image Allocation Tests With Inference
// =================================================

TEST_CASE("ir_allocate_image_resolve_operation") {
	// Multisampled source
	ICI ms_ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 512, 512, 1 }, Samples::e4);
	auto ms_view = allocate<>("ms_img", ms_ici);
	clear_image(ms_view, ClearColor{ 0.2f, 0.2f, 0.2f, 0.2f });

	// Single-sampled destination
	ICI ss_ici = {};
	auto ss_view = allocate<>("ss_img", ss_ici);

	auto resolved = resolve_into(ms_view, ss_view);

	// Download and verify all pixels are cleared to the expected value
	auto extent = Extent3D{ 512, 512, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.2f, 0.2f, 0.2f, 0.2f });
	verify_image_data(resolved, std::span(expected_data), Format::eR8G8B8A8Unorm, Extent3D{ 512, 512, 1 }, { .dump_graph = true });
}

TEST_CASE("ir_allocate_image_blit_operation") {
	// Source image with known parameters
	ICI src_ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);
	auto src_view = allocate<>("src_img", src_ici);
	clear_image(src_view, ClearColor{ 0.5f, 0.5f, 0.5f, 0.5f });

	// Destination with different size - inference should derive format and sample count
	ICI dst_ici = {};
	dst_ici.extent = Extent3D{ 512, 512, 1 };
	auto dst_view = allocate<>("dst_img", dst_ici);

	auto blitted = blit_image(src_view, dst_view, Filter::eLinear);

	// Verify the blit operation completes successfully
	auto extent = Extent3D{ 512, 512, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.5f, 0.5f, 0.5f, 0.5f });
	verify_image_data(blitted, std::span(expected_data), Format::eR8G8B8A8Unorm, extent, { .dump_graph = true });
}

TEST_CASE("ir_allocate_image_copy_operation") {
	// Source image with known parameters
	ICI src_ici = from_preset(Preset::eRTT2DUnmipped, Format::eR32G32B32A32Sfloat, Extent3D{ 128, 128, 1 }, Samples::e1);
	auto src_view = allocate<>("src_img", src_ici);
	clear_image(src_view, ClearColor{ 1.0f, 0.0f, 0.0f, 1.0f });

	// Destination with empty ICI - inference should derive all parameters
	ICI dst_ici = {};
	auto dst_view = allocate<>("dst_img", dst_ici);

	auto copied = copy(src_view, dst_view);

	// Verify the copy operation completes successfully
	auto extent = Extent3D{ 128, 128, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR32G32B32A32Sfloat>{ 1.0f, 0.0f, 0.0f, 1.0f });
	verify_image_data(copied, std::span(expected_data), Format::eR32G32B32A32Sfloat, extent, { .dump_graph = true });
}

TEST_CASE("ir_allocate_image_same_format_constraint") {
	// Source image
	ICI src_ici = from_preset(Preset::eRTT2DUnmipped, Format::eR16G16B16A16Sfloat, Extent3D{ 64, 64, 1 }, Samples::e1);
	auto src_view = allocate<>("src_img", src_ici);
	clear_image(src_view, ClearColor{ 0.75f, 0.25f, 0.5f, 1.0f });

	// Destination with explicit extent but no format
	ICI dst_ici = {};
	dst_ici.extent = Extent3D{ 64, 64, 1 };
	auto dst_view = allocate<>("dst_img", dst_ici);
	dst_view.same_format_as(src_view);

	auto copied = copy(src_view, dst_view);

	// Verify format was inferred correctly
	auto extent = Extent3D{ 64, 64, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR16G16B16A16Sfloat>{ 0.75f, 0.25f, 0.5f, 1.0f });
	verify_image_data(copied, std::span(expected_data), Format::eR16G16B16A16Sfloat, extent, { .dump_graph = true });
}

TEST_CASE("ir_allocate_image_same_extent_constraint") {
	// Source image
	ICI src_ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 256, 128, 1 }, Samples::e1);
	auto src_view = allocate<>("src_img", src_ici);
	clear_image(src_view, ClearColor{ 0.3f, 0.6f, 0.9f, 1.0f });

	// Destination with format but no extent
	ICI dst_ici = {};
	dst_ici.format = Format::eR8G8B8A8Unorm;
	auto dst_view = allocate<>("dst_img", dst_ici);
	dst_view.same_extent_as(src_view);

	auto copied = copy(src_view, dst_view);

	// Verify extent was inferred correctly
	auto extent = Extent3D{ 256, 128, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.3f, 0.6f, 0.9f, 1.0f });
	verify_image_data(copied, std::span(expected_data), Format::eR8G8B8A8Unorm, extent, { .dump_graph = true });
}

TEST_CASE("ir_allocate_image_same_shape_constraint") {
	// Source image with multiple mip levels
	ICI src_ici = from_preset(Preset::eRTT2DMipped, Format::eR8G8B8A8Unorm, Extent3D{ 128, 128, 1 }, Samples::e1);
	src_ici.mipLevels = 4;
	auto src_view = allocate<>("src_img", src_ici);
	clear_image(src_view, ClearColor{ 0.1f, 0.2f, 0.3f, 0.4f });

	// Destination with format only
	ICI dst_ici = {};
	dst_ici.format = Format::eR8G8B8A8Unorm;
	auto dst_view = allocate<>("dst_img", dst_ici);
	dst_view.same_shape_as(src_view);

	auto copied = copy(src_view, dst_view);

	// Verify shape (extent, layers, levels) was inferred correctly
	auto extent = Extent3D{ 128, 128, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.1f, 0.2f, 0.3f, 0.4f });
	verify_image_data(copied, std::span(expected_data), Format::eR8G8B8A8Unorm, extent, { .dump_graph = true });
}

TEST_CASE("ir_allocate_image_framebuffer_attachments") {
	// Create a render pass with color and depth attachments
	// Color attachment has known parameters
	ICI color_ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 512, 512, 1 }, Samples::e1);
	auto color_view = allocate<>("color_att", color_ici);

	// Depth attachment infers extent and samples from color attachment via framebuffer constraints
	ICI depth_ici = {};
	depth_ici.format = Format::eD32Sfloat;
	auto depth_view = allocate<>("depth_att", depth_ici);
	depth_view.same_extent_as(color_view);

	// Simple render pass that uses both attachments
	auto render = make_pass(
	    "render_with_depth",
	    [](CommandBuffer& cbuf, VUK_IA(Access::eColorWrite) color, VUK_IA(Access::eDepthStencilWrite) depth) {
		    // Use fullscreen triangle to fill color attachment instead of clear_image
		    cbuf.set_viewport(0, Rect2D::framebuffer());
		    cbuf.set_scissor(0, Rect2D::framebuffer());
		    cbuf.set_rasterization({});
		    cbuf.set_color_blend(color, {});
		    
		    // Clear depth with a dedicated operation (depth clears are allowed)
		    cbuf.clear_image(depth, Clear{ .depth = 1.0f });
		    
		    return color;
	    },
	    DomainFlagBits::eGraphicsQueue);

	// Use the fullscreen helper to render the color
	auto with_color = render_fullscreen_color(color_view, { 0.8f, 0.4f, 0.2f, 1.0f });
	auto result = render(with_color, depth_view);

	// Verify the color attachment
	auto extent = Extent3D{ 512, 512, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.8f, 0.4f, 0.2f, 1.0f });
	verify_image_data(result, std::span(expected_data), Format::eR8G8B8A8Unorm, extent, { .dump_graph = true });
}

TEST_CASE("ir_allocate_image_clear_operation") {
	// Image with partial ICI - needs inference
	ICI ici = {};
	ici.extent = Extent3D{ 128, 128, 1 };
	ici.format = Format::eR8G8B8A8Unorm;
	auto view = allocate<>("clear_target", ici);

	auto cleared = clear_image(view, ClearColor{ 0.25f, 0.5f, 0.75f, 1.0f });

	// Verify clear succeeded
	auto extent = Extent3D{ 128, 128, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.25f, 0.5f, 0.75f, 1.0f });
	verify_image_data(cleared, std::span(expected_data), Format::eR8G8B8A8Unorm, extent, { .dump_graph = true });
}

TEST_CASE("ir_allocate_image_generate_mips") {
	// Base mip level with known parameters
	ICI src_ici = from_preset(Preset::eRTT2DMipped, Format::eR8G8B8A8Unorm, Extent3D{ 512, 512, 1 }, Samples::e1);
	src_ici.mipLevels = 4;
	auto src_view = allocate<>("mipped_img", src_ici);
	clear_image(src_view.mip(0), ClearColor{ 0.9f, 0.1f, 0.5f, 1.0f });

	// Generate mips - this involves multiple blits with inferred parameters
	auto result = generate_mips(src_view, 0, 3);

	// Verify base mip level
	auto extent = Extent3D{ 512, 512, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.9f, 0.1f, 0.5f, 1.0f });
	verify_image_data(result.mip(0), std::span(expected_data), Format::eR8G8B8A8Unorm, extent, { .dump_graph = true });
}

TEST_CASE("ir_allocate_image_chain_inference") {
	// Test inference chain: src -> intermediate -> dst
	ICI src_ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 64, 64, 1 }, Samples::e1);
	auto src_view = allocate<>("src", src_ici);
	clear_image(src_view, ClearColor{ 0.4f, 0.5f, 0.6f, 0.7f });

	// Intermediate infers from source
	ICI mid_ici = {};
	auto mid_view = allocate<>("mid", mid_ici);
	mid_view.same_format_as(src_view);
	mid_view.same_extent_as(src_view);

	// Destination infers from intermediate
	ICI dst_ici = {};
	auto dst_view = allocate<>("dst", dst_ici);
	dst_view.same_format_as(mid_view);
	dst_view.same_extent_as(mid_view);

	auto step1 = copy(src_view, mid_view);
	auto step2 = copy(step1, dst_view);

	// Verify final result
	auto extent = Extent3D{ 64, 64, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.4f, 0.5f, 0.6f, 0.7f });
	verify_image_data(step2, std::span(expected_data), Format::eR8G8B8A8Unorm, extent, { .dump_graph = true });
}

TEST_CASE("ir_allocate_image_multiple_framebuffer_attachments") {
	// Test inference with multiple color attachments
	ICI color0_ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);
	auto color0 = allocate<>("color0", color0_ici);

	// Second color attachment infers extent from first
	ICI color1_ici = {};
	color1_ici.format = Format::eR16G16B16A16Sfloat;
	auto color1 = allocate<>("color1", color1_ici);
	color1.same_extent_as(color0);

	// Depth attachment infers extent from color0
	ICI depth_ici = {};
	depth_ici.format = Format::eD24UnormS8Uint;
	auto depth = allocate<>("depth", depth_ici);
	depth.same_extent_as(color0);

	// Fill color attachments with fullscreen triangles before the multi-attachment pass
	auto filled_color0 = render_fullscreen_color(color0, { 1.0f, 0.0f, 0.0f, 1.0f });
	auto filled_color1 = render_fullscreen_color(color1, { 0.0f, 1.0f, 0.0f, 1.0f });

	auto render = make_pass(
	    "multi_attachment_pass",
	    [](CommandBuffer& cbuf,
	       VUK_IA(Access::eColorWrite) c0,
	       VUK_IA(Access::eColorWrite) c1,
	       VUK_IA(Access::eDepthStencilWrite) d) {
		    // Only clear depth within the render pass
		    cbuf.clear_image(d, Clear{ .depth = 0.5f, .stencil = 0 });
		    // Color attachments are already filled by previous passes
		    return c0;
	    },
	    DomainFlagBits::eGraphicsQueue);

	auto result = render(filled_color0, filled_color1, depth);

	// Verify first color attachment
	auto extent = Extent3D{ 256, 256, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 1.0f, 0.0f, 0.0f, 1.0f });
	verify_image_data(result, std::span(expected_data), Format::eR8G8B8A8Unorm, extent, { .dump_graph = true });
}