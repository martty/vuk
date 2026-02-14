#include "TestContext.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/Value.hpp"

using namespace vuk;

// =================================================
// Common Test Colors
// =================================================
namespace {
	// Black - for whole image background
	constexpr ClearColor kClearBlack{ 0, 0, 0, 255 };
	constexpr ImageLike<Format::eR8G8B8A8Uint> kPixelBlack{ 0, 0, 0, 255 };

	// Red - for subregion content
	constexpr ClearColor kClearRed{ 255, 0, 0, 255 };
	constexpr ImageLike<Format::eR8G8B8A8Uint> kPixelRed{ 255, 0, 0, 255 };
} // namespace

// =================================================
// Basic Runtime Sanity Checks
// =================================================
/*
TEST_CASE("imageview_subregion_metadata") {
  ICI ici = from_preset(Preset::eGeneric2D, Format::eR8G8B8A8Uint, Extent3D{ 1024, 1024, 1 }, SampleCountFlagBits::e1);
  auto image = *allocate_image(*test_context.allocator, ici);

  auto view = image->default_view().subregion(Offset3D{ 100, 100, 0 }, Extent3D{ 256, 256, 1 });

  auto& meta = view.get_meta();
  CHECK(meta.offset.x == 100);
  CHECK(meta.offset.y == 100);
  CHECK(meta.offset.z == 0);
  CHECK(meta.extent.width == 256);
  CHECK(meta.extent.height == 256);
  CHECK(meta.extent.depth == 1);
}

TEST_CASE("imageview_is_full_view") {
  TestContext ctx;

  ICI ici = from_preset(Preset::eGeneric2D, Format::eR8G8B8A8Uint, Extent3D{ 1024, 1024, 1 }, SampleCountFlagBits::e1);
  auto image = *allocate_image(*test_context.allocator, ici);
  auto base_image = image->default_view();

  CHECK(base_image.is_full_view());

  auto offset_view = base_image.subregion(Offset3D{ 100, 100, 0 }, Extent3D{ 924, 924, 1 });
  CHECK_FALSE(offset_view.is_full_view());

  auto full_view = base_image.subregion(Offset3D{ 0, 0, 0 }, Extent3D{ 1024, 1024, 1 });
  CHECK(full_view.is_full_view());

  auto mip_view = base_image.mip(0);
  CHECK_FALSE(mip_view.is_full_view());

  auto layer_view = base_image.layer(0);
  CHECK(layer_view.is_full_view());
}*/

// =================================================
// IR-based Subregion Operations
// =================================================

TEST_CASE("ir_subregion_clear") {
	ICI ici = from_preset(Preset::eGeneric2D, Format::eR8G8B8A8Uint, Extent3D{ 1024, 1024, 1 }, SampleCountFlagBits::e1);
	ici.level_count = 1;
	auto full_image = allocate<>("full_img", ici);
	clear_image(full_image, kClearBlack);

	auto subregion = full_image.subregion(Offset3D{ 256, 256, 0 }, Extent3D{ 512, 512, 1 });
	auto cleared = clear_image(subregion, kClearRed);

	auto extent = Extent3D{ 512, 512, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, kPixelRed);
	verify_image_data(cleared, std::span(expected_data), Format::eR8G8B8A8Uint, extent);

	// Verify surrounding area still black
	auto corner = full_image.subregion(Value<Offset3D>{ Offset3D{ 0, 0, 0 } }, Value<Extent3D>{ Extent3D{ 128, 128, 1 } });
	auto corner_extent = Extent3D{ 128, 128, 1 };
	size_t corner_pixel_count = corner_extent.width * corner_extent.height * corner_extent.depth;
	std::vector corner_expected(corner_pixel_count, kPixelBlack);
	verify_image_data(corner, std::span(corner_expected), Format::eR8G8B8A8Uint, corner_extent);
}

TEST_CASE("ir_subregion_copy") {
	ICI src_ici = from_preset(Preset::eGeneric2DUnmipped, Format::eR8G8B8A8Uint, Extent3D{ 512, 512, 1 }, SampleCountFlagBits::e1);
	auto src_image = allocate<>("src", src_ici);
	clear_image(src_image, kClearRed);

	ICI dst_ici = from_preset(Preset::eGeneric2DUnmipped, Format::eR8G8B8A8Uint, Extent3D{ 1024, 1024, 1 }, SampleCountFlagBits::e1);
	auto dst_image = allocate<>("dst", dst_ici);
	clear_image(dst_image, kClearBlack);

	auto dst_subregion = dst_image.subregion(Value<Offset3D>{ Offset3D{ 256, 256, 0 } }, Value<Extent3D>{ Extent3D{ 512, 512, 1 } });
	auto copied = copy(src_image, dst_subregion);

	auto extent = Extent3D{ 512, 512, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, kPixelRed);
	verify_image_data(copied, std::span(expected_data), Format::eR8G8B8A8Uint, extent);

	// Verify surrounding area still black
	auto left_area = dst_image.subregion(Value<Offset3D>{ Offset3D{ 0, 256, 0 } }, Value<Extent3D>{ Extent3D{ 128, 512, 1 } });
	auto left_extent = Extent3D{ 128, 512, 1 };
	size_t left_pixel_count = left_extent.width * left_extent.height * left_extent.depth;
	std::vector left_expected(left_pixel_count, kPixelBlack);
	verify_image_data(left_area, std::span(left_expected), Format::eR8G8B8A8Uint, left_extent);
}

TEST_CASE("ir_subregion_blit") {
	ICI src_ici = from_preset(Preset::eGeneric2DUnmipped, Format::eR8G8B8A8Uint, Extent3D{ 256, 256, 1 }, SampleCountFlagBits::e1);
	auto src_image = allocate<>("src", src_ici);
	clear_image(src_image, kClearRed);

	ICI dst_ici = from_preset(Preset::eGeneric2DUnmipped, Format::eR8G8B8A8Uint, Extent3D{ 2048, 2048, 1 }, SampleCountFlagBits::e1);
	auto dst_image = allocate<>("dst", dst_ici);
	clear_image(dst_image, kClearBlack);

	auto dst_subregion = dst_image.subregion(Value<Offset3D>{ Offset3D{ 512, 512, 0 } }, Value<Extent3D>{ Extent3D{ 1024, 1024, 1 } });
	auto blitted = blit_image(src_image, dst_subregion, Filter::eNearest);

	auto extent = Extent3D{ 1024, 1024, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, kPixelRed);
	verify_image_data(blitted, std::span(expected_data), Format::eR8G8B8A8Uint, extent);
}

TEST_CASE("ir_subregion_viewport_rendering") {
	ICI ici = from_preset(Preset::eGeneric2DUnmipped, Format::eR8G8B8A8Uint, Extent3D{ 1920, 1080, 1 }, SampleCountFlagBits::e1);
	auto framebuffer = allocate<>("fb", ici);
	clear_image(framebuffer, kClearBlack);

	auto left_viewport = framebuffer.subregion(Value<Offset3D>{ Offset3D{ 0, 0, 0 } }, Value<Extent3D>{ Extent3D{ 960, 1080, 1 } });
	auto left_cleared = clear_image(left_viewport, kClearBlack);

	auto right_viewport = framebuffer.subregion(Value<Offset3D>{ Offset3D{ 960, 0, 0 } }, Value<Extent3D>{ Extent3D{ 960, 1080, 1 } });
	auto right_cleared = clear_image(right_viewport, kClearRed);

	auto left_extent = Extent3D{ 960, 1080, 1 };
	size_t left_pixel_count = left_extent.width * left_extent.height * left_extent.depth;
	std::vector left_expected(left_pixel_count, kPixelBlack);
	verify_image_data(left_cleared, std::span(left_expected), Format::eR8G8B8A8Uint, left_extent);

	auto right_extent = Extent3D{ 960, 1080, 1 };
	size_t right_pixel_count = right_extent.width * right_extent.height * right_extent.depth;
	std::vector right_expected(right_pixel_count, kPixelRed);
	verify_image_data(right_cleared, std::span(right_expected), Format::eR8G8B8A8Uint, right_extent);
}

TEST_CASE("ir_subregion_with_mip") {
	ICI ici = from_preset(Preset::eGeneric2D, Format::eR8G8B8A8Uint, Extent3D{ 1024, 1024, 1 }, SampleCountFlagBits::e1);
	ici.level_count = 2;
	auto mipped_image = allocate<>("mipped", ici);
	clear_image(mipped_image.mip(0), kClearBlack);

	auto mip0_subregion = mipped_image.mip(0).subregion(Value<Offset3D>{ Offset3D{ 256, 256, 0 } }, Value<Extent3D>{ Extent3D{ 512, 512, 1 } });
	auto cleared = clear_image(mip0_subregion, kClearRed);

	auto extent = Extent3D{ 512, 512, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, kPixelRed);
	verify_image_data(cleared, std::span(expected_data), Format::eR8G8B8A8Uint, extent);

	// Verify area outside subregion still black
	auto outside = mipped_image.mip(0).subregion(Value<Offset3D>{ Offset3D{ 0, 0, 0 } }, Value<Extent3D>{ Extent3D{ 128, 128, 1 } });
	auto outside_extent = Extent3D{ 128, 128, 1 };
	size_t outside_pixel_count = outside_extent.width * outside_extent.height * outside_extent.depth;
	std::vector outside_expected(outside_pixel_count, kPixelBlack);
	verify_image_data(outside, std::span(outside_expected), Format::eR8G8B8A8Uint, outside_extent);
}
/*
//TODO: support 3d image clears
TEST_CASE("ir_subregion_3d_image") {
  ICI ici = from_preset(Preset::eMap3D, Format::eR8G8B8A8Uint, Extent3D{ 256, 256, 256 }, SampleCountFlagBits::e1);
  auto volume = allocate<>("volume", ici);
  clear_image(volume, kClearBlack);

  auto subvolume = volume.subregion(Value<Offset3D>{ Offset3D{ 64, 64, 64 } }, Value<Extent3D>{ Extent3D{ 128, 128, 128 } });
  auto cleared = clear_image(subvolume, kClearRed);

  auto extent = Extent3D{ 128, 128, 128 };
  size_t pixel_count = extent.width * extent.height * extent.depth;
  std::vector expected_data(pixel_count, kPixelRed);
  verify_image_data(cleared, std::span(expected_data), Format::eR8G8B8A8Uint, extent);

  // Verify surrounding volume still black
  auto corner = volume.subregion(Value<Offset3D>{ Offset3D{ 0, 0, 0 } }, Value<Extent3D>{ Extent3D{ 32, 32, 32 } });
  auto corner_extent = Extent3D{ 32, 32, 32 };
  size_t corner_pixel_count = corner_extent.width * corner_extent.height * corner_extent.depth;
  std::vector corner_expected(corner_pixel_count, kPixelBlack);
  verify_image_data(corner, std::span(corner_expected), Format::eR8G8B8A8Uint, corner_extent);
}*/

// =================================================
// Advanced Subregion Tests
// =================================================

TEST_CASE("ir_subregion_clear_usage_propagation") {
	// Create an image with minimal usage (just transfer) - no color attachment usage
	ICI ici;
	ici.format = Format::eR8G8B8A8Uint;
	ici.extent = Extent3D{ 1024, 1024, 1 };
	ici.sample_count = SampleCountFlagBits::e1;
	ici.level_count = 1;
	ici.layer_count = 1;
	ici.image_type = ImageType::e2D;
	ici.usage = ImageUsageFlagBits::eTransferDst | ImageUsageFlagBits::eTransferSrc;

	auto image = allocate<>("img", ici);

	// This clear on graphics queue should cause ColorAttachment usage to be added
	auto subregion = image.subregion(Offset3D{ 256, 256, 0 }, Extent3D{ 512, 512, 1 });
	auto cleared = clear_image(subregion, kClearRed);

	auto extent = Extent3D{ 512, 512, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, kPixelRed);
	verify_image_data(cleared, std::span(expected_data), Format::eR8G8B8A8Uint, extent);
}

TEST_CASE("ir_subregion_explicit_viewport_scissor") {
	ICI ici = from_preset(Preset::eGeneric2DUnmipped, Format::eR8G8B8A8Uint, Extent3D{ 1920, 1080, 1 }, SampleCountFlagBits::e1);
	auto framebuffer = allocate<>("fb", ici);
	clear_image(framebuffer, kClearBlack);

	// Create a subregion for the right half starting at x=960
	auto right_half = framebuffer.subregion(Value<Offset3D>{ Offset3D{ 960, 0, 0 } }, Value<Extent3D>{ Extent3D{ 960, 1080, 1 } });

	// Render to the subregion with explicit viewport and scissor
	auto pass = make_pass(
	    "viewport_test",
	    [](CommandBuffer& cbuf, VUK_IA(Access::eColorWrite) target) {
		    // Set viewport to cover the entire render area (which is the subregion)
		    // Coordinates are relative to the render area, so (0,0) is the start of the subregion
		    cbuf.set_viewport(0, Rect2D::absolute(0, 0, 960, 1080));
		    cbuf.set_scissor(0, Rect2D::absolute(0, 0, 960, 1080));

		    // Clear with attachment clear (should respect the renderpass area)
		    cbuf.clear_image(target, kClearRed);

		    return target;
	    },
	    DomainFlagBits::eGraphicsQueue);

	auto result = pass(std::move(right_half));

	auto extent = Extent3D{ 960, 1080, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, kPixelRed);
	verify_image_data(result, std::span(expected_data), Format::eR8G8B8A8Uint, extent);

	// Verify left half is still black
	auto left_half = framebuffer.subregion(Value<Offset3D>{ Offset3D{ 0, 0, 0 } }, Value<Extent3D>{ Extent3D{ 960, 1080, 1 } });
	std::vector left_expected(pixel_count, kPixelBlack);
	verify_image_data(left_half, std::span(left_expected), Format::eR8G8B8A8Uint, extent);
}

TEST_CASE("ir_subregion_multiqueue_clear") {
	// Create three separate images for different queue clears
	// (Transfer/compute queues require spanning views, so we can't use subregions on the same image)

	// Graphics queue clear - supports non-spanning subregions
	ICI graphics_ici = from_preset(Preset::eGeneric2DUnmipped, Format::eR8G8B8A8Uint, Extent3D{ 1024, 1024, 1 }, SampleCountFlagBits::e1);
	auto graphics_image = allocate<>("graphics_img", graphics_ici);
	clear_image(graphics_image, kClearBlack);

	auto graphics_subregion = graphics_image.subregion(Value<Offset3D>{ Offset3D{ 256, 256, 0 } }, Value<Extent3D>{ Extent3D{ 512, 512, 1 } });
	auto graphics_cleared = clear_image(graphics_subregion, kClearRed);
	graphics_cleared.schedule_on(DomainFlagBits::eGraphicsQueue);

	// Transfer queue clear - requires spanning view
	ICI transfer_ici = from_preset(Preset::eGeneric2DUnmipped, Format::eR8G8B8A8Uint, Extent3D{ 512, 512, 1 }, SampleCountFlagBits::e1);
	auto transfer_image = allocate<>("transfer_img", transfer_ici);
	auto transfer_cleared = clear_image(transfer_image, kClearRed);
	transfer_cleared.schedule_on(DomainFlagBits::eTransferQueue);

	// Compute queue clear - requires spanning view
	ICI compute_ici = from_preset(Preset::eGeneric2DUnmipped, Format::eR8G8B8A8Uint, Extent3D{ 512, 512, 1 }, SampleCountFlagBits::e1);
	auto compute_image = allocate<>("compute_img", compute_ici);
	auto compute_cleared = clear_image(compute_image, kClearRed);
	compute_cleared.schedule_on(DomainFlagBits::eComputeQueue);

	// Verify each queue's clear worked correctly
	auto extent = Extent3D{ 512, 512, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;

	std::vector graphics_expected(pixel_count, kPixelRed);
	verify_image_data(graphics_cleared, std::span(graphics_expected), Format::eR8G8B8A8Uint, extent);

	/* TODO: check for error
	std::vector transfer_expected(pixel_count, kPixelRed);
	verify_image_data(transfer_cleared, std::span(transfer_expected), Format::eR8G8B8A8Uint, extent);
	*/

	std::vector compute_expected(pixel_count, kPixelRed);
	verify_image_data(compute_cleared, std::span(compute_expected), Format::eR8G8B8A8Uint, extent);

	// Verify graphics image area outside subregion is still black
	auto outside = graphics_image.subregion(Value<Offset3D>{ Offset3D{ 0, 0, 0 } }, Value<Extent3D>{ Extent3D{ 128, 128, 1 } });
	auto outside_extent = Extent3D{ 128, 128, 1 };
	size_t outside_pixel_count = outside_extent.width * outside_extent.height * outside_extent.depth;
	std::vector outside_expected(outside_pixel_count, kPixelBlack);
	verify_image_data(outside, std::span(outside_expected), Format::eR8G8B8A8Uint, outside_extent);
}
