#include "TestContext.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/Value.hpp"

using namespace vuk;

// =================================================
// Basic Runtime Sanity Checks
// =================================================
/*
TEST_CASE("imageview_subregion_metadata") {
  ICI ici = from_preset(Preset::eGeneric2D, Format::eR8G8B8A8Unorm, Extent3D{ 1024, 1024, 1 }, SampleCountFlagBits::e1);
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

  ICI ici = from_preset(Preset::eGeneric2D, Format::eR8G8B8A8Unorm, Extent3D{ 1024, 1024, 1 }, SampleCountFlagBits::e1);
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
	ICI ici = from_preset(Preset::eGeneric2D, Format::eR8G8B8A8Unorm, Extent3D{ 1024, 1024, 1 }, SampleCountFlagBits::e1);
	ici.level_count = 1;
	auto full_image = allocate<>("full_img", ici);
	clear_image(full_image, ClearColor{ 0.0f, 0.0f, 0.0f, 1.0f });

	auto subregion = full_image.subregion(Offset3D{ 256, 256, 0 }, Extent3D{ 512, 512, 1 });
	auto cleared = clear_image(subregion, ClearColor{ 1.0f, 0.0f, 0.0f, 1.0f });

	auto extent = Extent3D{ 512, 512, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 1.0f, 0.0f, 0.0f, 1.0f });
	verify_image_data(cleared, std::span(expected_data), Format::eR8G8B8A8Unorm, extent);

	// Verify surrounding area still black
	auto corner = full_image.subregion(Value<Offset3D>{ Offset3D{ 0, 0, 0 } }, Value<Extent3D>{ Extent3D{ 128, 128, 1 } });
	auto corner_extent = Extent3D{ 128, 128, 1 };
	size_t corner_pixel_count = corner_extent.width * corner_extent.height * corner_extent.depth;
	std::vector corner_expected(corner_pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.0f, 0.0f, 0.0f, 1.0f });
	verify_image_data(corner, std::span(corner_expected), Format::eR8G8B8A8Unorm, corner_extent);
}

TEST_CASE("ir_subregion_copy") {
	ICI src_ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 512, 512, 1 }, SampleCountFlagBits::e1);
	auto src_image = allocate<>("src", src_ici);
	clear_image(src_image, ClearColor{ 0.5f, 0.3f, 0.7f, 1.0f });

	ICI dst_ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 1024, 1024, 1 }, SampleCountFlagBits::e1);
	auto dst_image = allocate<>("dst", dst_ici);
	clear_image(dst_image, ClearColor{ 0.0f, 0.0f, 0.0f, 1.0f });

	auto dst_subregion = dst_image.subregion(Value<Offset3D>{ Offset3D{ 256, 256, 0 } }, Value<Extent3D>{ Extent3D{ 512, 512, 1 } });
	auto copied = copy(src_image, dst_subregion);

	auto extent = Extent3D{ 512, 512, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.5f, 0.3f, 0.7f, 1.0f });
	verify_image_data(copied, std::span(expected_data), Format::eR8G8B8A8Unorm, extent);

	// Verify surrounding area still black
	auto left_area = dst_image.subregion(Value<Offset3D>{ Offset3D{ 0, 256, 0 } }, Value<Extent3D>{ Extent3D{ 128, 512, 1 } });
	auto left_extent = Extent3D{ 128, 512, 1 };
	size_t left_pixel_count = left_extent.width * left_extent.height * left_extent.depth;
	std::vector left_expected(left_pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.0f, 0.0f, 0.0f, 1.0f });
	verify_image_data(left_area, std::span(left_expected), Format::eR8G8B8A8Unorm, left_extent);
}

TEST_CASE("ir_subregion_blit") {
	ICI src_ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, SampleCountFlagBits::e1);
	auto src_image = allocate<>("src", src_ici);
	clear_image(src_image, ClearColor{ 0.2f, 0.8f, 0.4f, 1.0f });

	ICI dst_ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 2048, 2048, 1 }, SampleCountFlagBits::e1);
	auto dst_image = allocate<>("dst", dst_ici);
	clear_image(dst_image, ClearColor{ 0.0f, 0.0f, 0.0f, 1.0f });

	auto dst_subregion = dst_image.subregion(Value<Offset3D>{ Offset3D{ 512, 512, 0 } }, Value<Extent3D>{ Extent3D{ 1024, 1024, 1 } });
	auto blitted = blit_image(src_image, dst_subregion, Filter::eLinear);

	auto extent = Extent3D{ 1024, 1024, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.2f, 0.8f, 0.4f, 1.0f });
	verify_image_data(blitted, std::span(expected_data), Format::eR8G8B8A8Unorm, extent);
}

TEST_CASE("ir_subregion_viewport_rendering") {
	ICI ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 1920, 1080, 1 }, SampleCountFlagBits::e1);
	auto framebuffer = allocate<>("fb", ici);
	clear_image(framebuffer, ClearColor{ 0.0f, 0.0f, 0.0f, 1.0f });

	auto left_viewport = framebuffer.subregion(Value<Offset3D>{ Offset3D{ 0, 0, 0 } }, Value<Extent3D>{ Extent3D{ 960, 1080, 1 } });
	auto left_cleared = clear_image(left_viewport, ClearColor{ 1.0f, 0.0f, 0.0f, 1.0f });

	auto right_viewport = framebuffer.subregion(Value<Offset3D>{ Offset3D{ 960, 0, 0 } }, Value<Extent3D>{ Extent3D{ 960, 1080, 1 } });
	auto right_cleared = clear_image(right_viewport, ClearColor{ 0.0f, 1.0f, 0.0f, 1.0f });

	auto left_extent = Extent3D{ 960, 1080, 1 };
	size_t left_pixel_count = left_extent.width * left_extent.height * left_extent.depth;
	std::vector left_expected(left_pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 1.0f, 0.0f, 0.0f, 1.0f });
	verify_image_data(left_cleared, std::span(left_expected), Format::eR8G8B8A8Unorm, left_extent);

	auto right_extent = Extent3D{ 960, 1080, 1 };
	size_t right_pixel_count = right_extent.width * right_extent.height * right_extent.depth;
	std::vector right_expected(right_pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.0f, 1.0f, 0.0f, 1.0f });
	verify_image_data(right_cleared, std::span(right_expected), Format::eR8G8B8A8Unorm, right_extent);
}

TEST_CASE("ir_subregion_tiled_rendering") {
	ICI ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 2048, 2048, 1 }, SampleCountFlagBits::e1);
	auto render_target = allocate<>("rt", ici);
	clear_image(render_target, ClearColor{ 0.0f, 0.0f, 0.0f, 1.0f });

	auto tile_0_0 = render_target.subregion(Value<Offset3D>{ Offset3D{ 0, 0, 0 } }, Value<Extent3D>{ Extent3D{ 1024, 1024, 1 } });
	auto cleared_0_0 = clear_image(tile_0_0, ClearColor{ 1.0f, 0.0f, 0.0f, 1.0f });

	auto tile_1_1 = render_target.subregion(Value<Offset3D>{ Offset3D{ 1024, 1024, 0 } }, Value<Extent3D>{ Extent3D{ 1024, 1024, 1 } });
	auto cleared_1_1 = clear_image(tile_1_1, ClearColor{ 0.0f, 1.0f, 0.0f, 1.0f });

	auto extent = Extent3D{ 1024, 1024, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;

	std::vector expected_0_0(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 1.0f, 0.0f, 0.0f, 1.0f });
	verify_image_data(cleared_0_0, std::span(expected_0_0), Format::eR8G8B8A8Unorm, extent);

	std::vector expected_1_1(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.0f, 1.0f, 0.0f, 1.0f });
	verify_image_data(cleared_1_1, std::span(expected_1_1), Format::eR8G8B8A8Unorm, extent);

	// Verify untouched tiles still black
	auto tile_0_1 = render_target.subregion(Value<Offset3D>{ Offset3D{ 0, 1024, 0 } }, Value<Extent3D>{ Extent3D{ 1024, 1024, 1 } });
	std::vector expected_0_1(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.0f, 0.0f, 0.0f, 1.0f });
	verify_image_data(tile_0_1, std::span(expected_0_1), Format::eR8G8B8A8Unorm, extent);
}

TEST_CASE("ir_subregion_with_mip") {
	ICI ici = from_preset(Preset::eRTT2D, Format::eR8G8B8A8Unorm, Extent3D{ 1024, 1024, 1 }, SampleCountFlagBits::e1);
	ici.level_count = 2;
	auto mipped_image = allocate<>("mipped", ici);
	clear_image(mipped_image.mip(0), ClearColor{ 0.0f, 0.0f, 0.0f, 1.0f });

	auto mip0_subregion = mipped_image.mip(0).subregion(Value<Offset3D>{ Offset3D{ 256, 256, 0 } }, Value<Extent3D>{ Extent3D{ 512, 512, 1 } });
	auto cleared = clear_image(mip0_subregion, ClearColor{ 0.9f, 0.5f, 0.1f, 1.0f });

	auto extent = Extent3D{ 512, 512, 1 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.9f, 0.5f, 0.1f, 1.0f });
	verify_image_data(cleared, std::span(expected_data), Format::eR8G8B8A8Unorm, extent);

	// Verify area outside subregion still black
	auto outside = mipped_image.mip(0).subregion(Value<Offset3D>{ Offset3D{ 0, 0, 0 } }, Value<Extent3D>{ Extent3D{ 128, 128, 1 } });
	auto outside_extent = Extent3D{ 128, 128, 1 };
	size_t outside_pixel_count = outside_extent.width * outside_extent.height * outside_extent.depth;
	std::vector outside_expected(outside_pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.0f, 0.0f, 0.0f, 1.0f });
	verify_image_data(outside, std::span(outside_expected), Format::eR8G8B8A8Unorm, outside_extent);
}

TEST_CASE("ir_subregion_3d_image") {
	ICI ici = from_preset(Preset::eMap3D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 256 }, SampleCountFlagBits::e1);
	auto volume = allocate<>("volume", ici);
	clear_image(volume, ClearColor{ 0.0f, 0.0f, 0.0f, 1.0f });

	auto subvolume = volume.subregion(Value<Offset3D>{ Offset3D{ 64, 64, 64 } }, Value<Extent3D>{ Extent3D{ 128, 128, 128 } });
	auto cleared = clear_image(subvolume, ClearColor{ 0.6f, 0.4f, 0.8f, 1.0f });

	auto extent = Extent3D{ 128, 128, 128 };
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector expected_data(pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.6f, 0.4f, 0.8f, 1.0f });
	verify_image_data(cleared, std::span(expected_data), Format::eR8G8B8A8Unorm, extent);

	// Verify surrounding volume still black
	auto corner = volume.subregion(Value<Offset3D>{ Offset3D{ 0, 0, 0 } }, Value<Extent3D>{ Extent3D{ 32, 32, 32 } });
	auto corner_extent = Extent3D{ 32, 32, 32 };
	size_t corner_pixel_count = corner_extent.width * corner_extent.height * corner_extent.depth;
	std::vector corner_expected(corner_pixel_count, ImageLike<Format::eR8G8B8A8Unorm>{ 0.0f, 0.0f, 0.0f, 1.0f });
	verify_image_data(corner, std::span(corner_expected), Format::eR8G8B8A8Unorm, corner_extent);
}
