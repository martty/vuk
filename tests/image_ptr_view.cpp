#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <doctest/doctest.h>
#include <string_view>

using namespace vuk;

// ============================================================================
// Basic Image Pointer Tests
// ============================================================================

TEST_CASE("image ptr basic") {
	// Default construction
	{
		Image<> img;
		CHECK(!img);
		CHECK(img.device_address == 0);
	}

	// Bool conversion and comparison
	{
		auto ici = from_preset(Preset::eMap2D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);

		Image<> img1;
		Image<> img2;

		CHECK(img1 == img2);

		test_context.allocator->allocate_images(std::span{ &img1, 1 }, std::span{ &ici, 1 });

		CHECK(img1);
		CHECK(img1.device_address != 0);
		CHECK(img1 != img2);

		test_context.allocator->deallocate(std::span{ &img1, 1 });
	}

	// Format templating
	{
		Image<Format::eR8G8B8A8Unorm> typed_img;
		Image<> generic_img;

		CHECK(!typed_img);
		CHECK(!generic_img);
		CHECK(sizeof(typed_img) == sizeof(generic_img));
	}

	// ICI from preset
	{
		auto ici_map = from_preset(Preset::eMap2D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);

		CHECK(ici_map.format == Format::eR8G8B8A8Unorm);
		CHECK(ici_map.extent.width == 256);
		CHECK(ici_map.image_type == ImageType::e2D);
		CHECK(ici_map.layer_count == 1);
		CHECK(!!(ici_map.usage & ImageUsageFlagBits::eSampled));
		CHECK(!!(ici_map.usage & ImageUsageFlagBits::eTransferDst));

		auto ici_rtt = from_preset(Preset::eRTT2D, Format::eR8G8B8A8Unorm, Extent3D{ 512, 512, 1 }, Samples::e1);

		CHECK(!!(ici_rtt.usage & ImageUsageFlagBits::eColorAttachment));
		CHECK(!!(ici_rtt.usage & ImageUsageFlagBits::eSampled));
	}
}

// ============================================================================
// ImageView Tests
// ============================================================================

TEST_CASE("image_view basic") {
	CHECK(test_context.runtime->get_image_count() == 0);
	CHECK(test_context.runtime->get_active_image_view_count() == 0);

	// Default construction
	{
		ImageView<> view;
		CHECK(!view);
		CHECK(view.view_key == 0);
	}

	// From image
	{
		auto ici = from_preset(Preset::eMap2D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);

		Image<> img;
		test_context.allocator->allocate_images(std::span{ &img, 1 }, std::span{ &ici, 1 });

		auto view = img.default_view();
		CHECK(view);
		CHECK(view.view_key != 0);

		test_context.allocator->deallocate(std::span{ &img, 1 });
	}

	// Comparison
	{
		ImageView<> view1;
		ImageView<> view2;

		CHECK(view1 == view2);

		auto ici = from_preset(Preset::eMap2D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);

		Image<> img;
		test_context.allocator->allocate_images(std::span{ &img, 1 }, std::span{ &ici, 1 });

		view1 = img.default_view();
		CHECK(view1 != view2);

		test_context.allocator->deallocate(std::span{ &img, 1 });
	}

	// Format constraints
	{
		ImageView<Format::eR8G8B8A8Unorm> typed_view;
		ImageView<> generic_view;

		CHECK(!typed_view);
		CHECK(!generic_view);

		CHECK(typed_view.constraints == Format::eR8G8B8A8Unorm);
		CHECK(generic_view.constraints == Format::eUndefined);
	}
}

// ============================================================================
// Mip and Layer Manipulation
// ============================================================================

TEST_CASE("image_view_mip_selection") {
	CHECK(test_context.runtime->get_image_count() == 0);
	CHECK(test_context.runtime->get_active_image_view_count() == 0);
	auto ici = from_preset(Preset::eMap2D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);
	ici.level_count = 4; // 256->128->64->32

	Image<> img;
	test_context.allocator->allocate_images(std::span{ &img, 1 }, std::span{ &ici, 1 });

	auto base_view = img.default_view();
	auto mip1_view = base_view.mip(1);

	CHECK(mip1_view);
	CHECK(mip1_view != base_view);

	auto& meta = mip1_view.get_meta();
	CHECK(meta.base_level == 1);
	CHECK(meta.level_count == 1);

	test_context.allocator->deallocate(std::span{ &img, 1 });
}

TEST_CASE("image_view_mip_range") {
	CHECK(test_context.runtime->get_image_count() == 0);
	CHECK(test_context.runtime->get_active_image_view_count() == 0);
	auto ici = from_preset(Preset::eMap2D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);
	ici.level_count = 4;

	Image<> img;
	test_context.allocator->allocate_images(std::span{ &img, 1 }, std::span{ &ici, 1 });

	auto base_view = img.default_view();
	auto mip_range_view = base_view.mip_range(1, 2);

	auto& meta = mip_range_view.get_meta();
	CHECK(meta.base_level == 1);
	CHECK(meta.level_count == 2);

	test_context.allocator->deallocate(std::span{ &img, 1 });
}

TEST_CASE("image_view_layer_selection") {
	CHECK(test_context.runtime->get_image_count() == 0);
	CHECK(test_context.runtime->get_active_image_view_count() == 0);
	auto ici = from_preset(Preset::eMap2D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);
	ici.layer_count = 6; // Array texture

	Image<> img;
	test_context.allocator->allocate_images(std::span{ &img, 1 }, std::span{ &ici, 1 });

	auto base_view = img.default_view();
	auto layer2_view = base_view.layer(2);

	auto& meta = layer2_view.get_meta();
	CHECK(meta.base_layer == 2);
	CHECK(meta.layer_count == 1);

	test_context.allocator->deallocate(std::span{ &img, 1 });
}

TEST_CASE("image_view_layer_range") {
	CHECK(test_context.runtime->get_image_count() == 0);
	CHECK(test_context.runtime->get_active_image_view_count() == 0);
	auto ici = from_preset(Preset::eMap2D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);
	ici.layer_count = 6;

	Image<> img;
	test_context.allocator->allocate_images(std::span{ &img, 1 }, std::span{ &ici, 1 });

	auto base_view = img.default_view();
	auto layer_range_view = base_view.layer_range(1, 3);

	auto& meta = layer_range_view.get_meta();
	CHECK(meta.base_layer == 1);
	CHECK(meta.layer_count == 3);

	test_context.allocator->deallocate(std::span{ &img, 1 });
}

TEST_CASE("image_view_chaining") {
	CHECK(test_context.runtime->get_image_count() == 0);
	CHECK(test_context.runtime->get_active_image_view_count() == 0);
	auto ici = from_preset(Preset::eMap2D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);
	ici.level_count = 4;
	ici.layer_count = 6;

	Image<> img;
	test_context.allocator->allocate_images(std::span{ &img, 1 }, std::span{ &ici, 1 });

	auto view = img.default_view();
	auto chained_view = view.mip(1).layer(2);

	auto& meta = chained_view.get_meta();
	CHECK(meta.base_level == 1);
	CHECK(meta.base_layer == 2);
	CHECK(meta.level_count == 1);
	CHECK(meta.layer_count == 1);

	test_context.allocator->deallocate(std::span{ &img, 1 });
}

TEST_CASE("image_view_extent_calculation") {
	CHECK(test_context.runtime->get_image_count() == 0);
	CHECK(test_context.runtime->get_active_image_view_count() == 0);
	auto ici = from_preset(Preset::eMap2D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);
	ici.level_count = 4;

	Image<> img;
	test_context.allocator->allocate_images(std::span{ &img, 1 }, std::span{ &ici, 1 });

	auto base_view = img.default_view();
	auto extent0 = base_view.base_mip_extent();
	CHECK(extent0.width == 256);
	CHECK(extent0.height == 256);

	auto mip1_view = base_view.mip(1);
	auto extent1 = mip1_view.base_mip_extent();
	CHECK(extent1.width == 128);
	CHECK(extent1.height == 128);

	auto mip2_view = base_view.mip(2);
	auto extent2 = mip2_view.base_mip_extent();
	CHECK(extent2.width == 64);
	CHECK(extent2.height == 64);

	test_context.allocator->deallocate(std::span{ &img, 1 });
}

// ============================================================================
// Allocation and Deallocation
// ============================================================================

TEST_CASE("allocate_image_with_preset") {
	Allocator allocator(test_context.runtime->get_vk_resource());

	auto ici = from_preset(Preset::eMap2D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);

	auto& resolver = allocator.get_context();
	auto initial_image_count = resolver.get_image_count();
	auto initial_view_count = resolver.get_active_image_view_count();

	Image<> img;
	auto result = allocator.allocate_images(std::span{ &img, 1 }, std::span{ &ici, 1 });

	CHECK(result);
	CHECK(img);

	// Should have added 1 image and 1 default view
	CHECK(resolver.get_image_count() == initial_image_count + 1);
	CHECK(resolver.get_active_image_view_count() == initial_view_count + 1);

	allocator.deallocate(std::span{ &img, 1 });

	// Image and view should be removed
	CHECK(resolver.get_image_count() == initial_image_count);
	// View should be cleaned up
	CHECK(resolver.get_active_image_view_count() == initial_view_count);
}

TEST_CASE("unique_image_ownership") {
	Allocator allocator(test_context.runtime->get_vk_resource());

	auto ici = from_preset(Preset::eMap2D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);

	auto& resolver = allocator.get_context();
	auto initial_image_count = resolver.get_image_count();
	auto initial_view_count = resolver.get_active_image_view_count();

	uint64_t img_key = 0;
	{
		Unique<Image<>> img(allocator);
		allocator.allocate_images(std::span{ &*img, 1 }, std::span{ &ici, 1 });

		CHECK(*img);
		CHECK(resolver.get_image_count() == initial_image_count + 1);
		CHECK(resolver.get_active_image_view_count() == initial_view_count + 1);

		img_key = img->device_address;
		CHECK(img_key != 0);
		// Should automatically deallocate when going out of scope
	}

	// Image and view should be removed after scope ends
	CHECK(resolver.get_image_count() == initial_image_count);
	CHECK(resolver.get_active_image_view_count() == initial_view_count);

	// Allocate a new image - should reuse the key from colony freelist
	Image<> img2;
	allocator.allocate_images(std::span{ &img2, 1 }, std::span{ &ici, 1 });
	CHECK(img2.device_address == img_key);

	allocator.deallocate(std::span{ &img2, 1 });
}

TEST_CASE("unique_image_view_ownership") {
	Allocator allocator(test_context.runtime->get_vk_resource());

	auto ici = from_preset(Preset::eMap2D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);
	ici.level_count = 4; // Need multiple mip levels

	auto& resolver = allocator.get_context();
	auto initial_active_view_count = resolver.get_active_image_view_count();

	Unique<Image<>> img(allocator);
	allocator.allocate_images(std::span{ &*img, 1 }, std::span{ &ici, 1 });

	// Default view is created automatically
	CHECK(resolver.get_active_image_view_count() == initial_active_view_count + 1);

	uint32_t view_key = 0;
	{
		// Create a mip view (non-default view)
		Unique<ImageView<>> view(allocator, img->default_view().mip(1));
		CHECK(*view);
		view_key = view->view_key;
		CHECK(view_key != 0);

		// Now we have default view + mip view
		CHECK(resolver.get_active_image_view_count() == initial_active_view_count + 2);
		// Should automatically deallocate when going out of scope
	}

	// Only default view should remain active
	CHECK(resolver.get_active_image_view_count() == initial_active_view_count + 1);

	// Create the same mip view again - should reuse the key from freelist
	auto new_view = img->default_view().mip(1);
	CHECK(new_view.view_key == view_key);

	// Now we have default view + mip view active again
	CHECK(resolver.get_active_image_view_count() == initial_active_view_count + 2);
}