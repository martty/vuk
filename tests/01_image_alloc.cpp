#include "TestContext.hpp"
#include "vuk/ir/IRPass.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <algorithm>
#include <doctest/doctest.h>
#include <string_view>

using namespace vuk;

// ============================================================================
// Helper Functions
// ============================================================================

// Helper to verify image contents match expected data
template<typename T>
void verify_image_data(Value<ImageView<>> image, std::span<T> expected_data, Format format, Extent3D extent) {
	size_t alignment = format_to_texel_block_size(format);
	size_t size = compute_image_size(format, extent);
	auto download_buf = *allocate_buffer<T>(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto download_buf_value = discard("verify_download", *download_buf);
	auto res = download_buffer(copy(image, std::move(download_buf_value))).get(*test_context.allocator, test_context.compiler);
	auto actual_data = res->to_span();
	CHECK(std::equal(actual_data.begin(), actual_data.end(), expected_data.begin()));
}

// Helper to clear an image and verify it executes successfully with expected clear value
template<typename T>
void clear_and_verify(Value<ImageView<>> image, Clear clear_value, Format format, Extent3D extent, T expected_clear_value) {
	auto cleared = clear_image(image, clear_value);

	// Download and verify all pixels are cleared to the expected value
	size_t pixel_count = extent.width * extent.height * extent.depth;
	std::vector<T> expected_data(pixel_count, expected_clear_value);
	verify_image_data(cleared, std::span(expected_data), format, extent);
}

// Helper to clear an image and verify the clear value with custom expected data
template<typename T>
void clear_and_verify_data(Value<ImageView<>> image, Clear clear_value, Format format, Extent3D extent, std::span<const T> expected_data) {
	auto cleared = clear_image(image, clear_value);
	verify_image_data(cleared, expected_data, format, extent);
}

// ============================================================================
// IR-based Image Allocation and Parameter Inference Tests
// ============================================================================

TEST_CASE("ir_allocate_image_basic") {
	// Create ICI and allocate image entirely in IR
	ICI ici = {};
	ici.format = Format::eR8G8B8A8Unorm;
	ici.extent = Extent3D{ 256, 256, 1 };
	ici.sample_count = Samples::e1;
	ici.usage = ImageUsageFlagBits::eSampled | ImageUsageFlagBits::eTransferDst | ImageUsageFlagBits::eTransferSrc;
	ici.level_count = 1;
	ici.layer_count = 1;

	auto ici_value = make_constant("test_ici", ici);
	auto view_value = allocate<>("test_img", ici_value);

	// Clear and execute to verify allocation worked
	ClearColor clear_value{ 0.0f, 0.0f, 0.0f, 1.0f };
	// Expected: R8G8B8A8Unorm black = (0, 0, 0, 255)
	uint32_t expected_pixel = 0xFF000000; // ABGR format in memory
	clear_and_verify(view_value, clear_value, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, expected_pixel);
}

TEST_CASE("ir_allocate_image_infer_from_copy_source") {
	// Create source image with data
	auto data = { 1u, 2u, 3u, 4u };
	auto src_ici = from_preset(Preset::eGeneric2D, Format::eR32Uint, Extent3D{ 2, 2, 1 }, Samples::e1);
	src_ici.level_count = 1;
	auto [src_view, src_fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, src_ici, std::span(data));

	// Create destination ICI with parameters that should match source
	ICI dst_ici = {};
	dst_ici.format = Format::eR32Uint;
	dst_ici.extent = Extent3D{ 2, 2, 1 };
	dst_ici.sample_count = Samples::e1;
	dst_ici.usage = ImageUsageFlagBits::eSampled | ImageUsageFlagBits::eTransferDst | ImageUsageFlagBits::eTransferSrc;
	dst_ici.level_count = 1;
	dst_ici.layer_count = 1;

	// Allocate destination image in IR
	auto dst_ici_value = make_constant("dst_ici", dst_ici);
	auto dst_view_value = allocate<>("dst_img", dst_ici_value);

	// Copy from source to destination and verify
	auto copied = copy(src_fut, dst_view_value);
	verify_image_data(copied, std::span(data), Format::eR32Uint, Extent3D{ 2, 2, 1 });
}

TEST_CASE("ir_allocate_image_infer_extent_from_copy") {
	// Create source with specific dimensions
	auto src_ici = from_preset(Preset::eGeneric2D, Format::eR16G16B16A16Sfloat, Extent3D{ 64, 64, 1 }, Samples::e1);
	src_ici.level_count = 1;
	auto src_img = *allocate_image(*test_context.allocator, src_ici);
	auto src_view = discard("src", src_img->default_view());

	// Allocate destination in IR with extent that will be validated during copy
	ICI dst_ici = {};
	dst_ici.format = Format::eR16G16B16A16Sfloat;
	dst_ici.extent = Extent3D{ 64, 64, 1 }; // Must match source for copy
	dst_ici.sample_count = Samples::e1;
	dst_ici.usage = ImageUsageFlagBits::eSampled | ImageUsageFlagBits::eTransferDst;
	dst_ici.level_count = 1;
	dst_ici.layer_count = 1;

	auto dst_ici_value = make_constant("dst_ici", dst_ici);
	auto dst_view_value = allocate<>("dst_img", dst_ici_value);

	// Copy validates extent compatibility
	auto copied = copy(src_view, dst_view_value);
	auto result = copied.wait(*test_context.allocator, test_context.compiler);
	REQUIRE(result);
}

TEST_CASE("ir_allocate_image_chain_copy") {
	// Create source
	auto data = { 10u, 20u, 30u, 40u };
	auto src_ici = from_preset(Preset::eGeneric2D, Format::eR32Uint, Extent3D{ 2, 2, 1 }, Samples::e1);
	src_ici.level_count = 1;
	auto [src_view, src_fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, src_ici, std::span(data));

	// First intermediate - allocated in IR
	auto int1_ici_value = make_constant("int1_ici", src_ici);
	auto int1_view = allocate<>("int1_img", int1_ici_value);
	auto copied1 = copy(src_fut, int1_view);

	// Second intermediate - allocated in IR, parameters inferred from chain
	auto int2_ici_value = make_constant("int2_ici", src_ici);
	auto int2_view = allocate<>("int2_img", int2_ici_value);
	auto copied2 = copy(copied1, int2_view);

	// Final destination
	auto dst_ici_value = make_constant("dst_ici", src_ici);
	auto dst_view = allocate<>("dst_img", dst_ici_value);
	auto final_copy = copy(copied2, dst_view);

	// Verify data propagated through the chain
	verify_image_data(final_copy, std::span(data), Format::eR32Uint, Extent3D{ 2, 2, 1 });
}

TEST_CASE("ir_allocate_image_clear_verify") {
	// Allocate image in IR
	ICI ici = from_preset(Preset::eGeneric2D, Format::eR8G8B8A8Unorm, Extent3D{ 2, 2, 1 }, Samples::e1);
	ici.level_count = 1;
	auto ici_value = make_constant("clear_test_ici", ici);
	auto view = allocate<>("clear_test_img", ici_value);

	// Clear to a specific color and verify
	ClearColor clear_value{ 0.5f, 0.25f, 0.75f, 1.0f };

	// Expected data: RGBA8 Unorm with values (128, 64, 191, 255)
	auto value = 255u << 24 | 191u << 16 | 64u << 8 | 128u;
	auto expected_data = { value, value, value, value };
	clear_and_verify_data(view, clear_value, Format::eR8G8B8A8Unorm, Extent3D{ 2, 2, 1 }, std::span(expected_data));
}

struct RGBA32F {
	float r, g, b, a;

	constexpr bool operator==(const RGBA32F& other) const = default;
};

ADAPT_STRUCT_FOR_IR(RGBA32F, r, g, b, a);

TEST_CASE("ir_allocate_image_different_formats") {
	// R8 format
	{
		ICI ici = from_preset(Preset::eGeneric2D, Format::eR8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);
		ici.level_count = 1;
		auto ici_value = make_constant("r8_ici", ici);
		auto view = allocate<>("r8_img", ici_value);

		ClearColor clear_value{ 0.5f, 0.0f, 0.0f, 1.0f };
		uint8_t expected_pixel = 128; // 0.5 * 255
		clear_and_verify(view, clear_value, Format::eR8Unorm, Extent3D{ 256, 256, 1 }, expected_pixel);
	}

	// R16G16 format
	{
		ICI ici = from_preset(Preset::eGeneric2D, Format::eR16G16Sfloat, Extent3D{ 256, 256, 1 }, Samples::e1);
		ici.level_count = 1;
		auto ici_value = make_constant("r16g16_ici", ici);
		auto view = allocate<>("r16g16_img", ici_value);

		ClearColor clear_value{ 0.0f, 0.5f, 0.0f, 1.0f };
		// Half precision: 0.0f = 0x0000, 0.5f = 0x3800
		uint32_t expected_pixel = 0x38000000; // RG as two half floats
		clear_and_verify(view, clear_value, Format::eR16G16Sfloat, Extent3D{ 256, 256, 1 }, expected_pixel);
	}

	// R32G32B32A32 format
	{
		ICI ici = from_preset(Preset::eGeneric2D, Format::eR32G32B32A32Sfloat, Extent3D{ 256, 256, 1 }, Samples::e1);
		ici.level_count = 1;
		auto ici_value = make_constant("r32_ici", ici);
		auto view = allocate<>("r32_img", ici_value);

		ClearColor clear_value{ 0.0f, 0.0f, 0.5f, 1.0f };
		// Four float32 values: (0.0f, 0.0f, 0.5f, 1.0f)
		RGBA32F expected_pixel = { 0.0f, 0.0f, 0.5f, 1.0f };
		clear_and_verify(view, clear_value, Format::eR32G32B32A32Sfloat, Extent3D{ 256, 256, 1 }, expected_pixel);
	}
}

TEST_CASE("ir_allocate_image_different_usages") {
	// Sampled usage
	{
		ICI ici = from_preset(Preset::eMap2D, Format::eR8G8B8A8Srgb, Extent3D{ 256, 256, 1 }, Samples::e1);
		ici.level_count = 1;
		ici.usage |= ImageUsageFlagBits::eTransferSrc;
		auto ici_value = make_constant("sampled_ici", ici);
		auto view = allocate<>("sampled_img", ici_value);

		ClearColor clear_value{ 1.0f, 0.0f, 0.0f, 1.0f };
		uint32_t expected_pixel = 0xFF0000FF; // ABGR: red
		clear_and_verify(view, clear_value, Format::eR8G8B8A8Srgb, Extent3D{ 256, 256, 1 }, expected_pixel);
	}

	// Render target usage
	{
		ICI ici = from_preset(Preset::eRTT2D, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, Samples::e1);
		ici.level_count = 1;
		ici.usage |= ImageUsageFlagBits::eTransferSrc;
		auto ici_value = make_constant("rtt_ici", ici);
		auto view = allocate<>("rtt_img", ici_value);

		ClearColor clear_value{ 0.0f, 1.0f, 0.0f, 1.0f };
		uint32_t expected_pixel = 0xFF00FF00; // ABGR: green
		clear_and_verify(view, clear_value, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, expected_pixel);
	}

	// Storage usage
	{
		ICI ici = from_preset(Preset::eSTT2D, Format::eR32G32B32A32Sfloat, Extent3D{ 256, 256, 1 }, Samples::e1);
		ici.level_count = 1;
		ici.usage |= ImageUsageFlagBits::eTransferSrc;
		auto ici_value = make_constant("storage_ici", ici);
		auto view = allocate<>("storage_img", ici_value);

		ClearColor clear_value{ 0.0f, 0.0f, 1.0f, 1.0f };
		RGBA32F expected_pixel = { 0.0f, 0.0f, 1.0f, 1.0f };
		clear_and_verify(view, clear_value, Format::eR32G32B32A32Sfloat, Extent3D{ 256, 256, 1 }, expected_pixel);
	}
}

TEST_CASE("ir_allocate_image_different_dimensions") {
	// 1D image
	{
		ICI ici = from_preset(Preset::eMap1D, Format::eR32G32B32A32Uint, Extent3D{ 128, 1, 1 }, Samples::e1);
		ici.level_count = 1;
		ici.usage |= ImageUsageFlagBits::eTransferSrc;
		auto ici_value = make_constant("1d_ici", ici);
		auto view = allocate<>("1d_img", ici_value);

		ClearColor clear_value{ 255u, 255u, 255u, 255u };
		uint32_t expected_pixel = 0xFFFFFFFF; // All white as uint
		clear_and_verify(view, clear_value, Format::eR32G32B32A32Uint, Extent3D{ 128, 1, 1 }, expected_pixel);
	}

	// 2D image
	{
		ICI ici = from_preset(Preset::eMap2D, Format::eR32G32B32A32Uint, Extent3D{ 128, 128, 1 }, Samples::e1);
		ici.level_count = 1;
		ici.usage |= ImageUsageFlagBits::eTransferSrc;
		auto ici_value = make_constant("2d_ici", ici);
		auto view = allocate<>("2d_img", ici_value);

		ClearColor clear_value{ 127u, 127u, 127u, 127u };
		uint32_t expected_pixel = 0x80808080; // Mid-gray as uint
		clear_and_verify(view, clear_value, Format::eR32G32B32A32Uint, Extent3D{ 128, 128, 1 }, expected_pixel);
	}

	// 3D image
	{
		ICI ici = from_preset(Preset::eMap3D, Format::eR32G32B32A32Uint, Extent3D{ 64, 64, 64 }, Samples::e1);
		ici.level_count = 1;
		ici.usage |= ImageUsageFlagBits::eTransferSrc;
		auto ici_value = make_constant("3d_ici", ici);
		auto view = allocate<>("3d_img", ici_value);

		ClearColor clear_value{ 0x0u, 0xBFu, 0x80u, 0x40u };
		uint32_t expected_pixel = 0x4080BF00; // Mixed color as uint (approximation)
		clear_and_verify(view, clear_value, Format::eR32G32B32A32Uint, Extent3D{ 64, 64, 64 }, expected_pixel);
	}
}

TEST_CASE("ir_allocate_image_with_mips") {
	ICI ici = from_preset(Preset::eMap2D, Format::eR8G8B8A8Srgb, Extent3D{ 256, 256, 1 }, Samples::e1);
	auto ici_value = make_constant("mipped_ici", ici);
	auto view = allocate<>("mipped_img", ici_value);

	ClearColor clear_value{ 0.8f, 0.2f, 0.6f, 1.0f };
	uint32_t expected_pixel = 0xFF9933CC; // ABGR approximation
	clear_and_verify(view, clear_value, Format::eR8G8B8A8Srgb, Extent3D{ 256, 256, 1 }, expected_pixel);
}

TEST_CASE("ir_allocate_image_multisampled") {
	ICI ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 512, 512, 1 }, Samples::e4);
	auto ici_value = make_constant("ms_ici", ici);
	auto view = allocate<>("ms_img", ici_value);

	ClearColor clear_value{ 0.3f, 0.7f, 0.9f, 1.0f };
	uint32_t expected_pixel = 0xFFE6B34D; // ABGR approximation
	clear_and_verify(view, clear_value, Format::eR8G8B8A8Unorm, Extent3D{ 512, 512, 1 }, expected_pixel);
}

TEST_CASE("ir_allocate_image_resolve_operation") {
	// Multisampled source
	ICI ms_ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 512, 512, 1 }, Samples::e4);
	auto ms_ici_value = make_constant("ms_ici", ms_ici);
	auto ms_view = allocate<>("ms_img", ms_ici_value);

	// Single-sampled destination
	ICI ss_ici = from_preset(Preset::eGeneric2D, Format::eR8G8B8A8Unorm, Extent3D{ 512, 512, 1 }, Samples::e1);
	auto ss_ici_value = make_constant("ss_ici", ss_ici);
	auto ss_view = allocate<>("ss_img", ss_ici_value);

	auto resolved = resolve_into(ms_view, ss_view);
	auto result = resolved.wait(*test_context.allocator, test_context.compiler);
	REQUIRE(result);
}

TEST_CASE("ir_allocate_image_cubemap") {
	ICI ici = from_preset(Preset::eMapCube, Format::eR8G8B8A8Srgb, Extent3D{ 256, 256, 1 }, Samples::e1);
	auto ici_value = make_constant("cube_ici", ici);
	auto view = allocate<>("cube_img", ici_value);

	ClearColor clear_value{ 0.1f, 0.4f, 0.7f, 1.0f };
	uint32_t expected_pixel = 0xFFB3661A; // ABGR approximation
	clear_and_verify(view, clear_value, Format::eR8G8B8A8Srgb, Extent3D{ 256, 256, 1 }, expected_pixel);
}

TEST_CASE("ir_allocate_image_compressed") {
	// BC1 - can't clear compressed formats, so just verify allocation works
	{
		ICI ici = from_preset(Preset::eMap2D, Format::eBc1RgbaSrgbBlock, Extent3D{ 512, 512, 1 }, Samples::e1);
		auto ici_value = make_constant("bc1_ici", ici);
		auto view = allocate<>("bc1_img", ici_value);

		// Just verify the allocation completes
		auto result = view.wait(*test_context.allocator, test_context.compiler);
		REQUIRE(result);
	}

	// BC7
	{
		ICI ici = from_preset(Preset::eMap2D, Format::eBc7SrgbBlock, Extent3D{ 512, 512, 1 }, Samples::e1);
		auto ici_value = make_constant("bc7_ici", ici);
		auto view = allocate<>("bc7_img", ici_value);

		auto result = view.wait(*test_context.allocator, test_context.compiler);
		REQUIRE(result);
	}
}

TEST_CASE("ir_allocate_image_depth_stencil") {
	// Depth only
	{
		ICI ici = from_preset(Preset::eRTT2DUnmipped, Format::eD32Sfloat, Extent3D{ 1024, 768, 1 }, Samples::e1);
		auto ici_value = make_constant("depth_ici", ici);
		auto view = allocate<>("depth_img", ici_value);

		ClearDepthStencil clear_value{ 1.0f, 0 };
		// Depth is a single float32
		float expected_pixel = 1.0f;
		clear_and_verify(view, clear_value, Format::eD32Sfloat, Extent3D{ 1024, 768, 1 }, expected_pixel);
	}

	// Depth-stencil
	{
		ICI ici = from_preset(Preset::eRTT2DUnmipped, Format::eD24UnormS8Uint, Extent3D{ 1024, 768, 1 }, Samples::e1);
		auto ici_value = make_constant("ds_ici", ici);
		auto view = allocate<>("ds_img", ici_value);

		ClearDepthStencil clear_value{ 0.5f, 128 };
		// D24S8: 24 bits depth (0.5 * 0xFFFFFF = 0x7FFFFF) + 8 bits stencil (128 = 0x80)
		// Packed as uint32: depth in lower 24 bits, stencil in upper 8 bits
		uint32_t expected_pixel = (128u << 24) | 0x7FFFFF;
		clear_and_verify(view, clear_value, Format::eD24UnormS8Uint, Extent3D{ 1024, 768, 1 }, expected_pixel);
	}
}

TEST_CASE("ir_allocate_buffer_to_image_copy") {
	// Create source buffer
	auto data = { 50u, 60u, 70u, 80u };
	size_t buf_size = data.size() * sizeof(uint32_t);
	auto src_buf = *allocate_buffer<uint32_t>(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUtoGPU, buf_size, alignof(uint32_t) });
	std::copy(data.begin(), data.end(), &(*src_buf)[0]);
	auto src_buf_value = discard("src_buf", *src_buf);

	// Allocate destination image in IR
	ICI dst_ici = from_preset(Preset::eGeneric2D, Format::eR32Uint, Extent3D{ 2, 2, 1 }, Samples::e1);
	auto dst_ici_value = make_constant("dst_ici", dst_ici);
	auto dst_view = allocate<>("dst_img", dst_ici_value);

	// Copy buffer to image and verify
	auto copied = copy(src_buf_value, dst_view);
	verify_image_data(copied, std::span(data), Format::eR32Uint, Extent3D{ 2, 2, 1 });
}

TEST_CASE("ir_allocate_image_to_buffer_copy") {
	// Create source image
	auto data = { 11u, 22u, 33u, 44u };
	auto src_ici = from_preset(Preset::eGeneric2D, Format::eR32Uint, Extent3D{ 2, 2, 1 }, Samples::e1);
	auto [src_view, src_fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, src_ici, std::span(data));
	auto src_value = discard("src_img", *src_view);

	// Allocate destination buffer
	size_t buf_size = data.size() * sizeof(uint32_t);
	auto dst_buf = *allocate_buffer<uint32_t>(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, buf_size, alignof(uint32_t) });
	auto dst_buf_value = discard("dst_buf", *dst_buf);

	// Copy and verify
	auto res = download_buffer(copy(src_value, std::move(dst_buf_value))).get(*test_context.allocator, test_context.compiler);
	REQUIRE(res);
	auto updata = res->to_span();
	CHECK(updata == std::span(data));
}
/*
TEST_CASE("ir_allocate_custom_image_view") {
  // Allocate image in IR
  ICI ici = from_preset(Preset::eGeneric2D, Format::eR8G8B8A8Unorm, Extent3D{ 512, 512, 1 }, Samples::e1);
  auto ici_value = make_constant("img_ici", ici);
  auto img = allocate<>("img", ici_value);

  // Create custom IVCI for view
  IVCI ivci = {};
  ivci.format = Format::eR8G8B8A8Unorm;
  ivci.view_type = ImageViewType::e2D;
  ivci.base_level = 0;
  ivci.level_count = 1;
  ivci.base_layer = 0;
  ivci.layer_count = 1;

  auto ivci_value = make_constant("custom_ivci", ivci);
  auto view = allocate("custom_view", img, ivci_value);

  ClearColor clear_value{ 0.6f, 0.3f, 0.9f, 1.0f };
  uint32_t expected_pixel = 0xFFE64D99; // ABGR approximation
  clear_and_verify(view, clear_value, Format::eR8G8B8A8Unorm, Extent3D{ 512, 512, 1 }, expected_pixel);
}
*/