#include "TestContext.hpp"
#include "vuk/ir/IRPass.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <algorithm>
#include <doctest/doctest.h>
#include <string_view>

using namespace vuk;

// =================================================================
// IR-based Buffer Allocation Tests With Size Inference
// =================================================================

TEST_CASE("ir_allocate_buffer_infer_size_from_copy") {
	// Source buffer with known size
	std::vector<uint32_t> data(128);
	for (size_t i = 0; i < data.size(); i++) {
		data[i] = i * 3;
	}
	auto [src_buf, src_fut] = create_buffer(*test_context.allocator, MemoryUsage::eCPUonly, DomainFlagBits::eAny, std::span(data));

	// Destination buffer with size left unspecified - should be inferred from copy source
	BufferCreateInfo dst_bci{};
	dst_bci.memory_usage = MemoryUsage::eGPUonly;
	// Note: size NOT set - should be inferred from source!
	auto dst_buffer = allocate<uint32_t>("inferred_dst", dst_bci);

	auto copied = copy(src_fut, dst_buffer);

	verify_buffer_data(copied, std::span(data));
}

TEST_CASE("ir_allocate_buffer_chain_inference") {
	// Test inference chain: src -> intermediate -> dst
	std::vector<float> data(64);
	for (size_t i = 0; i < data.size(); i++) {
		data[i] = static_cast<float>(i) * 0.5f;
	}
	auto [src_buf, src_fut] = create_buffer(*test_context.allocator, MemoryUsage::eCPUonly, DomainFlagBits::eAny, std::span(data));

	// Intermediate buffer - size inferred from source
	BufferCreateInfo mid_bci{};
	mid_bci.memory_usage = MemoryUsage::eGPUonly;
	// Size not specified - will be inferred
	auto mid_buffer = allocate<float>("mid", mid_bci);

	// Destination buffer - size inferred from intermediate
	BufferCreateInfo dst_bci{};
	dst_bci.memory_usage = MemoryUsage::eGPUonly;
	// Size not specified - will be inferred
	auto dst_buffer = allocate<float>("dst", dst_bci);

	auto step1 = copy(src_fut, mid_buffer);
	auto step2 = copy(step1, dst_buffer);

	verify_buffer_data(step2, std::span(data));
}

TEST_CASE("ir_allocate_buffer_infer_size_from_image") {
	// Create a source image with known parameters
	ICI src_ici = from_preset(Preset::eRTT2DUnmipped, Format::eR8G8B8A8Unorm, Extent3D{ 256, 256, 1 }, SampleCountFlagBits::e1);
	auto src_view = allocate<>("src_img", src_ici);
	clear_image(src_view, ClearColor{ 0.5f, 0.25f, 0.75f, 1.0f });

	// Destination buffer with size left unspecified - should be inferred from image
	BufferCreateInfo dst_bci{};
	dst_bci.memory_usage = MemoryUsage::eGPUonly;
	// Note: size NOT set - should be inferred from image!
	auto dst_buffer = allocate<uint32_t>("inferred_from_image", dst_bci);

	// Copy from image to buffer
	auto copied = copy(src_view, dst_buffer);

	// Verify the buffer received the correct data
	// Expected: R8G8B8A8Unorm with values (128, 64, 191, 255) for each pixel
	auto expected_pixel = 255u << 24 | 191u << 16 | 64u << 8 | 128u;
	size_t pixel_count = 256 * 256;
	std::vector<uint32_t> expected_data(pixel_count, expected_pixel);
	verify_buffer_data(copied, std::span(expected_data));
}
