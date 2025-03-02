#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <doctest/doctest.h>

#include "vuk/extra/SPD.hpp"

using namespace vuk;

TEST_CASE("SPD") {
	std::vector<float> data(256 * 256);
	for (int i = 0; i < 256 * 256; i++) {
		auto row = (float)i / 256;
		auto col = i % 256;
		data[i] = sinf((float)row / 64) * sinf((float)col / 64);
	}
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 256, 256, 1 }, Samples::e1);
	auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

	size_t alignment = format_to_texel_block_size(fut->format);
	size_t size = compute_image_size(fut->format, fut->extent);

	auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto dst_buf = discard_buf("dst", *dst);
	extra::generate_mips_spd(*test_context.runtime, fut);
	copy(fut, dst_buf);
	auto res = download_buffer(dst_buf).get(*test_context.allocator, test_context.compiler);
}
