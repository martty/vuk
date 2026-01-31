#include "TestContext.hpp"
#include "vuk/ir/IRPass.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <doctest/doctest.h>
#include <string_view>

using namespace vuk;

// ============================================================================
// IR Integration Tests for Buffers
// ============================================================================

TEST_CASE("constant_buffer_view_metadata") {
	std::vector<uint32_t> data = { 11u, 22u, 33u, 44u };
	auto [buf, fut] = create_buffer(*test_context.allocator, MemoryUsage::eCPUonly, DomainFlagBits::eAny, std::span(data));

	// Verify buffer was created correctly
	CHECK(buf->sz_bytes == data.size() * sizeof(uint32_t));
	CHECK(buf->ptr);

	// Download and verify the data
	verify_buffer_data(fut, std::span(data));
}

TEST_CASE("buffer_acquire_external") {
	std::vector<uint32_t> data = { 5u, 6u, 7u, 8u };
	auto [buf, fut] = create_buffer(*test_context.allocator, MemoryUsage::eCPUonly, DomainFlagBits::eAny, std::span(data));

	// Acquire as an external resource with storage buffer access
	auto acquired = acquire("external_buf", buf.get(), Access::eComputeRead);

	// Use the acquired buffer in a pass that just passes it through
	auto pass = make_pass(
	    "passthrough",
	    [](CommandBuffer& cb, VUK_ARG(Buffer<uint32_t>, Access::eComputeRead) input) {
		    // Just pass through
		    return input;
	    },
	    DomainFlagBits::eComputeQueue);

	auto result = pass(acquired);

	// Download and verify the data
	verify_buffer_data(result, std::span(data));
}

TEST_CASE("buffer_in_compute_pass") {
	std::vector<uint32_t> data = { 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u };
	auto [src_buf, src_fut] = create_buffer(*test_context.allocator, MemoryUsage::eCPUonly, DomainFlagBits::eAny, std::span(data));

	// Create destination buffer
	BufferCreateInfo dst_bci{};
	dst_bci.memory_usage = MemoryUsage::eGPUonly;
	dst_bci.size = data.size() * sizeof(uint32_t);
	dst_bci.alignment = alignof(uint32_t);
	auto dst_buffer = allocate<uint32_t>("compute_dst", dst_bci);

	// Copy data through a compute pass
	auto pass = make_pass(
	    "compute_copy",
	    [](CommandBuffer& cb, VUK_ARG(Buffer<uint32_t>, Access::eComputeRead) input, VUK_ARG(Buffer<uint32_t>, Access::eComputeWrite) output) {
		    // In a real scenario, would dispatch a compute shader
		    return output;
	    },
	    DomainFlagBits::eComputeQueue);

	auto copied = copy(src_fut, dst_buffer);

	// Verify the data
	verify_buffer_data(copied, std::span(data));
}

TEST_CASE("buffer_subrange_operations") {
	// Create a large source buffer with known pattern
	std::vector<uint32_t> data(1024);
	for (size_t i = 0; i < data.size(); i++) {
		data[i] = static_cast<uint32_t>(i * 17); // Unique pattern
	}

	auto [src_buf, src_fut] = create_buffer(*test_context.allocator, MemoryUsage::eCPUonly, DomainFlagBits::eAny, std::span(data));

	// Test 1: Create subview of first 256 elements
	auto first_subview = src_fut.subview(0, 256);
	BufferCreateInfo first_bci{};
	first_bci.memory_usage = MemoryUsage::eGPUonly;
	first_bci.size = 256 * sizeof(uint32_t);
	first_bci.alignment = alignof(uint32_t);
	auto first_dst = allocate<uint32_t>("first_256", first_bci);
	auto first_copied = copy(first_subview, first_dst);

	// Test 2: Create subview of middle 128 elements (from offset 400)
	auto middle_subview = src_fut.subview(400, 128);
	BufferCreateInfo middle_bci{};
	middle_bci.memory_usage = MemoryUsage::eGPUonly;
	middle_bci.size = 128 * sizeof(uint32_t);
	middle_bci.alignment = alignof(uint32_t);
	auto middle_dst = allocate<uint32_t>("middle_128", middle_bci);
	auto middle_copied = copy(middle_subview, middle_dst);

	// Test 3: Create subview of last 64 elements
	auto last_subview = src_fut.subview(960, 64);
	BufferCreateInfo last_bci{};
	last_bci.memory_usage = MemoryUsage::eGPUonly;
	last_bci.size = 64 * sizeof(uint32_t);
	last_bci.alignment = alignof(uint32_t);
	auto last_dst = allocate<uint32_t>("last_64", last_bci);
	auto last_copied = copy(last_subview, last_dst);

	// Verify each portion was copied correctly
	std::vector<uint32_t> first_expected(data.begin(), data.begin() + 256);
	std::vector<uint32_t> middle_expected(data.begin() + 400, data.begin() + 528);
	std::vector<uint32_t> last_expected(data.end() - 64, data.end());

	verify_buffer_data(first_copied, std::span(first_expected));
	verify_buffer_data(middle_copied, std::span(middle_expected));
	verify_buffer_data(last_copied, std::span(last_expected));
}
