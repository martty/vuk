#include "TestContext.hpp"
#include "vuk/ir/IRPass.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <algorithm>
#include <doctest/doctest.h>
#include <string_view>

using namespace vuk;

// ============================================================================
// IR-based Buffer Allocation and Parameter Inference Tests
// ============================================================================

TEST_CASE("ir_allocate_buffer_basic") {
	// Create BufferCreateInfo and allocate buffer entirely in IR
	BufferCreateInfo bci{};
	bci.memory_usage = MemoryUsage::eGPUonly;
	bci.size = 1024 * sizeof(uint32_t);
	bci.alignment = alignof(uint32_t);

	auto buffer_value = allocate<uint32_t>("test_buf", bci);

	// Fill and verify
	fill_and_verify(buffer_value, 42u, 1024);
}

TEST_CASE("ir_allocate_buffer_cpu_to_gpu") {
	// Allocate CPU->GPU buffer in IR
	BufferCreateInfo bci{};
	bci.memory_usage = MemoryUsage::eCPUtoGPU;
	bci.size = 256 * sizeof(float);
	bci.alignment = alignof(float);

	auto buffer_value = allocate<float>("cpu_gpu_buf", bci);

	// Create source data
	std::vector<float> data(256);
	for (size_t i = 0; i < data.size(); i++) {
		data[i] = static_cast<float>(i) * 0.5f;
	}

	// Upload and verify
	auto [src_buf, src_fut] = create_buffer(*test_context.allocator, MemoryUsage::eCPUonly, DomainFlagBits::eAny, std::span(data));
	auto copied = copy(src_fut, buffer_value);
	verify_buffer_data(copied, std::span(data));
}

TEST_CASE("ir_allocate_buffer_gpu_to_cpu") {
	// Allocate GPU->CPU buffer for readback
	BufferCreateInfo bci{};
	bci.memory_usage = MemoryUsage::eGPUtoCPU;
	bci.size = 128 * sizeof(uint32_t);
	bci.alignment = alignof(uint32_t);

	auto buffer_value = allocate<uint32_t>("gpu_cpu_buf", bci);

	// Fill with pattern
	fill_and_verify(buffer_value, 0xDEADBEEFu, 128);
}

TEST_CASE("ir_allocate_buffer_infer_from_copy_source") {
	// Create source buffer with data
	std::array data = { 10u, 20u, 30u, 40u, 50u };
	auto [src_buf, src_fut] = create_buffer(*test_context.allocator, MemoryUsage::eCPUonly, DomainFlagBits::eAny, std::span(data));

	// Allocate destination buffer in IR with matching size
	BufferCreateInfo dst_bci{};
	dst_bci.memory_usage = MemoryUsage::eGPUonly;
	dst_bci.size = data.size() * sizeof(uint32_t);
	dst_bci.alignment = alignof(uint32_t);

	auto dst_buffer = allocate<uint32_t>("dst_buf", dst_bci);

	// Copy and verify
	auto copied = copy(src_fut, dst_buffer);
	verify_buffer_data(copied, std::span(data));
}

TEST_CASE("ir_allocate_buffer_different_types") {
	// uint8_t buffer
	{
		BufferCreateInfo bci{};
		bci.memory_usage = MemoryUsage::eGPUonly;
		bci.size = 512 * sizeof(uint8_t);
		bci.alignment = alignof(uint8_t);
		auto buffer = allocate<uint8_t>("u8_buf", bci);

		fill_and_verify(buffer, static_cast<uint8_t>(0xAB), 512);
	}

	// uint16_t buffer
	{
		BufferCreateInfo bci{};
		bci.memory_usage = MemoryUsage::eGPUonly;
		bci.size = 256 * sizeof(uint16_t);
		bci.alignment = alignof(uint16_t);
		auto buffer = allocate<uint16_t>("u16_buf", bci);

		fill_and_verify(buffer, static_cast<uint16_t>(0xBEEF), 256);
	}

	// float buffer
	{
		BufferCreateInfo bci{};
		bci.memory_usage = MemoryUsage::eGPUonly;
		bci.size = 128 * sizeof(float);
		bci.alignment = alignof(float);
		auto buffer = allocate<float>("f32_buf", bci);

		fill_and_verify(buffer, 3.14159f, 128);
	}

	// double buffer
	{
		BufferCreateInfo bci{};
		bci.memory_usage = MemoryUsage::eGPUonly;
		bci.size = 64 * sizeof(double);
		bci.alignment = alignof(double);
		auto buffer = allocate<double>("f64_buf", bci);

		auto data = std::vector<double>(64, 2.718281828);
		auto [src_buf, src_fut] = create_buffer(*test_context.allocator, MemoryUsage::eCPUonly, DomainFlagBits::eAny, std::span(data));
		auto copied = copy(src_fut, buffer);
		verify_buffer_data(copied, std::span(data));
	}
}

TEST_CASE("ir_allocate_buffer_different_sizes") {
	// Small buffer (1 KB)
	{
		BufferCreateInfo bci{};
		bci.memory_usage = MemoryUsage::eGPUonly;
		bci.size = 1024;
		bci.alignment = 4;
		auto buffer = allocate<uint32_t>("small_buf", bci);

		fill_and_verify(buffer, 0x11111111u, 256);
	}

	// Medium buffer (64 KB)
	{
		BufferCreateInfo bci{};
		bci.memory_usage = MemoryUsage::eGPUonly;
		bci.size = 65536;
		bci.alignment = 4;
		auto buffer = allocate<uint32_t>("medium_buf", bci);

		fill_and_verify(buffer, 0x22222222u, 16384);
	}

	// Large buffer (1 MB)
	{
		BufferCreateInfo bci{};
		bci.memory_usage = MemoryUsage::eGPUonly;
		bci.size = 1048576;
		bci.alignment = 4;
		auto buffer = allocate<uint32_t>("large_buf", bci);

		fill_and_verify(buffer, 0x33333333u, 262144);
	}
}

TEST_CASE("ir_allocate_buffer_different_memory_usages") {
	size_t count = 256;
	size_t size = count * sizeof(uint32_t);

	// GPUonly
	{
		BufferCreateInfo bci{};
		bci.memory_usage = MemoryUsage::eGPUonly;
		bci.size = size;
		bci.alignment = alignof(uint32_t);
		auto buffer = allocate<uint32_t>("gpu_only_buf", bci);

		fill_and_verify(buffer, 0xAAAAAAAAu, count);
	}

	// CPUonly
	{
		BufferCreateInfo bci{};
		bci.memory_usage = MemoryUsage::eCPUonly;
		bci.size = size;
		bci.alignment = alignof(uint32_t);
		auto bci_value = make_constant("cpu_only_bci", bci);
		auto buffer = allocate<uint32_t>("cpu_only_buf", bci_value);

		fill_and_verify(buffer, 0xBBBBBBBBu, count);
	}

	// CPUtoGPU
	{
		BufferCreateInfo bci{};
		bci.memory_usage = MemoryUsage::eCPUtoGPU;
		bci.size = size;
		bci.alignment = alignof(uint32_t);
		auto buffer = allocate<uint32_t>("cpu_to_gpu_buf", bci);

		fill_and_verify(buffer, 0xCCCCCCCCu, count);
	}

	// GPUtoCPU
	{
		BufferCreateInfo bci{};
		bci.memory_usage = MemoryUsage::eGPUtoCPU;
		bci.size = size;
		bci.alignment = alignof(uint32_t);
		auto buffer = allocate<uint32_t>("gpu_to_cpu_buf", bci);

		fill_and_verify(buffer, 0xDDDDDDDDu, count);
	}
}

struct TestStruct {
	float x, y, z, w;
	uint32_t id;

	bool operator==(const TestStruct& other) const = default;
};

ADAPT_STRUCT_FOR_IR(TestStruct, x, y, z, w, id);

TEST_CASE("ir_allocate_buffer_with_struct") {
	BufferCreateInfo bci{};
	bci.memory_usage = MemoryUsage::eGPUonly;
	bci.size = 64 * sizeof(TestStruct);
	bci.alignment = alignof(TestStruct);
	auto buffer = allocate<TestStruct>("struct_buf", bci);

	// Create test data
	std::vector<TestStruct> data(64);
	for (size_t i = 0; i < data.size(); i++) {
		data[i] = { static_cast<float>(i), static_cast<float>(i * 2), static_cast<float>(i * 3), static_cast<float>(i * 4), static_cast<uint32_t>(i) };
	}

	auto [src_buf, src_fut] = create_buffer(*test_context.allocator, MemoryUsage::eCPUonly, DomainFlagBits::eAny, std::span(data));
	auto copied = copy(src_fut, buffer);
	verify_buffer_data(copied, std::span(data));
}

TEST_CASE("ir_allocate_buffer_multiple") {
	// Allocate multiple buffers and verify they're independent
	BufferCreateInfo bci1{};
	bci1.memory_usage = MemoryUsage::eGPUonly;
	bci1.size = 128 * sizeof(uint32_t);
	bci1.alignment = alignof(uint32_t);
	auto buffer1 = allocate<uint32_t>("buf1", bci1);

	BufferCreateInfo bci2{};
	bci2.memory_usage = MemoryUsage::eGPUonly;
	bci2.size = 256 * sizeof(uint32_t);
	bci2.alignment = alignof(uint32_t);
	auto buffer2 = allocate<uint32_t>("buf2", bci2);

	BufferCreateInfo bci3{};
	bci3.memory_usage = MemoryUsage::eGPUonly;
	bci3.size = 64 * sizeof(uint32_t);
	bci3.alignment = alignof(uint32_t);
	auto buffer3 = allocate<uint32_t>("buf3", bci3);

	// Fill each with different values
	fill(buffer1, 0x11111111u);
	fill(buffer2, 0x22222222u);
	fill(buffer3, 0x33333333u);

	// Verify each buffer independently
	{
		std::vector<uint32_t> expected(128, 0x11111111u);
		verify_buffer_data(buffer1, std::span(expected), RenderGraphCompileOptions{});
	}
	{
		std::vector<uint32_t> expected(256, 0x22222222u);
		auto db = download_buffer(buffer2);
		auto res = db.get(*test_context.allocator, test_context.compiler, RenderGraphCompileOptions{});
		REQUIRE(res);
		auto actual_data = res->to_span();
		CHECK(std::equal(actual_data.begin(), actual_data.end(), expected.begin()));
	}
	{
		std::vector<uint32_t> expected(64, 0x33333333u);
		verify_buffer_data(buffer3, std::span(expected));
	}
}

TEST_CASE("ir_allocate_buffer_alignment") {
	// Test various alignment values
	{
		BufferCreateInfo bci{};
		bci.memory_usage = MemoryUsage::eGPUonly;
		bci.size = 256 * sizeof(uint32_t);
		bci.alignment = 4;
		auto bci_value = make_constant("align4_bci", bci);
		auto buffer = allocate<uint32_t>("align4_buf", bci_value);
		fill_and_verify(buffer, 0x44444444u, 256);
	}

	{
		BufferCreateInfo bci{};
		bci.memory_usage = MemoryUsage::eGPUonly;
		bci.size = 64 * 16; // 16-byte aligned data
		bci.alignment = 16;
		auto bci_value = make_constant("align16_bci", bci);
		auto buffer = allocate<uint32_t>("align16_buf", bci_value);
		fill_and_verify(buffer, 0x16161616u, 256);
	}
}

TEST_CASE("ir_allocate_buffer_reuse_bci") {
	// Create a single BCI and use it for multiple allocations
	BufferCreateInfo bci{};
	bci.memory_usage = MemoryUsage::eGPUonly;
	bci.size = 512 * sizeof(uint32_t);
	bci.alignment = alignof(uint32_t);
	auto bci_value = make_constant("shared_bci", bci);

	auto buffer1 = allocate<uint32_t>("buf1", bci_value);
	auto buffer2 = allocate<uint32_t>("buf2", bci_value);
	auto buffer3 = allocate<uint32_t>("buf3", bci_value);

	// Fill each with different patterns
	fill(buffer1, 0xAAAAAAAAu);
	fill(buffer2, 0xBBBBBBBBu);
	fill(buffer3, 0xCCCCCCCCu);

	// Verify independently
	{
		std::vector<uint32_t> expected(512, 0xAAAAAAAAu);
		verify_buffer_data(buffer1, std::span(expected));
	}
	{
		std::vector<uint32_t> expected(512, 0xBBBBBBBBu);
		verify_buffer_data(buffer2, std::span(expected));
	}
	{
		std::vector<uint32_t> expected(512, 0xCCCCCCCCu);
		verify_buffer_data(buffer3, std::span(expected));
	}
}

TEST_CASE("ir_allocate_buffer_with_acquire") {
	// Allocate and immediately acquire for use
	BufferCreateInfo bci{};
	bci.memory_usage = MemoryUsage::eGPUonly;
	bci.size = 128 * sizeof(float);
	bci.alignment = alignof(float);
	auto bci_value = make_constant("acquire_bci", bci);
	auto buffer = allocate<float>("acquire_buf", bci_value);

	// Create test data
	std::vector<float> data(128);
	for (size_t i = 0; i < data.size(); i++) {
		data[i] = static_cast<float>(i) + 0.5f;
	}

	// Upload and verify
	auto [src_buf, src_fut] = create_buffer(*test_context.allocator, MemoryUsage::eCPUonly, DomainFlagBits::eAny, std::span(data));
	auto copied = copy(src_fut, buffer);
	verify_buffer_data(copied, std::span(data));
}

TEST_CASE("ir_allocate_buffer_repeated") {
	// Test repeated allocations in sequence to verify proper resource management
	BufferCreateInfo bci{};
	bci.memory_usage = MemoryUsage::eGPUonly;
	bci.size = 256 * sizeof(uint32_t);
	bci.alignment = alignof(uint32_t);

	// First allocation and use
	{
		auto bci_value = make_constant("repeat1_bci", bci);
		auto buffer = allocate<uint32_t>("repeat1_buf", bci_value);
		fill_and_verify(buffer, 0xAAAAAAAAu, 256);
	}

	// Second allocation with different pattern
	{
		auto bci_value = make_constant("repeat2_bci", bci);
		auto buffer = allocate<uint32_t>("repeat2_buf", bci_value);
		fill_and_verify(buffer, 0xBBBBBBBBu, 256);
	}

	// Third allocation with yet another pattern
	{
		auto bci_value = make_constant("repeat3_bci", bci);
		auto buffer = allocate<uint32_t>("repeat3_buf", bci_value);
		fill_and_verify(buffer, 0xCCCCCCCCu, 256);
	}

	// Multiple allocations in the same scope
	{
		auto bci_value1 = make_constant("multi1_bci", bci);
		auto buffer1 = allocate<uint32_t>("multi1_buf", bci_value1);

		auto bci_value2 = make_constant("multi2_bci", bci);
		auto buffer2 = allocate<uint32_t>("multi2_buf", bci_value2);

		auto bci_value3 = make_constant("multi3_bci", bci);
		auto buffer3 = allocate<uint32_t>("multi3_buf", bci_value3);

		// Use them in sequence
		fill(buffer1, 0x11111111u);
		fill(buffer2, 0x22222222u);
		fill(buffer3, 0x33333333u);

		// Verify each independently
		{
			std::vector<uint32_t> expected(256, 0x11111111u);
			verify_buffer_data(buffer1, std::span(expected));
		}
		{
			std::vector<uint32_t> expected(256, 0x22222222u);
			verify_buffer_data(buffer2, std::span(expected));
		}
		{
			std::vector<uint32_t> expected(256, 0x33333333u);
			verify_buffer_data(buffer3, std::span(expected));
		}
	}
}

TEST_CASE("ir_allocate_buffer_chain_copy") {
	// Create source
	std::array data = { 100u, 200u, 300u, 400u };
	auto [src_buf, src_fut] = create_buffer(*test_context.allocator, MemoryUsage::eCPUonly, DomainFlagBits::eAny, std::span(data));

	// First intermediate - allocated in IR
	BufferCreateInfo int1_bci{};
	int1_bci.memory_usage = MemoryUsage::eGPUonly;
	int1_bci.size = data.size() * sizeof(uint32_t);
	int1_bci.alignment = alignof(uint32_t);
	auto int1_bci_value = make_constant("int1_bci", int1_bci);
	auto int1_buffer = allocate<uint32_t>("int1_buf", int1_bci_value);
	auto copied1 = copy(src_fut, int1_buffer);

	// Second intermediate
	auto int2_bci_value = make_constant("int2_bci", int1_bci);
	auto int2_buffer = allocate<uint32_t>("int2_buf", int2_bci_value);
	auto copied2 = copy(copied1, int2_buffer);

	// Final destination
	auto dst_bci_value = make_constant("dst_bci", int1_bci);
	auto dst_buffer = allocate<uint32_t>("dst_buf", dst_bci_value);
	auto final_copy = copy(copied2, dst_buffer);

	// Verify data propagated through the chain
	verify_buffer_data<uint32_t>(final_copy, std::span(data));
}
