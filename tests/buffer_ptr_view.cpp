#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <doctest/doctest.h>
#include <string_view>

using namespace vuk;

// ============================================================================
// Basic Buffer Pointer Tests
// ============================================================================

TEST_CASE("buffer ptr basic") {
	// Default construction
	{
		ptr<BufferLike<float>> buf;
		CHECK(!buf);
		CHECK(buf.device_address == 0);
	}

	// Bool conversion and comparison
	{
		Allocator alloc(test_context.runtime->get_vk_resource());
		BufferCreateInfo bci{ .memory_usage = MemoryUsage::eCPUonly, .size = 1024 };

		ptr<BufferLike<float>> buf1;
		ptr<BufferLike<float>> buf2;

		CHECK(buf1 == buf2);

		alloc.allocate_memory(std::span{ static_cast<ptr_base*>(&buf1), 1 }, std::span{ &bci, 1 });

		CHECK(buf1);
		CHECK(buf1.device_address != 0);
		CHECK(buf1 != buf2);

		alloc.deallocate(std::span{ static_cast<ptr_base*>(&buf1), 1 });
	}

	// Type templating
	{
		ptr<BufferLike<uint32_t>> typed_buf;
		ptr<BufferLike<float>> float_buf;

		CHECK(!typed_buf);
		CHECK(!float_buf);
		CHECK(sizeof(typed_buf) == sizeof(float_buf));
	}

	// BCI construction
	{
		BufferCreateInfo bci{ .memory_usage = MemoryUsage::eGPUonly, .size = 2048, .alignment = 16 };

		CHECK(bci.memory_usage == MemoryUsage::eGPUonly);
		CHECK(bci.size == 2048);
		CHECK(bci.alignment == 16);
	}
}

TEST_CASE("ptr alloc") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	BufferCreateInfo bci{ .memory_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
	ptr<BufferLike<float>> foo;
	alloc.allocate_memory(std::span{ static_cast<ptr_base*>(&foo), 1 }, std::span{ &bci, 1 });

	*foo = 4;

	*foo *= 3;

	CHECK(*foo == 12);

	alloc.deallocate(std::span{ static_cast<ptr_base*>(&foo), 1 });
}

TEST_CASE("ptr with struct") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	BufferCreateInfo bci{ .memory_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
	ptr<BufferLike<std::pair<size_t, size_t>>> foo;
	alloc.allocate_memory(std::span{ static_cast<ptr_base*>(&foo), 1 }, std::span{ &bci, 1 });

	foo->first = 3;

	foo->second = 6;

	foo->second *= 3;

	CHECK(foo->first == 3);
	CHECK(foo->second == 18);
	alloc.deallocate(std::span{ static_cast<ptr_base*>(&foo), 1 });
}

TEST_CASE("ptr with array") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	BufferCreateInfo bci{ .memory_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
	ptr<float> foo;
	alloc.allocate_memory(std::span{ static_cast<ptr_base*>(&foo), 1 }, std::span{ &bci, 1 });

	for (int i = 0; i < 5; i++) {
		foo[i] = i;
	}

	for (int i = 0; i < 5; i++) {
		foo[i] *= i;
	}

	for (int i = 0; i < 5; i++) {
		CHECK(foo[i] == i * i);
	}
	alloc.deallocate(std::span{ static_cast<ptr_base*>(&foo), 1 });
}

TEST_CASE("ptr with helper") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_ptr<BufferLike<float>> foo = *allocate_memory<float>(alloc, MemoryUsage::eCPUonly);

	**foo = 4;

	**foo *= 3;

	CHECK(**foo == 12);
}

TEST_CASE("array with helper") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_view<BufferLike<float>> foo = *allocate_array<float>(alloc, 16, MemoryUsage::eCPUonly);

	for (int i = 0; i < 5; i++) {
		foo[i] = i;
	}

	for (int i = 0; i < 5; i++) {
		foo[i] *= i;
	}

	for (int i = 0; i < 5; i++) {
		CHECK(foo[i] == i * i);
	}
}

TEST_CASE("memory view from array with helper") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_view<BufferLike<float>> foo = *allocate_array<float>(alloc, 16, MemoryUsage::eCPUonly);
	// concrete views don't need allocations
	view<BufferLike<float>> view = foo.get();

	for (int i = 0; i < 4; i++) {
		view[i] = i;
	}

	for (int i = 0; i < 4; i++) {
		view[i] *= i;
	}

	for (int i = 0; i < 4; i++) {
		CHECK(view[i] == i * i);
	}
}

void sqr_specific(view<BufferLike<float>> view) {
	for (int i = 0; i < view.count(); i++) {
		view[i] *= i;
	}
}

TEST_CASE("function taking views") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_view<BufferLike<float>> foo = *allocate_array<float>(alloc, 16, MemoryUsage::eCPUonly);
	// concrete views don't need allocations
	view<BufferLike<float>> v = foo.get();

	for (int i = 0; i < 4; i++) {
		v[i] = i;
	}

	sqr_specific(v);

	for (int i = 0; i < 4; i++) {
		CHECK(v[i] == i * i);
	}
}

// ============================================================================
// BufferView Tests
// ============================================================================

TEST_CASE("buffer_view basic") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	// Default construction
	{
		view<BufferLike<float>> v;
		CHECK(!v);
		CHECK(v.count() == 0);
	}

	// From buffer
	{
		Unique_view<BufferLike<uint32_t>> buf = *allocate_array<uint32_t>(alloc, 256, MemoryUsage::eCPUonly);
		view<BufferLike<uint32_t>> v = buf.get();

		CHECK(v);
		CHECK(v.count() == 256);
	}

	// Comparison
	{
		view<BufferLike<float>> v1;
		view<BufferLike<float>> v2;

		CHECK(v1 == v2);

		Unique_view<BufferLike<float>> buf = *allocate_array<float>(alloc, 128, MemoryUsage::eCPUonly);
		v1 = buf.get();
		CHECK(v1 != v2);
	}
}

TEST_CASE("buffer_view_slice") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_view<BufferLike<uint32_t>> buf = *allocate_array<uint32_t>(alloc, 256, MemoryUsage::eCPUonly);

	// Fill with test data
	for (int i = 0; i < 256; i++) {
		buf[i] = i;
	}

	// Create a slice
	view<BufferLike<uint32_t>> full_view = buf.get();
	auto slice = full_view.slice(64, 128);

	CHECK(slice);
	CHECK(slice.count() == 128);

	// Verify slice data
	for (int i = 0; i < 128; i++) {
		CHECK(slice[i] == (64 + i));
	}
}

TEST_CASE("buffer_view_first_last") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_view<BufferLike<uint32_t>> buf = *allocate_array<uint32_t>(alloc, 100, MemoryUsage::eCPUonly);

	// Fill with test data
	for (int i = 0; i < 100; i++) {
		buf[i] = i * 10;
	}

	view<BufferLike<uint32_t>> full_view = buf.get();

	// Test first
	auto first_50 = full_view.first(50);
	CHECK(first_50.count() == 50);
	CHECK(first_50[0] == 0);
	CHECK(first_50[49] == 490);

	// Test last
	auto last_30 = full_view.last(30);
	CHECK(last_30.count() == 30);
	CHECK(last_30[0] == 700);
	CHECK(last_30[29] == 990);
}

TEST_CASE("buffer_view_chaining") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_view<BufferLike<uint32_t>> buf = *allocate_array<uint32_t>(alloc, 1000, MemoryUsage::eCPUonly);

	// Fill with test data
	for (int i = 0; i < 1000; i++) {
		buf[i] = i;
	}

	view<BufferLike<uint32_t>> full_view = buf.get();

	// Chain operations
	auto chained = full_view.slice(100, 800).first(400).last(200);

	CHECK(chained.count() == 200);
	CHECK(chained[0] == 300);
	CHECK(chained[199] == 499);
}

TEST_CASE("buffer_view_size_calculation") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_view<BufferLike<uint32_t>> buf = *allocate_array<uint32_t>(alloc, 512, MemoryUsage::eCPUonly);
	view<BufferLike<uint32_t>> full_view = buf.get();

	CHECK(full_view.count() == 512);
	CHECK(full_view.size_bytes() == 512 * sizeof(uint32_t));

	auto half = full_view.first(256);
	CHECK(half.count() == 256);
	CHECK(half.size_bytes() == 256 * sizeof(uint32_t));
}

// ============================================================================
// Allocation and Deallocation
// ============================================================================

TEST_CASE("allocate_buffer_with_bci") {
	Allocator allocator(test_context.runtime->get_vk_resource());

	BufferCreateInfo bci{ .memory_usage = MemoryUsage::eGPUonly, .size = 4096, .alignment = 16 };

	ptr<BufferLike<uint32_t>> buf;
	auto result = allocator.allocate_memory(std::span{ static_cast<ptr_base*>(&buf), 1 }, std::span{ &bci, 1 });

	CHECK(result);
	CHECK(buf);
	CHECK(buf.device_address != 0);

	allocator.deallocate(std::span{ static_cast<ptr_base*>(&buf), 1 });
}

TEST_CASE("unique_buffer_ownership") {
	Allocator allocator(test_context.runtime->get_vk_resource());

	uint64_t buf_addr = 0;
	{
		Unique_ptr<BufferLike<float>> buf = *allocate_memory<float>(allocator, MemoryUsage::eCPUonly);

		CHECK(*buf);
		buf_addr = buf->device_address;
		CHECK(buf_addr != 0);

		// Set a value
		**buf = 42.0f;
		CHECK(**buf == 42.0f);
		// Should automatically deallocate when going out of scope
	}

	// Allocate a new buffer - might reuse the address from freelist
	Unique_ptr<BufferLike<float>> buf2 = *allocate_memory<float>(allocator, MemoryUsage::eCPUonly);
	CHECK(*buf2);
}

TEST_CASE("unique_buffer_view_ownership") {
	Allocator allocator(test_context.runtime->get_vk_resource());

	Unique_view<BufferLike<uint32_t>> buf = *allocate_array<uint32_t>(allocator, 256, MemoryUsage::eCPUonly);

	// Fill with data
	for (int i = 0; i < 256; i++) {
		buf[i] = i * 2;
	}

	{
		// Create a slice view
		view<BufferLike<uint32_t>> slice = buf.get().slice(50, 100);
		CHECK(slice);
		CHECK(slice.count() == 100);

		// Verify data
		for (int i = 0; i < 100; i++) {
			CHECK(slice[i] == (50 + i) * 2);
		}
		// Slice goes out of scope but buffer remains valid
	}

	// Buffer still accessible
	CHECK(buf[0] == 0);
	CHECK(buf[255] == 510);
}
