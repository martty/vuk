#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <doctest/doctest.h>
#include <string_view>

using namespace vuk;

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
