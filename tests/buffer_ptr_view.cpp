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

	Unique_ptr<BufferLike<float>> foo = *allocate_array<float>(alloc, 16, MemoryUsage::eCPUonly);

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

TEST_CASE("shader ptr access") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_ptr<BufferLike<float>> foo = *allocate_array<float>(alloc, 4, MemoryUsage::eCPUonly);
	for (int i = 0; i < 4; i++) {
		foo[i] = (i + 1);
	}

	auto buf0 = vuk::acquire("b0", foo.get(), vuk::Access::eNone);

	auto pass = lift_compute(test_context.runtime->get_pipeline(vuk::PipelineBaseCreateInfo::from_inline_glsl(R"(#version 460
#pragma shader_stage(compute)
#include <runtime>

layout (push_constant) uniform data {
	REF(float) data_in;
};

layout (local_size_x = 1) in;

void main() {
	ARRAY(data_in)[gl_GlobalInvocationID.x] *= 2;
}
)")));
	pass(4, 1, 1, buf0);
	buf0.wait(*test_context.allocator, test_context.compiler);
	auto test = { 2.f, 4.f, 6.f, 8.f };
	auto schpen = std::span(&foo[0], 4);
	CHECK(schpen == std::span(test));
}
/*
TEST_CASE("shader buffer access (ptr)") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_ptr<BufferLike<float>> foo = *allocate_array<float>(alloc, 4, MemoryUsage::eCPUonly);
	for (int i = 0; i < 4; i++) {
		foo[i] = (i + 1);
	}

	auto buf0 = vuk::acquire("b0", foo.get(), vuk::Access::eNone);

	auto pass = lift_compute(test_context.runtime->get_pipeline(vuk::PipelineBaseCreateInfo::from_inline_glsl(R"(#version 460
#pragma shader_stage(compute)
#include <runtime>

layout (std430, binding = 0) buffer coherent BufferIn {
	float[] data_in;
};

layout (local_size_x = 1) in;

void main() {
	data_in[gl_GlobalInvocationID.x] *= 2;
}
)")));
	pass(4, 1, 1, buf0);
	buf0.wait(*test_context.allocator, test_context.compiler);
	auto test = { 2.f, 4.f, 6.f, 8.f };
	auto schpen = std::span(&foo[0], 4);
	CHECK(schpen == std::span(test));
}*/

TEST_CASE("generic view from array") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_ptr<BufferLike<float>> foo = *allocate_array<float>(alloc, 16, MemoryUsage::eCPUonly);
	BVCI bvci{ .ptr = foo.get(), .vci = { .elem_size = sizeof(float), .count = 16 } };
	view<float> view;
	alloc.allocate_memory_views({ static_cast<generic_view_base*>(&view), 1 }, { &bvci, 1 });

	for (int i = 0; i < 4; i++) {
		view[i] = i;
	}

	for (int i = 0; i < 4; i++) {
		view[i] *= i;
	}

	for (int i = 0; i < 4; i++) {
		CHECK(view[i] == i * i);
	}

	alloc.deallocate(std::span{ static_cast<generic_view_base*>(&view), 1 });
}

TEST_CASE("generic view from array with helper") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_ptr<BufferLike<float>> foo = *allocate_array<float>(alloc, 16, MemoryUsage::eCPUonly);
	Unique<view<float>> view = *generic_view_from_array(alloc, foo.get(), 16);

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

TEST_CASE("memory view from array with helper") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_ptr<BufferLike<float>> foo = *allocate_array<float>(alloc, 16, MemoryUsage::eCPUonly);
	// concrete views don't need allocations
	view<BufferLike<float>> view = ::view<BufferLike<float>>{ foo.get(), 16 };

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

void sqr_generic(view<float> view) {
	for (int i = 0; i < view.count(); i++) {
		view[i] *= i;
	}
}

void sqr_specific(view<BufferLike<float>> view) {
	for (int i = 0; i < view.count(); i++) {
		view[i] *= i;
	}
}

TEST_CASE("function taking views") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_ptr<BufferLike<float>> foo = *allocate_array<float>(alloc, 16, MemoryUsage::eCPUonly);
	// concrete views don't need allocations
	view<BufferLike<float>> v = ::view<BufferLike<float>>{ foo.get(), 16 };

	for (int i = 0; i < 4; i++) {
		v[i] = i;
	}

	sqr_generic(v);
	sqr_specific(v);

	for (int i = 0; i < 4; i++) {
		CHECK(v[i] == i * i * i);
	}
}

TEST_CASE("shader buffer access (view)") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_ptr<BufferLike<float>> foo = *allocate_array<float>(alloc, 4, MemoryUsage::eCPUonly);
	for (int i = 0; i < 4; i++) {
		foo[i] = (i + 1);
	}

	view<BufferLike<float>> v = ::view<BufferLike<float>>{ foo.get(), 16 };

	auto buf0 = vuk::acquire("b0", v, vuk::Access::eNone);

	auto pass = lift_compute(test_context.runtime->get_pipeline(vuk::PipelineBaseCreateInfo::from_inline_glsl(R"(#version 460
#pragma shader_stage(compute)
#include <runtime>

layout (std430, binding = 0) buffer coherent BufferIn {
	float[] data_in;
};

layout (local_size_x = 1) in;

void main() {
	data_in[gl_GlobalInvocationID.x] *= 2;
}
)")));
	pass(4, 1, 1, buf0);
	buf0.wait(*test_context.allocator, test_context.compiler);
	auto test = { 2.f, 4.f, 6.f, 8.f };
	auto schpen = std::span(&foo[0], 4);
	CHECK(schpen == std::span(test));
}

TEST_CASE("adapt type to IR") {
	BufferCreateInfo bci{ .memory_usage = MemoryUsage::eCPUonly, .size = 5, .alignment = 3 };
	using adaptor = erased_tuple_adaptor<BufferCreateInfo>;

	CHECK(*reinterpret_cast<size_t*>(adaptor::get(&bci, 1)) == 5);

	char storage[sizeof(BufferCreateInfo)];
	std::array args = { (void*)&bci.memory_usage, (void*)&bci.size, (void*)&bci.alignment };
	adaptor::construct(&storage, std::span(args));
	CHECK(*reinterpret_cast<BufferCreateInfo*>(storage) == bci);

	CHECK(strcmp(adaptor::member_names[0], "memory_usage") == 0);

	auto float_ty = to_IR_type<float>();
	auto u32_ty = to_IR_type<uint32_t>();
	auto bci_ty = to_IR_type<BufferCreateInfo>();
	auto vty = to_IR_type<view<BufferLike<float[]>>>();
}

template<class T>
inline val_view<BufferLike<T>> clear(val_view<BufferLike<T>> in, T clear_value, VUK_CALLSTACK) {
	auto clear = make_pass(
	    "clear",
	    [=](CommandBuffer& cbuf, VUK_ARG(Buffer<T>, Access::eTransferRW) dst) {
		    cbuf.fill_buffer(dst->to_byte_view(), clear_value);
		    return dst;
	    },
	    DomainFlagBits::eAny);

	return clear(std::move(in), VUK_CALL);
}
/*
TEST_CASE("allocate ptr and view in IR") {
	auto buf0 = vuk::declare_ptr<float>("jacob", { .memory_usage = MemoryUsage::eCPUonly, .size = 16 });

	auto view = buf0.implicit_view();
	clear(buf0.implicit_view(), 0.f);

	auto pass = lift_compute(test_context.runtime->get_pipeline(vuk::PipelineBaseCreateInfo::from_inline_glsl(R"(#version 460
#pragma shader_stage(compute)
#include <runtime>

layout (push_constant) uniform data {
	REF(float) data_in;
};

layout (local_size_x = 1) in;

void main() {
	ARRAY(data_in)[gl_GlobalInvocationID.x] = (gl_GlobalInvocationID.x + 1);
}
)")));
	pass(4, 1, 1, buf0);
	auto res = *buf0.get(*test_context.allocator, test_context.compiler);
	auto test = { 1.f, 2.f, 3.f, 4.f };
	auto schpen = std::span(&res[0], 4);
	CHECK(schpen == std::span(test));
}*/
/*
TEST_CASE("allocate view in IR") {
	auto buf0 = vuk::declare_buf<float>("jacob", { .memory_usage = MemoryUsage::eCPUonly, .size = 16 });

	clear(buf0, 0.f);

	auto pass = lift_compute(test_context.runtime->get_pipeline(vuk::PipelineBaseCreateInfo::from_inline_glsl(R"(#version 460
#pragma shader_stage(compute)
#include <runtime>

layout (push_constant) uniform data {
	REF(float) data_in;
};

layout (local_size_x = 1) in;

void main() {
	ARRAY(data_in)[gl_GlobalInvocationID.x] = (gl_GlobalInvocationID.x + 1);
}
)")));
	pass(4, 1, 1, buf0->ptr);
	auto res = *buf0.get(*test_context.allocator, test_context.compiler, { .dump_graph = true });
	auto test = { 1.f, 2.f, 3.f, 4.f };
	auto schpen = res.to_span();
	CHECK(schpen == std::span(test));
}*/

struct Bigbog {
	ptr<BufferLike<float>> the_boof;
	ptr<BufferLike<uint32_t>> the_beef;
	float a_milkshake;
	uint32_t a_pilkshake;
};

ADAPT_STRUCT_FOR_IR(Bigbog, the_boof, the_beef, a_milkshake, a_pilkshake);

TEST_CASE("composite transport") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Bigbog boog{ .a_milkshake = 14.f };
	Unique_ptr<BufferLike<float>> foo = *allocate_array<float>(alloc, 4, MemoryUsage::eCPUonly);
	boog.the_boof = static_cast<ptr<BufferLike<float>>>(foo.get());
	Unique_ptr<BufferLike<uint32_t>> foo2 = *allocate_array<uint32_t>(alloc, 4, MemoryUsage::eCPUonly);
	boog.the_beef = static_cast<ptr<BufferLike<uint32_t>>>(foo2.get());

	auto buf0 = vuk::acquire("jacobious_boog", boog, vuk::Access::eNone);
	auto dogget = vuk::acquire("dogget", 12u, vuk::Access::eNone);

	auto pass = vuk::make_pass("transport", [](CommandBuffer& cb, VUK_ARG(Bigbog, Access::eTransferWrite) bogbig, VUK_ARG(uint32_t, Access::eNone) doggets) {
		cb.fill_buffer(Buffer<uint32_t>{ bogbig->the_beef, 4 }.to_byte_view(), doggets);
		uint32_t a;
		memcpy(&a, &bogbig->a_milkshake, sizeof(float)); // yes this will go away
		cb.fill_buffer(Buffer<float>{ bogbig->the_boof, 4 }.to_byte_view(), a);
	});
	pass(buf0, dogget);
	auto res = *buf0.get(*test_context.allocator, test_context.compiler);
	{
		auto test = { res.a_milkshake, res.a_milkshake, res.a_milkshake, res.a_milkshake };
		auto schpen = std::span(&res.the_boof[0], 4);
		CHECK(schpen == std::span(test));
	}
	{
		auto test = { 12u, 12u, 12u, 12u };
		auto schpen = std::span(&res.the_beef[0], 4);
		CHECK(schpen == std::span(test));
	}
}

TEST_CASE("composite support for Value") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Bigbog boog{ .a_milkshake = 14.f, .a_pilkshake = 14u };
	Unique_ptr<BufferLike<float>> foo = *allocate_array<float>(alloc, 4, MemoryUsage::eCPUonly);
	boog.the_boof = static_cast<ptr<BufferLike<float>>>(foo.get());
	Unique_ptr<BufferLike<uint32_t>> foo2 = *allocate_array<uint32_t>(alloc, 4, MemoryUsage::eCPUonly);
	boog.the_beef = static_cast<ptr<BufferLike<uint32_t>>>(foo2.get());

	auto buf0 = vuk::acquire("jacobious_boog", boog, vuk::Access::eNone);

	auto pass = vuk::make_pass("transport", [](CommandBuffer& cb, VUK_ARG(Bigbog, Access::eTransferWrite) bogbig, VUK_ARG(uint32_t, Access::eNone) doggets) {
		cb.fill_buffer(Buffer<float>(bogbig->the_boof, 4).to_byte_view(), *(uint32_t*)&bogbig->a_milkshake);
		cb.fill_buffer(Buffer<uint32_t>(bogbig->the_beef, 4).to_byte_view(), doggets);
	});

	pass(buf0, buf0->a_pilkshake);
	auto res = *buf0.get(*test_context.allocator, test_context.compiler);
	{
		auto test = { res.a_milkshake, res.a_milkshake, res.a_milkshake, res.a_milkshake };
		auto schpen = std::span(&res.the_boof[0], 4);
		CHECK(schpen == std::span(test));
	}
	{
		auto test = { 14u, 14u, 14u, 14u };
		auto schpen = std::span(&res.the_beef[0], 4);
		CHECK(schpen == std::span(test));
	}
}