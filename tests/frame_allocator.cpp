#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <doctest/doctest.h>
#include <string_view>

using namespace vuk;

struct AllocatorChecker : DeviceNestedResource {
	int32_t counter = 0;

	AllocatorChecker(DeviceResource& upstream) : DeviceNestedResource(upstream) {}

	Result<void, AllocateException> allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override {
		counter += cis.size();
		return upstream->allocate_buffers(dst, cis, loc);
	}

	void deallocate_buffers(std::span<const Buffer> src) override {
		counter -= src.size();
		upstream->deallocate_buffers(src);
	}

	Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) override {
		counter += cis.size();
		return upstream->allocate_images(dst, cis, loc);
	}

	void deallocate_images(std::span<const Image> src) override {
		counter -= src.size();
		upstream->deallocate_images(src);
	}
};

TEST_CASE("ptr alloc") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	BufferCreateInfo bci{ .mem_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
	ptr<BufferLike<float>> foo;
	alloc.allocate_memory(std::span{ static_cast<ptr_base*>(&foo), 1 }, std::span{ &bci, 1 });

	*foo = 4;

	*foo *= 3;

	CHECK(*foo == 12);

	alloc.deallocate(std::span{ static_cast<ptr_base*>(&foo), 1 });
}

TEST_CASE("ptr with struct") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	BufferCreateInfo bci{ .mem_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
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

	BufferCreateInfo bci{ .mem_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
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

	*foo = 4;

	*foo *= 3;

	CHECK(*foo == 12);
}

TEST_CASE("array with helper") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_ptr<BufferLike<float[]>> foo = *allocate_array<float>(alloc, 16, MemoryUsage::eCPUonly);

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

	Unique_ptr<BufferLike<float[]>> foo = *allocate_array<float>(alloc, 4, MemoryUsage::eCPUonly);
	for (int i = 0; i < 4; i++) {
		foo[i] = (i + 1);
	}

	auto buf0 = vuk::acquire_ptr("b0", foo.get(), vuk::Access::eNone);

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

TEST_CASE("shader buffer access (ptr)") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_ptr<BufferLike<float[]>> foo = *allocate_array<float>(alloc, 4, MemoryUsage::eCPUonly);
	for (int i = 0; i < 4; i++) {
		foo[i] = (i + 1);
	}

	auto buf0 = vuk::acquire_ptr("b0", foo.get(), vuk::Access::eNone);

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
	pass(4, 1, 1, buf0.implicit_view());
	buf0.wait(*test_context.allocator, test_context.compiler);
	auto test = { 2.f, 4.f, 6.f, 8.f };
	auto schpen = std::span(&foo[0], 4);
	CHECK(schpen == std::span(test));
}

TEST_CASE("generic view from array") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_ptr<BufferLike<float[]>> foo = *allocate_array<float>(alloc, 16, MemoryUsage::eCPUonly);
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

	Unique_ptr<BufferLike<float[]>> foo = *allocate_array<float>(alloc, 16, MemoryUsage::eCPUonly);
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

	Unique_ptr<BufferLike<float[]>> foo = *allocate_array<float>(alloc, 16, MemoryUsage::eCPUonly);
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

	Unique_ptr<BufferLike<float[]>> foo = *allocate_array<float>(alloc, 16, MemoryUsage::eCPUonly);
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

	Unique_ptr<BufferLike<float[]>> foo = *allocate_array<float>(alloc, 4, MemoryUsage::eCPUonly);
	for (int i = 0; i < 4; i++) {
		foo[i] = (i + 1);
	}

	view<BufferLike<float>> v = ::view<BufferLike<float>>{ foo.get(), 16 };

	auto buf0 = vuk::acquire_view("b0", v, vuk::Access::eNone);

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
	BufferCreateInfo bci{ .mem_usage = MemoryUsage::eCPUonly, .size = 5, .alignment = 3 };
	using adaptor = erased_tuple_adaptor<BufferCreateInfo>;

	CHECK(*reinterpret_cast<size_t*>(adaptor::get(&bci, 1)) == 5);

	char storage[sizeof(BufferCreateInfo)];
	std::array args = { (void*)&bci.mem_usage, (void*)&bci.size, (void*)&bci.alignment };
	adaptor::construct(&storage, std::span(args));
	CHECK(*reinterpret_cast<BufferCreateInfo*>(storage) == bci);

	CHECK(strcmp(adaptor::member_names[0], "mem_usage") == 0);

	auto float_ty = to_IR_type<float>();
	auto u32_ty = to_IR_type<uint32_t>();
	auto bci_ty = to_IR_type<BufferCreateInfo>();
	auto vty = to_IR_type<view<BufferLike<float[]>>>();
}

template<class W, class T, class U>
void set(Value<W>& t, U T::*ptr, U arg)
  requires std::is_base_of_v<ptr_base, W>
{
	auto index = index_of<decltype(ptr)>(erased_tuple_adaptor<BufferCreateInfo>::members);
	auto def_or_v = get_def(t.get_head());
	if (!def_or_v || !def_or_v->is_ref) {
		return;
	}
	auto def = def_or_v->ref;
	def.node->construct.args[index] = current_module->make_constant(arg);
}

template<class T>
inline val_view<BufferLike<T>> clear(val_view<BufferLike<T>> in, T clear_value, VUK_CALLSTACK) {
	auto clear = make_pass(
	    "clear",
	    [=](CommandBuffer& cbuf, VUK_ARG(Buffer2<T>, Access::eTransferRW) dst) {
		    cbuf.fill_buffer(dst->to_byte_view(), clear_value);
		    return dst;
	    },
	    DomainFlagBits::eAny);

	return clear(std::move(in), VUK_CALL);
}

TEST_CASE("allocate ptr and view in IR") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	auto buf0 = vuk::declare_ptr<float>("jacob");
	// set directly on struct from an immediate
	buf0.def()->mem_usage = MemoryUsage::eCPUonly;
	// set from function that could take Value
	buf0.set_size_bytes(16);

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
}

/*
TEST_CASE("superframe allocator, uncached resource") {
  AllocatorChecker ac(*test_context.sfa_resource);
  DeviceSuperFrameResource sfr(ac, 2);

  Buffer buf;
  BufferCreateInfo bci{ .mem_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
  sfr.allocate_buffers(std::span{ &buf, 1 }, std::span{ &bci, 1 }, {});
  sfr.deallocate_buffers(std::span{ &buf, 1 });
  REQUIRE(ac.counter == 1);
  sfr.get_next_frame();
  REQUIRE(ac.counter == 1);
  sfr.get_next_frame();
  REQUIRE(ac.counter == 0);
}

TEST_CASE("superframe allocator, uncached resource") {
  AllocatorChecker ac(*test_context.sfa_resource);
  DeviceSuperFrameResource sfr(ac, 2);

  Buffer buf;
  BufferCreateInfo bci{ .mem_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
  sfr.allocate_buffers(std::span{ &buf, 1 }, std::span{ &bci, 1 }, {});
  sfr.deallocate_buffers(std::span{ &buf, 1 });
  REQUIRE(ac.counter == 1);
  sfr.get_next_frame();
  REQUIRE(ac.counter == 1);
  sfr.get_next_frame();
  REQUIRE(ac.counter == 0);
}

 TEST_CASE("frame allocator, uncached resource") {
  REQUIRE(test_context.prepare());

  AllocatorChecker ac(*test_context.sfa_resource);
  DeviceSuperFrameResource sfr(ac, 2);

  Buffer buf;
  BufferCreateInfo bci{ .mem_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
  auto& fa = sfr.get_next_frame();
  fa.allocate_buffers(std::span{ &buf, 1 }, std::span{ &bci, 1 }, {});
  REQUIRE(ac.counter == 1);
  sfr.get_next_frame();
  REQUIRE(ac.counter == 1);
  sfr.get_next_frame();
  REQUIRE(ac.counter == 0);
}

TEST_CASE("frame allocator, cached resource") {
  REQUIRE(test_context.prepare());

  AllocatorChecker ac(*test_context.sfa_resource);
  DeviceSuperFrameResource sfr(ac, 2);

  Image im;
  ImageCreateInfo ici{ .format = vuk::Format::eR8G8B8A8Srgb, .extent = vuk::Extent3D{100, 100, 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment };
  auto& fa = sfr.get_next_frame();
  fa.allocate_images(std::span{ &im, 1 }, std::span{ &ici, 1 }, {});
  REQUIRE(ac.counter == 1);
  sfr.get_next_frame();
  sfr.force_collect();
  REQUIRE(ac.counter == 1);
  sfr.get_next_frame();
  REQUIRE(ac.counter == 1);
  sfr.get_next_frame();
  REQUIRE(ac.counter == 0);
}

TEST_CASE("frame allocator, cached resource identity") {
  REQUIRE(test_context.prepare());

  AllocatorChecker ac(*test_context.sfa_resource);
  DeviceSuperFrameResource sfr(ac, 2);

  Image im1;
  Image im2;
  ImageCreateInfo ici{ .format = vuk::Format::eR8G8B8A8Srgb, .extent = vuk::Extent3D{ 100, 100, 1 }, .usage = vuk::ImageUsageFlagBits::eColorAttachment };
  {
    auto& fa = sfr.get_next_frame();
    fa.allocate_images(std::span{ &im1, 1 }, std::span{ &ici, 1 }, {});
    fa.allocate_images(std::span{ &im2, 1 }, std::span{ &ici, 1 }, {});
  }
  REQUIRE(im1 != im2);
  Image im3;
  Image im4;
  {
    auto& fa = sfr.get_next_frame();
    fa.allocate_images(std::span{ &im3, 1 }, std::span{ &ici, 1 }, {});
    fa.allocate_images(std::span{ &im4, 1 }, std::span{ &ici, 1 }, {});
  }
  REQUIRE((im1 == im3 || im1 == im4));
  REQUIRE((im2 == im3 || im2 == im4));
}

TEST_CASE("multiframe allocator, uncached resource") {
  REQUIRE(test_context.prepare());

  AllocatorChecker ac(*test_context.sfa_resource);
  DeviceSuperFrameResource sfr(ac, 2);

  Buffer buf;
  BufferCreateInfo bci{ .mem_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
  auto& mfa = sfr.get_multiframe_allocator(3);
  mfa.allocate_buffers(std::span{ &buf, 1 }, std::span{ &bci, 1 }, {});
  REQUIRE(ac.counter == 1);
  sfr.get_next_frame();
  sfr.get_next_frame();
  sfr.get_next_frame();
  REQUIRE(ac.counter == 0);
}


TEST_CASE("multiframe allocator, cached resource") {
  REQUIRE(test_context.prepare());

  AllocatorChecker ac(*test_context.sfa_resource);
  DeviceSuperFrameResource sfr(ac, 2);

  Image im;
  ImageCreateInfo ici{ .format = vuk::Format::eR8G8B8A8Srgb, .extent = vuk::Extent3D{ 100, 100, 1 }, .usage = vuk::ImageUsageFlagBits::eColorAttachment };
  auto& mfa = sfr.get_multiframe_allocator(3);
  mfa.allocate_images(std::span{ &im, 1 }, std::span{ &ici, 1 }, {});
  REQUIRE(ac.counter == 1);
  sfr.get_next_frame();
  sfr.get_next_frame();
  sfr.get_next_frame();
  sfr.force_collect();
  REQUIRE(ac.counter == 1);
  sfr.get_next_frame();
  REQUIRE(ac.counter == 1);
  sfr.get_next_frame();
  REQUIRE(ac.counter == 0);
}

TEST_CASE("multiframe allocator, cached resource identity for different MFAs") {
  REQUIRE(test_context.prepare());

  AllocatorChecker ac(*test_context.sfa_resource);
  DeviceSuperFrameResource sfr(ac, 2);

  Image im1;
  Image im2;
  ImageCreateInfo ici{ .format = vuk::Format::eR8G8B8A8Srgb, .extent = vuk::Extent3D{ 100, 100, 1 }, .usage = vuk::ImageUsageFlagBits::eColorAttachment };
  {
    auto& mfa = sfr.get_multiframe_allocator(3);
    mfa.allocate_images(std::span{ &im1, 1 }, std::span{ &ici, 1 }, {});
    mfa.allocate_images(std::span{ &im2, 1 }, std::span{ &ici, 1 }, {});
  }
  REQUIRE(im1 != im2);
  Image im3;
  Image im4;
  {
    auto& mfa = sfr.get_multiframe_allocator(3);
    mfa.allocate_images(std::span{ &im3, 1 }, std::span{ &ici, 1 }, {});
    mfa.allocate_images(std::span{ &im4, 1 }, std::span{ &ici, 1 }, {});
  }
  REQUIRE(im3 != im4);
  REQUIRE((im3 != im1 && im3 != im2));
  REQUIRE((im4 != im1 && im4 != im2));
}*/