#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <doctest/doctest.h>

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
	ptr<float> foo;
	alloc.allocate_memory(std::span{ static_cast<ptr_base*>(&foo), 1 }, std::span{ &bci, 1 });

	*foo = 4;

	*foo *= 3;

	CHECK(*foo == 12);

	alloc.deallocate(std::span{ static_cast<ptr_base*>(&foo), 1 });
}

TEST_CASE("ptr with struct") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	BufferCreateInfo bci{ .mem_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
	ptr<std::pair<size_t, size_t>> foo;
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

	Unique_ptr<float> foo = *allocate_memory<float>(alloc, MemoryUsage::eCPUonly);

	*foo = 4;

	*foo *= 3;

	CHECK(*foo == 12);
}

TEST_CASE("array with helper") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_ptr<float[]> foo = *allocate_array<float>(alloc, 16, MemoryUsage::eCPUonly);

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

	Unique_ptr<float[]> foo = *allocate_array<float>(alloc, 4, MemoryUsage::eCPUonly);
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