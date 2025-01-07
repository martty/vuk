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



/*
TEST_CASE("superframe allocator, uncached resource") {
  AllocatorChecker ac(*test_context.sfa_resource);
  DeviceSuperFrameResource sfr(ac, 2);

  Buffer buf;
  BufferCreateInfo bci{ .memory_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
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
  BufferCreateInfo bci{ .memory_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
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
  BufferCreateInfo bci{ .memory_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
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
  BufferCreateInfo bci{ .memory_usage = vuk::MemoryUsage::eCPUonly, .size = 1024 };
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