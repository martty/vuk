#include "TestContext.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Partials.hpp"
#include <doctest/doctest.h>

using namespace vuk;

#include <string>

auto make_unary_computation(std::string name, std::string& trace) {
	return make_pass(Name(name.c_str()), [=, &trace](CommandBuffer& cbuf, VUK_BA(Access::eTransferWrite) dst) {
		trace += name;
		trace += " ";
		return dst;
	});
}

auto make_binary_computation(std::string name, std::string& trace) {
	return make_pass(Name(name.c_str()), [=, &trace](CommandBuffer& cbuf, VUK_BA(Access::eTransferWrite) a, VUK_BA(Access::eTransferWrite) b) {
		trace += name;
		trace += " ";
		return a;
	});
}

TEST_CASE("minimal graph is submitted") {
	std::string trace = "";

	auto a = make_unary_computation("a", trace)(declare_buf("_a", { .size = sizeof(uint32_t) * 4, .memory_usage = MemoryUsage::eGPUonly }));
	auto b = make_unary_computation("b", trace)(declare_buf("_b", { .size = sizeof(uint32_t) * 4, .memory_usage = MemoryUsage::eGPUonly }));

	auto d = make_binary_computation("d", trace)(a, b); // d->a, d->b
	auto e = make_unary_computation("e", trace)(a);     // e->a
	e.submit(*test_context.allocator, test_context.compiler);

	trace = trace.substr(0, trace.size() - 1);
	CHECK(trace == "a e");
}

TEST_CASE("computation is never duplicated") {
	std::string trace = "";

	auto a = make_unary_computation("a", trace)(declare_buf("_a", { .size = sizeof(uint32_t) * 4, .memory_usage = MemoryUsage::eGPUonly }));
	auto b = make_unary_computation("b", trace)(declare_buf("_b", { .size = sizeof(uint32_t) * 4, .memory_usage = MemoryUsage::eGPUonly }));

	auto d = make_binary_computation("d", trace)(a, b); // d->a, d->b
	auto e = make_unary_computation("e", trace)(a);     // e->a

	e.submit(*test_context.allocator, test_context.compiler);
	d.submit(*test_context.allocator, test_context.compiler);
	trace = trace.substr(0, trace.size() - 1);
	CHECK(trace == "a e b d");
}

TEST_CASE("computation is never duplicated 2") {
	std::string trace = "";

	auto a = make_unary_computation("a", trace)(declare_buf("_a", { .size = sizeof(uint32_t) * 4, .memory_usage = MemoryUsage::eGPUonly }));
	auto b = make_unary_computation("b", trace)(declare_buf("_b", { .size = sizeof(uint32_t) * 4, .memory_usage = MemoryUsage::eGPUonly }));

	auto d = make_binary_computation("d", trace)(a, b); // d->a, d->b
	d.submit(*test_context.allocator, test_context.compiler);
	d.submit(*test_context.allocator, test_context.compiler);
	trace = trace.substr(0, trace.size() - 1);
	CHECK(trace == "a b d");
}

TEST_CASE("not moving Values will emit relacqs") {
	std::string trace = "";

	auto a = make_unary_computation("a", trace)(declare_buf("_a", { .size = sizeof(uint32_t) * 4, .memory_usage = MemoryUsage::eGPUonly }));
	auto b = make_unary_computation("b", trace)(declare_buf("_b", { .size = sizeof(uint32_t) * 4, .memory_usage = MemoryUsage::eGPUonly }));

	auto d = make_binary_computation("d", trace)(a, b); // d->a, d->b
	d.submit(*test_context.allocator, test_context.compiler);
	trace = trace.substr(0, trace.size() - 1);
	CHECK(trace == "a b d");
}

TEST_CASE("moving Values allows for more efficient building (but no semantic change)") {
	std::string trace = "";

	auto a = make_unary_computation("a", trace)(declare_buf("_a", { .size = sizeof(uint32_t) * 4, .memory_usage = MemoryUsage::eGPUonly }));
	auto b = make_unary_computation("b", trace)(declare_buf("_b", { .size = sizeof(uint32_t) * 4, .memory_usage = MemoryUsage::eGPUonly }));

	auto d = make_binary_computation("d", trace)(std::move(a), std::move(b)); // d->a, d->b
	d.submit(*test_context.allocator, test_context.compiler);
	trace = trace.substr(0, trace.size() - 1);
	CHECK(trace == "a b d");
}

TEST_CASE("moving Values doesn't help if it was leaked before") {
	std::string trace = "";

	auto a = make_unary_computation("a", trace)(declare_buf("_a", { .size = sizeof(uint32_t) * 4, .memory_usage = MemoryUsage::eGPUonly }));
	auto b = make_unary_computation("b", trace)(declare_buf("_b", { .size = sizeof(uint32_t) * 4, .memory_usage = MemoryUsage::eGPUonly }));

	auto d = make_binary_computation("d", trace)(a, b);        // d->a, d->b
	auto e = make_unary_computation("e", trace)(std::move(a)); // e->a <--- a cannot be consumed here! since previously we made d depend on a
	e.submit(*test_context.allocator, test_context.compiler);
	d.submit(*test_context.allocator, test_context.compiler);
	trace = trace.substr(0, trace.size() - 1);
	CHECK(trace == "a e b d");
}
/*
TEST_CASE("can't make Futures out of Values that have already been submitted") {
	std::string trace = "";

	auto a = make_unary_computation("a", trace)(declare_buf("_a", { .size = sizeof(uint32_t) * 4, .memory_usage = MemoryUsage::eGPUonly }));
	auto e = make_unary_computation("e", trace)(a); // e->a
	e.submit(*test_context.allocator, test_context.compiler);
    CHECK_THROWS(a.release()); // we don't allow this, because this Future no longer represents the computation of 'a'
}
*/

/*
TEST_CASE("scheduling single-queue") {
  {
    std::string execution;

    auto buf0 = allocate_buffer(*test_context.allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(uint32_t) * 4 });

    auto write = make_pass("write", [&](CommandBuffer& cbuf, VUK_BA(Access::eTransferWrite) dst) {
      execution += "w";
      return dst;
    });
    auto read = make_pass("read", [&](CommandBuffer& cbuf, VUK_BA(Access::eTransferRead) dst) {
      execution += "r";
      return dst;
    });

    {
      auto b0 = declare_buf("src0", **buf0);
      write(write(b0)).wait(*test_context.allocator, test_context.compiler);
      CHECK(execution == "ww");
      execution = "";
    }
    {
      auto b0 = declare_buf("src0", **buf0);
      read(write(b0)).wait(*test_context.allocator, test_context.compiler);
      CHECK(execution == "wr");
      execution = "";
    }
    {
      auto b0 = declare_buf("src0", **buf0);
      write(read(write(b0))).wait(*test_context.allocator, test_context.compiler);
      CHECK(execution == "wrw");
      execution = "";
    }
    {
      auto b0 = declare_buf("src0", **buf0);
      write(read(read(write(b0)))).wait(*test_context.allocator, test_context.compiler);
      CHECK(execution == "wrrw");
    }
  }
}

TEST_CASE("scheduling with submitted") {
  {
    std::string execution;

    auto buf0 = allocate_buffer(*test_context.allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(uint32_t) * 4 });

    auto write = make_pass("write", [&](CommandBuffer& cbuf, VUK_BA(Access::eTransferWrite) dst) {
      execution += "w";
      return dst;
    });
    auto read = make_pass("read", [&](CommandBuffer& cbuf, VUK_BA(Access::eTransferRead) dst) {
      execution += "r";
      return dst;
    });

    {
      auto written = write(declare_buf("src0", **buf0));
      written.wait(*test_context.allocator, test_context.compiler);
      read(written).wait(*test_context.allocator, test_context.compiler);
      CHECK(execution == "wr");
      execution = "";
    }
    {
      auto written = write(declare_buf("src0", **buf0));
      written.wait(*test_context.allocator, test_context.compiler);
      read(std::move(written)).wait(*test_context.allocator, test_context.compiler);
      CHECK(execution == "wr");
      execution = "";
    }
    {
      auto written = write(declare_buf("src0", **buf0));
      written.wait(*test_context.allocator, test_context.compiler);
      auto res = write(std::move(written));
      res.wait(*test_context.allocator, test_context.compiler);
      CHECK(execution == "ww");
      execution = "";
    }
  }
}

TEST_CASE("multi-queue buffers") {
  {
    std::string execution;

    auto buf0 = allocate_buffer(*test_context.allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(uint32_t) * 4 });

    auto write = make_pass(
        "write_A",
        [&](CommandBuffer& cbuf, VUK_BA(Access::eTransferWrite) dst) {
          cbuf.fill_buffer(dst, 0xf);
          execution += "w";
          return dst;
        },
        DomainFlagBits::eTransferQueue);
    auto read = make_pass(
        "read_B",
        [&](CommandBuffer& cbuf, VUK_BA(Access::eTransferRead) dst) {
          auto dummy = allocate_buffer(*test_context.allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(uint32_t) * 4 });
          cbuf.copy_buffer(**dummy, dst);
          execution += "r";
          return dst;
        },
        DomainFlagBits::eGraphicsQueue);

    {
      auto written = write(declare_buf("src0", **buf0));
      written.wait(*test_context.allocator, test_context.compiler);
      read(written).wait(*test_context.allocator, test_context.compiler);
      CHECK(execution == "wr");
      execution = "";
    }
    {
      auto written = write(declare_buf("src0", **buf0));
      written.wait(*test_context.allocator, test_context.compiler);
      read(std::move(written)).wait(*test_context.allocator, test_context.compiler);
      CHECK(execution == "wr");
      execution = "";
    }
    /* {
  auto written = write(declare_buf("src0", **buf0));
  written.wait(*test_context.allocator, test_context.compiler);
  write(read(std::move(written))).wait(*test_context.allocator, test_context.compiler);
  CHECK(execution == "wrw");
  execution = "";
}*//*
		{
			auto written = write(declare_buf("src0", **buf0));
			read(written).wait(*test_context.allocator, test_context.compiler);
			CHECK(execution == "wr");
			execution = "";
		}
		{
			auto written = write(declare_buf("src0", **buf0));
			read(std::move(written)).wait(*test_context.allocator, test_context.compiler);
			CHECK(execution == "wr");
			execution = "";
		}
		{
			auto written = write(declare_buf("src0", **buf0));
			write(read(std::move(written))).wait(*test_context.allocator, test_context.compiler);
			CHECK(execution == "wrw");
			execution = "";
		}
	}
}*/