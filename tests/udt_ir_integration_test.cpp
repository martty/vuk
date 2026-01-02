#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <cstring>
#include <doctest/doctest.h>

using namespace vuk;

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
}

struct Bigbog {
	ptr<BufferLike<float>> the_boof;
	ptr<BufferLike<uint32_t>> the_beef;
	float a_milkshake;
	uint32_t a_pilkshake;
};

std::string format_as(const Bigbog&) {
	return "bigbog";
}

ADAPT_STRUCT_FOR_IR(Bigbog, the_boof, the_beef, a_milkshake, a_pilkshake);

namespace vuk {
	void synchronize(Bigbog, struct SyncHelper&) {}
} // namespace vuk

TEST_CASE("composite transport") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Bigbog boog{ .a_milkshake = 14.f };
	Unique_view<BufferLike<float>> foo = *allocate_array<float>(alloc, 4, MemoryUsage::eCPUonly);
	boog.the_boof = static_cast<ptr<BufferLike<float>>>(foo->ptr);
	Unique_view<BufferLike<uint32_t>> foo2 = *allocate_array<uint32_t>(alloc, 4, MemoryUsage::eCPUonly);
	boog.the_beef = static_cast<ptr<BufferLike<uint32_t>>>(foo2->ptr);

	auto buf0 = vuk::acquire("jacobious_boog", boog, vuk::Access::eNone);

	auto pass = vuk::make_pass("transport", [](CommandBuffer& cb, VUK_ARG(Bigbog, Access::eTransferWrite) bogbig, uint32_t doggets) {
		cb.fill_buffer(Buffer<uint32_t>{ bogbig->the_beef, 4 }.to_byte_view(), doggets);
		uint32_t a;
		memcpy(&a, &bogbig->a_milkshake, sizeof(float)); // yes this will go away
		cb.fill_buffer(Buffer<float>{ bogbig->the_boof, 4 }.to_byte_view(), a);
	});
	pass(buf0, 12u);
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
TEST_CASE("composite support for Value") {
  Allocator alloc(test_context.runtime->get_vk_resource());

  Bigbog boog{ .a_milkshake = 15.f, .a_pilkshake = 15u };
  Unique_ptr<BufferLike<float>> foo = *allocate_array<float>(alloc, 4, MemoryUsage::eCPUonly);
  boog.the_boof = *foo;
  Unique_ptr<BufferLike<uint32_t>> foo2 = *allocate_array<uint32_t>(alloc, 4, MemoryUsage::eCPUonly);
  boog.the_beef = *foo2;

  auto buf0 = vuk::acquire("jacobious_boog", boog, vuk::Access::eNone);

  auto pass = vuk::make_pass("transport", [](CommandBuffer& cb, Arg<view<BufferLike<uint32_t>, 4>, Access::eTransferWrite> the_beef, uint32_t pilkshake) {
    cb.fill_buffer(the_beef->to_byte_view(), pilkshake);
  });

  pass(make_view(buf0->the_beef, 4), buf0->a_pilkshake);
  auto res = *buf0.get(*test_context.allocator, test_context.compiler, { .dump_graph = true });
  {
    auto test = { res.a_pilkshake, res.a_pilkshake, res.a_pilkshake, res.a_pilkshake };
    auto schpen = std::span(&res.the_beef[0], 4);
    CHECK(schpen == std::span(test));
  }
}*/
/*
TEST_CASE("aliased constant composite support for Value") {
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
  auto res = *buf0.get(*test_context.allocator, test_context.compiler, {.dump_graph = true});
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
}*/
