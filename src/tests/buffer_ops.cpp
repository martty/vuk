#include "TestContext.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Partials.hpp"
#include "vuk/partials/Map.hpp"
#include <doctest/doctest.h>

using namespace vuk;

TEST_CASE("test text_context preparation") {
	REQUIRE(test_context.prepare());
}

TEST_CASE("test buffer harness") {
	REQUIRE(test_context.prepare());
	auto data = { 1u, 2u, 3u };
	auto [buf, fut] = create_buffer_cross_device(*test_context.allocator, MemoryUsage::eCPUtoGPU, std::span(data));
	auto res = fut.get<Buffer>(*test_context.allocator, test_context.compiler);
	CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(data));
}

TEST_CASE("test buffer upload/download") {
	REQUIRE(test_context.prepare());
	{
		auto data = { 1u, 2u, 3u };
		auto [buf, fut] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));

		auto res = download_buffer(fut).get<Buffer>(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(data));
	}
	{
		auto data = { 1u, 2u, 3u, 4u, 5u };
		auto [buf, fut] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));

		auto res = download_buffer(fut).get<Buffer>(*test_context.allocator, test_context.compiler);
		CHECK(std::span((uint32_t*)res->mapped_ptr, 5) == std::span(data));
	}
}

TEST_CASE("test unary_map") {
	REQUIRE(test_context.prepare());
	{
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector data = { 1u, 2u, 3u };
		// function to apply
		auto func = [](auto A) {
			return A + 3u + 33u;
		};
		std::vector<uint32_t> expected;
		// cpu result
		std::transform(data.begin(), data.end(), std::back_inserter(expected), func);

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 64 };
		auto [_2, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = unary_map<uint32_t>(*test_context.context, func, src, {}, cnt);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((uint32_t*)res->mapped_ptr, data.size());
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
	{
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector data = { 1u, 2u, 3u };
		// function to apply
		auto func = [](auto A) {
			return spirv::select(A > 1u, 1u, 2u);
		};
		std::vector<uint32_t> expected;
		// cpu result
		std::transform(data.begin(), data.end(), std::back_inserter(expected), func);

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 64 };
		auto [_2, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = unary_map<uint32_t>(*test_context.context, func, src, {}, cnt);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((uint32_t*)res->mapped_ptr, data.size());
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
	{
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector data = { 1.f, 2.f, 3.f };
		// function to apply
		auto func = [](auto A) {
			return spirv::select(A > 1.f, 3.f + A, 4.f) * spirv::select(A >= 1.f, 3.f + A, -A);
		};
		std::vector<float> expected;
		// cpu result
		std::transform(data.begin(), data.end(), std::back_inserter(expected), func);

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 64 };
		auto [_2, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = unary_map<float>(*test_context.context, func, src, {}, cnt);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((float*)res->mapped_ptr, data.size());
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
}

TEST_CASE("test binary_map") {
	REQUIRE(test_context.prepare());
	{
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector data0 = { 1.f, 2.f, 3.f };
		std::vector data1 = { 1.f, 2.f, 3.f };
		// function to apply
		auto func = [](auto A, auto B) {
			return spirv::select(A > 1.f, 3.f + A, 4.f) * spirv::select(B >= 1.f, 3.f + B, -B);
		};
		std::vector<float> expected;
		// cpu result
		std::transform(data0.begin(), data0.end(), data1.begin(), std::back_inserter(expected), func);

		// put data on gpu
		auto [_1, src_a] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data0));
		auto [_2, src_b] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data1));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data0.size(), 64 };
		auto [_3, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = binary_map<float>(*test_context.context, src_a, src_b, {}, cnt, func);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((float*)res->mapped_ptr, data0.size());
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
}

struct POD {
	unsigned foo;
	float bar;

	bool operator==(const POD&) const = default;
};

constexpr bool operator==(const std::span<POD>& lhs, const std::span<POD>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<POD>& lhs, const std::span<const POD>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<const POD>& lhs, const std::span<const POD>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

template<>
struct spirv::Type<POD> : spirv::TypeStruct<spirv::Member<Type<uint32_t>>, spirv::Member<Type<float>>> {
	using type = POD;
};

template<class Ctx>
struct spirv::TypeContext<Ctx, spirv::Type<POD>> {
	uint32_t cnt = 0;
	SpvExpression<CompositeExtract<Type<uint32_t>, Ctx, Id>> foo = { static_cast<Ctx&>(*this), cnt++ };
	SpvExpression<CompositeExtract<Type<float>, Ctx, Id>> bar = { static_cast<Ctx&>(*this), cnt++ };
};

TEST_CASE("test unary_map, custom type, casting") {
	REQUIRE(test_context.prepare());
	{
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data

		std::vector data = { POD{ 1, 2.f }, POD{ 1, 3.f }, POD{ 1, 4.f } };
		// function to apply
		auto func = [](auto A) {
			return spirv::make<POD>(spirv::cast<uint32_t>(A.foo) * 2u + spirv::cast<uint32_t>(A.bar), spirv::cast<float>(A.foo) + spirv::cast<float>(A.bar) * 2.f);
		};
		std::vector<POD> expected;
		// cpu result
		std::transform(data.begin(), data.end(), std::back_inserter(expected), func);

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 64 };
		auto [_2, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = unary_map<POD>(*test_context.context, func, src, {}, cnt);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((POD*)res->mapped_ptr, data.size());
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
}

namespace vuk::spirv {
	template<class T,
	         class WT = TypeStruct<Member<Type<T>>>,
	         class VT = Variable<Type<ptr<spv::StorageClassUniform, WT>>, spv::StorageClassUniform>,
	         class MAC = MemberAccessChain<0, VT>,
	         class LV = Load<MAC>>
	struct Uniform : LV {
		using Variable = VT;

		constexpr Uniform(LV ce) : LV(ce) {}
	};

	template<class T,
	         class WT = TypeStruct<Member<Type<T>>>,
	         class VT = Variable<Type<ptr<spv::StorageClassStorageBuffer, WT>>, spv::StorageClassStorageBuffer>,
	         class MAC = MemberAccessChain<0, VT>,
	         class LV = Load<MAC>>
	struct Buffer : LV {
		using Variable = VT;

		//constexpr Buffer(uint32_t id_counter, LV ce) : LV(ce), spvmodule(id_counter) {}
		constexpr Buffer(LV ce) : LV(ce) {}

		constexpr auto& operator&() {
			auto& vt = std::get<0>(this->children);
			return vt;
		}
	};
} // namespace vuk::spirv

TEST_CASE("test unary_map, impure (uniform)") {
	REQUIRE(test_context.prepare());
	{
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector data = { 1u, 2u, 3u };
		uint32_t uni_data = 55u;
		// function to apply
		auto func = [](auto A, spirv::Uniform<uint32_t> v) {
			return A + 3u + v;
		};
		std::vector<uint32_t> expected;
		// cpu result
		std::transform(data.begin(), data.end(), std::back_inserter(expected), [=](auto A) { return A + 3u + uni_data; });

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		auto [_2, unif] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span<uint32_t>(&uni_data, 1));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 64 };
		auto [_3, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = unary_map<uint32_t>(*test_context.context, func, src, {}, cnt, unif);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((uint32_t*)res->mapped_ptr, data.size());
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(out == std::span(expected));
	}
}


TEST_CASE("test unary_map, impure (buffer, multiple variadics)") {
	REQUIRE(test_context.prepare());
	{
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector data = { 1u, 2u, 3u };
		uint32_t initial_data = 0u;
		uint32_t uni_data = 32u;
		// function to apply
		auto func = [](auto A, spirv::Buffer<uint32_t>& v, spirv::Uniform<uint32_t> vv) {
			spirv::atomicIncrement(&v);
			return A + vv;
		};

		std::vector<uint32_t> expected;
		// cpu result
		std::transform(data.begin(), data.end(), std::back_inserter(expected), [=](auto A) { return A + uni_data; });
		uint32_t atomic_expected = data.size();

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		auto [_2, buff] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span<uint32_t>(&initial_data, 1));
		auto [_3, unif] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span<uint32_t>(&uni_data, 1));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 64 };
		auto [_4, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = unary_map<uint32_t>(*test_context.context, func, src, {}, cnt, buff, unif);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((uint32_t*)res->mapped_ptr, data.size());
		auto atomic_res = download_buffer(buff).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto atomic_out = *(uint32_t*)atomic_res->mapped_ptr;
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(atomic_out == atomic_expected);
		CHECK(out == std::span(expected));
	}
}

TEST_CASE("test sideeffects") {
	REQUIRE(test_context.prepare());
	{
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector data = { 1u, 2u, 3u };
		uint32_t initial_data = 0u;
		uint32_t uni_data = 32u;
		// function to apply
		auto func = [](auto A, spirv::Buffer<uint32_t>& v, spirv::Uniform<uint32_t> vv) {
			atomicIncrement(&v);
			atomicIncrement(&v);
			return A + vv;
		};

		std::vector<uint32_t> expected;
		// cpu result
		std::transform(data.begin(), data.end(), std::back_inserter(expected), [=](auto A) { return A + uni_data; });
		uint32_t atomic_expected = 2*data.size();

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		auto [_2, buff] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span<uint32_t>(&initial_data, 1));
		auto [_3, unif] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span<uint32_t>(&uni_data, 1));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 64 };
		auto [_4, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = unary_map<uint32_t>(*test_context.context, func, src, {}, cnt, buff, unif);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((uint32_t*)res->mapped_ptr, data.size());
		auto atomic_res = download_buffer(buff).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto atomic_out = *(uint32_t*)atomic_res->mapped_ptr;
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(atomic_out == atomic_expected);
		CHECK(out == std::span(expected));
	}
}

TEST_CASE("test sideeffects 2") {
	REQUIRE(test_context.prepare());
	{
		if (test_context.rdoc_api)
			test_context.rdoc_api->StartFrameCapture(NULL, NULL);
		// src data
		std::vector data = { 1u, 2u, 3u };
		uint32_t initial_data = 0u;
		uint32_t uni_data = 32u;
		// function to apply
		auto func = [](auto A, spirv::Buffer<uint32_t>& v, spirv::Uniform<uint32_t> vv) {
			auto preop = atomicIncrement(&v);
			return A + vv + preop;
		};

		std::vector<uint32_t> expected;
		// cpu result
		uint32_t atomic_cnt = 0;
		std::transform(data.begin(), data.end(), std::back_inserter(expected), [&](auto A) { return A + uni_data + (atomic_cnt++); });

		// put data on gpu
		auto [_1, src] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(data));
		auto [_2, buff] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span<uint32_t>(&initial_data, 1));
		auto [_3, unif] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span<uint32_t>(&uni_data, 1));
		// put count on gpu
		CountWithIndirect count_data{ (uint32_t)data.size(), 64 };
		auto [_4, cnt] = create_buffer_gpu(*test_context.allocator, DomainFlagBits::eAny, std::span(&count_data, 1));

		// apply function on gpu
		auto calc = unary_map<uint32_t>(*test_context.context, func, src, {}, cnt, buff, unif);
		// bring data back to cpu
		auto res = download_buffer(calc).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto out = std::span((uint32_t*)res->mapped_ptr, data.size());
		auto atomic_res = download_buffer(buff).get<Buffer>(*test_context.allocator, test_context.compiler);
		auto atomic_out = *(uint32_t*)atomic_res->mapped_ptr;
		if (test_context.rdoc_api)
			test_context.rdoc_api->EndFrameCapture(NULL, NULL);
		CHECK(atomic_out == atomic_cnt);
		CHECK(out == std::span(expected));
	}
}