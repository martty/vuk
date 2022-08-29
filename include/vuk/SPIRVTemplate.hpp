#pragma once

#include "vuk/CommandBuffer.hpp"
#include "vuk/Future.hpp"
#include "vuk/RenderGraph.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <spirv-headers/spirv.hpp>

namespace vuk {
	namespace spirv {
		template<typename T, std::size_t M, std::size_t N, std::size_t... MIndexes, std::size_t... NIndexes>
		constexpr auto concat_array_impl(std::array<T, M> const& m, std::array<T, N> const& n, std::index_sequence<MIndexes...>, std::index_sequence<NIndexes...>) {
			return std::array<T, M + N>{ m[MIndexes]..., n[NIndexes]... };
		}

		template<typename T, std::size_t M, std::size_t N>
		constexpr auto operator<<(std::array<T, M> const& m, std::array<T, N> const& n) {
			return concat_array_impl<T, M, N>(m, n, std::make_index_sequence<M>(), std::make_index_sequence<N>());
		}

		template<typename... Ts>
		constexpr auto concat_array(Ts&&... arrs) {
			return (arrs << ...);
		}

		template<std::size_t N, typename T>
		constexpr auto array_copy(const T* src) {
			std::array<T, N> res;
			std::copy(src, src + N, res.begin());
			return res;
		}

		static constexpr std::array<uint32_t, 0> no_spirv = {};

		constexpr uint32_t op(spv::Op inop, uint32_t word_count) {
			uint32_t lower = inop & spv::OpCodeMask;
			uint32_t upper = (word_count << spv::WordCountShift) & 0xFFFF0000u;
			return upper | lower;
		}

		constexpr uint32_t count(auto v) {
			return (uint32_t)v.first.size() + (uint32_t)v.second.size();
		}

		template<typename E>
		constexpr auto compile_to_spirv(const E& expr, uint32_t max_id) {
			return expr.to_spirv(max_id);
		}

		template<typename E>
		struct SpvExpression {
			constexpr auto to_spirv(uint32_t counter) const {
				return static_cast<const E*>(this)->to_spirv(counter);
			}
		};

		template<typename E1, typename E2>
		struct Mul : public SpvExpression<Mul<E1, E2>> {
			E1 e1;
			E2 e2;

			constexpr Mul(E1 e1, E2 e2) : e1(e1), e2(e2) {}

		public:
			constexpr auto to_spirv(uint32_t counter) const {
				auto e1id = counter - 1 - count(e2.to_spirv(counter));
				auto [re1a, re1b] = e1.to_spirv(e1id);
				auto [re2a, re2b] = e2.to_spirv(counter - 1);
				auto us = std::array{ op(spv::OpIMul, 5), 6u, counter, counter - 1, e1id };
				return std::pair{ concat_array(re1a, re2a), concat_array(re1b, re2b, us) };
			}
		};

		template<typename E1, typename E2>
		Mul<E1, E2> constexpr operator*(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return Mul<E1, E2>{ *static_cast<const E1*>(&u), *static_cast<const E2*>(&v) };
		}

		template<uint32_t id>
		struct LoadId : public SpvExpression<LoadId<id>> {
			constexpr auto to_spirv(uint32_t counter) const {
				auto us = std::array{ op(spv::OpLoad, 4), 6u, counter, id };
				return std::pair{ no_spirv, us };
			}
		};

		struct Constant : public SpvExpression<Constant> {
			uint32_t value;
			constexpr Constant(uint32_t v) : value(v) {}

			constexpr auto to_spirv(uint32_t counter) const {
				auto us = std::array{ op(spv::OpConstant, 4), 6u, counter, value };
				return std::pair{ us, no_spirv };
			}
		};

		template<typename E1>
		Mul<E1, Constant> constexpr operator*(SpvExpression<E1> const& u, Constant const& v) {
			return Mul<E1, Constant>(*static_cast<const E1*>(&u), v);
		}

		template<typename E2>
		struct StoreId : SpvExpression<StoreId<E2>> {
			const uint32_t id;
			E2 e2;

			constexpr StoreId(const uint32_t& id, E2 e2) : id(id), e2(e2) {}

			constexpr auto to_spirv(uint32_t counter) const {
				auto [re2a, re2b] = e2.to_spirv(counter);
				auto us = std::array{ op(spv::OpStore, 3), id, uint32_t(counter) };
				return std::pair{ re2a, concat_array(re2b, us) };
			}
		};

		template<class Derived>
		struct SPIRVTemplate {
			template<class F>
			static constexpr auto compile(F&& f) {
				constexpr auto first_bit = array_copy<0x00000388 / 4>(Derived::template_bytes);
				constexpr auto second_bit = array_copy<0x0000071c / 4 - 0x00000388 / 4>(Derived::template_bytes + 0x00000388 / 4);
				constexpr auto epilogue = array_copy<6>(Derived::template_bytes + 0x00000728 / 4);

				auto res = compile_to_spirv(Derived::specialize(f), 200);
				return concat_array(first_bit, res.first, second_bit, res.second, epilogue);
			}
		};
	} // namespace spirv

	struct SPIRVBinaryMap : public spirv::SPIRVTemplate<SPIRVBinaryMap> {
		static constexpr const uint32_t template_bytes[] = {
			0x07230203, 0x00010000, 0x0008000a, 0x00000171, 0x00000000, 0x00020011, 0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
			0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000f, 0x0000002d, 0x00060010,
			0x00000004, 0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
			0x00070005, 0x0000000f, 0x4e5f6c67, 0x6f576d75, 0x72476b72, 0x7370756f, 0x00000000, 0x00080005, 0x0000002d, 0x475f6c67, 0x61626f6c, 0x766e496c,
			0x7461636f, 0x496e6f69, 0x00000044, 0x00050005, 0x00000032, 0x66667542, 0x6f437265, 0x00746e75, 0x00040006, 0x00000032, 0x00000000, 0x0000006e,
			0x00030005, 0x00000034, 0x00000000, 0x00050005, 0x00000040, 0x66667542, 0x754f7265, 0x00000074, 0x00060006, 0x00000040, 0x00000000, 0x61746164,
			0x74756f5f, 0x00000000, 0x00030005, 0x00000042, 0x00000000, 0x00050005, 0x00000045, 0x66667542, 0x6e497265, 0x00000030, 0x00060006, 0x00000045,
			0x00000000, 0x61746164, 0x306e695f, 0x00000000, 0x00030005, 0x00000047, 0x00000000, 0x00050005, 0x0000004c, 0x66667542, 0x6e497265, 0x00000031,
			0x00060006, 0x0000004c, 0x00000000, 0x61746164, 0x316e695f, 0x00000000, 0x00030005, 0x0000004e, 0x00000000, 0x00040047, 0x0000000f, 0x0000000b,
			0x00000018, 0x00040047, 0x0000002d, 0x0000000b, 0x0000001c, 0x00040048, 0x00000032, 0x00000000, 0x00000018, 0x00050048, 0x00000032, 0x00000000,
			0x00000023, 0x0000000c, 0x00030047, 0x00000032, 0x00000003, 0x00040047, 0x00000034, 0x00000022, 0x00000000, 0x00040047, 0x00000034, 0x00000021,
			0x00000004, 0x00040047, 0x0000003f, 0x00000006, 0x00000004, 0x00040048, 0x00000040, 0x00000000, 0x00000017, 0x00050048, 0x00000040, 0x00000000,
			0x00000023, 0x00000000, 0x00030047, 0x00000040, 0x00000003, 0x00040047, 0x00000042, 0x00000022, 0x00000000, 0x00040047, 0x00000042, 0x00000021,
			0x00000001, 0x00040047, 0x00000044, 0x00000006, 0x00000004, 0x00040048, 0x00000045, 0x00000000, 0x00000017, 0x00050048, 0x00000045, 0x00000000,
			0x00000023, 0x00000000, 0x00030047, 0x00000045, 0x00000003, 0x00040047, 0x00000047, 0x00000022, 0x00000000, 0x00040047, 0x00000047, 0x00000021,
			0x00000000, 0x00040047, 0x0000004b, 0x00000006, 0x00000004, 0x00040048, 0x0000004c, 0x00000000, 0x00000017, 0x00050048, 0x0000004c, 0x00000000,
			0x00000023, 0x00000000, 0x00030047, 0x0000004c, 0x00000003, 0x00040047, 0x0000004e, 0x00000022, 0x00000000, 0x00040047, 0x0000004e, 0x00000021,
			0x00000002, 0x00040047, 0x00000013, 0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006,
			0x00000020, 0x00000000, 0x00040017, 0x00000007, 0x00000006, 0x00000003, 0x00040020, 0x0000000e, 0x00000001, 0x00000007, 0x0004003b, 0x0000000e,
			0x0000000f, 0x00000001, 0x0004002b, 0x00000006, 0x00000011, 0x00000040, 0x0004002b, 0x00000006, 0x00000012, 0x00000001, 0x0006002c, 0x00000007,
			0x00000013, 0x00000011, 0x00000012, 0x00000012, 0x0004002b, 0x00000006, 0x0000001c, 0x00000000, 0x0004003b, 0x0000000e, 0x0000002d, 0x00000001,
			0x0003001e, 0x00000032, 0x00000006, 0x00040020, 0x00000033, 0x00000002, 0x00000032, 0x0004003b, 0x00000033, 0x00000034, 0x00000002, 0x00040015,
			0x00000035, 0x00000020, 0x00000001, 0x0004002b, 0x00000035, 0x00000036, 0x00000000, 0x00040020, 0x00000037, 0x00000002, 0x00000006, 0x00020014,
			0x0000003a, 0x0003001d, 0x0000003f, 0x00000006, 0x0003001e, 0x00000040, 0x0000003f, 0x00040020, 0x00000041, 0x00000002, 0x00000040, 0x0004003b,
			0x00000041, 0x00000042, 0x00000002, 0x0003001d, 0x00000044, 0x00000006, 0x0003001e, 0x00000045, 0x00000044, 0x00040020, 0x00000046, 0x00000002,
			0x00000045, 0x0004003b, 0x00000046, 0x00000047, 0x00000002, 0x0003001d, 0x0000004b, 0x00000006, 0x0003001e, 0x0000004c, 0x0000004b, 0x00040020,
			0x0000004d, 0x00000002, 0x0000004c, 0x0004003b, 0x0000004d, 0x0000004e, 0x00000002, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
			0x000200f8, 0x00000005, 0x000300f7, 0x00000054, 0x00000000, 0x000300fb, 0x0000001c, 0x00000055, 0x000200f8, 0x00000055, 0x0004003d, 0x00000007,
			0x0000002f, 0x0000002d, 0x0004003d, 0x00000007, 0x0000005d, 0x0000000f, 0x00050084, 0x00000007, 0x0000005e, 0x0000005d, 0x00000013, 0x00050051,
			0x00000006, 0x00000060, 0x0000002f, 0x00000002, 0x00050051, 0x00000006, 0x00000062, 0x0000005e, 0x00000001, 0x00050084, 0x00000006, 0x00000063,
			0x00000060, 0x00000062, 0x00050051, 0x00000006, 0x00000065, 0x0000005e, 0x00000000, 0x00050084, 0x00000006, 0x00000066, 0x00000063, 0x00000065,
			0x00050051, 0x00000006, 0x00000068, 0x0000002f, 0x00000001, 0x00050084, 0x00000006, 0x0000006b, 0x00000068, 0x00000065, 0x00050080, 0x00000006,
			0x0000006c, 0x00000066, 0x0000006b, 0x00050051, 0x00000006, 0x0000006e, 0x0000002f, 0x00000000, 0x00050080, 0x00000006, 0x00000070, 0x0000006c,
			0x0000006e, 0x00050041, 0x00000037, 0x00000038, 0x00000034, 0x00000036, 0x0004003d, 0x00000006, 0x00000039, 0x00000038, 0x000500ae, 0x0000003a,
			0x0000003b, 0x00000070, 0x00000039, 0x000300f7, 0x0000003d, 0x00000000, 0x000400fa, 0x0000003b, 0x0000003c, 0x0000003d, 0x000200f8, 0x0000003c,
			0x000200f9, 0x00000054, 0x000200f8, 0x0000003d, 0x00060041, 0x00000037, 0x00000049, 0x00000047, 0x00000036, 0x00000070, 0x0004003d, 0x00000006,
			0x0000004a, 0x00000049, 0x00060041, 0x00000037, 0x00000050, 0x0000004e, 0x00000036, 0x00000070, 0x0004003d, 0x00000006, 0x00000051, 0x00000050,
			0x00050080, 0x00000006, 0x00000052, 0x0000004a, 0x00000051, 0x00060041, 0x00000037, 0x00000053, 0x00000042, 0x00000036, 0x00000070, 0x0003003e,
			0x00000053, 0x00000052, 0x000200f9, 0x00000054, 0x000200f8, 0x00000054, 0x000100fd, 0x00010038
		};

		static constexpr const auto A = spirv::LoadId<73>{};
		static constexpr const auto B = spirv::LoadId<80>{};

		template<class F>
		static constexpr auto specialize(F&& f) {
			return spirv::StoreId{ 83u, f(A, B) };
		}
	};

	struct CountWithIndirect {
		CountWithIndirect(uint32_t count, uint32_t wg_size) : workgroup_count((uint32_t)idivceil(count, wg_size)), count(count) {}

		uint32_t workgroup_count;
		uint32_t yz[2] = {1, 1};
		uint32_t count;
	};

	PipelineBaseInfo* static_compute_pbi(const uint32_t* ptr, size_t size, std::string ident) {
		vuk::PipelineBaseCreateInfo pci;
		pci.add_static_spirv(ptr, size, std::move(ident));
		return test_context.context->get_pipeline(pci);
	}

	template<class F>
	inline Future unary_map(Future src, Future dst, Future count, const F& fn) {
		constexpr auto spirv = SPIRVBinaryMap::compile([](auto A, auto B) { return F{}(A); });
		static auto pbi = static_compute_pbi(spirv.data(), spirv.size(), "unary");
		std::shared_ptr<RenderGraph> rgp = std::make_shared<RenderGraph>("unary_map");
		rgp->attach_in("src", std::move(src));
		if (dst) {
			rgp->attach_in("dst", std::move(dst));
		} else {
			rgp->attach_buffer("dst", Buffer{ .memory_usage = vuk::MemoryUsage::eGPUonly });
			rgp->inference_rule("dst", same_size_as("src"));
		}
		rgp->attach_in("count", std::move(count));
		rgp->add_pass(
		    { .name = "unary_map",
		      .resources = { "src"_buffer >> eComputeRead, "dst"_buffer >> eComputeWrite, "count"_buffer >> eComputeRead, "count"_buffer >> eIndirectRead },
		      .execute = [](CommandBuffer& command_buffer) {
			      command_buffer.bind_buffer(0, 0, "src");
			      command_buffer.bind_buffer(0, 1, "dst");
			      command_buffer.bind_buffer(0, 2, "src");
			      command_buffer.bind_buffer(0, 4, "count");
			      command_buffer.bind_compute_pipeline(pbi);
			      command_buffer.dispatch_indirect("count");
		      } });
		return { rgp, "dst+" };
	}
} // namespace vuk