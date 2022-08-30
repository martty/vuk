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

		template<class T>
		constexpr uint32_t type_id() {
			if constexpr (std::is_same_v<T, uint32_t>) {
				return 6u;
			} else if constexpr (std::is_same_v<T, bool>) {
				return 58u;
			} else {
				assert(0);
			}
		}

		template<typename E>
		struct SpvExpression {
			constexpr auto to_spirv(uint32_t counter) const {
				return static_cast<const E*>(this)->to_spirv(counter);
			}
		};

		template<class T>
		struct Type : public SpvExpression<Type<T>> {
			constexpr Type(T v) {}

			constexpr auto to_spirv(uint32_t counter) const {
				assert(0);
				return std::pair{ no_spirv, no_spirv };
			}
		};

		template<>
		struct Type<bool> {
			constexpr auto to_spirv(uint32_t counter) const {
				auto us = std::array{ op(spv::OpTypeBool, 2), counter };
				return std::pair{ us, no_spirv };
			}
		};

		template<typename E1, typename E2>
		struct Add : public SpvExpression<Add<E1, E2>> {
			static_assert(std::is_same_v<typename E1::type, typename E2::type>);
			using type = typename E1::type;

			E1 e1;
			E2 e2;

			constexpr Add(E1 e1, E2 e2) : e1(e1), e2(e2) {}

		public:
			constexpr auto to_spirv(uint32_t counter) const {
				auto e1id = counter - 1 - count(e2.to_spirv(counter));
				auto [re1a, re1b] = e1.to_spirv(e1id);
				auto [re2a, re2b] = e2.to_spirv(counter - 1);
				auto us = std::array{ op(std::is_floating_point_v<type> ? spv::OpFAdd : spv::OpIAdd, 5), 6u, counter, counter - 1, e1id };
				return std::pair{ concat_array(re1a, re2a), concat_array(re1b, re2b, us) };
			}
		};

		template<typename E1, typename E2>
		Add<E1, E2> constexpr operator+(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return Add<E1, E2>{ *static_cast<const E1*>(&u), *static_cast<const E2*>(&v) };
		}

		template<typename E1, typename E2>
		struct Mul : public SpvExpression<Mul<E1, E2>> {
			static_assert(std::is_same_v<typename E1::type, typename E2::type>);
			using type = typename E1::type;

			E1 e1;
			E2 e2;

			constexpr Mul(E1 e1, E2 e2) : e1(e1), e2(e2) {}

		public:
			constexpr auto to_spirv(uint32_t counter) const {
				auto e1id = counter - 1 - count(e2.to_spirv(counter));
				auto [re1a, re1b] = e1.to_spirv(e1id);
				auto [re2a, re2b] = e2.to_spirv(counter - 1);
				auto us = std::array{ op(std::is_floating_point_v<type> ? spv::OpFMul : spv::OpIMul, 5), 6u, counter, counter - 1, e1id };
				return std::pair{ concat_array(re1a, re2a), concat_array(re1b, re2b, us) };
			}
		};

		template<typename E1, typename E2>
		Mul<E1, E2> constexpr operator*(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return Mul<E1, E2>{ *static_cast<const E1*>(&u), *static_cast<const E2*>(&v) };
		}

		template<class T>
		struct LoadId : public SpvExpression<LoadId<T>> {
			using type = T;
			const uint32_t id;

			constexpr LoadId(const uint32_t& id) : id(id) {}

			constexpr auto to_spirv(uint32_t counter) const {
				auto us = std::array{ op(spv::OpLoad, 4), type_id<T>(), counter, id };
				return std::pair{ no_spirv, us };
			}
		};

		template<class T>
		struct Constant : public SpvExpression<Constant<T>> {
			using type = T;
			T value;
			constexpr Constant(T v) : value(v) {}

			constexpr auto to_spirv(uint32_t counter) const {
				constexpr size_t num_uints = sizeof(T) / sizeof(uint32_t);
				auto as_uints = std::bit_cast<std::array<uint32_t, num_uints>>(value);
				auto us = std::array{ op(spv::OpConstant, 3 + num_uints), type_id<T>(), counter };
				auto with_value = concat_array(us, as_uints);
				return std::pair{ with_value, no_spirv };
			}
		};

		template<typename E1, typename T>
		Add<E1, Constant<T>> constexpr operator+(SpvExpression<E1> const& u, T const& v) {
			return Add<E1, Constant<T>>(*static_cast<const E1*>(&u), Constant<T>(v));
		}

		template<typename E1, typename T>
		requires std::integral<T> Mul<E1, Constant<T>>
		constexpr operator*(SpvExpression<E1> const& u, T const& v) {
			return Mul<E1, Constant<T>>(*static_cast<const E1*>(&u), Constant<T>(v));
		}

		template<typename E2>
		struct StoreId : SpvExpression<StoreId<E2>> {
			using type = void;

			const uint32_t id;
			E2 e2;

			constexpr StoreId(const uint32_t& id, E2 e2) : id(id), e2(e2) {}

			constexpr auto to_spirv(uint32_t counter) const {
				auto [re2a, re2b] = e2.to_spirv(counter);
				auto us = std::array{ op(spv::OpStore, 3), id, counter };
				return std::pair{ re2a, concat_array(re2b, us) };
			}
		};

		template<typename E1, typename E2>
		struct Cmp : SpvExpression<Cmp<E1, E2>> {
			static_assert(std::is_same_v<typename E1::type, typename E2::type>);
			using type = bool; // TODO: vector types

			E1 e1;
			E2 e2;

			constexpr Cmp(E1 e1, E2 e2) : e1(e1), e2(e2) {}

			constexpr auto to_spirv(uint32_t counter) const {
				auto e1id = counter - 1 - count(e2.to_spirv(counter));
				auto [re1a, re1b] = e1.to_spirv(e1id);
				auto [re2a, re2b] = e2.to_spirv(counter - 1);
				auto us = std::array{ op(spv::OpUGreaterThan, 5), type_id<type>(), counter, e1id, counter - 1 };
				return std::pair{ concat_array(re1a, re2a), concat_array(re1b, re2b, us) };
			}
		};

		template<typename E1, typename E2>
		Cmp<E1, E2> constexpr operator>(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return Cmp<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
		}

		template<typename E1, typename T>
		requires std::integral<T> Cmp<E1, Constant<T>>
		constexpr operator>(SpvExpression<E1> const& u, T const& v) {
			return Cmp<E1, Constant<T>>(*static_cast<const E1*>(&u), Constant<T>(v));
		}

		template<typename Cond, typename E1, typename E2>
		struct Select : SpvExpression<Select<Cond, E1, E2>> {
			static_assert(std::is_same_v<typename E1::type, typename E2::type>);
			using type = typename E1::type;

			Cond cond;
			E1 e1;
			E2 e2;

			constexpr Select(Cond cond, E1 e1, E2 e2) : cond(cond), e1(e1), e2(e2) {}

			constexpr auto to_spirv(uint32_t counter) const {
				auto e1id = counter - 1 - count(e2.to_spirv(counter));
				auto condid = e1id - count(e1.to_spirv(counter));
				auto [rca, rcb] = cond.to_spirv(condid);
				auto [re1a, re1b] = e1.to_spirv(e1id);
				auto [re2a, re2b] = e2.to_spirv(counter - 1);
				auto us = std::array{ op(spv::OpSelect, 6), type_id<type>(), counter, condid, e1id, counter - 1 };
				return std::pair{ concat_array(rca, re1a, re2a), concat_array(rcb, re1b, re2b, us) };
			}
		};

		template<class T>
		requires std::derived_from<T, SpvExpression<T>>
		constexpr auto fwd(T t) {
			return t;
		}

		template<class T>
		constexpr auto fwd(T t) {
			return Constant<T>(t);
		}

		template<class... Ts>
		constexpr bool any_spir(const Ts&... ts) {
			return std::disjunction_v<std::is_base_of<SpvExpression<Ts>, Ts>...>;
		}

		template<typename Cond, typename E1, typename E2>
		constexpr auto select(Cond cond, E1 e1, E2 e2) {
			if constexpr (any_spir(cond, e1, e2)) {
				return Select(fwd(cond), fwd(e1), fwd(e2));
			} else {
				return cond ? e1 : e2;
			}
		}

		template<typename E>
		constexpr auto compile_to_spirv(const E& expr, uint32_t max_id) {
			auto res = expr.to_spirv(max_id);
			auto prologue = concat_array(no_spirv);
			return std::pair{ concat_array(prologue, res.first), res.second };
		}

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

		template<class F>
		static constexpr auto specialize(F&& f) {
			constexpr const auto A = spirv::LoadId<uint32_t>{ 73 };
			constexpr const auto B = spirv::LoadId<uint32_t>{ 83 };

			return spirv::StoreId{ 83u, f(A, B) };
		}
	};

	struct CountWithIndirect {
		CountWithIndirect(uint32_t count, uint32_t wg_size) : workgroup_count((uint32_t)idivceil(count, wg_size)), count(count) {}

		uint32_t workgroup_count;
		uint32_t yz[2] = { 1, 1 };
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