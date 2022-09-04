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

		template<class T>
		concept numeric = std::is_integral_v<T> || std::is_floating_point_v<T>;

		static constexpr std::array<uint32_t, 0> no_spirv = {};

		constexpr uint32_t op(spv::Op inop, uint32_t word_count) {
			uint32_t lower = inop & spv::OpCodeMask;
			uint32_t upper = (word_count << spv::WordCountShift) & 0xFFFF0000u;
			return upper | lower;
		}

		template<typename T>
		constexpr std::string_view type_name() {
#ifdef __clang__
			return __PRETTY_FUNCTION__;
#elif defined(__GNUC__)
			return __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
			return __FUNCSIG__;
#else
#error "Unsupported compiler"
#endif
		}

		struct SPIRType {
			std::string_view type_identifier;
			uint32_t spirv_id;
		};

		static constexpr std::array<SPIRType, 0> no_types = {};

		template<size_t Section8Len, size_t Section9Len, size_t Section11Len, size_t NTypes>
		struct SPIRVModule {
			uint32_t counter;
			std::array<uint32_t, Section8Len> annotations;
			std::array<uint32_t, Section9Len> type_decls;
			std::array<uint32_t, Section11Len> codes;

			std::array<SPIRType, NTypes> types;

			constexpr SPIRVModule(uint32_t id_counter,
			                      std::array<uint32_t, Section8Len> annotations,
			                      std::array<uint32_t, Section9Len> type_decls,
			                      std::array<uint32_t, Section11Len> codes,
			                      std::array<SPIRType, NTypes> types) :
			    counter(id_counter),
			    annotations(annotations),
			    type_decls(type_decls),
			    codes(codes),
			    types(types) {}

			template<class T>
			constexpr auto type_id() {
				auto tn = type_name<T>();
				for (auto& t : types) {
					if (tn == t.type_identifier) {
						return std::pair(t.spirv_id, *this + SPIRVModule<0, 0, 0, 1>{ counter + 1, no_spirv, no_spirv, no_spirv, {} });
					}
				}
				auto new_types_arr = std::array{ SPIRType{ tn, counter + 1 } };
				return std::pair(counter + 1, *this + SPIRVModule<0, 0, 0, 1>{ counter + 1, no_spirv, no_spirv, no_spirv, new_types_arr });
			}

			template<size_t OSection8Len, size_t OSection9Len, size_t OSection11Len, size_t ONTypes>
			constexpr auto operator+(SPIRVModule<OSection8Len, OSection9Len, OSection11Len, ONTypes> o) {
				return SPIRVModule<Section8Len + OSection8Len, Section9Len + OSection9Len, Section11Len + OSection11Len, NTypes + ONTypes>{
					std::max(counter, o.counter),
					concat_array(annotations, o.annotations),
					concat_array(type_decls, o.type_decls),
					concat_array(codes, o.codes),
					concat_array(types, o.types)
				};
			}

			template<size_t N>
			constexpr auto code(uint32_t counter, std::array<uint32_t, N> v) {
				return *this + SPIRVModule<0, 0, N, 0>{ counter, no_spirv, no_spirv, v, no_types };
			}

			template<size_t N>
			constexpr auto constant(uint32_t counter, std::array<uint32_t, N> v) {
				return *this + SPIRVModule<0, N, 0, 0>{ counter, no_spirv, v, no_spirv, no_types };
			}

			template<size_t N>
			constexpr auto annotation(std::array<uint32_t, N> v) {
				return *this + SPIRVModule<N, 0, 0, 0>{ v, no_spirv, no_spirv, no_types };
			}
		};

		/* constexpr uint32_t count(auto v) {
		  return (uint32_t)v.annotations.size() + (uint32_t)v.type_decls.size() + (uint32_t)v.code.size();
		}*/

		template<spv::StorageClass sc, class Pointee>
		struct ptr {
			using pointee = Pointee;
		};

		/* template<class T>
		constexpr uint32_t type_id() {
		  if constexpr (std::is_same_v<T, uint32_t>) {
		    return 6u;
		  } else if constexpr (std::is_same_v<T, bool>) {
		    return 58u;
		  } else if constexpr (std::is_same_v<T, float>) {
		    return 201u;
		  } else if constexpr (std::is_same_v<T, ptr<spv::StorageClassStorageBuffer, uint32_t>>) {
		    return 55u;
		  } else if constexpr (std::is_same_v<T, ptr<spv::StorageClassStorageBuffer, float>>) {
		    return 202u;
		  } else {
		    assert(0);
		  }
		}*/

		template<typename E>
		struct SpvExpression {
			constexpr auto to_spirv(uint32_t counter) const {
				return static_cast<const E*>(this)->to_spirv(counter);
			}
		};

		/* template<class T>
		struct Type : public SpvExpression<Type<T>> {
		  constexpr auto to_spirv(uint32_t counter) const {
		    assert(0);
		    return type_or_constant(no_spirv);
		  }
		};

		template<>
		struct Type<bool> {
		  constexpr auto to_spirv(uint32_t counter) const {
		    auto us = std::array{ op(spv::OpTypeBool, 2), counter };
		    return type_or_constant(us);
		  }
		};

		template<>
		struct Type<float> {
		  constexpr auto to_spirv(uint32_t counter) const {
		    auto us = std::array{ op(spv::OpTypeFloat, 3), counter, 32u };
		    return type_or_constant(us);
		  }
		};

		template<spv::StorageClass sc, class Pointee>
		struct Type<ptr<sc, Pointee>> {
		  constexpr auto to_spirv(uint32_t counter) const {
		    auto us = std::array{ op(spv::OpTypePointer, 4), counter, uint32_t(sc), type_id<Pointee>() };
		    return type_or_constant(us);
		  }
		};

		template<class T>
		struct TypeRuntimeArray : public SpvExpression<TypeRuntimeArray<T>> {
		  constexpr auto to_spirv(uint32_t counter) const {
		    auto us = std::array{ op(spv::OpTypeRuntimeArray, 3), counter, type_id<T>() };
		    return type_or_constant(us);
		  }
		};

		template<class T>
		struct TypeUStruct : public SpvExpression<TypeUStruct<T>> {
		  constexpr auto to_spirv(uint32_t counter) const {
		    auto us = std::array{ op(spv::OpTypeStruct, 3), counter, type_id<T>() };
		    return type_or_constant(us);
		  }
		};

		template<class T>
		struct Variable : public SpvExpression<TypeRuntimeArray<T>> {
		  constexpr auto to_spirv(uint32_t counter) const {
		    auto us = std::array{ op(spv::OpVariable, 4), counter, type_id<T>() };
		    return type_or_constant(us);
		  }
		};*/

		template<typename E1, typename E2>
		struct Add : public SpvExpression<Add<E1, E2>> {
			static_assert(std::is_same_v<typename E1::type, typename E2::type>);
			using type = typename E1::type;

			E1 e1;
			E2 e2;

			constexpr Add(E1 e1, E2 e2) : e1(e1), e2(e2) {}

		public:
			constexpr auto to_spirv(auto mod) const {
				auto mod1 = e1.to_spirv(mod);
				auto mod2 = e2.to_spirv(mod1);
				auto [tid, mod3] = mod2.type_id<type>();
				auto us = std::array{ op(std::is_floating_point_v<type> ? spv::OpFAdd : spv::OpIAdd, 5), tid, mod3.counter + 1, mod1.counter, mod2.counter };
				return mod3.code(mod3.counter + 1, us);
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
				auto e1m = e1.to_spirv(e1id);
				auto e2m = e2.to_spirv(counter - 1);
				auto us = std::array{ op(std::is_floating_point_v<type> ? spv::OpFMul : spv::OpIMul, 5), 6u, counter, counter - 1, e1id };
				return e1m + e2m + code(us);
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

			constexpr auto to_spirv(auto mod) const {
				auto [tid, mod1] = mod.type_id<type>();
				auto us = std::array{ op(spv::OpLoad, 4), tid, mod1.id_counter + 1, id };
				return mod1.code(mod1.id_counter + 1, us);
			}
		};

		template<typename E1>
		struct Load : public SpvExpression<Load<E1>> {
			using type = typename E1::type::pointee;

			E1 e1;
			constexpr Load(E1 e1) : e1(e1) {}

			constexpr auto to_spirv(auto mod) const {
				auto [tid, mod1] = mod.type_id<type>();
				auto mod2 = e1.to_spirv(mod1);
				auto us = std::array{ op(spv::OpLoad, 4), tid, mod2.counter + 1, mod2.counter };
				return mod2.code(mod2.counter + 1, us);
			}
		};

		template<class T>
		struct Constant : public SpvExpression<Constant<T>> {
			using type = T;
			T value;
			constexpr Constant(T v) : value(v) {}

			constexpr auto to_spirv(auto mod) const {
				constexpr size_t num_uints = sizeof(T) / sizeof(uint32_t);
				auto as_uints = std::bit_cast<std::array<uint32_t, num_uints>>(value);
				auto [tid, modt] = mod.type_id<T>();
				auto us = std::array{ op(spv::OpConstant, 3 + num_uints), tid, modt.counter + 1 };
				auto with_value = concat_array(us, as_uints);
				return modt.constant(modt.counter + 1, with_value);
			}
		};

		template<typename E1, typename T>
		Add<E1, Constant<T>> constexpr operator+(SpvExpression<E1> const& u, T const& v) {
			return Add<E1, Constant<T>>(*static_cast<const E1*>(&u), Constant<T>(v));
		}

		template<typename E1, typename T>
		requires numeric<T> Mul<E1, Constant<T>>
		constexpr operator*(SpvExpression<E1> const& u, T const& v) {
			return Mul<E1, Constant<T>>(*static_cast<const E1*>(&u), Constant<T>(v));
		}

		template<typename E1, typename T>
		requires numeric<T>
		constexpr auto operator*(T const& v, SpvExpression<E1> const& u) {
			return u * v;
		}

		template<typename E1>
		struct StoreId : SpvExpression<StoreId<E1>> {
			using type = void;

			const uint32_t id;
			E1 e1;

			constexpr StoreId(uint32_t id, E1 e1) : id(id), e1(e1) {}

			constexpr auto to_spirv(auto mod) const {
				auto mod1 = e1.to_spirv(mod);
				auto us = std::array{ op(spv::OpStore, 3), id, mod1.counter };
				return mod1.code(mod1.counter, us);
			}
		};

		template<typename T, typename... Indices>
		struct AccessChainId : SpvExpression<AccessChainId<T, Indices...>> {
			using type = T;

			const uint32_t base;
			std::tuple<Indices...> indices;

			constexpr AccessChainId(uint32_t base, Indices... inds) : base(base), indices(inds...) {}

			/* constexpr auto to_spirv(uint32_t counter) const {
			  uint32_t base_counter = counter;
			  std::apply([&](auto... exprs) { return ... << exprs; }, indices);
			  auto counts = std::apply([&](auto& expr) { return count(expr.to_spirv(200)); }, indices);
			  uint32_t ac = base_counter;
			  std::apply([&](auto& expr) { expr.to_spirv(base_counter); }, indices);
			  auto a = concat_array(.to_spirv(counter);
			  auto us = std::array{ op(spv::OpAccessChain, 3), type_id<T>(), counter, base };
			  return std::pair{ re2a, concat_array(re2b, us) };
			}*/

			constexpr auto to_spirv(auto mod) const {
				static_assert(sizeof...(Indices) == 1);
				auto [tid, modt] = mod.type_id<type>();
				auto mod1 = std::get<0>(indices).to_spirv(modt);
				auto us = std::array{ op(spv::OpAccessChain, 5), tid, mod1.counter + 1, base, mod1.counter };
				return mod1.code(mod1.counter + 1, us);
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
				auto e1m = e1.to_spirv(e1id);
				auto e2m = e2.to_spirv(counter - 1);
				auto us = std::array{ op(spv::OpUGreaterThan, 5), type_id<type>(), counter, e1id, counter - 1 };
				return e1m + e2m + code(us);
			}
		};

		template<typename E1, typename E2>
		Cmp<E1, E2> constexpr operator>(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return Cmp<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
		}

		template<typename E1, typename T>
		requires numeric<T> Cmp<E1, Constant<T>>
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
				auto condm = cond.to_spirv(condid);
				auto e1m = e1.to_spirv(e1id);
				auto e2m = e2.to_spirv(counter - 1);
				auto us = std::array{ op(spv::OpSelect, 6), type_id<type>(), counter, condid, e1id, counter - 1 };
				return condm + e1m + e2m + code(us);
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
			std::array predef_types = { SPIRType{ type_name<uint32_t>(), 6u },
				                          SPIRType{ type_name<bool>(), 58u },
				                          SPIRType{ type_name<ptr<spv::StorageClassStorageBuffer, uint32_t>>(), 55u } };
			SPIRVModule spvmodule{ max_id, no_spirv, no_spirv, no_spirv, predef_types };
			auto res = expr.to_spirv(spvmodule);
			/* auto prologue = Type<float>{}.to_spirv(201u) + Type<ptr<spv::StorageClassStorageBuffer, float>>{}.to_spirv(202u);
			return prologue + res;*/
			return res;
		}

		template<class Derived>
		struct SPIRVTemplate {
			template<class F>
			static constexpr auto compile(F&& f) {
				constexpr auto first_bit = array_copy<0x00000398 / 4>(Derived::template_bytes);
				constexpr auto second_bit = array_copy<0x0000072c / 4 - 0x00000398 / 4>(Derived::template_bytes + 0x00000398 / 4);
				constexpr auto epilogue = array_copy<6>(Derived::template_bytes + 0x00000738 / 4);

				auto res = compile_to_spirv(Derived::specialize(f), 200);
				return concat_array(first_bit, res.type_decls, second_bit, res.codes, epilogue);
			}
		};
	} // namespace spirv

	template<class T1, class T2>
	struct SPIRVBinaryMap : public spirv::SPIRVTemplate<SPIRVBinaryMap<T1, T2>> {
		static constexpr const uint32_t template_bytes[] = {
			0x07230203, 0x00010500, 0x0008000a, 0x00000171, 0x00000000, 0x00020011, 0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
			0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x000b000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000f, 0x0000002d, 0x00000034,
			0x00000042, 0x00000047, 0x0000004e, 0x00060010, 0x00000004, 0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2,
			0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00070005, 0x0000000f, 0x4e5f6c67, 0x6f576d75, 0x72476b72, 0x7370756f, 0x00000000, 0x00080005,
			0x0000002d, 0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00050005, 0x00000032, 0x66667542, 0x6f437265, 0x00746e75,
			0x00040006, 0x00000032, 0x00000000, 0x0000006e, 0x00030005, 0x00000034, 0x00000000, 0x00050005, 0x00000040, 0x66667542, 0x754f7265, 0x00000074,
			0x00060006, 0x00000040, 0x00000000, 0x61746164, 0x74756f5f, 0x00000000, 0x00030005, 0x00000042, 0x00000000, 0x00050005, 0x00000045, 0x66667542,
			0x6e497265, 0x00000030, 0x00060006, 0x00000045, 0x00000000, 0x61746164, 0x306e695f, 0x00000000, 0x00030005, 0x00000047, 0x00000000, 0x00050005,
			0x0000004c, 0x66667542, 0x6e497265, 0x00000031, 0x00060006, 0x0000004c, 0x00000000, 0x61746164, 0x316e695f, 0x00000000, 0x00030005, 0x0000004e,
			0x00000000, 0x00040047, 0x0000000f, 0x0000000b, 0x00000018, 0x00040047, 0x0000002d, 0x0000000b, 0x0000001c, 0x00040048, 0x00000032, 0x00000000,
			0x00000018, 0x00050048, 0x00000032, 0x00000000, 0x00000023, 0x0000000c, 0x00030047, 0x00000032, 0x00000002, 0x00040047, 0x00000034, 0x00000022,
			0x00000000, 0x00040047, 0x00000034, 0x00000021, 0x00000004, 0x00040047, 0x0000003f, 0x00000006, 0x00000004, 0x00040048, 0x00000040, 0x00000000,
			0x00000017, 0x00050048, 0x00000040, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000040, 0x00000002, 0x00040047, 0x00000042, 0x00000022,
			0x00000000, 0x00040047, 0x00000042, 0x00000021, 0x00000001, 0x00040047, 0x00000044, 0x00000006, 0x00000004, 0x00040048, 0x00000045, 0x00000000,
			0x00000017, 0x00050048, 0x00000045, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000045, 0x00000002, 0x00040047, 0x00000047, 0x00000022,
			0x00000000, 0x00040047, 0x00000047, 0x00000021, 0x00000000, 0x00040047, 0x0000004b, 0x00000006, 0x00000004, 0x00040048, 0x0000004c, 0x00000000,
			0x00000017, 0x00050048, 0x0000004c, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x0000004c, 0x00000002, 0x00040047, 0x0000004e, 0x00000022,
			0x00000000, 0x00040047, 0x0000004e, 0x00000021, 0x00000002, 0x00040047, 0x00000013, 0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021,
			0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040017, 0x00000007, 0x00000006, 0x00000003, 0x00040020, 0x0000000e,
			0x00000001, 0x00000007, 0x0004003b, 0x0000000e, 0x0000000f, 0x00000001, 0x0004002b, 0x00000006, 0x00000011, 0x00000040, 0x0004002b, 0x00000006,
			0x00000012, 0x00000001, 0x0006002c, 0x00000007, 0x00000013, 0x00000011, 0x00000012, 0x00000012, 0x0004002b, 0x00000006, 0x0000001c, 0x00000000,
			0x0004003b, 0x0000000e, 0x0000002d, 0x00000001, 0x0003001e, 0x00000032, 0x00000006, 0x00040020, 0x00000033, 0x0000000c, 0x00000032, 0x0004003b,
			0x00000033, 0x00000034, 0x0000000c, 0x00040015, 0x00000035, 0x00000020, 0x00000001, 0x0004002b, 0x00000035, 0x00000036, 0x00000000, 0x00040020,
			0x00000037, 0x0000000c, 0x00000006, 0x00020014, 0x0000003a, 0x0003001d, 0x0000003f, 0x00000006, 0x0003001e, 0x00000040, 0x0000003f, 0x00040020,
			0x00000041, 0x0000000c, 0x00000040, 0x0004003b, 0x00000041, 0x00000042, 0x0000000c, 0x0003001d, 0x00000044, 0x00000006, 0x0003001e, 0x00000045,
			0x00000044, 0x00040020, 0x00000046, 0x0000000c, 0x00000045, 0x0004003b, 0x00000046, 0x00000047, 0x0000000c, 0x0003001d, 0x0000004b, 0x00000006,
			0x0003001e, 0x0000004c, 0x0000004b, 0x00040020, 0x0000004d, 0x0000000c, 0x0000004c, 0x0004003b, 0x0000004d, 0x0000004e, 0x0000000c, 0x00050036,
			0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x000300f7, 0x00000054, 0x00000000, 0x000300fb, 0x0000001c, 0x00000055,
			0x000200f8, 0x00000055, 0x0004003d, 0x00000007, 0x0000002f, 0x0000002d, 0x0004003d, 0x00000007, 0x0000005d, 0x0000000f, 0x00050084, 0x00000007,
			0x0000005e, 0x0000005d, 0x00000013, 0x00050051, 0x00000006, 0x00000060, 0x0000002f, 0x00000002, 0x00050051, 0x00000006, 0x00000062, 0x0000005e,
			0x00000001, 0x00050084, 0x00000006, 0x00000063, 0x00000060, 0x00000062, 0x00050051, 0x00000006, 0x00000065, 0x0000005e, 0x00000000, 0x00050084,
			0x00000006, 0x00000066, 0x00000063, 0x00000065, 0x00050051, 0x00000006, 0x00000068, 0x0000002f, 0x00000001, 0x00050084, 0x00000006, 0x0000006b,
			0x00000068, 0x00000065, 0x00050080, 0x00000006, 0x0000006c, 0x00000066, 0x0000006b, 0x00050051, 0x00000006, 0x0000006e, 0x0000002f, 0x00000000,
			0x00050080, 0x00000006, 0x00000070, 0x0000006c, 0x0000006e, 0x00050041, 0x00000037, 0x00000038, 0x00000034, 0x00000036, 0x0004003d, 0x00000006,
			0x00000039, 0x00000038, 0x000500ae, 0x0000003a, 0x0000003b, 0x00000070, 0x00000039, 0x000300f7, 0x0000003d, 0x00000000, 0x000400fa, 0x0000003b,
			0x0000003c, 0x0000003d, 0x000200f8, 0x0000003c, 0x000200f9, 0x00000054, 0x000200f8, 0x0000003d, 0x00060041, 0x00000037, 0x00000049, 0x00000047,
			0x00000036, 0x00000070, 0x0004003d, 0x00000006, 0x0000004a, 0x00000049, 0x00060041, 0x00000037, 0x00000050, 0x0000004e, 0x00000036, 0x00000070,
			0x0004003d, 0x00000006, 0x00000051, 0x00000050, 0x00050080, 0x00000006, 0x00000052, 0x0000004a, 0x00000051, 0x00060041, 0x00000037, 0x00000053,
			0x00000042, 0x00000036, 0x00000070, 0x0003003e, 0x00000053, 0x00000052, 0x000200f9, 0x00000054, 0x000200f8, 0x00000054, 0x000100fd, 0x00010038
		};

		template<class F>
		static constexpr auto specialize(F&& f) {
			constexpr auto ldA = spirv::AccessChainId<spirv::ptr<spv::StorageClassStorageBuffer, T1>, spirv::Constant<uint32_t>>(71u, spirv::Constant(0u));
			constexpr const auto A = spirv::Load(ldA);
			constexpr auto ldB = spirv::AccessChainId<spirv::ptr<spv::StorageClassStorageBuffer, T2>, spirv::Constant<uint32_t>>(78u, spirv::Constant(0u));
			constexpr const auto B = spirv::Load(ldB);

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

	template<class T, class F>
	inline Future unary_map(Future src, Future dst, Future count, const F& fn) {
		constexpr auto spirv = SPIRVBinaryMap<T, uint32_t>::compile([](auto A, auto B) { return F{}(A); });
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