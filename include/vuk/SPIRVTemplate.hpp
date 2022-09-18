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
			uint32_t annotation_size = 0;
			std::array<uint32_t, Section9Len> decls;
			uint32_t decl_size = 0;
			std::array<uint32_t, Section11Len> codes;

			std::array<SPIRType, NTypes> types;

			constexpr SPIRVModule(uint32_t id_counter,
			                      std::array<uint32_t, Section8Len> annotations,
			                      uint32_t annotation_size,
			                      std::array<uint32_t, Section9Len> decls,
			                      uint32_t decl_size,
			                      std::array<uint32_t, Section11Len> codes,
			                      std::array<SPIRType, NTypes> types) :
			    counter(id_counter),
			    annotations(annotations),
			    annotation_size(annotation_size),
			    decls(decls),
			    decl_size(decl_size),
			    codes(codes),
			    types(types) {}

			template<class T>
			constexpr auto type_id();

			template<size_t OSection8Len, size_t OSection9Len, size_t OSection11Len, size_t ONTypes>
			constexpr auto operator+(SPIRVModule<OSection8Len, OSection9Len, OSection11Len, ONTypes> o) {
				std::array<uint32_t, Section8Len + OSection8Len> merged_annots = {};
				{
					auto endit = std::copy_n(annotations.begin(), annotation_size, merged_annots.begin());
					std::copy_n(o.annotations.begin(), o.annotation_size, endit);
				}

				std::array<uint32_t, Section9Len + OSection9Len> merged_decls = {};
				{
					auto endit = std::copy_n(decls.begin(), decl_size, merged_decls.begin());
					std::copy_n(o.decls.begin(), o.decl_size, endit);
				}
				return SPIRVModule<Section8Len + OSection8Len, Section9Len + OSection9Len, Section11Len + OSection11Len, NTypes + ONTypes>{
					std::max(counter, o.counter), merged_annots, annotation_size + o.annotation_size, merged_decls, decl_size + o.decl_size, concat_array(codes, o.codes),
					concat_array(types, o.types)
				};
			}

			template<size_t N>
			constexpr auto code(uint32_t counter, std::array<uint32_t, N> v) {
				return std::pair(counter, *this + SPIRVModule<0, 0, N, 0>{ counter, no_spirv, 0, no_spirv, 0, v, no_types });
			}

			template<size_t N>
			constexpr auto constant(uint32_t counter, std::array<uint32_t, N> v) {
				return std::pair(counter, *this + SPIRVModule<0, N, 0, 0>{ counter, no_spirv, 0, v, N, no_spirv, no_types });
			}

			template<size_t N>
			constexpr auto annotation(uint32_t counter, std::array<uint32_t, N> v) {
				return *this + SPIRVModule<N, 0, 0, 0>{ counter, v, N, no_spirv, 0, no_spirv, no_types };
			}
		};

		template<size_t Section8Len, size_t Section9Len, size_t Section11Len, size_t NTypes>
		template<class T>
		constexpr auto SPIRVModule<Section8Len, Section9Len, Section11Len, NTypes>::type_id() {
			auto tn = type_name<T>();
			uint32_t id;
			bool found = false;
			for (auto& t : types) {
				if (tn == t.type_identifier) {
					id = t.spirv_id;
					found = true;
					break;
				}
			}
			auto [declid, declmod] = T::to_spirv(*this);
			auto new_types_arr = std::array{ SPIRType{ tn, declid } };
			if (!found) {
				return std::pair(declid, declmod + SPIRVModule<0, 0, 0, 1>{ declid, no_spirv, 0, no_spirv, 0, no_spirv, new_types_arr });
			} else {
				declmod.counter = counter;
				declmod.decl_size = decl_size;
				return std::pair(id, declmod + SPIRVModule<0, 0, 0, 1>{ id, no_spirv, 0, no_spirv, 0, no_spirv, {} });
			}
		}

		template<spv::StorageClass sc, class Pointee>
		struct ptr {
			using pointee = Pointee;
			static constexpr auto storage_class = sc;

			pointee operator*() const {
				return {};
			}
		};

		template<typename T, size_t N>
		struct Deref {
			using type = typename Deref<typename T::pointee, N - 1>::type;
		};

		template<typename T>
		struct Deref<T, 0> {
			using type = T;
		};

		template<typename E>
		struct SpvExpression {};

		template<class T>
		struct Type : public SpvExpression<Type<T>> {};

		template<>
		struct Type<uint32_t> {
			using type = uint32_t;

			static constexpr auto to_spirv(auto mod) {
				auto us = std::array{ op(spv::OpTypeInt, 4), mod.counter + 1, 32u, 0u };
				return mod.constant(mod.counter + 1, us);
			}
		};

		template<>
		struct Type<bool> {
			using type = bool;

			static constexpr auto to_spirv(auto mod) {
				auto us = std::array{ op(spv::OpTypeBool, 2), mod.counter + 1 };
				return mod.constant(mod.counter + 1, us);
			}
		};

		template<>
		struct Type<float> {
			using type = float;

			static constexpr auto to_spirv(auto mod) {
				auto us = std::array{ op(spv::OpTypeFloat, 3), mod.counter + 1, 32u };
				return mod.constant(mod.counter + 1, us);
			}
		};

		template<spv::StorageClass sc, class Pointee>
		struct Type<ptr<sc, Pointee>> {
			using type = ptr<sc, Pointee>;
			static constexpr auto storage_class = sc;
			using pointee = Pointee;

			static constexpr auto to_spirv(auto mod) {
				auto [tid, modt] = mod.template type_id<Pointee>();
				auto us = std::array{ op(spv::OpTypePointer, 4), modt.counter + 1, uint32_t(sc), tid };
				return modt.constant(modt.counter + 1, us);
			}
		};

		template<class T>
		struct TypeRuntimeArray : public SpvExpression<TypeRuntimeArray<T>> {
			using pointee = T;

			static constexpr auto to_spirv(auto mod) {
				auto [tid, modt] = mod.template type_id<T>();
				auto us = std::array{ op(spv::OpTypeRuntimeArray, 3), modt.counter + 1, tid };
				auto deco =
				    std::array{ op(spv::OpDecorate, 4), modt.counter + 1, uint32_t(spv::Decoration::DecorationArrayStride), (uint32_t)sizeof(typename T::type) };
				return std::pair(modt.counter + 1, modt.annotation(modt.counter + 1, deco).constant(modt.counter + 1, us).second);
			}
		};

		template<class T, uint32_t Offset>
		struct Member {
			using type = T;

			static constexpr auto to_spirv(auto mod, uint32_t parent, uint32_t index) {
				// auto [tid, modt] = mod.template type_id<T>();
				auto deco = std::array{ op(spv::OpMemberDecorate, 5), parent, index, uint32_t(spv::Decoration::DecorationOffset), Offset };
				return mod.annotation(mod.counter, deco);
			}
		};

		template<class... Members>
		struct TypeStruct : public SpvExpression<TypeStruct<Members...>> {
			using members = std::tuple<Members...>;

			static constexpr auto to_spirv(auto mod) {
				static_assert(sizeof...(Members) == 1);
				auto [tid, modt] = mod.template type_id<typename member<0>::type>();
				auto str_id = modt.counter + 1;
				auto mod1 = member<0>::to_spirv(modt, str_id, 0);
				auto deco = std::array{ op(spv::OpDecorate, 3), str_id, uint32_t(spv::Decoration::DecorationBlock) };
				auto mod2 = mod1.annotation(mod1.counter, deco);
				auto us = std::array{ op(spv::OpTypeStruct, 3), str_id, tid };
				return mod2.constant(str_id, us);
			}

			template<uint32_t idx>
			using member = typename std::remove_reference_t<decltype(std::get<idx>(std::declval<members>()))>;

			template<uint32_t idx>
			using deref = typename std::remove_reference_t<decltype(std::get<idx>(std::declval<members>()))>::type;
		};

		template<class T, spv::StorageClass sc>
		struct Variable : public SpvExpression<Variable<T, sc>> {
			using type = T;

			uint32_t descriptor_set;
			uint32_t binding;

			std::tuple<> children;
			uint32_t id = 0;

			constexpr Variable(uint32_t descriptor_set, uint32_t binding) : descriptor_set(descriptor_set), binding(binding) {}

			constexpr auto to_spirv(auto mod) {
				auto [tid, modt] = mod.template type_id<T>();
				id = modt.counter + 1;
				auto mod1 = modt.annotation(id, std::array{ op(spv::OpDecorate, 4), id, uint32_t(spv::Decoration::DecorationDescriptorSet), descriptor_set })
				                .annotation(id, std::array{ op(spv::OpDecorate, 4), id, uint32_t(spv::Decoration::DecorationBinding), binding });
				auto us = std::array{ op(spv::OpVariable, 4), tid, id, uint32_t(sc) };
				return mod1.constant(id, us);
			}
		};

		template<class T>
		struct is_variable {
			static constexpr bool value = false;
		};

		template<class T, spv::StorageClass sc>
		struct is_variable<Variable<T, sc>> {
			static constexpr bool value = true;
		};

		struct Id : public SpvExpression<Id> {
			uint32_t id;
			std::tuple<> children;

			constexpr Id(uint32_t id) : id(id) {}

			constexpr auto to_spirv(auto mod) {
				return std::pair(id, mod);
			}
		};

		template<size_t N, typename... Ts>
		constexpr auto _emit_children(auto mod, std::tuple<Ts...>& children) {
			if constexpr (N == 0) {
				auto [resid, resmod] = std::get<N>(children).to_spirv(mod);
				return std::pair{ std::array{ resid }, resmod };
			} else {
				auto r = _emit_children<N - 1>(mod, children);
				auto [resid, resmod] = std::get<N>(children).to_spirv(r.second);
				return std::pair{ concat_array(std::array{ resid }, r.first), resmod };
			}
		}

		template<class... Ts>
		constexpr auto emit_children(auto mod, std::tuple<Ts...>& children) {
			auto [resids, resmod] = _emit_children<sizeof...(Ts) - 1>(mod, children);
			return std::pair{ resids, resmod };
		}

		template<size_t N, class F, typename... Ts>
		constexpr auto visit_children(std::tuple<Ts...>& children, F&& fn) {
			if constexpr (N == 0) {
				visit(std::get<0>(children), fn);
			} else {
				visit_children<N - 1>(children, fn);
				visit(std::get<N>(children), fn);
			}
		}

		template<class F>
		constexpr auto visit(auto root, F&& fn) {
			constexpr auto n_children = std::tuple_size_v<decltype(root.children)>;
			if constexpr (n_children > 0) {
				visit_children<n_children - 1>(root.children, fn);
			}
			fn(root);
		}

		template<typename E1, typename E2>
		struct Add : public SpvExpression<Add<E1, E2>> {
			static_assert(std::is_same_v<typename E1::type, typename E2::type>);
			using type = typename E1::type;

			std::tuple<E1, E2> children;
			uint32_t id = 0;

			constexpr Add(E1 e1, E2 e2) : children(e1, e2) {}

		public:
			constexpr auto to_spirv(auto mod) {
				auto [eids, resmod] = emit_children(mod, children);
				auto [tid, mod3] = resmod.template type_id<type>();
				auto us = std::array{ op(std::is_floating_point_v<type> ? spv::OpFAdd : spv::OpIAdd, 5), tid, mod3.counter + 1 } << eids;
				id = mod3.counter + 1;
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

			constexpr auto to_spirv(auto mod) {
				auto [tid, mod1] = mod.template type_id<type>();
				auto us = std::array{ op(spv::OpLoad, 4), tid, mod1.id_counter + 1, id };
				return mod1.code(mod1.id_counter + 1, us);
			}
		};

		template<typename E1>
		struct Load : public SpvExpression<Load<E1>> {
			using type = typename Deref<typename E1::type, 1u>::type;

			uint32_t id = 0;
			std::tuple<E1> children;
			constexpr Load(E1 e1) : children(e1) {}

			constexpr auto to_spirv(auto mod) {
				auto [tid, modt] = mod.template type_id<type>();
				auto [e1id, mod1] = emit_children(modt, children);
				auto us = std::array{ op(spv::OpLoad, 4), tid, mod1.counter + 1 } << e1id;
				id = mod1.counter + 1;
				return mod1.code(mod1.counter + 1, us);
			}
		};

		template<class T>
		struct Constant : public SpvExpression<Constant<T>> {
			using type = Type<T>;
			T value;
			std::tuple<> children;
			uint32_t id = 0;
			constexpr Constant(T v) : value(v) {}

			constexpr auto to_spirv(auto mod) {
				constexpr size_t num_uints = sizeof(T) / sizeof(uint32_t);
				auto as_uints = std::bit_cast<std::array<uint32_t, num_uints>>(value);
				auto [tid, modt] = mod.template type_id<type>();
				auto us = std::array{ op(spv::OpConstant, 3 + num_uints), tid, modt.counter + 1 };
				auto with_value = concat_array(us, as_uints);
				id = modt.counter + 1;
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

			uint32_t dst_id = 0;
			const uint32_t id = ~0u;
			std::tuple<E1> children;

			constexpr StoreId(uint32_t dst_id, E1 e1) : dst_id(dst_id), children(e1) {}

			constexpr auto to_spirv(auto mod) {
				auto [e1id, mod1] = emit_children(mod, children);
				auto us = std::array{ op(spv::OpStore, 3), dst_id } << e1id;
				return mod1.code(mod1.counter, us);
			}
		};

		template<typename T, typename E1, typename... Indices>
		struct AccessChain : SpvExpression<AccessChain<E1, Indices...>> {
			using type = T; // typename Deref<typename E1::type, Indices...>::type;

			std::tuple<E1, Indices...> children;
			uint32_t id = 0;

			constexpr AccessChain(T type, E1 base, Indices... inds) : children(base, inds...) {}

			constexpr auto to_spirv(auto mod) {
				auto [eids, resmod] = emit_children(mod, children);
				auto [tid, modt] = resmod.template type_id<type>();
				std::reverse(eids.begin(), eids.end());
				auto us = std::array{ op(spv::OpAccessChain, sizeof...(Indices) + 4), tid, modt.counter + 1 } << eids;
				id = modt.counter + 1;
				return modt.code(modt.counter + 1, us);
			}
		};

		template<auto CIndex, typename E1, typename... VIndices>
		constexpr auto access_chain(E1 base, VIndices... inds) {
			using value_t = typename Deref<typename E1::type, 1>::type;
			constexpr auto sc = E1::type::storage_class;
			using deref_t = typename Deref<typename value_t::template deref<CIndex>, sizeof...(VIndices)>::type;
			return AccessChain(Type<ptr<sc, deref_t>>{}, base, Constant{ CIndex }, inds...);
		}

		enum class CmpOp {
			eGreaterThan
		};

		template<CmpOp Op, typename E1, typename E2>
		struct Cmp : SpvExpression<Cmp<Op, E1, E2>> {
			static_assert(std::is_same_v<typename E1::type, typename E2::type>);
			using type = Type<bool>; // TODO: vector types

			std::tuple<E2, E1> children;
			uint32_t id = 0;

			constexpr Cmp(E1 e1, E2 e2) : children(e2, e1) {}

			constexpr auto to_spirv(auto mod) {
				auto [eids, resmod] = emit_children(mod, children);
				auto [tid, modt] = resmod.template type_id<type>();
				id = modt.counter + 1;
				auto us = std::array{ op(spv::OpUGreaterThan, 5), tid, id } << eids;
				return modt.code(modt.counter + 1, us);
			}
		};

		template<typename E1, typename E2>
		Cmp<CmpOp::eGreaterThan, E1, E2> constexpr operator>(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return Cmp<CmpOp::eGreaterThan, E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
		}

		template<typename E1, typename T>
		requires numeric<T> Cmp<CmpOp::eGreaterThan, E1, Constant<T>>
		constexpr operator>(SpvExpression<E1> const& u, T const& v) {
			return Cmp<CmpOp::eGreaterThan, E1, Constant<T>>(*static_cast<const E1*>(&u), Constant<T>(v));
		}

		template<typename Cond, typename E1, typename E2>
		struct Select : SpvExpression<Select<Cond, E1, E2>> {
			static_assert(std::is_same_v<typename E1::type, typename E2::type>);
			using type = typename E1::type;

			std::tuple<E2, E1, Cond> children;
			uint32_t id = 0;

			constexpr Select(Cond cond, E1 e1, E2 e2) : children(e2, e1, cond) {}

			constexpr auto to_spirv(auto mod) {
				auto [eids, resmod] = emit_children(mod, children);
				auto [tid, modt] = resmod.template type_id<type>();
				id = modt.counter + 1;
				auto us = std::array{ op(spv::OpSelect, 6), tid, id} << eids;
				return modt.code(modt.counter + 1, us);
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
		constexpr auto compile_to_spirv(E& expr, uint32_t max_id) {
			std::array predef_types = { SPIRType{ type_name<Type<uint32_t>>(), 6u },
				                          SPIRType{ type_name<Type<bool>>(), 58u },
				                          SPIRType{ type_name<Type<ptr<spv::StorageClassStorageBuffer, uint32_t>>>(), 55u } };
			SPIRVModule spvmodule{ max_id, no_spirv, 0, no_spirv, 0, no_spirv, predef_types };
			auto res = expr.to_spirv(spvmodule).second;
			/* auto prologue = Type<float>{}.to_spirv(201u) + Type<ptr<spv::StorageClassStorageBuffer, float>>{}.to_spirv(202u);
			return prologue + res;*/
			return res;
		}

		template<class Derived>
		struct SPIRVTemplate {
			template<class F>
			static constexpr auto compile(F&& f) {
				constexpr auto prelude = array_copy<0x0000040 / 4>(Derived::template_bytes);
				constexpr auto prologue = array_copy<0x00000354 / 4 - 0x0000006c / 4>(Derived::template_bytes + +0x0000006c / 4);
				constexpr auto builtin_decls = array_copy<0x0000050c / 4 - 0x00000354 / 4>(Derived::template_bytes + 0x00000354 / 4);
				constexpr auto second_bit = array_copy<0x0000072c / 4 - 0x0000050c / 4>(Derived::template_bytes + 0x0000050c / 4);
				constexpr auto epilogue = array_copy<6>(Derived::template_bytes + 0x00000738 / 4);

				auto specialized = Derived::specialize(f);
				auto res = compile_to_spirv(specialized, 200);
				std::vector<uint32_t> variable_ids;
				visit(specialized, [&]<typename T>(const T& node) {
					if (is_variable<T>::value) {
						variable_ids.push_back(node.id);
					}
				});
				constexpr std::string_view t = "main";
				std::array<uint32_t, (t.size() + sizeof(uint32_t) + 1 - 1) / sizeof(uint32_t)> maintext = {};
				for (int i = 0; i < t.size(); i++) {
					auto word = i / sizeof(uint32_t);
					auto byte = i % sizeof(uint32_t);
					maintext[word] = maintext[word] | (t[i] << 8 * byte);
				}
				std::array builtin_variables = { 15u, 45u, 52u, 66u, 71u, 78u };

				std::array fixed_op_entry = std::array{ op(spv::OpEntryPoint, 5 + (uint32_t)builtin_variables.size() + (uint32_t)variable_ids.size()),
					                                      uint32_t(spv::ExecutionModelGLCompute),
					                                      4u }
				    << maintext << builtin_variables;
				std::vector<uint32_t> op_entry(fixed_op_entry.begin(), fixed_op_entry.end());
				op_entry.insert(op_entry.end(), variable_ids.begin(), variable_ids.end());
				
				std::vector<uint32_t> final_bc(prelude.begin(), prelude.end());
				auto it = final_bc.insert(final_bc.end(), op_entry.begin(), op_entry.end());
				it = final_bc.insert(final_bc.end(), prologue.begin(), prologue.end());
				it = final_bc.insert(final_bc.end(), res.annotations.begin(), res.annotations.begin() + res.annotation_size);
				it = final_bc.insert(final_bc.end(), builtin_decls.begin(), builtin_decls.end());
				it = final_bc.insert(final_bc.end(), res.decls.begin(), res.decls.begin() + res.decl_size);
				it = final_bc.insert(final_bc.end(), second_bit.begin(), second_bit.end());
				it = final_bc.insert(final_bc.end(), res.codes.begin(), res.codes.begin() + res.codes.size());
				it = final_bc.insert(final_bc.end(), epilogue.begin(), epilogue.end());
				auto arr = concat_array(prelude, prologue, res.annotations, builtin_decls, res.decls, second_bit, res.codes, epilogue, std::array<uint32_t, 25>{});
				std::copy(final_bc.begin(), final_bc.end(), arr.begin());
				return std::pair(final_bc.size(), arr);
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
			using namespace spirv;

			constexpr TypeStruct<Member<TypeRuntimeArray<Type<T1>>, 0>> strT = {};
			constexpr auto ptr_to_struct = Type<ptr<spv::StorageClassStorageBuffer, decltype(strT)>>{};
			constexpr auto vA = Variable<decltype(ptr_to_struct), spv::StorageClassStorageBuffer>(0, 0);
			constexpr auto ldA = access_chain<0u>(vA, Id(112u));
			constexpr auto A = Load(ldA);
			constexpr auto vB = Variable<decltype(ptr_to_struct), spv::StorageClassStorageBuffer>(0, 1);
			constexpr auto ldB = access_chain<0u>(vB, Id(112u));
			constexpr auto B = Load(ldB);

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
		FILE* fo = fopen("dumb.spv", "wb");
		fwrite(ptr, sizeof(uint32_t), size, fo);
		fclose(fo);
		return test_context.context->get_pipeline(pci);
	}

	template<class T, class F>
	inline Future unary_map(Future src, Future dst, Future count, const F& fn) {
		auto spv_result = SPIRVBinaryMap<T, uint32_t>::compile([](auto A, auto B) { return F{}(A); });
		static auto pbi = static_compute_pbi(spv_result.second.data(), spv_result.first, "unary");
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