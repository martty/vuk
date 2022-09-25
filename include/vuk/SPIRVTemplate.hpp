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

		enum TypeClass { eUint = 0, eSint, eFloat, eBool };

		template<class T>
		constexpr TypeClass to_typeclass() {
			using type = typename T::type;
			if constexpr (std::is_same_v<type, unsigned>) {
				return TypeClass::eUint;
			}
			if constexpr (std::is_same_v<type, signed>) {
				return TypeClass::eSint;
			}
			if constexpr (std::is_same_v<type, float>) {
				return TypeClass::eFloat;
			}
			if constexpr (std::is_same_v<type, bool>) {
				return TypeClass::eBool;
			}
		}

		struct SPIRVModule {
			uint32_t counter;
			std::vector<uint32_t> annotations;
			std::vector<uint32_t> decls;
			std::vector<uint32_t> codes;

			std::vector<SPIRType> types;

			constexpr SPIRVModule(uint32_t id_counter,
			                      std::vector<uint32_t> annotations,
			                      std::vector<uint32_t> decls,
			                      std::vector<uint32_t> codes,
			                      std::vector<SPIRType> types) :
			    counter(id_counter),
			    annotations(std::move(annotations)),
			    decls(std::move(decls)),
			    codes(std::move(codes)),
			    types(std::move(types)) {}

			template<class T>
			constexpr auto type_id();

			constexpr auto operator+(SPIRVModule&& o) {
				annotations.insert(annotations.end(), o.annotations.begin(), o.annotations.end());
				decls.insert(decls.end(), o.decls.begin(), o.decls.end());
				codes.insert(codes.end(), o.codes.begin(), o.codes.end());
				types.insert(types.end(), o.types.begin(), o.types.end());

				return SPIRVModule{ std::max(counter, o.counter), std::move(annotations), std::move(decls), std::move(codes), std::move(types) };
			}

			template<size_t N>
			constexpr auto code(uint32_t counter, const std::array<uint32_t, N>& v) {
				codes.insert(codes.end(), v.begin(), v.end());

				return std::pair(counter, SPIRVModule{ counter, std::move(annotations), std::move(decls), std::move(codes), std::move(types) });
			}

			template<size_t N>
			constexpr auto constant(uint32_t counter, const std::array<uint32_t, N>& v) {
				decls.insert(decls.end(), v.begin(), v.end());

				return std::pair(counter, SPIRVModule{ counter, std::move(annotations), std::move(decls), std::move(codes), std::move(types) });
			}

			template<size_t N>
			constexpr auto annotation(uint32_t counter, const std::array<uint32_t, N>& v) {
				annotations.insert(annotations.end(), v.begin(), v.end());

				return SPIRVModule{ counter, std::move(annotations), std::move(decls), std::move(codes), std::move(types) };
			}
		};

		template<class T>
		constexpr auto SPIRVModule::type_id() {
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
			auto old_declsize = decls.size();
			auto old_annsize = annotations.size();
			auto [declid, declmod] = T::to_spirv(*this);
			if (!found) {
				declmod.counter = declid;
				declmod.types.push_back(SPIRType{ tn, declid });
				return std::pair(declid, declmod);
			} else {
				declmod.counter = counter;
				declmod.decls.resize(old_declsize);
				declmod.annotations.resize(old_annsize);
				return std::pair(id, declmod);
			}
		}

		template<size_t N, typename... Ts>
		constexpr auto _emit_children(auto& mod, std::tuple<Ts...>& children) {
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
		constexpr auto emit_children(auto& mod, std::tuple<Ts...>& children) {
			if constexpr (sizeof...(Ts) == 1) {
				auto [resid, resmod] = std::get<0>(children).to_spirv(mod);
				return std::pair{ std::array{ resid }, resmod };
			} else if constexpr (sizeof...(Ts) == 2) {
				auto [resid1, resmod1] = std::get<0>(children).to_spirv(mod);
				auto [resid2, resmod2] = std::get<1>(children).to_spirv(resmod1);
				return std::pair{ std::array{ resid2, resid1 }, resmod2 };
			} else {
				auto [resids, resmod] = _emit_children<sizeof...(Ts) - 1>(mod, children);
				return std::pair{ resids, resmod };
			}
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

			static constexpr uint32_t count = 4;

			static constexpr auto to_spirv(SPIRVModule& mod) {
				auto us = std::array{ op(spv::OpTypeInt, 4), mod.counter + 1, 32u, 0u };
				return mod.constant(mod.counter + 1, us);
			}
		};

		template<>
		struct Type<bool> {
			using type = bool;

			static constexpr uint32_t count = 2;

			static constexpr auto to_spirv(SPIRVModule& mod) {
				auto us = std::array{ op(spv::OpTypeBool, 2), mod.counter + 1 };
				return mod.constant(mod.counter + 1, us);
			}
		};

		template<>
		struct Type<float> {
			using type = float;

			static constexpr uint32_t count = 3;

			static constexpr auto to_spirv(SPIRVModule& mod) {
				auto us = std::array{ op(spv::OpTypeFloat, 3), mod.counter + 1, 32u };
				return mod.constant(mod.counter + 1, us);
			}
		};

		template<spv::StorageClass sc, class Pointee>
		struct Type<ptr<sc, Pointee>> {
			using type = ptr<sc, Pointee>;
			static constexpr auto storage_class = sc;
			using pointee = Pointee;

			static constexpr uint32_t count = pointee::count + 4;

			static constexpr auto to_spirv(SPIRVModule& mod) {
				auto [tid, modt] = mod.template type_id<Pointee>();
				auto us = std::array{ op(spv::OpTypePointer, 4), modt.counter + 1, uint32_t(sc), tid };
				return modt.constant(modt.counter + 1, us);
			}
		};

		template<class T>
		struct TypeRuntimeArray : public SpvExpression<TypeRuntimeArray<T>> {
			using pointee = T;

			static constexpr uint32_t count = pointee::count + 3 + 4;

			static constexpr auto to_spirv(SPIRVModule& mod) {
				auto [tid, modt] = mod.template type_id<T>();
				auto us = std::vector{ op(spv::OpTypeRuntimeArray, 3), modt.counter + 1, tid };
				auto deco =
				    std::vector{ op(spv::OpDecorate, 4), modt.counter + 1, uint32_t(spv::Decoration::DecorationArrayStride), (uint32_t)sizeof(typename T::type) };
				return std::pair(modt.counter + 1, modt + SPIRVModule{ modt.counter + 1, deco, us, {}, {} });
			}
		};

		template<class T, uint32_t Offset>
		struct Member {
			using type = T;

			static constexpr uint32_t count = type::count + 5;

			static constexpr auto to_spirv(auto mod, uint32_t parent, uint32_t index) {
				auto deco = std::array{ op(spv::OpMemberDecorate, 5), parent, index, uint32_t(spv::Decoration::DecorationOffset), Offset };
				return mod.annotation(mod.counter, deco);
			}
		};

		template<class... Members>
		struct TypeStruct : public SpvExpression<TypeStruct<Members...>> {
			using members = std::tuple<Members...>;

			static constexpr uint32_t count = 3 + 3 + (Members::count + ...);

			static constexpr auto to_spirv(SPIRVModule& mod) {
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
			static constexpr uint32_t count = 4 + 4 + 4;

			constexpr Variable(uint32_t descriptor_set, uint32_t binding) : descriptor_set(descriptor_set), binding(binding) {}

			constexpr auto to_spirv(SPIRVModule& mod) {
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
			static constexpr uint32_t count = 0;

			constexpr Id(uint32_t id) : id(id) {}

			constexpr auto to_spirv(SPIRVModule& mod) {
				return std::pair(id, mod);
			}
		};

		template<class T>
		struct Constant : public SpvExpression<Constant<T>> {
			using type = Type<T>;
			T value;
			std::tuple<> children;
			uint32_t id = 0;
			static constexpr size_t num_uints = sizeof(T) / sizeof(uint32_t);
			static constexpr uint32_t count = 3 + num_uints;

			constexpr Constant(T v) : value(v) {}

			constexpr auto to_spirv(SPIRVModule& mod) {
				auto as_uints = std::bit_cast<std::array<uint32_t, num_uints>>(value);
				auto [tid, modt] = mod.template type_id<type>();
				auto us = std::array{ op(spv::OpConstant, 3 + num_uints), tid, modt.counter + 1 };
				auto with_value = concat_array(us, as_uints);
				id = modt.counter + 1;
				return modt.constant(modt.counter + 1, with_value);
			}
		};

		template<typename E1, typename E2>
		struct Add : public SpvExpression<Add<E1, E2>> {
			static_assert(std::is_same_v<typename E1::type, typename E2::type>);
			using type = typename E1::type;

			std::tuple<E1, E2> children;
			uint32_t id = 0;
			static constexpr uint32_t count = E1::count + E2::count + 5;

			constexpr Add(E1 e1, E2 e2) : children(e1, e2) {}

		public:
			constexpr auto to_spirv(SPIRVModule& mod) {
				auto [eids, resmod] = emit_children(mod, children);
				auto [tid, mod3] = resmod.template type_id<type>();
				id = mod3.counter + 1;
				auto us = std::array{ op(std::is_floating_point_v<typename type::type> ? spv::OpFAdd : spv::OpIAdd, 5), tid, id } << eids;
				return mod3.code(mod3.counter + 1, us);
			}
		};

		template<typename E1, typename E2>
		Add<E1, E2> constexpr operator+(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return Add<E1, E2>{ *static_cast<const E1*>(&u), *static_cast<const E2*>(&v) };
		}

		template<typename E1, typename T>
		requires numeric<T> Add<E1, Constant<T>>
		constexpr operator+(SpvExpression<E1> const& u, T const& v) {
			return { *static_cast<const E1*>(&u), Constant<T>(v) };
		}

		template<typename E1, typename T>
		requires numeric<T> Add<Constant<T>, E1>
		constexpr operator+(T const& v, SpvExpression<E1> const& u) {
			return { Constant<T>(v), *static_cast<const E1*>(&u) };
		}

		template<typename E1, typename E2>
		struct Sub : public SpvExpression<Sub<E1, E2>> {
			static_assert(std::is_same_v<typename E1::type, typename E2::type>);
			using type = typename E1::type;

			std::tuple<E2, E1> children;
			uint32_t id = 0;
			static constexpr uint32_t count = type::count + E1::count + E2::count + 5;

			constexpr Sub(E1 e1, E2 e2) : children(e2, e1) {}

		public:
			constexpr auto to_spirv(SPIRVModule& mod) {
				auto [eids, resmod] = emit_children(mod, children);
				auto [tid, mod3] = resmod.template type_id<type>();
				id = mod3.counter + 1;
				auto us = std::array{ op(std::is_floating_point_v<typename type::type> ? spv::OpFSub : spv::OpISub, 5), tid, id } << eids;
				return mod3.code(mod3.counter + 1, us);
			}
		};

		template<typename E1, typename E2>
		Sub<E1, E2> constexpr operator-(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return Sub<E1, E2>{ *static_cast<const E1*>(&u), *static_cast<const E2*>(&v) };
		}

		template<typename E1, typename T>
		requires numeric<T> Sub<E1, Constant<T>>
		constexpr operator-(SpvExpression<E1> const& u, T const& v) {
			return { *static_cast<const E1*>(&u), Constant<T>(v) };
		}

		template<typename E1, typename T>
		requires numeric<T> Sub<Constant<T>, E1>
		constexpr operator-(T const& v, SpvExpression<E1> const& u) {
			return { Constant<T>(v), *static_cast<const E1*>(&u) };
		}

		template<typename E1, typename E2>
		struct Mul : public SpvExpression<Mul<E1, E2>> {
			static_assert(std::is_same_v<typename E1::type, typename E2::type>);
			using type = typename E1::type;

			std::tuple<E1, E2> children;
			static constexpr uint32_t count = type::count + E1::count + E2::count + 5;
			uint32_t id = 0;

			constexpr Mul(E1 e1, E2 e2) : children(e1, e2) {}

		public:
			constexpr auto to_spirv(SPIRVModule& mod) {
				auto [eids, resmod] = emit_children(mod, children);
				auto [tid, mod3] = resmod.template type_id<type>();
				id = mod3.counter + 1;
				auto us = std::array{ op(std::is_floating_point_v<typename type::type> ? spv::OpFMul : spv::OpIMul, 5), tid, id } << eids;
				return mod3.code(mod3.counter + 1, us);
			}
		};

		template<typename E1, typename E2>
		Mul<E1, E2> constexpr operator*(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return Mul<E1, E2>{ *static_cast<const E1*>(&u), *static_cast<const E2*>(&v) };
		}

		template<typename E1, typename T>
		requires numeric<T> Mul<E1, Constant<T>>
		constexpr operator*(SpvExpression<E1> const& u, T const& v) {
			return { *static_cast<const E1*>(&u), Constant<T>(v) };
		}

		template<typename E1, typename T>
		requires numeric<T> Mul<Constant<T>, E1>
		constexpr operator*(T const& v, SpvExpression<E1> const& u) {
			return { Constant<T>(v), *static_cast<const E1*>(&u) };
		}

		template<typename E1, typename E2>
		struct Div : public SpvExpression<Div<E1, E2>> {
			static_assert(std::is_same_v<typename E1::type, typename E2::type>);
			using type = typename E1::type;

			std::tuple<E2, E1> children;
			uint32_t id = 0;
			static constexpr uint32_t count = type::count + E1::count + E2::count + 5;

			constexpr Div(E1 e1, E2 e2) : children(e2, e1) {}

			static constexpr spv::Op divs[] = { spv::OpUDiv, spv::OpSDiv, spv::OpFDiv };

		public:
			constexpr auto to_spirv(SPIRVModule& mod) {
				auto [eids, resmod] = emit_children(mod, children);
				auto [tid, mod3] = resmod.template type_id<type>();
				id = mod3.counter + 1;
				auto us = std::array{ op(divs[to_typeclass<type>()], 5), tid, id } << eids;
				return mod3.code(mod3.counter + 1, us);
			}
		};

		template<typename E1, typename E2>
		Div<E1, E2> constexpr operator/(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return Div<E1, E2>{ *static_cast<const E1*>(&u), *static_cast<const E2*>(&v) };
		}

		template<typename E1, typename T>
		requires numeric<T> Div<E1, Constant<T>>
		constexpr operator/(SpvExpression<E1> const& u, T const& v) {
			return { *static_cast<const E1*>(&u), Constant<T>(v) };
		}

		template<typename E1, typename T>
		requires numeric<T> Div<Constant<T>, E1>
		constexpr operator/(T const& v, SpvExpression<E1> const& u) {
			return { Constant<T>(v), *static_cast<const E1*>(&u) };
		}

		template<typename E1>
		struct UnaryMinus : public SpvExpression<UnaryMinus<E1>> {
			using type = typename E1::type;

			static_assert(std::is_signed_v<typename type::type> || std::is_floating_point_v<typename type::type>);

			std::tuple<E1> children;
			uint32_t id = 0;
			static constexpr uint32_t count = type::count + E1::count + 4;

			constexpr UnaryMinus(E1 e1) : children(e1) {}

		public:
			constexpr auto to_spirv(SPIRVModule& mod) {
				auto [eids, resmod] = emit_children(mod, children);
				auto [tid, mod3] = resmod.template type_id<type>();
				id = mod3.counter + 1;
				auto us = std::array{ op(std::is_floating_point_v<typename type::type> ? spv::OpFNegate : spv::OpSNegate, 4), tid, id } << eids;
				return mod3.code(mod3.counter + 1, us);
			}
		};

		template<typename E1>
		UnaryMinus<E1> constexpr operator-(SpvExpression<E1> const& u) {
			return { *static_cast<const E1*>(&u) };
		}

		template<typename E1>
		struct Load : public SpvExpression<Load<E1>> {
			using type = typename Deref<typename E1::type, 1u>::type;

			uint32_t id = 0;
			std::tuple<E1> children;
			static constexpr uint32_t count = type::count + E1::count + 4;

			constexpr Load(E1 e1) : children(e1) {}

			constexpr auto to_spirv(SPIRVModule& mod) {
				auto [tid, modt] = mod.template type_id<type>();
				auto [e1id, mod1] = emit_children(modt, children);
				auto us = std::array{ op(spv::OpLoad, 4), tid, mod1.counter + 1 } << e1id;
				id = mod1.counter + 1;
				return mod1.code(mod1.counter + 1, us);
			}
		};

		template<typename E1, typename E2>
		struct Store : SpvExpression<Store<E1, E2>> {
			using type = void;

			const uint32_t id = ~0u;
			std::tuple<E2, E1> children;
			static constexpr uint32_t count = E1::count + E2::count + 3;

			constexpr Store(E1 ptr, E2 value) : children(value, ptr) {}

			constexpr auto to_spirv(SPIRVModule& mod) {
				auto [e1id, mod1] = emit_children(mod, children);
				auto us = std::array{ op(spv::OpStore, 3) } << e1id;
				return mod1.code(mod1.counter, us);
			}
		};

		template<typename T, typename E1, typename... Indices>
		struct AccessChain : SpvExpression<AccessChain<E1, Indices...>> {
			using type = T; // typename Deref<typename E1::type, Indices...>::type;

			std::tuple<E1, Indices...> children;
			uint32_t id = 0;
			static constexpr uint32_t count = sizeof...(Indices) + 4 + type::count + E1::count + (Indices::count + ...);

			constexpr AccessChain(T type, E1 base, Indices... inds) : children(base, inds...) {}

			constexpr auto to_spirv(SPIRVModule& mod) {
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

		enum CmpOp { eGreaterThan = 0, eGreaterThanEqual, eEqual, eNotEqual, eLessThanEqual, eLessThan };

		template<CmpOp Op, typename E1, typename E2>
		struct Cmp : SpvExpression<Cmp<Op, E1, E2>> {
			static_assert(std::is_same_v<typename E1::type, typename E2::type>);
			using type = Type<bool>; // TODO: vector types

			std::tuple<E2, E1> children;
			uint32_t id = 0;
			static constexpr uint32_t count = type::count + E1::count + E2::count + 5;

			static constexpr const std::array<spv::Op, 6> compares[3] = {
				{ spv::OpUGreaterThan, spv::OpUGreaterThanEqual, spv::OpIEqual, spv::OpINotEqual, spv::OpULessThanEqual, spv::OpULessThan },
				{ spv::OpSGreaterThan, spv::OpSGreaterThanEqual, spv::OpIEqual, spv::OpINotEqual, spv::OpSLessThanEqual, spv::OpSLessThan },
				{ spv::OpFOrdGreaterThan, spv::OpFOrdGreaterThanEqual, spv::OpFOrdEqual, spv::OpFOrdNotEqual, spv::OpFOrdLessThanEqual, spv::OpFOrdLessThan }
			};

			constexpr Cmp(E1 e1, E2 e2) : children(e2, e1) {}

			constexpr auto to_spirv(SPIRVModule& mod) {
				auto [eids, resmod] = emit_children(mod, children);
				auto [tid, modt] = resmod.template type_id<type>();
				id = modt.counter + 1;
				auto tc = to_typeclass<typename E1::type>();
				auto actualop = compares[tc][Op];
				auto us = std::array{ op(actualop, 5), tid, id } << eids;
				return modt.code(modt.counter + 1, us);
			}
		};

		template<typename E1, typename E2>
		Cmp<CmpOp::eGreaterThan, E1, E2> constexpr operator>(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return { *static_cast<const E1*>(&u), *static_cast<const E2*>(&v) };
		}

		template<typename E1, typename T>
		requires numeric<T> Cmp<CmpOp::eGreaterThan, E1, Constant<T>>
		constexpr operator>(SpvExpression<E1> const& u, T const& v) {
			return { *static_cast<const E1*>(&u), Constant<T>(v) };
		}

		template<typename E1, typename E2>
		Cmp<CmpOp::eGreaterThanEqual, E1, E2> constexpr operator>=(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return { *static_cast<const E1*>(&u), *static_cast<const E2*>(&v) };
		}

		template<typename E1, typename T>
		requires numeric<T> Cmp<CmpOp::eGreaterThanEqual, E1, Constant<T>>
		constexpr operator>=(SpvExpression<E1> const& u, T const& v) {
			return { *static_cast<const E1*>(&u), Constant<T>(v) };
		}

		template<typename E1, typename E2>
		Cmp<CmpOp::eEqual, E1, E2> constexpr operator==(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return { *static_cast<const E1*>(&u), *static_cast<const E2*>(&v) };
		}

		template<typename E1, typename T>
		requires numeric<T> Cmp<CmpOp::eEqual, E1, Constant<T>>
		constexpr operator==(SpvExpression<E1> const& u, T const& v) {
			return { *static_cast<const E1*>(&u), Constant<T>(v) };
		}

		template<typename E1, typename E2>
		Cmp<CmpOp::eNotEqual, E1, E2> constexpr operator!=(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return { *static_cast<const E1*>(&u), *static_cast<const E2*>(&v) };
		}

		template<typename E1, typename T>
		requires numeric<T> Cmp<CmpOp::eNotEqual, E1, Constant<T>>
		constexpr operator!=(SpvExpression<E1> const& u, T const& v) {
			return { *static_cast<const E1*>(&u), Constant<T>(v) };
		}

		template<typename E1, typename E2>
		Cmp<CmpOp::eLessThan, E1, E2> constexpr operator<(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return { *static_cast<const E1*>(&u), *static_cast<const E2*>(&v) };
		}

		template<typename E1, typename T>
		requires numeric<T> Cmp<CmpOp::eLessThan, E1, Constant<T>>
		constexpr operator<(SpvExpression<E1> const& u, T const& v) {
			return { *static_cast<const E1*>(&u), Constant<T>(v) };
		}

		template<typename E1, typename E2>
		Cmp<CmpOp::eLessThanEqual, E1, E2> constexpr operator<=(SpvExpression<E1> const& u, SpvExpression<E2> const& v) {
			return { *static_cast<const E1*>(&u), *static_cast<const E2*>(&v) };
		}

		template<typename E1, typename T>
		requires numeric<T> Cmp<CmpOp::eLessThanEqual, E1, Constant<T>>
		constexpr operator<=(SpvExpression<E1> const& u, T const& v) {
			return { *static_cast<const E1*>(&u), Constant<T>(v) };
		}

		template<typename Cond, typename E1, typename E2>
		struct Select : SpvExpression<Select<Cond, E1, E2>> {
			static_assert(std::is_same_v<typename E1::type, typename E2::type>);
			using type = typename E1::type;

			std::tuple<E2, E1, Cond> children;
			uint32_t id = 0;
			static constexpr uint32_t count = type::count + Cond::count + E1::count + E2::count + 6;

			constexpr Select(Cond cond, E1 e1, E2 e2) : children(e2, e1, cond) {}

			constexpr auto to_spirv(SPIRVModule& mod) {
				auto [eids, resmod] = emit_children(mod, children);
				auto [tid, modt] = resmod.template type_id<type>();
				id = modt.counter + 1;
				auto us = std::array{ op(spv::OpSelect, 6), tid, id } << eids;
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

		template<class Derived>
		struct SPIRVTemplate {
			template<class F>
			static constexpr auto compile(F&& f) {
				auto specialized = Derived::specialize(f);

				SPIRVModule spvmodule{ Derived::max_id, {}, {}, {}, std::vector<SPIRType>(Derived::predef_types.begin(), Derived::predef_types.end()) };
				spvmodule.types.reserve(100);
				auto res = specialized.to_spirv(spvmodule).second;

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
				std::array fixed_op_entry = std::array{ op(spv::OpEntryPoint, 5 + (uint32_t)Derived::builtin_variables.size() + (uint32_t)variable_ids.size()),
					                                      uint32_t(spv::ExecutionModelGLCompute),
					                                      4u }
				                            << maintext << Derived::builtin_variables;
				std::vector<uint32_t> op_entry(fixed_op_entry.begin(), fixed_op_entry.end());
				op_entry.insert(op_entry.end(), variable_ids.begin(), variable_ids.end());

				std::vector<uint32_t> final_bc(Derived::prelude.begin(), Derived::prelude.end());
				auto it = final_bc.insert(final_bc.end(), op_entry.begin(), op_entry.end());
				it = final_bc.insert(final_bc.end(), Derived::prologue.begin(), Derived::prologue.end());
				it = final_bc.insert(final_bc.end(), res.annotations.begin(), res.annotations.end());
				it = final_bc.insert(final_bc.end(), Derived::builtin_decls.begin(), Derived::builtin_decls.end());
				it = final_bc.insert(final_bc.end(), res.decls.begin(), res.decls.end());
				it = final_bc.insert(final_bc.end(), Derived::second_bit.begin(), Derived::second_bit.end());
				it = final_bc.insert(final_bc.end(), res.codes.begin(), res.codes.end());
				it = final_bc.insert(final_bc.end(), Derived::epilogue.begin(), Derived::epilogue.end());
				std::array<uint32_t,
				           Derived::prelude.size() + Derived::prologue.size() + specialized.count + Derived::builtin_decls.size() + Derived::second_bit.size() +
				               Derived::epilogue.size() + 25>
				    arr{};
				std::copy(final_bc.begin(), final_bc.end(), arr.begin());
				return std::pair(final_bc.size(), arr);
			}
		};
	} // namespace spirv
} // namespace vuk