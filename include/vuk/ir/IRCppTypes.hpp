#pragma once

#include "vuk/ErasedTupleAdaptor.hpp"
#include "vuk/ir/IR.hpp"
#include <fmt/format.h>

namespace vuk {
	template<class T>
	struct ir_type_provider {
		static constexpr bool has_custom_ir_type = false;
		static std::shared_ptr<Type> get_ir_type() {
			static_assert(sizeof(T) == 0, "No IR type provider for this type");
			return nullptr;
		}
	};

	template<class>
	struct dependent_false : std::false_type {};

	template<template<typename...> class T, typename U>
	struct is_specialization_of : std::false_type {};

	template<template<typename...> class T, typename... Us>
	struct is_specialization_of<T, T<Us...>> : std::true_type {};

	template<class T>
	struct is_imagelike : std::false_type {};

	template<Format f>
	struct is_imagelike<ImageLike<f>> : std::true_type {};

	template<class T>
	struct is_imageview : std::false_type {};

	template<Format f>
	struct is_imageview<view<ImageLike<f>, dynamic_extent>> : std::true_type {};

	template<class T>
	inline std::shared_ptr<Type> to_IR_type() {
		if constexpr (ir_type_provider<T>::has_custom_ir_type) {
			return ir_type_provider<T>::get_ir_type();
		} else if constexpr (!std::is_same_v<T, typename detail::unwrap<T>::T>) {
			return to_IR_type<typename detail::unwrap<T>::T>();
		}
		if constexpr (std::is_array_v<T> && std::extent_v<T> == 0) {
			return to_IR_type<std::remove_all_extents_t<T>>();
		}

		if constexpr (std::is_void_v<T>) {
			return current_module->types.make_void_ty();
		} else if constexpr (std::is_integral_v<T>) {
			return current_module->types.make_scalar_ty(Type::INTEGER_TY, sizeof(T) * 8);
		} else if constexpr (std::is_floating_point_v<T>) {
			return current_module->types.make_scalar_ty(Type::FLOAT_TY, sizeof(T) * 8);
		} else if constexpr (std::is_enum_v<T>) {
			return current_module->types.make_scalar_ty(Type::INTEGER_TY, sizeof(T) * 8);
		} else if constexpr (is_imageview<T>::value) {
			return current_module->types.make_imageview_ty();
		} else if constexpr (std::is_base_of_v<ptr_base, T>) {
			return current_module->types.make_pointer_ty(to_IR_type<typename T::UnwrappedT>());
		} else if constexpr (erased_tuple_adaptor<T>::value) {
			std::vector<std::shared_ptr<Type>> child_types =
			    std::apply([&](auto... member_tys) { return std::vector<std::shared_ptr<Type>>{ to_IR_type<decltype(member_tys)>()... }; },
			               erased_tuple_adaptor<T>::member_types);
			auto offsets = std::vector<size_t>(erased_tuple_adaptor<T>::offsets.begin(), erased_tuple_adaptor<T>::offsets.end());
			auto composite_type = current_module->types.emplace_type(
			    std::shared_ptr<Type>(new Type{ .kind = Type::COMPOSITE_TY,
			                                    .size = sizeof(T),
			                                    .debug_info = current_module->types.allocate_type_debug_info(erased_tuple_adaptor<T>::name),
			                                    .child_types = child_types,
			                                    .offsets = offsets,
			                                    .member_names = { erased_tuple_adaptor<T>::member_names.begin(), erased_tuple_adaptor<T>::member_names.end() },
			                                    .composite = { .types = child_types,
			                                                   .tag = std::hash<const char*>{}(erased_tuple_adaptor<T>::name),
			                                                   .construct = &erased_tuple_adaptor<T>::construct,
			                                                   .get = &erased_tuple_adaptor<T>::get,
			                                                   .is_default = &erased_tuple_adaptor<T>::is_default,
			                                                   .destroy = &erased_tuple_adaptor<T>::destroy,
			                                                   .format_to = [](void* v, std::string& dst) {
				                                                   fmt::format_to(std::back_inserter(dst), "{}", *reinterpret_cast<T*>(v));
			                                                   } } }));
			return composite_type;
		} else {
			static_assert(dependent_false<T>::value, "Cannot convert type to IR");
		}
	}

} // namespace vuk