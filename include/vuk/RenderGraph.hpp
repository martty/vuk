#pragma once

#include "vuk/ErasedTupleAdaptor.hpp"
#include "vuk/Hash.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/ir/IR.hpp"
#include "vuk/ir/IRCppSupport.hpp"
#include "vuk/ir/IRCppTypes.hpp"
#include "vuk/Result.hpp"
#include "vuk/runtime/vk/Image.hpp"
#include "vuk/runtime/vk/Pipeline.hpp"
#include "vuk/runtime/vk/VkSwapchain.hpp"
#include "vuk/SourceLocation.hpp"
#include "vuk/Value.hpp"
#include "vuk/vuk_fwd.hpp"

#include <deque>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <vector>

#if false // seems like clang still has issues with this - lets keep it safe
#define VUK_IA(access, ...)        vuk::Arg<vuk::ImageAttachment, access, decltype([]() {}) __VA_OPT__(, ) __VA_ARGS__>
#define VUK_BA(access, ...)        vuk::Arg<vuk::Buffer, access, decltype([]() {}) __VA_OPT__(, ) __VA_ARGS__>
#define VUK_ARG(type, access, ...) vuk::Arg<type, access, decltype([]() {}) __VA_OPT__(, ) __VA_ARGS__>
#else
namespace vuk {
	template<size_t I>
	struct tag_type {};
}; // namespace vuk
#define VUK_IA(access)        vuk::Arg<vuk::ImageAttachment, access, vuk::tag_type<__COUNTER__>>
#define VUK_BA(access)        vuk::Arg<vuk::Buffer<>, access, vuk::tag_type<__COUNTER__>>
#define VUK_ARG(type, access) vuk::Arg<type, access, vuk::tag_type<__COUNTER__>>
#endif

#define VUK_CALLSTACK vuk::SourceLocationAtFrame _pscope = VUK_HERE_AND_NOW(), vuk::SourceLocationAtFrame _scope = VUK_HERE_AND_NOW()
#define VUK_CALL      (_pscope != _scope ? _scope.parent = &_pscope, _scope : _scope)

namespace vuk {
	ADAPT_TEMPLATED_STRUCT_FOR_IR(Buffer, ptr, sz_bytes);
	ADAPT_STRUCT_FOR_IR(BufferCreateInfo, memory_usage, size, alignment);
	ADAPT_STRUCT_FOR_IR(ImageAttachment,
	                    image,
	                    image_view,
	                    image_flags,
	                    image_type,
	                    tiling,
	                    usage,
	                    extent,
	                    format,
	                    sample_count,
	                    allow_srgb_unorm_mutable,
	                    image_view_flags,
	                    view_type,
	                    components,
	                    layout,
	                    base_level,
	                    level_count,
	                    base_layer,
	                    layer_count);

	static_assert(erased_tuple_adaptor<view<BufferLike<float>>>::value);
} // namespace vuk

namespace vuk {
	ResourceUse to_use(Access acc);

	template<typename... T>
	static auto make_ret(std::shared_ptr<ExtNode> extnode, const std::tuple<T...>& us) {
		if constexpr (sizeof...(T) > 0) {
			size_t i = 0;
			// FIXME: I think this is well defined but seems like compilers don't agree on the result
#if VUK_COMPILER_MSVC
			return std::tuple{ Value<typename T::type>{ ExtRef{ extnode, Ref{ extnode->get_node(), sizeof...(T) - (++i) } } }... };
#else
			return std::tuple{ Value<typename T::type>{ ExtRef{ extnode, Ref{ extnode->get_node(), i++ } } }... };
#endif
		}
	}

	template<size_t N, typename... T>
	static auto fill_ret_ty(std::array<size_t, sizeof...(T)> idxs, const std::tuple<T...>& args, fixed_vector<std::shared_ptr<Type>, N>& ret_types) {
		size_t i = 0;
		(ret_types.emplace_back(current_module->types.make_aliased_ty(Type::stripped(std::get<T>(args).src.type()), idxs[i++] + 1)), ...);
	}

	inline auto First = [](auto& first, auto&...) -> auto& {
		return first;
	};

	template<class T>
	struct AsDynamicExtentView {
		using type = T;
	};

	template<class T, size_t Extent>
	struct AsDynamicExtentView<Buffer<T, Extent>> {
		using type = Buffer<T>;
	};

	template<class T>
	struct UnwrapArg;

	template<class T, Access acc, class UniqueT>
	struct UnwrapArg<Arg<T, acc, UniqueT>> {
		using type = typename AsDynamicExtentView<T>::type;
	};

	template<Unsynchronized T>
	struct UnwrapArg<T> {
		using type = T;
	};

	template<class T>
	struct AsArg;

	template<class T, Access acc, class UniqueT>
	struct AsArg<Arg<T, acc, UniqueT>> {
		using type = Arg<T, acc, UniqueT>;
	};

	template<Unsynchronized T>
	struct AsArg<T> {
		using type = Arg<T, Access::eNone, int>;
	};

	template<typename... T>
	struct TupleMap<std::tuple<T...>> {
		using ret_tuple = std::tuple<Value<typename UnwrapArg<T>::type>...>;

		template<class Ret, class F>
		static auto make_lam(Name name, F&& body, SchedulingInfo scheduling_info, VUK_CALLSTACK) {
			size_t hash_code = typeid(body).hash_code();

			auto callback = [typed_cb = std::move(body)](CommandBuffer& cb, std::span<void*> args, std::span<void*> meta, std::span<void*> rets) mutable {
				// we do type recovery here -> convert untyped args to typed ones
				alignas(alignof(std::tuple<CommandBuffer&, typename AsArg<T>::type...>)) char storage[sizeof(std::tuple<CommandBuffer&, typename AsArg<T>::type...>)];
				pack_typed_tuple<typename AsArg<T>::type...>(args, meta, cb, storage);
				if constexpr (!std::is_same_v<void,
				                              decltype(std::apply(typed_cb, *reinterpret_cast<std::tuple<CommandBuffer&, typename AsArg<T>::type...>*>(storage)))>) {
					auto typed_ret = std::apply(typed_cb, *reinterpret_cast<std::tuple<CommandBuffer&, typename AsArg<T>::type...>*>(storage));
					// now we erase these types
					if constexpr (!is_tuple<Ret>::value) {
						rets[0] = typed_ret.ptr;
					} else {
						unpack_typed_tuple(typed_ret, rets);
					}
				} else {
					std::apply(typed_cb, *reinterpret_cast<std::tuple<CommandBuffer&, typename AsArg<T>::type...>*>(storage));
				}
			};

			std::shared_ptr<Type> opaque_fn_ty;

			// when this function is called, we weave in this call into the IR
			return [untyped_cb = std::move(callback), name, scheduling_info, hash_code, opaque_fn_ty, inner_scope = VUK_CALL](
			           Value<typename UnwrapArg<T>::type>... args, VUK_CALLSTACK) mutable {
				auto& first = First(args...);

				bool reuse_node = first.node.use_count() == 1 && first.node->acqrel->status == Signal::Status::eDisarmed;
				reuse_node = false;

				std::tuple arg_tuple_as_a = { typename AsArg<T>::type{ nullptr, args.get_head() }... };
				constexpr size_t arg_count = std::tuple_size_v<decltype(arg_tuple_as_a)>;

				if (!opaque_fn_ty) {
					std::array<std::shared_ptr<Type>, arg_count> arg_types = { current_module->types.make_imbued_ty(
						  typename AsArg<T>::type{ nullptr, args.get_head() }.src.type(), AsArg<T>::type::access)... };

					fixed_vector<std::shared_ptr<Type>, arg_count> ret_types;
					if constexpr (is_tuple<Ret>::value) {
						auto [idxs, ret_tuple] = intersect_tuples<std::tuple<typename AsArg<T>::type...>, Ret>(arg_tuple_as_a);
						fill_ret_ty(idxs, ret_tuple, ret_types);
					} else if constexpr (!std::is_same_v<Ret, void>) {
						auto [idxs, ret_tuple] = intersect_tuples<std::tuple<typename AsArg<T>::type...>, std::tuple<Ret>>(arg_tuple_as_a);
						fill_ret_ty(idxs, ret_tuple, ret_types);
					}

					std::array<bool, arg_count> existing_maps = {};
					fixed_vector<size_t, arg_count> maps_to_add;
					for (auto& ret_t : ret_types) {
						assert(ret_t->kind == Type::ALIASED_TY);
						existing_maps[ret_t->aliased.ref_idx - 1] = true;
					}

					auto old_ret_cnt = ret_types.size();
					for (size_t i = 0; i < arg_types.size(); i++) {
						if (!existing_maps[i]) {
							maps_to_add.push_back(i);
							ret_types.push_back(current_module->types.make_aliased_ty(Type::stripped(arg_types[i]), i + 1));
						}
					}

					if (maps_to_add.size() > 0) {
						auto wrapped_cb = [cb = std::move(untyped_cb), old_ret_cnt, maps_to_add](CommandBuffer& cbuf,
						                                                                         std::span<void*> opaque_args,
						                                                                         std::span<void*> opaque_meta,
						                                                                         std::span<void*> opaque_rets) mutable -> void {
							cb(cbuf, opaque_args, opaque_meta, opaque_rets.subspan(0, old_ret_cnt));
							for (auto i = 0; i < maps_to_add.size(); i++) {
								opaque_rets[old_ret_cnt + i] = opaque_args[maps_to_add[i]];
							}
						};
						opaque_fn_ty =
						    current_module->types.make_opaque_fn_ty(arg_types, ret_types, vuk::DomainFlagBits::eAny, hash_code, std::move(wrapped_cb), name.c_str());
					} else {
						opaque_fn_ty =
						    current_module->types.make_opaque_fn_ty(arg_types, ret_types, vuk::DomainFlagBits::eAny, hash_code, std::move(untyped_cb), name.c_str());
					}
				}
				auto opaque_fn = current_module->make_declare_fn(opaque_fn_ty);
				Node* node = current_module->make_call(opaque_fn, args.get_head()...);
				node->scheduling_info = new SchedulingInfo(scheduling_info);
				inner_scope.parent = &_scope;
				current_module->set_source_location(node, inner_scope);

				std::vector<std::shared_ptr<ExtNode>> dependent_nodes;
				[reuse_node, &dependent_nodes](auto& first, auto&... rest) {
					if (!reuse_node) {
						dependent_nodes.push_back(std::move(first.node));
					}
					(dependent_nodes.push_back(std::move(rest.node)), ...);
				}(args...);

				if (reuse_node) {
					first.node->mutate(node);
				}
				auto extnode = reuse_node ? std::move(first.node) : std::make_shared<ExtNode>(node, std::move(dependent_nodes));
				if (reuse_node) {
					extnode->deps.insert(extnode->deps.end(), std::make_move_iterator(dependent_nodes.begin()), std::make_move_iterator(dependent_nodes.end()));
				}

				for (auto& node : extnode->deps) {
					node->deps.push_back(extnode);
				}

				current_module->set_source_location(extnode->get_node(), inner_scope);

				if constexpr (is_tuple<Ret>::value) {
					auto [idxs, ret_tuple] = intersect_tuples<std::tuple<typename AsArg<T>::type...>, Ret>(arg_tuple_as_a);
					return make_ret(std::move(extnode), ret_tuple);
				} else if constexpr (!std::is_same_v<Ret, void>) {
					auto [idxs, ret_tuple] = intersect_tuples<std::tuple<typename AsArg<T>::type...>, std::tuple<Ret>>(arg_tuple_as_a);
					return std::get<0>(make_ret(std::move(extnode), ret_tuple));
				}
			};
		}
	};

	/// @brief Turn a lambda into a callable rendergraph computation (a pass)
	/// @tparam F Lambda type
	/// @param name Debug name for the pass
	/// @param body Callback lambda (body of the pass)
	/// @param scheduling_info Queue scheduling constraints
	template<class F>
	[[nodiscard]] auto make_pass(Name name, F&& body, SchedulingInfo scheduling_info = SchedulingInfo(DomainFlagBits::eAny), VUK_CALLSTACK) {
		using traits = closure_traits<decltype(&F::operator())>;
		static_assert(std::is_same_v<std::tuple_element_t<0, typename traits::types>, CommandBuffer&>, "First argument to pass MUST be CommandBuffer&");
		return TupleMap<drop_t<1, typename traits::types>>::template make_lam<typename traits::result_type, F>(
		    name, std::forward<F>(body), scheduling_info, VUK_CALL);
	}
	
	/// @brief Turn a compute pipeline create info into a callable compute pass
	inline auto lift_compute(PipelineBaseCreateInfo pbci, VUK_CALLSTACK) {
		return [pbci, inner_scope = VUK_CALL]<class... T>(size_t size_x, size_t size_y, size_t size_z, Value<T>... args) mutable { // no callstack for these :/
			Node* node = current_module->make_call(current_module->make_constant(pbci),
			                                       current_module->make_constant(size_x),
			                                       current_module->make_constant(size_y),
			                                       current_module->make_constant(size_z),
			                                       args.get_head()...);
			current_module->set_source_location(node, inner_scope);
			auto extnode = std::make_shared<ExtNode>(node);
			(args.node->deps.push_back(extnode), ...);
		};
	}

	/// @brief Turn a compute pipeline into a callable compute pass
	inline auto lift_compute(PipelineBaseInfo* compute_pipeline, VUK_CALLSTACK) {
		auto& flat_bindings = compute_pipeline->reflection_info.flat_bindings;

		std::vector<std::shared_ptr<Type>> arg_types;
		std::vector<std::shared_ptr<Type>> ret_types;
		std::shared_ptr<Type> base_ty;
		size_t i = 0;
		for (auto& [set_index, b] : flat_bindings) {
			Access acc = Access::eNone;
			switch (b->type) {
			case DescriptorType::eSampledImage:
				acc = Access::eComputeSampled;
				base_ty = current_module->types.get_builtin_image();
				break;
			case DescriptorType::eCombinedImageSampler:
				acc = Access::eComputeSampled;
				base_ty = current_module->types.get_builtin_sampled_image();
				break;
			case DescriptorType::eStorageImage:
				acc = b->non_writable ? Access::eComputeRead : (b->non_readable ? Access::eComputeWrite : Access::eComputeRW);
				base_ty = current_module->types.get_builtin_image();
				break;
			case DescriptorType::eUniformBuffer:
			case DescriptorType::eStorageBuffer:
				acc = b->non_writable ? Access::eComputeRead : (b->non_readable ? Access::eComputeWrite : Access::eComputeRW);
				base_ty = to_IR_type<view<BufferLike<float>>>();
				break;
			case DescriptorType::eSampler:
				acc = Access::eNone;
				base_ty = current_module->types.get_builtin_sampler();
				break;
			default:
				assert(0);
			}

			arg_types.push_back(current_module->types.make_imbued_ty(base_ty, acc));
			ret_types.emplace_back(current_module->types.make_aliased_ty(base_ty, i + 4));
			i++;
		}

		if (compute_pipeline->reflection_info.push_constant_ranges.size() > 0) {
			auto& pcr = compute_pipeline->reflection_info.push_constant_ranges[0];
			auto base_ty = current_module->types.make_pointer_ty(current_module->types.make_scalar_ty(Type::FLOAT_TY, 32)); // TODO: IR types from shader types
			for (auto j = 0; j < pcr.num_members; j++) {
				// TODO: check which args are pointers and dereference on host the once that are not
				arg_types.push_back(current_module->types.make_imbued_ty(base_ty, Access::eComputeRW));
				ret_types.emplace_back(current_module->types.make_aliased_ty(base_ty, i + 4));
				i++;
			}
		}
		return [arg_types, ret_types, compute_pipeline, inner_scope = VUK_CALL]<class... T>(
		           size_t size_x, size_t size_y, size_t size_z, Value<T>... args) mutable { // no callstack for these :/
			assert(sizeof...(args) == arg_types.size());

			auto shader_fn_ty =
			    current_module->types.make_shader_fn_ty(arg_types, ret_types, vuk::DomainFlagBits::eAny, compute_pipeline, compute_pipeline->pipeline_name.c_str());
			auto fn = current_module->make_declare_fn(shader_fn_ty);
			Node* node = current_module->make_call(
			    fn, current_module->make_constant(size_x), current_module->make_constant(size_y), current_module->make_constant(size_z), args.get_head()...);
			current_module->set_source_location(node, inner_scope);
			auto extnode = std::make_shared<ExtNode>(node);
			(args.node->deps.push_back(extnode), ...);
		};
	}

	inline ExtRef make_ext_ref(Ref ref, std::vector<std::shared_ptr<ExtNode>> deps = {}) {
		return ExtRef(std::make_shared<ExtNode>(ref.node, std::move(deps)), ref);
	}

	[[nodiscard]] inline Value<PipelineBaseInfo*> compile_pipeline(Value<PipelineBaseCreateInfo> pbci, VUK_CALLSTACK) {
		Ref ref = current_module->make_compile_pipeline(pbci.get_head());
		current_module->set_source_location(ref.node, VUK_CALL);
		return { make_ext_ref(ref) };
	}

	// acquire ~~ int* x = _existing_;
	// discard ~~ int* x = _existing_; invalidate(x);

	[[nodiscard]] inline Value<ImageAttachment> declare_ia(Name name, ImageAttachment ia = {}, VUK_CALLSTACK) {
		Ref ref = current_module->make_declare_image(ia);
		current_module->name_output(ref, name.c_str());
		current_module->set_source_location(ref.node, VUK_CALL);
		return { make_ext_ref(ref) };
	}

	[[nodiscard]] inline Value<ImageAttachment> discard_ia(Name name, ImageAttachment ia, VUK_CALLSTACK) {
		assert(ia.image_view != ImageView{});
		Ref ref = current_module->make_declare_image(ia);
		current_module->name_output(ref, name.c_str());
		current_module->set_source_location(ref.node, VUK_CALL);
		return { make_ext_ref(ref) };
	}

	/// @brief Adopt an existing resource into a Value.
	/// @param name
	/// @param ia
	/// @param previous_access
	[[nodiscard]] inline Value<ImageAttachment> acquire_ia(Name name, ImageAttachment ia, Access previous_access, VUK_CALLSTACK) {
		assert(ia.image_view != ImageView{});
		Ref ref = current_module->acquire(current_module->types.get_builtin_image(), nullptr, ia);
		auto ext_ref = ExtRef(std::make_shared<ExtNode>(ref.node, to_use(previous_access)), ref);
		current_module->name_output(ref, name.c_str());
		current_module->set_source_location(ref.node, VUK_CALL);
		return { std::move(ext_ref) };
	}

	// TODO: PAV: constrain to meaningful types?
	template<class T>
	[[nodiscard]] inline Value<T> discard(Name name, T buf, VUK_CALLSTACK) {
		assert(buf);
		return acquire(name, buf, Access::eNone, VUK_CALL);
	}

	template<class T = byte>
	[[nodiscard]] inline Value<Buffer<T>> allocate(Name name, Value<BufferCreateInfo> bci = {}, VUK_CALLSTACK) {
		std::array<Ref, 3> args = {};
		args[0] = bci->memory_usage.get_head();
		args[1] = bci->size.get_head();
		args[2] = bci->alignment.get_head();

		Ref bci_ref = current_module->make_construct(to_IR_type<BufferCreateInfo>(), nullptr, args);
		Ref ptr_ref = current_module->make_allocate(current_module->types.make_pointer_ty(to_IR_type<T>()), bci_ref);
		std::array arg_refs = { ptr_ref, bci->size.get_head() };
		Ref ref = current_module->make_construct(to_IR_type<Buffer<T>>(), nullptr, std::span(arg_refs));
		current_module->name_output(ref, name.c_str());
		current_module->set_source_location(ref.node, VUK_CALL);
		return { make_ext_ref(ref) };
	}

	template<class T>
	[[nodiscard]] inline Value<T> acquire(Name name, T value, Access access, VUK_CALLSTACK) {
		Ref ref = current_module->acquire(to_IR_type<T>(), nullptr, value);
		auto ext_ref = ExtRef(std::make_shared<ExtNode>(ref.node, to_use(access)), ref);
		current_module->name_output(ref, name.c_str());
		current_module->set_source_location(ref.node, VUK_CALL);
		return { std::move(ext_ref) };
	}

	template<class T>
	[[nodiscard]] inline val_view<BufferLike<T>> make_view(val_ptr<BufferLike<T>> buf, Value<size_t> size, VUK_CALLSTACK) {
		std::array<Ref, 2> args = { buf.get_head(), size.get_head() };
		Ref ref = current_module->make_construct(to_IR_type<view<BufferLike<T>>>(), nullptr, args);
		auto ext_ref = make_ext_ref(ref, { buf.node, size.node });
		current_module->set_source_location(ref.node, VUK_CALL);
		return { std::move(ext_ref) };
	}

	struct NameWithLocation : Name {
		SourceLocationAtFrame location;

		NameWithLocation(const char* name_, VUK_CALLSTACK) : Name(name_), location(VUK_CALL) {}
		explicit NameWithLocation(Name name_, VUK_CALLSTACK) : Name(name_), location(VUK_CALL) {}
	};

	template<class T, class... Args>
	[[nodiscard]] inline Value<T[]> declare_array(NameWithLocation name, Value<T> arg, Args... args) {
		std::vector<std::shared_ptr<ExtNode>> deps;
		std::array refs = { arg.get_head(), args.get_head()... };
		deps = { arg.node, args.node... };
		Ref ref = current_module->make_declare_array(Type::stripped(refs[0].type()), refs);
		current_module->name_output(ref, name.c_str());
		current_module->set_source_location(ref.node, name.location);
		return { make_ext_ref(ref, deps) };
	}

	template<class T>
	[[nodiscard]] inline Value<T[]> declare_array(Name name, std::span<const Value<T>> args, VUK_CALLSTACK) {
		std::vector<Ref> refs;
		std::vector<std::shared_ptr<ExtNode>> deps;
		for (size_t i = 0; i < args.size(); i++) {
			auto& arg = args[i];
			refs.push_back(arg.get_head());
			deps.push_back(arg.node);
		}
		std::shared_ptr<Type> t;
		if constexpr (std::is_same_v<T, vuk::ImageAttachment>) {
			t = current_module->types.get_builtin_image();
		} else if constexpr (std::is_same_v<T, vuk::Sampler>) {
			t = current_module->types.get_builtin_sampler();
		} else if constexpr (std::is_same_v<T, vuk::SampledImage>) {
			t = current_module->types.get_builtin_sampled_image();
		}
		Ref ref = current_module->make_declare_array(t, refs);
		current_module->name_output(ref, name.c_str());
		current_module->set_source_location(ref.node, VUK_CALL);
		return { make_ext_ref(ref, std::move(deps)) };
	}

	template<class T>
	[[nodiscard]] inline Value<T[]> declare_array(Name name, std::span<Value<T>> args, VUK_CALLSTACK) {
		std::vector<Ref> refs;
		std::vector<std::shared_ptr<ExtNode>> deps;
		for (size_t i = 0; i < args.size(); i++) {
			auto& arg = args[i];
			refs.push_back(arg.get_head());
			deps.push_back(arg.node);
		}
		std::shared_ptr<Type> t;
		if constexpr (std::is_same_v<T, vuk::ImageAttachment>) {
			t = current_module->types.get_builtin_image();
		} else if constexpr (std::is_same_v<T, vuk::Sampler>) {
			t = current_module->types.get_builtin_sampler();
		} else if constexpr (std::is_same_v<T, vuk::SampledImage>) {
			t = current_module->types.get_builtin_sampled_image();
		}
		Ref ref = current_module->make_declare_array(t, refs);
		current_module->name_output(ref, name.c_str());
		current_module->set_source_location(ref.node, VUK_CALL);
		return { make_ext_ref(ref, std::move(deps)) };
	}

	[[nodiscard]] inline Value<Swapchain> acquire_swapchain(Swapchain& bundle, VUK_CALLSTACK) {
		Ref ref = current_module->make_declare_swapchain(bundle);
		current_module->set_source_location(ref.node, VUK_CALL);
		return { make_ext_ref(ref) };
	}

	[[nodiscard]] inline Value<Sampler> acquire_sampler(Name name, SamplerCreateInfo sci, VUK_CALLSTACK) {
		Ref ref = current_module->acquire(current_module->types.get_builtin_sampler(), nullptr, sci);
		auto ext_ref = ExtRef(std::make_shared<ExtNode>(ref.node, to_use(Access::eNone)), ref);
		current_module->name_output(ref, name.c_str());
		current_module->set_source_location(ref.node, VUK_CALL);
		return { std::move(ext_ref) };
	}

	[[nodiscard]] inline Value<SampledImage> combine_image_sampler(Name name, Value<ImageAttachment> ia, Value<Sampler> sampler, VUK_CALLSTACK) {
		Ref ref = current_module->make_sampled_image(ia.get_head(), sampler.get_head());
		current_module->name_output(ref, name.c_str());
		current_module->set_source_location(ref.node, VUK_CALL);
		return { make_ext_ref(ref, { ia.node, sampler.node }) };
	}

	[[nodiscard]] inline Value<ImageAttachment> acquire_next_image(Name name, Value<Swapchain> in, VUK_CALLSTACK) {
		Ref ref = current_module->make_acquire_next_image(in.get_head());
		current_module->name_output(ref, name.c_str());
		current_module->set_source_location(ref.node, VUK_CALL);
		return std::move(in).transmute<ImageAttachment>(ref);
	}

	template<class T>
	[[nodiscard]] inline Value<T> make_constant(Name name, T in, VUK_CALLSTACK) {
		Ref ref = current_module->make_constant(in);
		current_module->name_output(ref, name.c_str());
		current_module->set_source_location(ref.node, VUK_CALL);
		return { make_ext_ref(ref) };
	}

	[[nodiscard]] inline Value<void> enqueue_presentation(Value<ImageAttachment> in) {
		return std::move(in).as_released<void>(Access::ePresent, DomainFlagBits::ePE);
	}

	struct Compiler {
		Compiler();
		~Compiler();

		void reset();

		/// @brief Build the graph, assign framebuffers, render passes and subpasses
		///	link automatically calls this, only needed if you want to use the reflection functions
		/// @param compile_options CompileOptions controlling compilation behaviour
		Result<void> compile(Allocator& allocator, std::span<std::shared_ptr<ExtNode>> rgs, const RenderGraphCompileOptions& compile_options);

		// reflection functions

		/// @brief retrieve usages of resources in the RenderGraph
		std::span<struct ChainLink*> get_use_chains() const;

		/// @brief compute ImageUsageFlags for given use chain
		ImageUsageFlags compute_usage(const struct ChainLink* chain);

		/// @brief Dump the pass dependency graph in graphviz format
		std::string dump_graph();

		/// @brief retrieve nodes in the scheduled order
		std::span<struct ScheduledItem*> get_scheduled_nodes() const;

		Result<void> execute(Allocator& allocator);

	private:
		struct RGCImpl* impl;

		// internal passes
		void queue_inference();
		void pass_partitioning();
		Result<void> validate_read_undefined();
		Result<void> validate_same_argument_different_access();
	};
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::Subrange::Image> {
		size_t operator()(vuk::Subrange::Image const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.base_layer, x.base_level, x.layer_count, x.level_count);
			return h;
		}
	};
}; // namespace std
