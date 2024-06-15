#pragma once

#include "vuk/IR.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/Types.hpp"
#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"
#include "vuk/vuk_fwd.hpp"

#include <memory>
#include <span>
#include <variant>

namespace vuk {
	class UntypedValue {
	public:
		UntypedValue() = default;
		UntypedValue(ExtRef extref) : node(std::move(extref.node)), index(extref.index) {}

		/// @brief Name the value currently referenced by this Value
		void set_name(std::string_view name) noexcept {
			current_module.name_output(get_head(), name);
		}

		Ref get_head() const noexcept {
			return { node->get_node(), index };
		}

		Ref get_peeled_head() noexcept {
			if (node.use_count() == 1 && get_head().node->kind == Node::SPLICE && can_peel) {
				Ref peeled_head = get_head().node->splice.src[get_head().index];
				return peeled_head;
			} else {
				return get_head();
			}
		}

		Ref peel_head() noexcept {
			if (node.use_count() == 1 && get_head().node->kind == Node::SPLICE && can_peel) {
				Ref peeled_head = get_head().node->splice.src[get_head().index];
				get_head().node->kind = Node::NOP;
				if (get_head().node->generic_node.arg_count == 255) {
					delete get_head().node->variable_node.args.data();
				}
				get_head().node->generic_node.arg_count = 0;
				current_module.garbage.push_back(get_head().node);
				return peeled_head;
			} else {
				return get_head();
			}
		}

		void release(Access access = Access::eNone, DomainFlagBits domain = DomainFlagBits::eAny) noexcept {
			assert(node->acqrel->status == Signal::Status::eDisarmed);
			auto ref = get_head();
			auto release = current_module.make_release(ref, nullptr, access, domain);
			node = std::make_shared<ExtNode>(release, node); // previous extnode is a dep
			release->release.release = node->acqrel.get();
			index = 0;
		}

		/// @brief Submit Value for execution
		Result<void> submit(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {});
		/// @brief If the Value has been submitted for execution, polls for status.
		[[nodiscard]] Result<Signal::Status> poll();

		Result<void> wait(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {});

		std::shared_ptr<ExtNode> node;

	protected:
		size_t index;
		bool can_peel = false;
	};

	template<class T>
	class Value : public UntypedValue {
	public:
		using UntypedValue::UntypedValue;

		template<class U>
		Value<U> transmute(Ref new_head) noexcept {
			node = std::make_shared<ExtNode>(new_head.node, node);
			index = new_head.index;
			return *reinterpret_cast<Value<U>*>(this); // TODO: not cool
		}

		T* operator->() noexcept {
			auto def_or_v = get_def(get_head());
			if (!def_or_v.is_ref) {
				return static_cast<T*>(def_or_v.value);
			}
			auto def = def_or_v.ref;
			return eval<T*>(def);
		}

		/// @brief Wait and retrieve the result of the Value on the host
		[[nodiscard]] Result<T> get(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {})
		  requires(!std::is_array_v<T>)
		{
			if (auto result = wait(allocator, compiler, options)) {
				return { expected_value, *operator->() };
			} else {
				return result;
			}
		}

		template<class U = T>
		Value<U> as_released(Access access = Access::eNone, DomainFlagBits domain = DomainFlagBits::eAny) {
			release(access, domain);
			return *reinterpret_cast<Value<U>*>(this); // TODO: not cool
		}

		// Image inferences
		void same_extent_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			if (get_head().node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				return;
			}
			node->deps.push_back(src.node);
			replace_arg_with_extract_or_constant(get_head(), src.get_head(), 0);
			replace_arg_with_extract_or_constant(get_head(), src.get_head(), 1);
			replace_arg_with_extract_or_constant(get_head(), src.get_head(), 2);
		}

		/// @brief Inference target has the same width & height as the source
		void same_2D_extent_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			if (get_head().node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				return;
			}
			node->deps.push_back(src.node);
			replace_arg_with_extract_or_constant(get_head(), src.get_head(), 0);
			replace_arg_with_extract_or_constant(get_head(), src.get_head(), 1);
		}

		/// @brief Inference target has the same format as the source
		void same_format_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			if (get_head().node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				return;
			}
			node->deps.push_back(src.node);
			replace_arg_with_extract_or_constant(get_head(), src.get_head(), 3);
		}

		/// @brief Inference target has the same shape(extent, layers, levels) as the source
		void same_shape_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			if (get_head().node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				return;
			}
			same_extent_as(src);

			for (auto i = 6; i < 10; i++) { /* 6 - 9 : layers, levels */
				replace_arg_with_extract_or_constant(get_head(), src.get_head(), i - 1);
			}
		}

		/// @brief Inference target is similar to(same shape, same format, same sample count) the source
		void similar_to(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			if (get_head().node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				return;
			}
			same_shape_as(src);
			same_format_as(src);
			replace_arg_with_extract_or_constant(get_head(), src.get_head(), 4);
		}

		// Buffer inferences

		void same_size(const Value<Buffer>& src)
		  requires std::is_same_v<T, Buffer>
		{
			node->deps.push_back(src.node);
			replace_arg_with_extract_or_constant(get_head(), src.get_head(), 0);
		}

		Value<uint64_t> get_size()
		  requires std::is_same_v<T, Buffer>
		{
			Ref extract = current_module.make_extract(get_head(), 0);
			return { ExtRef{ std::make_shared<ExtNode>(extract.node, node), extract } };
		}

		void set_size(Value<uint64_t> arg)
		  requires std::is_same_v<T, Buffer>
		{
			node->deps.push_back(arg.node);
			auto def_or_v = get_def(get_head());
			if (!def_or_v.is_ref) {
				return;
			}
			auto def = def_or_v.ref;
			def.node->construct.args[1] = arg.get_head();
		}

		auto operator[](size_t index)
		  requires std::is_array_v<T>
		{
			assert(get_head().type()->kind == Type::ARRAY_TY);
			Ref item = current_module.make_extract(get_head(), current_module.make_constant(index));
			return Value<std::remove_reference_t<decltype(std::declval<T>()[0])>>(ExtRef(std::make_shared<ExtNode>(item.node, node), item));
		}

		auto mip(uint32_t mip)
		  requires std::is_same_v<T, ImageAttachment>
		{
			can_peel = false;
			Ref item = current_module.make_slice(get_head(),
			                                     current_module.make_constant(mip),
			                                     current_module.make_constant(1u),
			                                     current_module.make_constant(0u),
			                                     current_module.make_constant(VK_REMAINING_ARRAY_LAYERS));
			return Value(ExtRef(std::make_shared<ExtNode>(item.node, node), item));
		}

		auto layer(uint32_t layer)
		  requires std::is_same_v<T, ImageAttachment>
		{
			can_peel = false;
			Ref item = current_module.make_slice(get_head(),
			                                     current_module.make_constant(0u),
			                                     current_module.make_constant(VK_REMAINING_MIP_LEVELS),
			                                     current_module.make_constant(layer),
			                                     current_module.make_constant(1u));
			return Value(ExtRef(std::make_shared<ExtNode>(item.node, node), item));
		}

		void replace_arg_with_extract_or_constant(Ref construct, Ref src_composite, uint64_t index) {
			auto def_or_v = get_def(construct);
			if (!def_or_v.is_ref) {
				return;
			}
			auto def = def_or_v.ref;
			Type* cty = current_module.u64();
			auto constant_node = Node{ .kind = Node::CONSTANT, .type = std::span{ &cty, 1 } };
			constant_node.constant.value = &index; // writing these out for clang workaround

			auto composite = src_composite;
			Type* ty;
			auto stripped = Type::stripped(composite.type());
			if (stripped->kind == Type::ARRAY_TY) {
				ty = stripped->array.T;
			} else if (stripped->kind == Type::COMPOSITE_TY) {
				ty = stripped->composite.types[index];
			}
			auto candidate_node = Node{ .kind = Node::EXTRACT, .type = std::span{ &ty, 1 } };
			candidate_node.extract.composite = composite; // writing these out for clang workaround
			candidate_node.extract.index = first(&constant_node);
			current_module.garbage.push_back(def.node->construct.args[index + 1].node);
			try {
				auto result = eval<uint64_t>(first(&candidate_node));
				def.node->construct.args[index + 1] = current_module.template make_constant<uint64_t>(result);
			} catch (...) {
				def.node->construct.args[index + 1] = current_module.make_extract(composite, index);
			}
		}
	};

	inline Value<uint64_t> operator+(Value<uint64_t> a, uint64_t b) {
		Ref ref = current_module.make_math_binary_op(Node::BinOp::ADD, a.get_head(), current_module.make_constant(b));
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator+(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module.make_math_binary_op(Node::BinOp::ADD, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator-(Value<uint64_t> a, uint64_t b) {
		Ref ref = current_module.make_math_binary_op(Node::BinOp::SUB, a.get_head(), current_module.make_constant(b));
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator-(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module.make_math_binary_op(Node::BinOp::SUB, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator*(Value<uint64_t> a, uint64_t b) {
		Ref ref = current_module.make_math_binary_op(Node::BinOp::MUL, a.get_head(), current_module.make_constant(b));
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator*(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module.make_math_binary_op(Node::BinOp::MUL, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator/(Value<uint64_t> a, uint64_t b) {
		Ref ref = current_module.make_math_binary_op(Node::BinOp::DIV, a.get_head(), current_module.make_constant(b));
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator/(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module.make_math_binary_op(Node::BinOp::DIV, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator%(Value<uint64_t> a, uint64_t b) {
		Ref ref = current_module.make_math_binary_op(Node::BinOp::MOD, a.get_head(), current_module.make_constant(b));
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator%(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module.make_math_binary_op(Node::BinOp::MOD, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Result<void> wait_for_values_explicit(Allocator& alloc, Compiler& compiler, std::span<UntypedValue> values) {
		std::vector<SyncPoint> waits;
		for (uint64_t i = 0; i < values.size(); i++) {
			auto& value = values[i];
			auto res = value.submit(alloc, compiler, {});
			if (!res) {
				return res;
			}
			if (value.node->acqrel->status != Signal::Status::eSynchronizable) {
				continue;
			}
			waits.emplace_back(value.node->acqrel->source);
		}
		if (waits.size() > 0) {
			// TODO: turn these into HostAvailable
			return alloc.get_context().wait_for_domains(std::span(waits));
		}

		return { expected_value };
	}

	template<class... Args>
	Result<void> wait_for_values(Allocator& alloc, Compiler& compiler, Args&&... futs) {
		auto cbs = std::array{ futs... };
		return wait_for_values_explicit(alloc, compiler, cbs);
	}
} // namespace vuk
