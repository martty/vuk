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
			current_module->name_output(get_head(), name);
		}

		Ref get_head() const noexcept {
			return { node->get_node(), index };
		}

		void release(Access access = Access::eNone, DomainFlagBits domain = DomainFlagBits::eDevice) noexcept {
			assert(node->acqrel && node->acqrel->status == Signal::Status::eDisarmed);
			node = std::make_shared<ExtNode>(Ref{ node->get_node(), index }, node, access, domain); // previous extnode is a dep
		}

		/// @brief Submit Value for execution
		Result<void> submit(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {});
		/// @brief Polls for the status of this Value.
		[[nodiscard]] Result<Signal::Status> poll();
		/// @brief Submit this Value and waits for it the be ready on the host.
		Result<void> wait(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {});

		std::shared_ptr<ExtNode> node;

	protected:
		size_t index;
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
			auto def_or_v = *get_def(get_head());
			if (!def_or_v.is_ref) {
				return static_cast<T*>(def_or_v.value);
			}
			auto def = def_or_v.ref;
			return *eval<T*>(def);
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
			node->deps.push_back(src.node);
			set_with_extract(get_head(), src.get_head(), 0);
			set_with_extract(get_head(), src.get_head(), 1);
			set_with_extract(get_head(), src.get_head(), 2);
		}

		/// @brief Inference target has the same width & height as the source
		void same_2D_extent_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			node->deps.push_back(src.node);
			set_with_extract(get_head(), src.get_head(), 0);
			set_with_extract(get_head(), src.get_head(), 1);
		}

		/// @brief Inference target has the same format as the source
		void same_format_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			node->deps.push_back(src.node);
			set_with_extract(get_head(), src.get_head(), 3);
		}

		/// @brief Inference target has the same shape(extent, layers, levels) as the source
		void same_shape_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			same_extent_as(src);

			for (auto i = 6; i < 10; i++) { /* 6 - 9 : layers, levels */
				set_with_extract(get_head(), src.get_head(), i - 1);
			}
		}

		/// @brief Inference target is similar to(same shape, same format, same sample count) the source
		void similar_to(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			same_shape_as(src);
			same_format_as(src);
			set_with_extract(get_head(), src.get_head(), 4);
		}

		// Buffer inferences

		Value<Buffer> subrange(uint64_t new_offset, uint64_t new_size)
		  requires std::is_same_v<T, Buffer>
		{
			Ref item =
			    current_module->make_slice(get_head(), 0, current_module->make_constant<uint64_t>(new_offset), current_module->make_constant<uint64_t>(new_size));
			return Value(ExtRef(std::make_shared<ExtNode>(item.node, node), item));
		}

		void same_size(const Value<Buffer>& src)
		  requires std::is_same_v<T, Buffer>
		{
			node->deps.push_back(src.node);
			set_with_extract(get_head(), src.get_head(), 0);
		}

		Value<uint64_t> get_size()
		  requires std::is_same_v<T, Buffer>
		{
			Ref extract = current_module->make_extract(get_head(), 0);
			return { ExtRef{ std::make_shared<ExtNode>(extract.node, node), extract } };
		}

		void set_size(Value<uint64_t> arg)
		  requires std::is_same_v<T, Buffer>
		{
			node->deps.push_back(arg.node);
			current_module->set_value(get_head(), 0, arg.get_head());
		}

		auto operator[](size_t index)
		  requires std::is_array_v<T>
		{
			assert(Type::stripped(get_head().type())->kind == Type::ARRAY_TY);
			Ref item = current_module->make_extract(get_head(), current_module->make_constant(index));
			return Value<std::remove_reference_t<decltype(std::declval<T>()[0])>>(ExtRef(std::make_shared<ExtNode>(item.node, node), item));
		}

		auto mip(uint32_t mip)
		  requires std::is_same_v<T, ImageAttachment>
		{
			Ref item = current_module->make_slice(
			    get_head(), Node::NamedAxis::MIP, current_module->make_constant<uint64_t>(mip), current_module->make_constant<uint64_t>(1u));
			return Value(ExtRef(std::make_shared<ExtNode>(item.node, node), item));
		}

		auto layer(uint32_t layer)
		  requires std::is_same_v<T, ImageAttachment>
		{
			Ref item = current_module->make_slice(
			    get_head(), Node::NamedAxis::LAYER, current_module->make_constant<uint64_t>(layer), current_module->make_constant<uint64_t>(1u));
			return Value(ExtRef(std::make_shared<ExtNode>(item.node, node), item));
		}

		void set_with_extract(Ref construct, Ref src_composite, uint64_t index) {
			current_module->set_value(construct, index, current_module->make_extract(src_composite, index));
		}
	};

	inline Value<uint64_t> operator+(Value<uint64_t> a, uint64_t b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::ADD, a.get_head(), current_module->make_constant(b));
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator+(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::ADD, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator-(Value<uint64_t> a, uint64_t b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::SUB, a.get_head(), current_module->make_constant(b));
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator-(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::SUB, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator*(Value<uint64_t> a, uint64_t b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::MUL, a.get_head(), current_module->make_constant(b));
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator*(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::MUL, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator/(Value<uint64_t> a, uint64_t b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::DIV, a.get_head(), current_module->make_constant(b));
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator/(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::DIV, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator%(Value<uint64_t> a, uint64_t b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::MOD, a.get_head(), current_module->make_constant(b));
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator%(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::MOD, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	Result<void> submit(Allocator& allocator, Compiler& compiler, std::span<UntypedValue> values, RenderGraphCompileOptions options);

	Result<void> wait_for_values_explicit(Allocator& alloc, Compiler& compiler, std::span<UntypedValue> values, RenderGraphCompileOptions options = {});

	template<class... Args>
	Result<void> wait_for_values(Allocator& alloc, Compiler& compiler, Args&&... futs) {
		auto cbs = std::array{ futs... };
		return wait_for_values_explicit(alloc, compiler, cbs);
	}
} // namespace vuk
