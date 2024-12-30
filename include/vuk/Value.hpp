#pragma once

#include "vuk/ImageAttachment.hpp"
#include "vuk/IR.hpp"
#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"
#include "vuk/Types.hpp"
#include "vuk/vuk_fwd.hpp"

#include <memory>
#include <span>
#include <variant>

namespace vuk {
	/// @brief Base class for typed Value, provides execution methods
	class UntypedValue {
	public:
		UntypedValue() = default;
		UntypedValue(ExtRef extref) : node(std::move(extref.node)), index(extref.index) {}

		/// @brief Set a debug name for this Value
		/// @param name Debug name to assign
		void set_name(std::string_view name) noexcept {
			current_module->name_output(get_head(), name);
		}

		/// @brief Get the internal IR reference for this Value
		/// @return Reference to the IR node
		Ref get_head() const noexcept {
			return { node->get_node(), index };
		}

		void release(Access access = Access::eNone, DomainFlagBits domain = DomainFlagBits::eDevice) noexcept {
			assert(node->acqrel && node->acqrel->status == Signal::Status::eDisarmed);
			node = std::make_shared<ExtNode>(Ref{ node->get_node(), index }, node, access, domain); // previous extnode is a dep
		}

		/// @brief Submit the render graph for execution without waiting
		/// @param allocator Allocator to use for resource allocation
		/// @param compiler Compiler to use for graph compilation
		/// @param options Optional compilation options
		/// @return Result indicating success or error
		Result<void> submit(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {});

		/// @brief Poll the execution status of this Value
		/// @return Current status of execution (pending, ready, etc.)
		[[nodiscard]] Result<Signal::Status> poll();

		/// @brief Submit the render graph and wait for completion
		/// @param allocator Allocator to use for resource allocation
		/// @param compiler Compiler to use for graph compilation
		/// @param options Optional compilation options
		/// @return Result indicating success or error
		Result<void> wait(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {});

		std::shared_ptr<ExtNode> node;

	protected:
		size_t index;
	};

	/// @brief Represents a GPU resource that will be available after some work completes
	/// @tparam T Type of the resource (Buffer, ImageAttachment, etc.)
	template<class T>
	class Value : public UntypedValue {
	public:
		using UntypedValue::UntypedValue;

		/// @brief Internal: Transmute this Value to a different type
		/// @tparam U New type for the Value
		/// @param new_head New IR reference
		/// @return Value with new type
		template<class U>
		Value<U> transmute(Ref new_head) noexcept {
			node = std::make_shared<ExtNode>(new_head.node, node);
			index = new_head.index;
			return *reinterpret_cast<Value<U>*>(this); // TODO: not cool
		}

		/// @brief Access the underlying resource (only after declare or wait/get)
		/// @return Pointer to the resource
		T* operator->() noexcept {
			auto v = eval(get_head());
			assert(v.holds_value());
			return (T*)v->value;
		}

		/// @brief Submit, wait, and retrieve the resource value on the host
		/// @param allocator Allocator to use for resource allocation
		/// @param compiler Compiler to use for graph compilation
		/// @param options Optional compilation options
		/// @return Result containing the resource, or an error
		[[nodiscard]] Result<T> get(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {})
		  requires(!std::is_array_v<T>)
		{
			if (auto result = wait(allocator, compiler, options)) {
				return { expected_value, *operator->() };
			} else {
				return result;
			}
		}

		/// @brief Mark this Value as released for use outside the render graph
		/// @tparam U Type of the returned Value (defaults to T)
		/// @param access The access pattern for future use
		/// @param domain The domain where the resource will be used
		/// @return New Value representing the released resource
		template<class U = T>
		Value<U> as_released(Access access = Access::eNone, DomainFlagBits domain = DomainFlagBits::eAny) {
			release(access, domain);
			return *reinterpret_cast<Value<U>*>(this); // TODO: not cool
		}

		// Image inferences

		/// @brief Infer extent (width, height, depth) from another image
		/// @param src Source image to copy extent from
		void same_extent_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			node->deps.push_back(src.node);
			set_with_extract(get_head(), src.get_head(), 0);
			set_with_extract(get_head(), src.get_head(), 1);
			set_with_extract(get_head(), src.get_head(), 2);
		}

		/// @brief Infer 2D extent (width, height) from another image
		/// @param src Source image to copy 2D extent from
		void same_2D_extent_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			node->deps.push_back(src.node);
			set_with_extract(get_head(), src.get_head(), 0);
			set_with_extract(get_head(), src.get_head(), 1);
		}

		/// @brief Infer format from another image
		/// @param src Source image to copy format from
		void same_format_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			node->deps.push_back(src.node);
			set_with_extract(get_head(), src.get_head(), 3);
		}

		/// @brief Infer shape (extent, layers, mip levels) from another image
		/// @param src Source image to copy shape from
		void same_shape_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			same_extent_as(src);

			for (auto i = 6; i < 10; i++) { /* 6 - 9 : layers, levels */
				set_with_extract(get_head(), src.get_head(), i - 1);
			}
		}

		/// @brief Infer all properties (shape, format, sample count) from another image
		/// @param src Source image to copy properties from
		void similar_to(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			same_shape_as(src);
			same_format_as(src);
			set_with_extract(get_head(), src.get_head(), 4);
		}

		// Buffer inferences

		/// @brief Create a subrange view of this buffer
		/// @param new_offset Offset in bytes from the start of the buffer
		/// @param new_size Size of the subrange in bytes
		/// @return Value representing the buffer subrange
		Value<Buffer> subrange(uint64_t new_offset, uint64_t new_size)
		  requires std::is_same_v<T, Buffer>
		{
			Ref item =
			    current_module->make_slice(get_head(), 0, current_module->make_constant<uint64_t>(new_offset), current_module->make_constant<uint64_t>(new_size));
			return Value(ExtRef(std::make_shared<ExtNode>(item.node, node), item));
		}

		/// @brief Infer buffer size from another buffer
		/// @param src Source buffer to copy size from
		void same_size(const Value<Buffer>& src)
		  requires std::is_same_v<T, Buffer>
		{
			node->deps.push_back(src.node);
			set_with_extract(get_head(), src.get_head(), 0);
		}

		Value<uint64_t> get_size_bytes()
		  requires std::is_same_v<T, Buffer>
		{
			Ref extract = current_module->make_extract(get_head(), 0);
			return { ExtRef{ std::make_shared<ExtNode>(extract.node, node), extract } };
		}

		void set_size_bytes(Value<uint64_t> arg)
		  requires std::is_base_of_v<ptr_base, T>
		{
			node->deps.push_back(arg.node);
			current_module->set_value(get_head(), 0, arg.get_head());
		}

		/// @brief Array subscript operator for array-typed Values
		/// @param index Index into the array
		/// @return Value representing the array element
		auto operator[](size_t index)
		  requires std::is_array_v<T>
		{
			assert(Type::stripped(get_head().type())->kind == Type::ARRAY_TY);
			Ref item = current_module->make_extract(get_head(), current_module->make_constant(index));
			return Value<std::remove_reference_t<decltype(std::declval<T>()[0])>>(ExtRef(std::make_shared<ExtNode>(item.node, node), item));
		}

		/// @brief Get a specific mip level of this image
		/// @param mip Mip level to extract
		/// @return Value representing the mip level
		auto mip(uint32_t mip)
		  requires std::is_same_v<T, ImageAttachment>
		{
			Ref item = current_module->make_slice(
			    get_head(), Node::NamedAxis::MIP, current_module->make_constant<uint64_t>(mip), current_module->make_constant<uint64_t>(1u));
			return Value(ExtRef(std::make_shared<ExtNode>(item.node, node), item));
		}

		/// @brief Get a specific array layer of this image
		/// @param layer Array layer to extract
		/// @return Value representing the array layer
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

		Value<view<T>> implicit_view()
		  requires std::is_base_of_v<ptr_base, T>
		{
			std::array args = { get_head(), current_module->make_get_allocation_size(get_head()) };
			auto imp_view = current_module->make_construct(current_module->types.make_bufferlike_view_ty(current_module->types.u32()), args);
			auto vval = Value<view<T>>{ make_ext_ref(imp_view, { node }) };
			node->deps.push_back(vval.node);
			return std::move(vval);
		}
	};

	template<class T = void, class... Ctrs>
	using val_ptr = Value<ptr<T, Ctrs...>>;

	template<class T = void, class... Ctrs>
	using val_view = Value<view<T, Ctrs...>>;

	// Arithmetic operators for Value<uint64_t>

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

	/// @brief Submit multiple Values for execution
	/// @param allocator Allocator to use for resource allocation
	/// @param compiler Compiler to use for graph compilation
	/// @param values Span of Values to submit
	/// @param options Optional compilation options
	/// @return Result indicating success or error
	Result<void> submit(Allocator& allocator, Compiler& compiler, std::span<UntypedValue> values, RenderGraphCompileOptions options);

	/// @brief Wait for multiple Values to complete execution
	/// @param alloc Allocator to use for resource allocation
	/// @param compiler Compiler to use for graph compilation
	/// @param values Span of Values to wait for
	/// @param options Optional compilation options
	/// @return Result indicating success or error
	Result<void> wait_for_values_explicit(Allocator& alloc, Compiler& compiler, std::span<UntypedValue> values, RenderGraphCompileOptions options = {});

	/// @brief Wait for multiple Values to complete execution (variadic)
	/// @tparam Args Types of the Values
	/// @param alloc Allocator to use for resource allocation
	/// @param compiler Compiler to use for graph compilation
	/// @param futs Values to wait for
	/// @return Result indicating success or error
	template<class... Args>
	Result<void> wait_for_values(Allocator& alloc, Compiler& compiler, Args&&... futs) {
		auto cbs = std::array{ futs... };
		return wait_for_values_explicit(alloc, compiler, cbs);
	}
} // namespace vuk
