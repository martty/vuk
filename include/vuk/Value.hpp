#pragma once

#include "vuk/ImageAttachment.hpp"
#include "vuk/ir/IR.hpp"
#include "vuk/ir/IRCppTypes.hpp"
#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"
#include "vuk/Types.hpp"
#include "vuk/vuk_fwd.hpp"

#include <memory>
#include <span>
#include <variant>

namespace vuk {
	struct SyncHelper;

	template<>
	struct erased_tuple_adaptor<ImageAttachment>;

	template<class T>
	struct is_value : std::false_type {};

	/// @brief Base class for typed Value, provides execution methods
	class UntypedValue {
	public:
		UntypedValue() = default;
		UntypedValue(ExtRef extref) : node(std::move(extref.node)), index(extref.index) {}
		UntypedValue(Ref ref) : node(std::make_shared<ExtNode>(ref.node)), index(ref.index) {}

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

	template<class T>
	class Value;

	/// @brief Represents a GPU resource that will be available after some work completes
	/// @tparam T Type of the resource (Buffer, ImageAttachment, etc.)
	template<class T>
	class ValueBase : public UntypedValue {
	public:
		using UntypedValue::UntypedValue;

		/// @brief Create a Value from a unsynchronized type (eg. int)
		template<class U = T>
		ValueBase(U s)
		  requires(Unsynchronized<U> && std::is_convertible_v<U, T>)
		{
			// support for placeholders
			if constexpr (erased_tuple_adaptor<U>::value) {
				constexpr size_t member_count = std::tuple_size_v<decltype(erased_tuple_adaptor<U>::members)>;
				std::array<Ref, member_count> args;

				std::apply(
				    [&](auto... member_ptrs) {
					    size_t i = 0;
					    ((args[i] = erased_tuple_adaptor<U>::is_default(&s, i)
					                    ? current_module->make_placeholder(to_IR_type<detail::member_type_t<decltype(member_ptrs)>>())
					                    : current_module->make_constant(to_IR_type<detail::member_type_t<decltype(member_ptrs)>>(), erased_tuple_adaptor<U>::get(&s, i)),
					      i++),
					     ...);
				    },
				    erased_tuple_adaptor<U>::members);

				Ref ref = current_module->make_construct(to_IR_type<U>(), nullptr, std::span(args));
				node = std::make_shared<ExtNode>(ref.node);
				index = ref.index;
			} else {
				// Regular constant for non-adapted types
				Ref ref = current_module->make_constant(to_IR_type<U>(), &s);
				node = std::make_shared<ExtNode>(ref.node);
				index = ref.index;
			}
		}

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
			return get_value<T>(get_head());
		}

		auto operator->() noexcept
		  requires(erased_tuple_adaptor<T>::value)
		{
			return std::apply(
			    [head = get_head(), this](auto... a) {
				    size_t i = 0;
				    return typename erased_tuple_adaptor<T>::proxy{ [&, this](auto a, auto i) {
					    Ref ref = current_module->make_extract(head, i);
					    ExtRef exref = make_ext_ref(ref);
					    node->deps.push_back(exref.node);
					    return exref;
					  }((a, head), i++)... };
			    },
			    erased_tuple_adaptor<T>::member_types);
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
				return { expected_value, *get_value<T>(get_head()) };
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

		void set_with_extract(Ref construct, Ref src_composite, uint64_t index) {
			current_module->set_value(construct, index, current_module->make_extract(src_composite, index));
		}
	};

	template<class T>
	class Value : public ValueBase<T> {
	public:
		using ValueBase<T>::ValueBase;

		auto operator[](size_t index) {
			assert(Type::stripped(this->get_head().type())->kind == Type::ARRAY_TY);
			Ref item = current_module->make_extract(this->get_head(), current_module->make_constant(index));
			return Value<std::remove_reference_t<decltype(std::declval<T>()[0])>>(item);
		}

		template<class U = T>
		U operator*()
		  requires(Unsynchronized<U> && !std::is_array_v<U>)
		{
			auto v = eval(this->get_head());
			if (v) {
				return *static_cast<T*>(*v);
			}
			assert(false);
			return U{}; // unreachable
		}
	};

	inline Value<uint64_t> operator+(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::ADD, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator-(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::SUB, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator*(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::MUL, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator/(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::DIV, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Value<uint64_t> operator%(Value<uint64_t> a, Value<uint64_t> b) {
		Ref ref = current_module->make_math_binary_op(Node::BinOp::MOD, a.get_head(), b.get_head());
		a.node->deps.push_back(b.node);
		return std::move(a).transmute<uint64_t>(ref);
	}

	template<class T>
	struct is_value<Value<T>> : std::true_type {};

	template<class T = void>
	using val_ptr = Value<ptr<T>>;

	template<class Type>
	struct Value<view<BufferLike<Type>, dynamic_extent>> : ValueBase<view<BufferLike<Type>, dynamic_extent>> {
		Value<ptr<BufferLike<Type>>> ptr;
		Value<uint64_t> sz_bytes;

		Value<view<BufferLike<Type>, dynamic_extent>>(Ref ref) : ValueBase<view<BufferLike<Type>, dynamic_extent>>(ref) {
			ptr = Value<vuk::ptr<BufferLike<Type>>>(current_module->make_extract(this->get_head(), 0));
			sz_bytes = Value<uint64_t>(current_module->make_extract(this->get_head(), 1));
		}

		Value<view<BufferLike<Type>, dynamic_extent>>(ExtRef extref) : ValueBase<view<BufferLike<Type>, dynamic_extent>>(extref) {
			ptr = Value<vuk::ptr<BufferLike<Type>>>(current_module->make_extract(this->get_head(), 0));
			sz_bytes = Value<uint64_t>(current_module->make_extract(this->get_head(), 1));
		}
		Value<view<BufferLike<Type>, dynamic_extent>>(Value<vuk::ptr<BufferLike<Type>>> ptr, Value<uint64_t> count)
		  requires(!std::is_array_v<Type>)
		    : ptr(ptr), sz_bytes(count * sizeof(Type)) {
			assert(false);
		}

		auto& operator[](Value<size_t> index)
		  requires(!std::is_same_v<Type, void>)
		{
			// assert(index < (sz_bytes / sizeof(Type)));
			// TODO: IR assert
			return ptr[index];
		}

		const auto& operator[](Value<size_t> index) const
		  requires(!std::is_same_v<Type, void>)
		{
			// assert(index < (sz_bytes / sizeof(Type)));
			return ptr[index];
		}

		explicit operator Value<bool>() const noexcept {
			return !!ptr;
		}

		[[nodiscard]] Value<vuk::ptr<BufferLike<Type>>> data() noexcept {
			return ptr;
		}

		[[nodiscard]] Value<uint64_t> size_bytes() const noexcept {
			return sz_bytes;
		}

		[[nodiscard]] Value<uint64_t> size() const noexcept {
			return sz_bytes / sizeof(Type);
		}

		template<class new_T>
		[[nodiscard]] Value<view<BufferLike<new_T>>> cast() noexcept {
			return { ptr.template transmute<vuk::ptr<BufferLike<new_T>>>(this->get_head()), sz_bytes };
		}

		[[nodiscard]] view<BufferLike<byte>> to_byte_view() const noexcept {
			return cast<byte>();
		}

		// TODO: PAV: operate on Ts, not bytes
		// TODO: PAV: this is completely wrong
		/// @brief Create a new view that is a subset of the original
		[[nodiscard]] Value<view<BufferLike<Type>>> subview(Value<uint64_t> offset, Value<uint64_t> new_count = ~(0ULL)) const {
			// TODO: IR assert
			// 	assert(offset + new_count <= count());
			Ref item = current_module->make_slice(this->get_head(), 0, offset.get_head(), new_count.get_head());
			return Value(item);
		}

		template<class U = Type>
		void same_size(const Value<view<BufferLike<U>>>& src) {
			this->node->deps.push_back(src.node);
			current_module->set_value(sz_bytes.get_head(), src.sz_bytes.get_head());
		}

		void set_memory_usage(MemoryUsage mu) {
			current_module->set_value_on_allocate_src(this->get_head(), 0, mu);
		}

		void set_size(Value<uint64_t> size) {
			current_module->set_value_on_allocate_src(this->get_head(), 1, size.get_head());
		}

		void set_alignment(Value<uint64_t> alignment) {
			current_module->set_value_on_allocate_src(this->get_head(), 2, alignment.get_head());
		}
	};

	template<class T = void>
	using val_view = Value<view<T>>;
	
	// Arithmetic operators for Value<uint64_t>

	template<>
	struct Value<ImageAttachment> : ValueBase<ImageAttachment> {
		Value<Image> image = {};
		Value<ImageView> image_view = {};

		Value<ImageCreateFlags> image_flags = {};
		Value<ImageType> image_type = ImageType::e2D;
		Value<ImageTiling> tiling = ImageTiling::eOptimal;
		Value<ImageUsageFlags> usage = {};
		Value<Extent3D> extent = {};
		Value<Format> format = Format::eUndefined;
		Value<Samples> sample_count = Samples::eInfer;
		Value<bool> allow_srgb_unorm_mutable = false;
		Value<ImageViewCreateFlags> image_view_flags = {};
		Value<ImageViewType> view_type = ImageViewType::eInfer;
		Value<ComponentMapping> components = {};
		Value<ImageLayout> layout = ImageLayout::eUndefined;

		Value<uint32_t> base_level = VK_REMAINING_MIP_LEVELS;
		Value<uint32_t> level_count = VK_REMAINING_MIP_LEVELS;

		Value<uint32_t> base_layer = VK_REMAINING_ARRAY_LAYERS;
		Value<uint32_t> layer_count = VK_REMAINING_ARRAY_LAYERS;

		Value<ImageAttachment>() = default;

		Value<ImageAttachment>(Ref ref) : ValueBase<ImageAttachment>(ref) {
			/* ptr = Value<vuk::ptr<BufferLike<Type>>>(current_module->make_extract(this->get_head(), 0));
			sz_bytes = Value<size_t>(current_module->make_extract(this->get_head(), 1));*/
		}

		Value<ImageAttachment>(ExtRef extref) : Value<ImageAttachment>(Ref(extref.node->get_node(), extref.index)) {}
		// Image inferences
		void same_extent_as(const Value<ImageAttachment>& src) {
			this->node->deps.push_back(src.node);
			this->set_with_extract(this->get_head(), src.get_head(), 0);
			this->set_with_extract(this->get_head(), src.get_head(), 1);
			this->set_with_extract(this->get_head(), src.get_head(), 2);
		}

		/// @brief Inference target has the same width & height as the source
		void same_2D_extent_as(const Value<ImageAttachment>& src) {
			this->node->deps.push_back(src.node);
			this->set_with_extract(this->get_head(), src.get_head(), 0);
			this->set_with_extract(this->get_head(), src.get_head(), 1);
		}

		/// @brief Inference target has the same format as the source
		void same_format_as(const Value<ImageAttachment>& src) {
			this->node->deps.push_back(src.node);
			this->set_with_extract(this->get_head(), src.get_head(), 3);
		}

		/// @brief Inference target has the same shape(extent, layers, levels) as the source
		void same_shape_as(const Value<ImageAttachment>& src) {
			same_extent_as(src);

			for (auto i = 6; i < 10; i++) { /* 6 - 9 : layers, levels */
				this->set_with_extract(this->get_head(), src.get_head(), i - 1);
			}
		}

		/// @brief Inference target is similar to(same shape, same format, same sample count) the source
		void similar_to(const Value<ImageAttachment>& src) {
			same_shape_as(src);
			same_format_as(src);
			this->set_with_extract(this->get_head(), src.get_head(), 4);
		}

		Value<ImageAttachment> mip(uint32_t mip) {
			Ref item = current_module->make_slice(
			    this->get_head(), Node::NamedAxis::MIP, current_module->make_constant<uint64_t>(mip), current_module->make_constant<uint64_t>(1u));
			return Value(item);
		}

		Value<ImageAttachment> layer(uint32_t layer) {
			Ref item = current_module->make_slice(
			    this->get_head(), Node::NamedAxis::LAYER, current_module->make_constant<uint64_t>(layer), current_module->make_constant<uint64_t>(1u));
			return Value(item);
		}
	};

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
