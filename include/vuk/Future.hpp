#pragma once

#include "vuk/Allocator.hpp"
#include "vuk/Context.hpp"
#include "vuk/IR.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/Types.hpp"
#include "vuk/vuk_fwd.hpp"

#include <memory>
#include <span>
#include <variant>

// futures
namespace vuk {
	class UntypedValue {
	public:
		UntypedValue(std::shared_ptr<ExtRef> head, Ref def) : head(head), def(def) {}
		UntypedValue(std::shared_ptr<ExtRef> head, Ref def, std::vector<std::shared_ptr<ExtRef>> deps) : head(head), def(def), deps(deps) {}

		/// @brief Get the referenced RenderGraph
		const std::shared_ptr<RG>& get_render_graph() const noexcept {
			return head->module;
		}

		/// @brief Name the value currently referenced by this Value
		void set_name(std::string_view name) noexcept {
			get_render_graph()->name_output(head->get_head(), std::string(name));
		}

		Ref get_head() const noexcept {
			return head->get_head();
		}

		Ref get_def() const noexcept {
			return def;
		}

		void release(Access access = Access::eNone, DomainFlagBits domain = DomainFlagBits::eAny) noexcept {
			assert(head->acqrel->status == Signal::Status::eDisarmed);
			head->to_release(access, domain);
		}

		void to_acquire() {
			auto current_value = get_constant_value(def.node);
			auto current_ty = def.type();
			// new RG with ACQUIRE node
			auto new_rg = std::make_shared<RG>();
			auto new_def = new_rg->make_acquire(current_ty, nullptr, current_value);
			auto new_extref = std::make_shared<ExtRef>(new_rg, new_def);
			new_def.node->acquire.acquire = head->acqrel.get();
			deps = { head };
			head = new_extref;
			def = new_def;
		}

		/// @brief Submit Future for execution
		Result<void> submit(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {});
		/// @brief If the Future has been submitted for execution, polls for status.
		[[nodiscard]] Result<Signal::Status> poll();

		Result<void> wait(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {});

		std::shared_ptr<ExtRef> head;

		std::vector<std::shared_ptr<ExtRef>> deps;
	protected:
		Ref def;
	};

	template<class T>
	class Value : public UntypedValue {
	public:
		using UntypedValue::UntypedValue;

		template<class U>
		Value<U> transmute(Ref new_head) noexcept {
			head = std::make_shared<ExtRef>(ExtRef{ head->module, new_head });
			def = new_head;
			return *reinterpret_cast<Value<U>*>(this); // TODO: not cool
		}

		T* operator->() noexcept {
			return reinterpret_cast<T*>(get_constant_value(def.node));
		}

		/// @brief Wait and retrieve the result of the Future on the host
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
			if (src.get_def().node->kind == Node::CONSTRUCT) {
				def.node->construct.args[1] = src.get_def().node->construct.args[1];
				def.node->construct.args[2] = src.get_def().node->construct.args[2];
				def.node->construct.args[3] = src.get_def().node->construct.args[3];
			} else if (src.get_def().node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				Swapchain& swp = *reinterpret_cast<Swapchain*>(src.get_def().node->acquire_next_image.swapchain.node->construct.args[0].node->constant.value);
				def.node->construct.args[1] = get_render_graph()->make_constant<uint32_t>(swp.images[0].extent.extent.width);
				def.node->construct.args[2] = get_render_graph()->make_constant<uint32_t>(swp.images[0].extent.extent.height);
				def.node->construct.args[3] = get_render_graph()->make_constant<uint32_t>(swp.images[0].extent.extent.depth);
			}
		}

		/// @brief Inference target has the same width & height as the source
		void same_2D_extent_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			if (src.get_def().type()->is_image()) {
				def.node->construct.args[1] = src.get_def().node->construct.args[1];
				def.node->construct.args[2] = src.get_def().node->construct.args[2];
			} else if (src.get_def().type()->kind == Type::SWAPCHAIN_TY) {
				Swapchain& swp = *reinterpret_cast<Swapchain*>(src.get_def().node->acquire_next_image.swapchain.node->construct.args[0].node->constant.value);
				def.node->construct.args[1] = get_render_graph()->make_constant<uint32_t>(swp.images[0].extent.extent.width);
				def.node->construct.args[2] = get_render_graph()->make_constant<uint32_t>(swp.images[0].extent.extent.height);
			}
		}

		/// @brief Inference target has the same format as the source
		void same_format_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			if (src.get_def().type()->is_image()) {
				def.node->construct.args[4] = src.get_def().node->construct.args[4];
			} else if (src.get_def().type()->kind == Type::SWAPCHAIN_TY) {
				Swapchain& swp = *reinterpret_cast<Swapchain*>(src.get_def().node->acquire_next_image.swapchain.node->construct.args[0].node->constant.value);
				def.node->construct.args[4] = get_render_graph()->make_constant(swp.images[0].format);
			}
		}

		/// @brief Inference target has the same shape(extent, layers, levels) as the source
		void same_shape_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			same_extent_as(src);
			if (src.get_def().type()->is_image()) {
				for (auto i = 6; i < 10; i++) { /* 6 - 9 : layers, levels */
					def.node->construct.args[i] = src.get_def().node->construct.args[i];
				}
			} else if (src.get_def().type()->kind == Type::SWAPCHAIN_TY) {
				Swapchain& swp = *reinterpret_cast<Swapchain*>(src.get_def().node->acquire_next_image.swapchain.node->construct.args[0].node->constant.value);
				def.node->construct.args[6] = get_render_graph()->make_constant(swp.images[0].base_layer);
				def.node->construct.args[7] = get_render_graph()->make_constant(swp.images[0].layer_count);
				def.node->construct.args[8] = get_render_graph()->make_constant(swp.images[0].base_level);
				def.node->construct.args[9] = get_render_graph()->make_constant(swp.images[0].level_count);
			}
		}

		/// @brief Inference target is similar to(same shape, same format, same sample count) the source
		void similar_to(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			same_shape_as(src);
			same_format_as(src);
			if (src.get_def().type()->is_image()) {
				def.node->construct.args[5] = src.get_def().node->construct.args[5]; // sample count
			} else if (src.get_def().type()->kind == Type::SWAPCHAIN_TY) {
				def.node->construct.args[5] = get_render_graph()->make_constant(Samples::e1); // swapchain is always single-sample
			}
		}

		// Buffer inferences

		void same_size(const Value<Buffer>& src)
		  requires std::is_same_v<T, Buffer>
		{
			assert(src.get_def().type()->is_buffer());
			def.node->construct.args[1] = src.get_def().node->construct.args[1];
		}

		Value<uint64_t> get_size()
		  requires std::is_same_v<T, Buffer>
		{
			return { std::make_shared<ExtRef>(get_render_graph(), def.node->construct.args[1]), {} };
		}

		void set_size(Value<uint64_t> arg)
		  requires std::is_same_v<T, Buffer>
		{
			get_render_graph()->subgraphs.push_back(arg.get_render_graph());
			def.node->construct.args[1] = arg.get_head();
		}

		auto operator[](size_t index)
		  requires std::is_array_v<T>
		{
			assert(def.node->kind == Node::CONSTRUCT);
			assert(def.type()->kind == Type::ARRAY_TY);
			auto item_def = def.node->construct.defs[index];
			Ref item = head->module->make_array_indexing(def.type()->array.T, get_head(), head->module->make_constant(index));
			return Value<std::remove_reference_t<decltype(std::declval<T>()[0])>>(std::make_shared<ExtRef>(get_render_graph(), item), item_def, {head});
		}
	};

	inline Value<uint64_t> operator*(Value<uint64_t> a, uint64_t b) {
		Ref ref = a.get_render_graph()->make_math_binary_op(Node::BinOp::MUL, a.get_head(), a.get_render_graph()->make_constant(b));
		return std::move(std::move(a).transmute<uint64_t>(ref));
	}

	inline Result<void> wait_for_futures_explicit(Allocator& alloc, Compiler& compiler, std::span<UntypedValue> futures) {
		std::vector<SyncPoint> waits;
		for (uint64_t i = 0; i < futures.size(); i++) {
			auto& future = futures[i];			
			auto res = future.submit(alloc, compiler, {});
			if (!res) {
				return res;
			}
			if (future.head->acqrel->status != Signal::Status::eSynchronizable) {
				continue;
			}
			waits.emplace_back(future.head->acqrel->source);
		}
		if (waits.size() > 0) {
			return alloc.get_context().wait_for_domains(std::span(waits));
		}

		return { expected_value };
	}

	template<class... Args>
	Result<void> wait_for_futures(Allocator& alloc, Compiler& compiler, Args&&... futs) {
		auto cbs = std::array{ futs... };
		return wait_for_futures_explicit(alloc, compiler, cbs);
	}
} // namespace vuk