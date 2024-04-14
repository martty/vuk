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
		UntypedValue() = default;
		UntypedValue(ExtRef extref, Ref def) : node(std::move(extref.node)), head{ node->get_node(), extref.index }, def(def) {}
		UntypedValue(ExtRef extref, Ref def, std::vector<std::shared_ptr<ExtNode>> deps) :
		    node(std::move(extref.node)),
		    head{ node->get_node(), extref.index },
		    deps(std::move(deps)),
		    def(def) {}

		/// @brief Get the referenced RenderGraph
		const std::shared_ptr<RG>& get_render_graph() const noexcept {
			return node->module;
		}

		/// @brief Name the value currently referenced by this Value
		void set_name(std::string_view name) noexcept {
			get_render_graph()->name_output(head, name);
		}

		Ref get_head() const noexcept {
			return head;
		}

		Ref get_def() const noexcept {
			return def;
		}

		Ref get_peeled_head() noexcept {
			if (node.use_count() == 1 && head.node->kind == Node::RELACQ && can_peel) {
				Ref peeled_head = head.node->relacq.src[head.index];
				return peeled_head;
			} else {
				return head;
			}
		}

		Ref peel_head() noexcept {
			if (node.use_count() == 1 && head.node->kind == Node::RELACQ && can_peel) {
				Ref peeled_head = head.node->relacq.src[head.index];
				head.node->kind = Node::NOP;
				return peeled_head;
			} else {
				return head;
			}
		}

		void release(Access access = Access::eNone, DomainFlagBits domain = DomainFlagBits::eAny) noexcept {
			assert(node->acqrel->status == Signal::Status::eDisarmed);
			auto ref = get_head();
			auto release = node->module->make_release(ref, nullptr, access, domain);
			deps.push_back(node); // previous extnode is a dep
			node = std::make_shared<ExtNode>(ExtNode{ node->module, release });
			release->release.release = node->acqrel;
			head = { node->get_node(), 0 };
		}

		/// @brief Submit Future for execution
		Result<void> submit(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {});
		/// @brief If the Future has been submitted for execution, polls for status.
		[[nodiscard]] Result<Signal::Status> poll();

		Result<void> wait(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {});

		std::shared_ptr<ExtNode> node;
		Ref head;
		std::vector<std::shared_ptr<ExtNode>> deps;

	protected:
		Ref def;
		bool can_peel = true;
	};

	template<class T>
	class Value : public UntypedValue {
	public:
		using UntypedValue::UntypedValue;

		template<class U>
		Value<U> transmute(Ref new_head) noexcept {
			node = std::make_shared<ExtNode>(ExtNode{ node->module, new_head.node });
			head = { node->get_node(), new_head.index };
			def = new_head;
			return *reinterpret_cast<Value<U>*>(this); // TODO: not cool
		}

		T* operator->() noexcept {
			return eval<T*>(def);
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
			if (def.node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				return;
			}
			def.node->construct.args[1] = get_render_graph()->make_extract(src.get_def(), 0);
			def.node->construct.args[2] = get_render_graph()->make_extract(src.get_def(), 1);
			def.node->construct.args[3] = get_render_graph()->make_extract(src.get_def(), 2);
		}

		/// @brief Inference target has the same width & height as the source
		void same_2D_extent_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			if (def.node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				return;
			}
			def.node->construct.args[1] = get_render_graph()->make_extract(src.get_def(), 0);
			def.node->construct.args[2] = get_render_graph()->make_extract(src.get_def(), 1);
		}

		/// @brief Inference target has the same format as the source
		void same_format_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			if (def.node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				return;
			}
			def.node->construct.args[4] = get_render_graph()->make_extract(src.get_def(), 3);
		}

		/// @brief Inference target has the same shape(extent, layers, levels) as the source
		void same_shape_as(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			if (def.node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				return;
			}
			same_extent_as(src);

			for (auto i = 6; i < 10; i++) { /* 6 - 9 : layers, levels */
				def.node->construct.args[i] = get_render_graph()->make_extract(src.get_def(), i - 1);
			}
		}

		/// @brief Inference target is similar to(same shape, same format, same sample count) the source
		void similar_to(const Value<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			if (def.node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				return;
			}
			same_shape_as(src);
			same_format_as(src);
			def.node->construct.args[5] = get_render_graph()->make_extract(src.get_def(), 4);
		}

		// Buffer inferences

		void same_size(const Value<Buffer>& src)
		  requires std::is_same_v<T, Buffer>
		{
			def.node->construct.args[1] = get_render_graph()->make_extract(src.get_def(), 0);
		}

		Value<uint64_t> get_size()
		  requires std::is_same_v<T, Buffer>
		{
			Ref extract = get_render_graph()->make_extract(get_def(), 0);
			return { ExtRef{ std::make_shared<ExtNode>(get_render_graph(), extract.node), extract }, {} };
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
			Ref item = node->module->make_extract(get_head(), node->module->make_constant(index));
			return Value<std::remove_reference_t<decltype(std::declval<T>()[0])>>(
			    ExtRef(std::make_shared<ExtNode>(get_render_graph(), item.node), item), item_def, { node });
		}

		auto mip(uint32_t mip)
		  requires std::is_same_v<T, ImageAttachment>
		{
			can_peel = false;
			auto item_def = get_def();
			Ref item = node->module->make_slice(get_head(),
			                                    node->module->make_constant(mip),
			                                    node->module->make_constant(1u),
			                                    node->module->make_constant(0u),
			                                    node->module->make_constant(VK_REMAINING_ARRAY_LAYERS));
			return Value(ExtRef(std::make_shared<ExtNode>(get_render_graph(), item.node), item), item_def, { node });
		}

		auto layer(uint32_t layer)
		  requires std::is_same_v<T, ImageAttachment>
		{
			can_peel = false;
			auto item_def = get_def();
			Ref item = node->module->make_slice(get_head(),
			                                    node->module->make_constant(0u),
			                                    node->module->make_constant(VK_REMAINING_MIP_LEVELS),
			                                    node->module->make_constant(layer),
			                                    node->module->make_constant(1u));
			return Value(ExtRef(std::make_shared<ExtNode>(get_render_graph(), item.node), item), item_def, { node });
		}
	};

	inline Value<uint64_t> operator*(Value<uint64_t> a, uint64_t b) {
		Ref ref = a.get_render_graph()->make_math_binary_op(Node::BinOp::MUL, a.get_head(), a.get_render_graph()->make_constant(b));
		return std::move(a).transmute<uint64_t>(ref);
	}

	inline Result<void> wait_for_futures_explicit(Allocator& alloc, Compiler& compiler, std::span<UntypedValue> futures) {
		std::vector<SyncPoint> waits;
		for (uint64_t i = 0; i < futures.size(); i++) {
			auto& future = futures[i];
			auto res = future.submit(alloc, compiler, {});
			if (!res) {
				return res;
			}
			if (future.node->acqrel->status != Signal::Status::eSynchronizable) {
				continue;
			}
			waits.emplace_back(future.node->acqrel->source);
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