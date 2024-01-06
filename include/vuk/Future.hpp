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
	class FutureControlBlock {
	public:
		/// @brief Submit Future for execution
		Result<void> submit(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {});
		/// @brief Submit and wait for Future to complete execution on host
		Result<void> wait(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {});

		/// @brief If the Future has been submitted for execution, polls for status.
		[[nodiscard]] Result<Signal::Status> poll();

		AcquireRelease acqrel;

		std::shared_ptr<RG> rg;

		/// @brief Get the referenced RenderGraph
		std::shared_ptr<RG>& get_render_graph() {
			return rg;
		}
	};

	template<class T>
	class Future;

	class UntypedFuture {
	public:
		UntypedFuture(std::shared_ptr<RG> rg, Ref ref, Ref def, std::vector<std::shared_ptr<FutureControlBlock>> dependent_blocks) {
			this->control = std::make_shared<FutureControlBlock>();

			this->head = { rg->make_release(ref, &this->control->acqrel, Access::eNone, DomainFlagBits::eAny), 0 };

			this->control->rg = std::move(rg);
			this->def = def;
			this->dependent_blocks = std::move(dependent_blocks);
		}

		UntypedFuture(const UntypedFuture& o) noexcept : control{ std::make_shared<FutureControlBlock>(*o.control) }, def{ o.def } {
			head = { control->rg->make_release(o.get_head(), &this->control->acqrel, Access::eNone, DomainFlagBits::eAny), 0 };
			dependent_blocks = o.dependent_blocks;
		}

		UntypedFuture(UntypedFuture&& o) noexcept :
		    control{ std::exchange(o.control, nullptr) },
		    dependent_blocks{ std::exchange(o.dependent_blocks, {}) },
		    def{ std::exchange(o.def, {}) },
		    head{ std::exchange(o.head, {}) } {}

		UntypedFuture& operator=(const UntypedFuture& o) noexcept {
			control = { std::make_shared<FutureControlBlock>(*o.control) };
			def = { o.def };

			head = { control->rg->make_release(o.get_head(), &this->control->acqrel, Access::eNone, DomainFlagBits::eAny), 0 };
			dependent_blocks = o.dependent_blocks;

			return *this;
		}

		UntypedFuture& operator=(UntypedFuture&& o) noexcept {
			std::swap(o.control, control);
			std::swap(o.dependent_blocks, dependent_blocks);
			std::swap(o.def, def);
			std::swap(o.head, head);

			return *this;
		}

		~UntypedFuture() {
			abandon();
		}

		/// @brief Get the referenced RenderGraph
		const std::shared_ptr<RG>& get_render_graph() const noexcept {
			return control->rg;
		}

		/// @brief Name the value currently referenced by this Future
		void set_name(std::string_view name) noexcept {
			get_render_graph()->name_output(head, std::string(name));
		}

		Ref get_head() const noexcept {
			return head.node->release.src;
		}

		Ref get_def() const noexcept {
			return def;
		}

		Result<void> wait(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {}) {
			auto result = control->wait(allocator, compiler, options);
			if (result.holds_value()) {
				// save value
				auto current_value = get_constant_value(def.node);
				auto current_ty = def.type();
				// new RG with ACQUIRE node
				auto new_rg = std::make_shared<RG>();
				this->def = { new_rg->make_acquire(current_ty, &this->control->acqrel, current_value) };
				// drop current RG
				this->control->rg = std::move(new_rg);
				this->head = { this->control->rg->make_release(this->def, &this->control->acqrel, Access::eNone, DomainFlagBits::eAny), 0 };
			}
			return result;
		}

		template<class U>
		Future<U> transmute(Ref ref) noexcept {
			head.node->release.src = ref;
			return *reinterpret_cast<Future<U>*>(this); // TODO: not cool
		}

		void abandon() {
			if (head.node) {
				assert(head.node->kind == Node::RELEASE || head.node->kind == Node::NOP);
				head.node->kind = Node::NOP;
			}
		}

		// TODO: remove this from public API
		std::shared_ptr<FutureControlBlock> control;
		std::vector<std::shared_ptr<FutureControlBlock>> dependent_blocks;

	protected:
		Ref def;
		Ref head;
	};

	template<class T>
	class Future : public UntypedFuture {
	public:
		using UntypedFuture::UntypedFuture;

		template<class U = T>
		Future<U> transmute(Ref new_head) noexcept {
			head.node->release.src = new_head;
			return *reinterpret_cast<Future<U>*>(this); // TODO: not cool
		}

		template<class U = T>
		Future<U> transmute(Ref new_head, Ref new_def) noexcept {
			head.node->release.src = new_head;
			def = new_def;
			return *reinterpret_cast<Future<U>*>(this); // TODO: not cool
		}

		template<class U = T>
		Future<U> release_to(Access access, DomainFlagBits domain) noexcept {
			assert(head.node->kind == Node::RELEASE);
			head.node->release.dst_access = access;
			head.node->release.dst_domain = domain;
			return std::move(*reinterpret_cast<Future<U>*>(this)); // TODO: not cool
		}

		T* operator->() noexcept {
			return reinterpret_cast<T*>(get_constant_value(def.node));
		}

		/// @brief Wait and retrieve the result of the Future on the host
		[[nodiscard]] Result<T> get(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {})
		  requires(!std::is_array_v<T>)
		{
			if (auto result = control->wait(allocator, compiler, options)) {
				return { expected_value, *operator->() };
			} else {
				return result;
			}
		}

		// Image inferences
		void same_extent_as(const Future<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			if (src.get_def().node->kind == Node::VALLOC) {
				def.node->valloc.args[1] = src.get_def().node->valloc.args[1];
				def.node->valloc.args[2] = src.get_def().node->valloc.args[2];
				def.node->valloc.args[3] = src.get_def().node->valloc.args[3];
			} else if (src.get_def().node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				Swapchain& swp = *reinterpret_cast<Swapchain*>(src.get_def().node->acquire_next_image.swapchain.node->valloc.args[0].node->constant.value);
				def.node->valloc.args[1] = get_render_graph()->make_constant<uint32_t>(swp.images[0].extent.extent.width);
				def.node->valloc.args[2] = get_render_graph()->make_constant<uint32_t>(swp.images[0].extent.extent.height);
				def.node->valloc.args[3] = get_render_graph()->make_constant<uint32_t>(swp.images[0].extent.extent.depth);
			}
		}

		/// @brief Inference target has the same width & height as the source
		void same_2D_extent_as(const Future<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			if (src.get_def().type()->is_image()) {
				def.node->valloc.args[1] = src.get_def().node->valloc.args[1];
				def.node->valloc.args[2] = src.get_def().node->valloc.args[2];
			} else if (src.get_def().type()->kind == Type::SWAPCHAIN_TY) {
				Swapchain& swp = *reinterpret_cast<Swapchain*>(src.get_def().node->acquire_next_image.swapchain.node->valloc.args[0].node->constant.value);
				def.node->valloc.args[1] = get_render_graph()->make_constant<uint32_t>(swp.images[0].extent.extent.width);
				def.node->valloc.args[2] = get_render_graph()->make_constant<uint32_t>(swp.images[0].extent.extent.height);
			}
		}

		/// @brief Inference target has the same format as the source
		void same_format_as(const Future<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			if (src.get_def().type()->is_image()) {
				def.node->valloc.args[4] = src.get_def().node->valloc.args[4];
			} else if (src.get_def().type()->kind == Type::SWAPCHAIN_TY) {
				Swapchain& swp = *reinterpret_cast<Swapchain*>(src.get_def().node->acquire_next_image.swapchain.node->valloc.args[0].node->constant.value);
				def.node->valloc.args[4] = get_render_graph()->make_constant(swp.images[0].format);
			}
		}

		/// @brief Inference target has the same shape(extent, layers, levels) as the source
		void same_shape_as(const Future<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			same_extent_as(src);
			if (src.get_def().type()->is_image()) {
				for (auto i = 6; i < 10; i++) { /* 6 - 9 : layers, levels */
					def.node->valloc.args[i] = src.get_def().node->valloc.args[i];
				}
			} else if (src.get_def().type()->kind == Type::SWAPCHAIN_TY) {
				Swapchain& swp = *reinterpret_cast<Swapchain*>(src.get_def().node->acquire_next_image.swapchain.node->valloc.args[0].node->constant.value);
				def.node->valloc.args[6] = get_render_graph()->make_constant(swp.images[0].base_layer);
				def.node->valloc.args[7] = get_render_graph()->make_constant(swp.images[0].layer_count);
				def.node->valloc.args[8] = get_render_graph()->make_constant(swp.images[0].base_level);
				def.node->valloc.args[9] = get_render_graph()->make_constant(swp.images[0].level_count);
			}
		}

		/// @brief Inference target is similar to(same shape, same format, same sample count) the source
		void similar_to(const Future<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			same_shape_as(src);
			same_format_as(src);
			if (src.get_def().type()->is_image()) {
				def.node->valloc.args[5] = src.get_def().node->valloc.args[5]; // sample count
			} else if (src.get_def().type()->kind == Type::SWAPCHAIN_TY) {
				def.node->valloc.args[5] = get_render_graph()->make_constant(Samples::e1); // swapchain is always single-sample
			}
		}

		// Buffer inferences

		void same_size(const Future<Buffer>& src)
		  requires std::is_same_v<T, Buffer>
		{
			assert(src.get_def().type()->is_buffer());
			def.node->valloc.args[1] = src.get_def().node->valloc.args[1];
		}

		Future<uint64_t> get_size()
		  requires std::is_same_v<T, Buffer>
		{
			return { get_render_graph(), def.node->valloc.args[1], {}, {control} };
		}

		void set_size(Future<uint64_t> arg)
		  requires std::is_same_v<T, Buffer>
		{
			get_render_graph()->subgraphs.push_back(arg.get_render_graph());
			def.node->valloc.args[1] = arg.get_head();
		}

		auto operator[](size_t index)
		  requires std::is_array_v<T>
		{
			auto item_def = def.node->aalloc.defs[index];
			Ref item = control->rg->make_array_indexing(def.type()->array.T, get_head(), control->rg->make_constant(index));
			assert(def.node->kind == Node::AALLOC);
			assert(def.type()->kind == Type::ARRAY_TY);
			return Future<std::remove_reference_t<decltype(std::declval<T>()[0])>>(get_render_graph(), item, item_def, {control});
		}
	};

	inline Future<uint64_t> operator*(Future<uint64_t> a, uint64_t b) {
		Ref ref = a.get_render_graph()->make_math_binary_op(Node::BinOp::MUL, a.get_head(), a.get_render_graph()->make_constant(b));
		return std::move(std::move(a).transmute<uint64_t>(ref));
	}

	inline Result<void> wait_for_futures_explicit(Allocator& alloc, Compiler& compiler, std::span<std::shared_ptr<FutureControlBlock>> futures) {
		std::vector<std::shared_ptr<RG>> rgs_to_run;
		for (uint64_t i = 0; i < futures.size(); i++) {
			auto& future = futures[i];
			if (future->acqrel.status == Signal::Status::eDisarmed && !futures[i]->get_render_graph()) {
				return { expected_error, RenderGraphException{} };
			} else if (future->acqrel.status == Signal::Status::eHostAvailable || future->acqrel.status == Signal::Status::eSynchronizable) {
				continue;
			} else {
				rgs_to_run.emplace_back(futures[i]->get_render_graph());
			}
		}
		if (rgs_to_run.size() != 0) {
			VUK_DO_OR_RETURN(link_execute_submit(alloc, compiler, std::span(rgs_to_run)));
		}

		std::vector<SyncPoint> waits;
		for (uint64_t i = 0; i < futures.size(); i++) {
			auto& future = futures[i];
			if (future->acqrel.status != Signal::Status::eSynchronizable) {
				continue;
			}
			waits.emplace_back(future->acqrel.source);
		}
		if (waits.size() > 0) {
			alloc.get_context().wait_for_domains(std::span(waits));
		}

		return { expected_value };
	}

	inline Result<void> wait_for_futures_explicit(Allocator& alloc, Compiler& compiler, std::span<UntypedFuture> futures) {
		std::vector<std::shared_ptr<FutureControlBlock>> cbs;
		for (auto& f : futures) {
			cbs.push_back(f.control);
		}
		return wait_for_futures_explicit(alloc, compiler, cbs);
	}

	template<class... Args>
	Result<void> wait_for_futures(Allocator& alloc, Compiler& compiler, Args&&... futs) {
		auto cbs = std::array{ futs.control... };
		return wait_for_futures_explicit(alloc, compiler, cbs);
	}
} // namespace vuk