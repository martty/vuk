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
	class FutureBase {
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
	class TypedFuture {
	public:
		TypedFuture(std::shared_ptr<RG> rg, Ref ref, Ref def) {
			this->control = std::make_shared<FutureBase>();
			this->head = { rg->make_release(ref, &this->control->acqrel), 0 };
			this->control->rg = std::move(rg);
			this->def = def;
		}

		TypedFuture(const TypedFuture& o) noexcept :
		    control{ std::make_shared<FutureBase>(*o.control) },
		    def{ o.def },
		    head{ control->rg->make_release(o.get_head(), &this->control->acqrel), 0 } {}

		TypedFuture(TypedFuture&& o) noexcept : control{ std::exchange(o.control, nullptr) }, def{ std::exchange(o.def, {}) }, head{ std::exchange(o.head, {}) } {}

		TypedFuture& operator=(const TypedFuture& o) noexcept {
			control = { std::make_shared<FutureBase>(*o.control) };
			def = { o.def };
			head = { control->rg->make_release(o.get_head(), &this->control->acqrel), 0 };

			return *this;
		}

		TypedFuture& operator=(TypedFuture&& o) noexcept {
			std::swap(o.control, control);
			std::swap(o.def, def);
			std::swap(o.head, head);

			return *this;
		}

		// TODO: add back copy/move
		~TypedFuture() {
			if (head.node) {
				assert(head.node->kind == Node::RELEASE);
				head.node->kind = Node::NOP;
			}
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

		template<class U = T>
		TypedFuture<U> transmute(Ref ref) noexcept {
			head.node->release.src = ref;
			return *reinterpret_cast<TypedFuture<U>*>(this); // TODO: not cool
		}

		T* operator->() noexcept {
			return reinterpret_cast<T*>(def.node->valloc.args[0].node->constant.value);
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

		Result<void> wait(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {}) {
			return control->wait(allocator, compiler, options);
		}

		void same_size(const TypedFuture<Buffer>& src)
		  requires std::is_same_v<T, Buffer>
		{
			assert(src.get_def().type()->is_buffer());
			def.node->valloc.args[1] = src.get_def().node->valloc.args[1];
		}

		auto operator[](size_t index)
		  requires std::is_array_v<T>
		{
			auto item_def = def.node->aalloc.defs[index];
			Ref item = control->rg->make_array_indexing(def.type()->array.T, get_head(), control->rg->make_constant(index));
			assert(def.node->kind == Node::AALLOC);
			assert(def.type()->kind == Type::ARRAY_TY);
			return TypedFuture<std::remove_reference_t<decltype(std::declval<T>()[0])>>(get_render_graph(), item, item_def);
		}

		// TODO: remove this from public API
		std::shared_ptr<FutureBase> control;

	private:
		Ref def;
		Ref head;
	};

	inline Result<void> wait_for_futures_explicit(Allocator& alloc, Compiler& compiler, std::span<FutureBase> futures) {
		std::vector<std::shared_ptr<RG>> rgs_to_run;
		for (uint64_t i = 0; i < futures.size(); i++) {
			auto& future = futures[i];
			if (future.acqrel.status == Signal::Status::eDisarmed && !futures[i].get_render_graph()) {
				return { expected_error, RenderGraphException{} };
			} else if (future.acqrel.status == Signal::Status::eHostAvailable || future.acqrel.status == Signal::Status::eSynchronizable) {
				continue;
			} else {
				rgs_to_run.emplace_back(futures[i].get_render_graph());
			}
		}
		if (rgs_to_run.size() != 0) {
			VUK_DO_OR_RETURN(link_execute_submit(alloc, compiler, std::span(rgs_to_run)));
		}

		std::vector<SyncPoint> waits;
		for (uint64_t i = 0; i < futures.size(); i++) {
			auto& future = futures[i];
			if (future.acqrel.status != Signal::Status::eSynchronizable) {
				continue;
			}
			waits.emplace_back(future.acqrel.source);
		}
		if (waits.size() > 0) {
			alloc.get_context().wait_for_domains(std::span(waits));
		}

		return { expected_value };
	}

	template<class... Args>
	Result<void> wait_for_futures(Allocator& alloc, Compiler& compiler, Args&&... futs) {
		return wait_for_futures_explicit(alloc, compiler, std::array{ futs... });
	}
} // namespace vuk