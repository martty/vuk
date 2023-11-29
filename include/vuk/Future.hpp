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
		TypedFuture(std::shared_ptr<RG> rg, Ref ref, T* value) {
			this->control = std::make_shared<FutureBase>();
			this->head = { rg->make_release(ref, &this->control->acqrel), 0 };
			this->control->rg = std::move(rg);
			this->value = value;
		}

		TypedFuture(const TypedFuture& o) noexcept :
		    control{ std::make_shared<FutureBase>(*o.control) },
		    value{ o.value },
		    head{ control->rg->make_release(o.get_head(), &this->control->acqrel), 0 } {}

		TypedFuture(TypedFuture&& o) noexcept :
		    control{ std::exchange(o.control, nullptr) },
		    value{ std::exchange(o.value, nullptr) },
		    head{ std::exchange(o.head, {}) } {}

		TypedFuture& operator=(const TypedFuture& o) noexcept {
			control = { std::make_shared<FutureBase>(*o.control) };
			value = { o.value };
			head = { control->rg->make_release(o.get_head(), &this->control->acqrel), 0 };

			return *this;
		}

		TypedFuture& operator=(TypedFuture&& o) noexcept {
			std::swap(o.control, control);
			std::swap(o.value, value);
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
		std::shared_ptr<RG>& get_render_graph() noexcept {
			return control->rg;
		}

		/// @brief Name the value currently referenced by this Future
		void set_name(std::string_view name) noexcept {
			get_render_graph()->name_output(head, std::string(name));
		}

		Ref get_head() const noexcept {
			return head.node->release.src;
		}

		TypedFuture transmute(Ref ref) noexcept {
			head.node->release.src = ref;
			return *this;
		}

		T* operator->() noexcept {
			return value;
		}

		/// @brief Wait and retrieve the result of the Future on the host
		[[nodiscard]] Result<T> get(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {}) {
			if (auto result = control->wait(allocator, compiler, options)) {
				return { expected_value, *value };
			} else {
				return result;
			}
		}

		// TODO: remove this from public API
		std::shared_ptr<FutureBase> control;
		T* value;

	private:
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