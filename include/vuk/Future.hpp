#pragma once

#include "vuk/Allocator.hpp"
#include "vuk/Context.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/Types.hpp"
#include "vuk/vuk_fwd.hpp"

#include <memory>
#include <span>
#include <variant>

// futures
namespace vuk {
	struct FutureBase {
		FutureBase() = default;
		FutureBase(Allocator&);

		enum class Status {
			eInitial,          // default-constructed future
			eSubmitted,        // the rendergraph referenced by this future was submitted (result is available on device with appropriate sync)
			eHostAvailable     // the result is available on host, available on device without sync
		} status = Status::eInitial;

		DomainFlagBits initial_domain = DomainFlagBits::eNone; // the domain where we submitted this Future to
		QueueResourceUse last_use;                             // the results of the future are available if waited for on the initial_domain
		uint64_t initial_visibility;                           // the results of the future are available if waited for {initial_domain, initial_visibility}

		std::variant<ImageAttachment, Buffer> result;

		template<class T>
		T& get_result() {
			return std::get<T>(result);
		}
	};

	class Future {
	public:
		Future() = default;

		/// @brief Create a Future with ownership of a RenderGraph and bind to an output
		/// @param rg
		/// @param output_binding
		Future(std::shared_ptr<RenderGraph> rg, Name output_binding, DomainFlags dst_domain = DomainFlagBits::eDevice);

		/// @brief Create a Future with ownership of a RenderGraph and bind to an output
		/// @param rg
		/// @param output_binding
		Future(std::shared_ptr<RenderGraph> rg, QualifiedName output_binding, DomainFlags dst_domain = DomainFlagBits::eDevice);
		
		/// @brief Create a Future from a value, automatically making the result available on the host
		/// @param value
		Future(ImageAttachment value) : control(std::make_shared<FutureBase>()) {
			control->result = std::move(value);
			control->status = FutureBase::Status::eHostAvailable;
			control->last_use.layout = ImageLayout::eUndefined;
		}

		/// @brief Create a Future from a value, automatically making the result available on the host
		/// @param value
		Future(Buffer value) : control(std::make_shared<FutureBase>()) {
			control->result = std::move(value);
			control->status = FutureBase::Status::eHostAvailable;
			control->last_use.layout = ImageLayout::eUndefined;
		}

		Future(const Future&) noexcept;
		Future& operator=(const Future&) noexcept;
		Future(Future&&) noexcept;
		Future& operator=(Future&&) noexcept;

		~Future();

		explicit operator bool() const {
			return rg.get() != nullptr;
		}

		/// @brief Get status of the Future
		FutureBase::Status& get_status() {
			return control->status;
		}

		/// @brief Get the referenced RenderGraph
		std::shared_ptr<RenderGraph> get_render_graph() {
			return rg;
		}

		QualifiedName get_bound_name() {
			return output_binding;
		}

		/// @brief Submit Future for execution
		Result<void> submit(Allocator& allocator, Compiler& compiler);
		/// @brief Wait for Future to complete execution on host
		Result<void> wait(Allocator& allocator, Compiler& compiler);
		/// @brief Wait and retrieve the result of the Future on the host
		template<class T>
		[[nodiscard]] Result<T> get(Allocator& allocator, Compiler& compiler);

		/// @brief Get control block for Future
		FutureBase* get_control() {
			return control.get();
		}

		bool is_image() const {
			return control->result.index() == 0;
		}

		bool is_buffer() const {
			return control->result.index() == 1;
		}

		template<class T>
		T& get_result() {
			return control->get_result<T>();
		}

	private:
		QualifiedName output_binding;

		std::shared_ptr<RenderGraph> rg;

		std::shared_ptr<FutureBase> control;

		friend struct RenderGraph;
	};

	template<class... Args>
	Result<void> wait_for_futures(Allocator& alloc, Compiler& compiler, Args&&... futs) {
		std::array controls = { futs.get_control()... };
		std::array rgs = { futs.get_render_graph()... };
		std::vector<std::shared_ptr<RenderGraph>> rgs_to_run;
		for (uint64_t i = 0; i < controls.size(); i++) {
			auto& control = controls[i];
			if (control->status == FutureBase::Status::eInitial && !rgs[i]) {
				return { expected_error, RenderGraphException{} };
			} else if (control->status == FutureBase::Status::eHostAvailable || control->status == FutureBase::Status::eSubmitted) {
				continue;
			} else {
				rgs_to_run.emplace_back(rgs[i]);
			}
		}
		if (rgs_to_run.size() != 0) {
			VUK_DO_OR_RETURN(link_execute_submit(alloc, compiler, std::span(rgs_to_run)));
		}

		std::vector<std::pair<DomainFlags, uint64_t>> waits;
		for (uint64_t i = 0; i < controls.size(); i++) {
			auto& control = controls[i];
			if (control->status != FutureBase::Status::eSubmitted) {
				continue;
			}
			waits.emplace_back(control->initial_domain, control->initial_visibility);
		}
		if (waits.size() > 0) {
			alloc.get_context().wait_for_domains(std::span(waits));
		}

		return { expected_value };
	}

	inline Result<void> wait_for_futures_explicit(Allocator& alloc, Compiler& compiler, std::span<Future> futures) {
		std::vector<std::shared_ptr<RenderGraph>> rgs_to_run;
		for (uint64_t i = 0; i < futures.size(); i++) {
			auto control = futures[i].get_control();
			if (control->status == FutureBase::Status::eInitial && !futures[i].get_render_graph()) {
				return { expected_error, RenderGraphException{} };
			} else if (control->status == FutureBase::Status::eHostAvailable || control->status == FutureBase::Status::eSubmitted) {
				continue;
			} else {
				rgs_to_run.emplace_back(futures[i].get_render_graph());
			}
		}
		if (rgs_to_run.size() != 0) {
			VUK_DO_OR_RETURN(link_execute_submit(alloc, compiler, std::span(rgs_to_run)));
		}

		std::vector<std::pair<DomainFlags, uint64_t>> waits;
		for (uint64_t i = 0; i < futures.size(); i++) {
			auto control = futures[i].get_control();
			if (control->status != FutureBase::Status::eSubmitted) {
				continue;
			}
			waits.emplace_back(control->initial_domain, control->initial_visibility);
		}
		if (waits.size() > 0) {
			alloc.get_context().wait_for_domains(std::span(waits));
		}

		return { expected_value };
	}
} // namespace vuk