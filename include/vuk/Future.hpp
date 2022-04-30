#pragma once

#include "vuk/Allocator.hpp"
#include "vuk/Context.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/vuk_fwd.hpp"

#include <memory>
#include <span>

// futures
namespace vuk {
	struct QueueResourceUse {
		vuk::Access original;
		vuk::PipelineStageFlags stages;
		vuk::AccessFlags access;
		vuk::ImageLayout layout; // ignored for buffers
		vuk::DomainFlagBits domain;
	};

	struct FutureBase {
		FutureBase() = default;
		FutureBase(Allocator&);

		Allocator* allocator;

		enum class Status {
			eInitial,          // default-constructed future
			eRenderGraphBound, // a rendergraph was bound to this future
			eInputAttached,    // this future was attached to a rendergraph as input
			eOutputAttached,   // this future was attached to a rendergraph as output
			eSubmitted,        // the rendergraph referenced by this future was submitted (result is available on device with appropriate sync)
			eHostAvailable     // the result is available on host, available on device without sync
		} status = Status::eInitial;

		Allocator& get_allocator() {
			return *allocator;
		}

		DomainFlagBits initial_domain = DomainFlagBits::eNone; // the domain where we submitted this Future to
		QueueResourceUse last_use;                             // the results of the future are available if waited for on the initial_domain
		uint64_t initial_visibility;                           // the results of the future are available if waited for {initial_domain, initial_visibility}

		ImageAttachment result_image;
		Buffer result_buffer;

		template<class T>
		T& get_result();

	};

	template<>
	inline ImageAttachment& FutureBase::get_result() {
		return result_image;
	}

	template<>
	inline Buffer& FutureBase::get_result() {
		return result_buffer;
	}

	template<class T>
	class Future {
	public:
		Future() = default;
		/// @brief Create a Future with ownership of a RenderGraph and bind to an output
		/// @param allocator
		/// @param rg
		/// @param output_binding
		Future(Allocator& allocator, std::unique_ptr<struct RenderGraph> rg, Name output_binding, DomainFlags dst_domain = DomainFlagBits::eDevice);
		/// @brief Create a Future without ownership of a RenderGraph and bind to an output
		/// @param allocator
		/// @param rg
		/// @param output_binding
		Future(Allocator& allocator, struct RenderGraph& rg, Name output_binding, DomainFlags dst_domain = DomainFlagBits::eDevice);
		/// @brief Create a Future from a value, automatically making the result available on the host
		/// @param allocator
		/// @param value
		Future(Allocator& allocator, T&& value);

		/// @brief Get status of the Future
		FutureBase::Status& get_status() {
			return control->status;
		}

		/// @brief Get associated Allocator
		Allocator& get_allocator() {
			return *control->allocator;
		}
		
		/// @brief Get the referenced RenderGraph
		RenderGraph* get_render_graph() {
			return rg;
		}

		/// @brief Submit Future for execution
		Result<void> submit();
		/// @brief Wait and retrieve the result of the Future on the host
		Result<T> get();
		/// @brief Get control block for Future
		FutureBase* get_control() {
			return control.get();
		}
	private:
		Name output_binding;

		std::unique_ptr<RenderGraph> owned_rg;
		RenderGraph* rg = nullptr;

		std::unique_ptr<FutureBase> control;

		friend struct RenderGraph;
	};

	template<class... Args>
	Result<void> wait_for_futures(Allocator& alloc, Args&... futs) {
		std::array controls = { futs.get_control()... };
		std::array rgs = { futs.get_render_graph()... };
		std::vector<std::pair<Allocator*, RenderGraph*>> rgs_to_run;
		for (uint64_t i = 0; i < controls.size(); i++) {
			auto& control = controls[i];
			if (control->status == FutureBase::Status::eInputAttached || control->status == FutureBase::Status::eInitial) {
				return { expected_error, RenderGraphException{} };
			} else if (control->status == FutureBase::Status::eHostAvailable || control->status == FutureBase::Status::eSubmitted) {
				continue;
			} else {
				rgs_to_run.emplace_back(&control->get_allocator(), rgs[i]);
			}
		}
		if (rgs_to_run.size() != 0) {
			VUK_DO_OR_RETURN(link_execute_submit(alloc, std::span(rgs_to_run)));
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

	inline Result<void> wait_for_futures_explicit(Allocator& alloc, std::span<FutureBase*> controls, std::span<RenderGraph*> render_graphs) {
		std::vector<std::pair<Allocator*, RenderGraph*>> rgs_to_run;
		for (uint64_t i = 0; i < controls.size(); i++) {
			auto& control = controls[i];
			if (control->status == FutureBase::Status::eInputAttached || control->status == FutureBase::Status::eInitial) {
				return { expected_error, RenderGraphException{} };
			} else if (control->status == FutureBase::Status::eHostAvailable || control->status == FutureBase::Status::eSubmitted) {
				continue;
			} else {
				rgs_to_run.emplace_back(&control->get_allocator(), render_graphs[i]);
			}
		}
		if (rgs_to_run.size() != 0) {
			VUK_DO_OR_RETURN(link_execute_submit(alloc, std::span(rgs_to_run)));
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
} // namespace vuk