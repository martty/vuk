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

		template<>
		ImageAttachment& get_result() {
			return result_image;
		}

		template<>
		Buffer& get_result() {
			return result_buffer;
		}
	};

	template<class T>
	struct Future {
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
		/// @brief Create a Future from a value, automatically making it host available
		/// @param allocator
		/// @param value
		Future(Allocator& allocator, T&& value);

		Name output_binding;

		std::unique_ptr<RenderGraph> owned_rg;
		RenderGraph* rg = nullptr;

		std::unique_ptr<FutureBase> control;

		FutureBase::Status& get_status() {
			return control->status;
		}

		Allocator& get_allocator() {
			return *control->allocator;
		}

		Result<void> submit(); // turn cmdbufs into possibly a TS
		Result<T> get();       // wait on host for T to be produced by the computation
	};

	template<class... Args>
	Result<void> wait_for_futures(Allocator& alloc, Args&... futs) {
		std::array controls = { futs.control.get()... };
		std::array rgs = { futs.rg... };
		std::vector<std::pair<Allocator*, RenderGraph*>> rgs_to_run;
		for (uint64_t i = 0; i < controls.size(); i++) {
			auto& control = controls[i];
			if (control->status == FutureBase::Status::eInputAttached || control->status == FutureBase::Status::eInitial) {
				return { expected_error };
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
} // namespace vuk