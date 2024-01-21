#include "vuk/Util.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Context.hpp"
#include "vuk/Future.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/SampledImage.hpp"
#include "vuk/Swapchain.hpp"

#include <atomic>
#ifndef DOCTEST_CONFIG_DISABLE
#include <doctest/doctest.h>
#endif
#include <mutex>
#include <sstream>
#include <utility>

namespace vuk {
	Swapchain::Swapchain(Allocator alloc, size_t image_count) :
	    allocator(alloc){
		semaphores.resize(image_count * 2);
		allocator.allocate_semaphores(std::span(semaphores));
	}

	Result<void> Context::wait_for_domains(std::span<SyncPoint> queue_waits) {
		std::array<uint32_t, 3> domain_to_sema_index = { ~0u, ~0u, ~0u };
		std::array<VkSemaphore, 3> queue_timeline_semaphores;
		std::array<uint64_t, 3> values = {};

		uint32_t count = 0;
		for (auto& [executor, v] : queue_waits) {
			assert(executor->type == Executor::Type::eVulkanDeviceQueue);
			auto vkq = static_cast<rtvk::QueueExecutor*>(executor);
			auto idx = vkq->get_queue_family_index();
			auto& mapping = domain_to_sema_index[idx];
			if (mapping == -1) {
				mapping = count++;
			}
			queue_timeline_semaphores[mapping] = vkq->get_semaphore();
			values[mapping] = values[mapping] > v ? values[mapping] : v;
		}

		VkSemaphoreWaitInfo swi{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
		swi.pSemaphores = queue_timeline_semaphores.data();
		swi.pValues = values.data();
		swi.semaphoreCount = count;
		VkResult result = this->vkWaitSemaphores(device, &swi, UINT64_MAX);
		if (result != VK_SUCCESS) {
			return { expected_error, VkException{ result } };
		}
		return { expected_value };
	}

	Result<void> link_execute_submit(Allocator& allocator, Compiler& compiler, std::span<std::shared_ptr<RG>> rgs, RenderGraphCompileOptions options) {
		/* auto erg = compiler.link(rgs, options);
		if (!erg) {
			return erg;
		}
		std::pair erg_and_alloc = std::pair{ &allocator, &*erg };
		return execute_submit(allocator, std::span(&erg_and_alloc, 1));*/
		assert(false);
		return { expected_value };
	}

	std::string_view to_name(vuk::DomainFlagBits d) {
		switch (d) {
		case DomainFlagBits::eTransferQueue:
			return "Transfer";
		case DomainFlagBits::eGraphicsQueue:
			return "Graphics";
		case DomainFlagBits::eComputeQueue:
			return "Compute";
		default:
			return "Unknown";
		}
	}

	// assume rgs are independent - they don't reference eachother
	Result<void> execute_submit(Allocator& allocator,
	                            std::span<std::pair<Allocator*, ExecutableRenderGraph*>> rgs) {
		for (auto& [alloc, rg] : rgs) {
			rg->execute(*alloc);
		}

		return { expected_value };
	}

	Result<void> execute_submit_and_wait(Allocator& allocator, ExecutableRenderGraph&& rg) {
		Context& ctx = allocator.get_context();
		std::pair v = { &allocator, &rg };
		VUK_DO_OR_RETURN(execute_submit(allocator, std::span{ &v, 1 }));
		ctx.wait_idle(); // TODO:
		return { expected_value };
	}

	SampledImage make_sampled_image(ImageView iv, SamplerCreateInfo sci) {
		return { SampledImage::Global{ iv, sci, ImageLayout::eReadOnlyOptimalKHR } };
	}

	SampledImage make_sampled_image(NameReference n, SamplerCreateInfo sci) {
		return { SampledImage::RenderGraphAttachment{ n, sci, {}, ImageLayout::eReadOnlyOptimalKHR } };
	}

	SampledImage make_sampled_image(NameReference n, ImageViewCreateInfo ivci, SamplerCreateInfo sci) {
		return { SampledImage::RenderGraphAttachment{ n, sci, ivci, ImageLayout::eReadOnlyOptimalKHR } };
	}

	Result<void> UntypedFuture::wait(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options) {
		/* if (acqrel.status == Signal::Status::eDisarmed && !rg) {
			return { expected_error,
				       RenderGraphException{} }; // can't get wait for future that has not been attached anything or has been attached into a rendergraph
		} else if (acqrel.status == Signal::Status::eHostAvailable) {
			return { expected_value };
		} else if (acqrel.status == Signal::Status::eSynchronizable) {
			allocator.get_context().wait_for_domains(std::span{ &acqrel.source, 1 });
			return { expected_value };
		} else {
			auto erg = compiler.link(std::span{ &rg, 1 }, options);
			if (!erg) {
				return erg;
			}
			std::pair v = { &allocator, &*erg };
			VUK_DO_OR_RETURN(execute_submit(allocator, std::span{ &v, 1 }));
			assert(acqrel.status != Signal::Status::eDisarmed);
			if (acqrel.status == Signal::Status::eSynchronizable) {
				allocator.get_context().wait_for_domains(std::span{ &acqrel.source, 1 });
			}
			acqrel.status = Signal::Status::eHostAvailable;
			return { expected_value };
		}

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
		}*/

		return { expected_value };
	}

	Result<void> UntypedFuture::submit(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options) {
		auto& acqrel = head->acqrel;
		if (acqrel->status == Signal::Status::eDisarmed && !head->module) {
			return { expected_error, RenderGraphException{} };
		} else if (acqrel->status == Signal::Status::eHostAvailable || acqrel->status == Signal::Status::eSynchronizable) {
			return { expected_value }; // nothing to do
		} else {
			acqrel->status = Signal::Status::eSynchronizable;
			auto erg = compiler.link(std::span{ &head, 1 }, options);
			if (!erg) {
				return erg;
			}
			std::pair v = { &allocator, &*erg };
			VUK_DO_OR_RETURN(execute_submit(allocator, std::span{ &v, 1 }));
			return { expected_value };
		}
	}

	std::string_view image_view_type_to_sv(ImageViewType view_type) noexcept {
		switch (view_type) {
		case ImageViewType::e1D:
			return "1D";
		case ImageViewType::e2D:
			return "2D";
		case ImageViewType::e3D:
			return "3D";
		case ImageViewType::eCube:
			return "Cube";
		case ImageViewType::e1DArray:
			return "1DArray";
		case ImageViewType::e2DArray:
			return "2DArray";
		case ImageViewType::eCubeArray:
			return "CubeArray";
		default:
			assert(0 && "not reached.");
			return "";
		}
	}
} // namespace vuk
