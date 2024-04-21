#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/runtime/vk/VkSwapchain.hpp"
#include "vuk/Value.hpp"

namespace vuk {
	// assume rgs are independent - they don't reference eachother
	Result<void> execute_submit(Allocator& allocator, std::span<std::pair<Allocator*, ExecutableRenderGraph*>> rgs) {
		for (auto& [alloc, rg] : rgs) {
			rg->execute(*alloc);
		}

		return { expected_value };
	}

	Result<void> execute_submit_and_wait(Allocator& allocator, ExecutableRenderGraph&& rg) {
		Runtime& ctx = allocator.get_context();
		std::pair v = { &allocator, &rg };
		VUK_DO_OR_RETURN(execute_submit(allocator, std::span{ &v, 1 }));
		ctx.wait_idle(); // TODO:
		return { expected_value };
	}

	Result<void> UntypedValue::wait(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options) {
		auto res = submit(allocator, compiler, options);
		if (!res) {
			return res;
		}
		assert(node->acqrel->status != Signal::Status::eDisarmed);
		if (node->acqrel->status == Signal::Status::eSynchronizable) {
			allocator.get_context().wait_for_domains(std::span{ &node->acqrel->source, 1 });
		}

		return { expected_value };
	}

	Result<void> UntypedValue::submit(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options) {
		if (node->acqrel->status == Signal::Status::eDisarmed && node->get_node()->kind == Node::SPLICE) { // splice -> release if unsubmitted
			release();
		}

		auto& acqrel = node->acqrel;
		if (acqrel->status == Signal::Status::eDisarmed && !node->module) {
			return { expected_error, RenderGraphException{ "Tried to submit without a module" } };
		} else if (acqrel->status == Signal::Status::eHostAvailable || acqrel->status == Signal::Status::eSynchronizable) {
			return { expected_value }; // nothing to do
		} else {
			auto erg = compiler.link(std::span{ &node, 1 }, options);
			if (!erg) {
				return erg;
			}
			std::pair v = { &allocator, &*erg };
			VUK_DO_OR_RETURN(execute_submit(allocator, std::span{ &v, 1 }));
			assert(acqrel->status != Signal::Status::eDisarmed);
			return { expected_value };
		}
	}
} // namespace vuk
