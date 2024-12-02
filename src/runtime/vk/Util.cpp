#include "vuk/RenderGraph.hpp"
#include "vuk/Value.hpp"
#include "vuk/runtime/Cache.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"
#include "vuk/runtime/vk/VkSwapchain.hpp"

namespace vuk {
	// assume rgs are independent - they don't reference eachother
	Result<void> execute_submit(Allocator& allocator, std::span<std::pair<Allocator*, ExecutableRenderGraph*>> rgs) {
		for (auto& [alloc, rg] : rgs) {
			auto res = rg->execute(*alloc);
			if (!res) {
				return res;
			}
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

	Result<void> submit(Allocator& allocator, Compiler& compiler, std::span<UntypedValue> values, RenderGraphCompileOptions options) {
		std::vector<std::shared_ptr<ExtNode>> extnodes;
		for (auto& value : values) {
			auto& node = value.node;
			if (node->acqrel->status == Signal::Status::eHostAvailable || node->acqrel->status == Signal::Status::eSynchronizable) {
				// nothing to do
			} else {
				if (node->get_node()->splice.dst_access == Access::eNone && node->get_node()->splice.dst_domain == DomainFlagBits::eAny) {
					value.release();
				}
				extnodes.push_back(node);
			}
		}
		if (extnodes.size() == 0) {
			compiler.reset();
			return { expected_value }; // nothing to do
		}
		auto erg = compiler.link(extnodes, options);
		if (!erg) {
			return erg;
		}

		std::pair v = { &allocator, &*erg };
		VUK_DO_OR_RETURN(execute_submit(allocator, std::span{ &v, 1 }));
		compiler.reset();
		return { expected_value };
	}

	Result<void> UntypedValue::wait(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options) {
		auto res = submit(allocator, compiler, options);
		if (!res) {
			compiler.reset();
			return res;
		}
		assert(node->acqrel->status != Signal::Status::eDisarmed);
		if (node->acqrel->status == Signal::Status::eSynchronizable) {
			allocator.get_context().wait_for_domains(std::span{ &node->acqrel->source, 1 });
		}

		return { expected_value };
	}

	Result<void> UntypedValue::submit(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options) {
		if (node->acqrel->status == Signal::Status::eHostAvailable || node->acqrel->status == Signal::Status::eSynchronizable) {
			compiler.reset();
			return { expected_value }; // nothing to do
		} else {
			if (node->get_node()->splice.dst_access == Access::eNone && node->get_node()->splice.dst_domain == DomainFlagBits::eAny) {
				release();
			}
			auto erg = compiler.link(std::span{ &node, 1 }, options);
			if (!erg) {
				return erg;
			}
			std::pair v = { &allocator, &*erg };
			VUK_DO_OR_RETURN(execute_submit(allocator, std::span{ &v, 1 }));
			assert(node->acqrel->status != Signal::Status::eDisarmed);
			compiler.reset();
			return { expected_value };
		}
	}
} // namespace vuk

size_t std::hash<vuk::ImageAttachment>::operator()(vuk::ImageAttachment const& x) const noexcept {
	size_t h = 0;
	if (x.image != vuk::Image{} && x.image_view != vuk::ImageView{}) { // both i and iv, thats the only thing that matter
		hash_combine(h, x.image.image, x.image_view.payload);
	} else if (x.image != vuk::Image{}) { // only i, take all params that form the iv
		hash_combine(h,
		             x.image.image,
		             x.view_type,
		             x.format,
		             x.base_layer,
		             x.layer_count,
		             x.base_level,
		             x.level_count,
		             x.components,
		             x.format,
		             x.image_view_flags,
		             x.layout);
	} else {
		hash_combine(h,
		             x.extent,
		             x.image_flags,
		             x.image_type,
		             x.sample_count.count,
		             x.tiling,
		             x.usage,
		             x.view_type,
		             x.format,
		             x.base_layer,
		             x.layer_count,
		             x.base_level,
		             x.level_count,
		             x.components,
		             x.format,
		             x.image_view_flags,
		             x.layout);
	}
	return h;
}

size_t std::hash<vuk::Buffer>::operator()(vuk::Buffer const& x) const noexcept {
	size_t h = 0;
	hash_combine(h, x.buffer, x.offset, x.size);
	return h;
}