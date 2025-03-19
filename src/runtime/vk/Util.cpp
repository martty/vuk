#include "vuk/RenderGraph.hpp"
#include "vuk/Value.hpp"
#include "vuk/runtime/Cache.hpp"
#include "vuk/runtime/vk/DeviceVkResource.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"
#include "vuk/runtime/vk/VkSwapchain.hpp"

namespace vuk {
	Result<void> submit(Allocator& allocator, Compiler& compiler, std::span<UntypedValue> values, RenderGraphCompileOptions options) {
		// DeviceVkResource is unsuitable for submits, because internally the lifetimes are not tracked
		if (dynamic_cast<DeviceVkResource*>(&allocator.get_device_resource()) != nullptr) {
			return { expected_error, RenderGraphException{ "DeviceVkResource is unsuitable for submits" } };
		}
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

		VUK_DO_OR_RETURN(erg->execute(allocator));
		compiler.reset();
		return { expected_value };
	}

	Result<void> wait_for_values_explicit(Allocator& alloc, Compiler& compiler, std::span<UntypedValue> values, RenderGraphCompileOptions options) {
		std::vector<SyncPoint> waits;
		VUK_DO_OR_RETURN(submit(alloc, compiler, values, options));
		for (uint64_t i = 0; i < values.size(); i++) {
			auto& value = values[i];
			if (value.node->acqrel->status != Signal::Status::eSynchronizable) {
				continue;
			}
			waits.emplace_back(value.node->acqrel->source);
		}
		if (waits.size() > 0) {
			VUK_DO_OR_RETURN(alloc.get_context().wait_for_domains(std::span(waits)));
			for (uint64_t i = 0; i < values.size(); i++) {
				auto& value = values[i];
				if (value.node->acqrel->status == Signal::Status::eSynchronizable) {
					value.node->acqrel->status = Signal::Status::eHostAvailable;
				}
			}
		}

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
		return vuk::submit(allocator, compiler, std::span{ this, 1 }, options);
	}

	Result<Signal::Status> UntypedValue::poll() {
		if (node->acqrel->status == Signal::Status::eDisarmed || node->acqrel->status == Signal::Status::eHostAvailable) {
			return { expected_value, node->acqrel->status };
		}
		auto res = Runtime::sync_point_ready(node->acqrel->source);
		if (!res) {
			return res;
		}
		if (*res) {
			node->acqrel->status = Signal::Status::eHostAvailable;
		}
		return { expected_value, node->acqrel->status };
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