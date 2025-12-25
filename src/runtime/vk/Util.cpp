#include "vuk/RenderGraph.hpp"
#include "vuk/runtime/Cache.hpp"
#include "vuk/runtime/vk/DeviceVkResource.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"
#include "vuk/runtime/vk/VkSwapchain.hpp"
#include "vuk/Value.hpp"

#include <fmt/format.h>

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
				if (node->deps.size() == 0) {
					// nothing to do
				} else {
					node = std::make_shared<ExtNode>(Ref{ node->get_node(), value.get_head().index }, node, Access::eNone, DomainFlagBits::eDevice);
					extnodes.push_back(node);
				}
			} else {
				if (node->get_node()->kind != Node::RELEASE) {
					auto rel_node = std::make_shared<ExtNode>(Ref{ node->get_node(), value.get_head().index }, node, Access::eNone, DomainFlagBits::eDevice);
					extnodes.push_back(rel_node);
				} else {
					extnodes.push_back(node);
				}
			}
		}
		if (extnodes.size() == 0) {
			compiler.reset();
			return { expected_value }; // nothing to do
		}
		auto erg = compiler.compile(allocator, extnodes, options);
		if (!erg) {
			return erg;
		}

		VUK_DO_OR_RETURN(compiler.execute(allocator));
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

	std::string format_as(const BufferCreateInfo& foo) {
		return fmt::format("BufferCreateInfo{{{}, {}}}", foo.memory_usage, foo.size);
	}

	std::string format_as(const view<BufferLike<byte>, dynamic_extent>& foo) {
		return fmt::format("buffer[{:x}:{}]", foo.ptr.device_address, foo.sz_bytes);
	}

	std::string format_as(const ImageView<Format::eUndefined>& foo) {
		return fmt::format("iv[{}]", foo.view_key);
	}

	std::string format_as(const Extent3D& extent) {
		return fmt::format("{}x{}x{}", extent.width, extent.height, extent.depth);
	}

	std::string format_as(const Samples& samples) {
		return fmt::format("{}x", (uint32_t)samples.count);
	}

	std::string format_as(const ImageViewEntry& entry) {
		return fmt::format("ImageViewEntry{{format={}, extent={}, samples={}, base_level={}, level_count={}, base_layer={}, layer_count={}}}",
		                   entry.format,
		                   entry.extent,
		                   entry.sample_count,
		                   entry.base_level,
		                   entry.level_count,
		                   entry.base_layer,
		                   entry.layer_count);
	}
} // namespace vuk

size_t std::hash<vuk::ImageView<>>::operator()(vuk::ImageView<> const& x) const noexcept {
	size_t h = std::hash<uint32_t>()(x.view_key);

	return h;
}