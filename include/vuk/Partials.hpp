#pragma once

#include <vuk/Context.hpp>
#include <vuk/RenderGraph.hpp>
#include <vuk/CommandBuffer.hpp>

namespace vuk {
	template<class Allocator>
	TokenWithContext copy_to_buffer(Allocator& allocator, Domain copy_domain, Buffer buffer, void* src_data, size_t size) {
		Token tok = allocator.allocate_token();
		auto& data = allocator.get_token_data(tok);

		// host-mapped buffers just get memcpys
		if (buffer.mapped_ptr) {
			memcpy(buffer.mapped_ptr, src_data, size);
			data.state = TokenData::State::eComplete;
			return { allocator.ctx, tok };
		}

		auto src = allocator.allocate_buffer(vuk::MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, size, 1);
		::memcpy(src.mapped_ptr, src_data, size);

		data.rg = new vuk::RenderGraph();
		data.rg->add_pass({
			.executes_on = copy_domain,
			.resources = {"_dst"_buffer(vuk::Access::eTransferDst), "_src"_buffer(vuk::Access::eTransferSrc)},
			.execute = [size](vuk::CommandBuffer& command_buffer) {
				command_buffer.copy_buffer("_src", "_dst", VkBufferCopy{.size = size});
			} });
		data.rg->attach_buffer("_src", src, vuk::Access::eNone, vuk::Access::eNone);
		data.rg->attach_buffer("_dst", buffer, vuk::Access::eNone, vuk::Access::eNone);
		data.state = TokenData::State::eArmed;
		return { allocator.ctx, tok };
	}

	template<class Allocator, class T>
	TokenWithContext copy_to_buffer(Allocator& allocator, Domain copy_domain, Buffer dst, std::span<T> data) {
		return copy_to_buffer(allocator, copy_domain, dst, data.data(), data.size_bytes());
	}

	template<class Allocator>
	vuk::TokenWithContext transition_image(Allocator& allocator, vuk::Texture& t, vuk::Access src_access, vuk::Access dst_access) {
		Token tok = allocator.allocate_token();
		auto& data = allocator.get_token_data(tok);
		data.rg = new vuk::RenderGraph();
		data.rg->attach_image("_transition", ImageAttachment::from_texture(t), src_access, dst_access);
		data.state = TokenData::State::eArmed;
		return { allocator.ctx, tok };
	}

	template<class Allocator>
	vuk::TokenWithContext copy_to_image(Allocator& allocator, vuk::Image dst, vuk::Format format, vuk::Extent3D extent, uint32_t base_layer, void* src_data, size_t size) {
		Token tok = allocator.allocate_token();
		auto& data = allocator.get_token_data(tok);

		// compute staging buffer alignment as texel block size
		size_t alignment = format_to_texel_block_size(format);
		BufferAllocationCreateInfo baci{ vuk::MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, size, alignment };
		auto src = allocator.allocate_buffer(baci, 0, VUK_HERE());
		memcpy(src->mapped_ptr, src_data, size);

		data.rg = new vuk::RenderGraph();

		data.rg->add_pass({
			.executes_on = vuk::Domain::eTransfer,
			.resources = {"_dst"_image(vuk::Access::eTransferDst), "_src"_buffer(vuk::Access::eTransferSrc)},
			.execute = [base_layer](vuk::CommandBuffer& command_buffer) {
				command_buffer.copy_buffer_to_image("_src", "_dst", vuk::BufferImageCopy{.imageSubresource = {.baseArrayLayer = base_layer}});
			}
			});

		Access src_access = vuk::eNone;
		vuk::ImageAttachment ia;
		ia.image = dst;
		ia.image_view = {};
		ia.format = format;
		ia.extent = vuk::Extent2D{ extent.width, extent.height };
		ia.sample_count = vuk::Samples::e1;

		data.rg->attach_buffer("_src", *src, vuk::Access::eNone, vuk::Access::eNone);
		data.rg->attach_image("_dst", ia, src_access, vuk::eFragmentRead);
		data.state = TokenData::State::eArmed;
		return { allocator.ctx, tok };
	}
}