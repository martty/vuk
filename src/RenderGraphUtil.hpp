#pragma once

#include "vuk/RenderGraph.hpp"
#include <vuk/ShortAlloc.hpp>

namespace std {
	template<> struct hash<vuk::Resource> {
		std::size_t operator()(vuk::Resource const& s) const noexcept {
			return (size_t)((uintptr_t)s.name.c_str());
		}
	};
}

namespace vuk {
	inline bool is_write_access(Access ia) {
		switch (ia) {
		case eColorResolveWrite:
		case eColorWrite:
		case eColorRW:
		case eDepthStencilRW:
		case eFragmentWrite:
		case eTransferWrite:
		case eComputeWrite:
		case eComputeRW:
		case eHostWrite:
		case eHostRW:
		case eMemoryWrite:
		case eMemoryRW:
			return true;
		default:
			return false;
		}
	}

	inline bool is_read_access(Access ia) {
		switch (ia) {
		case eColorResolveRead:
		case eColorRead:
		case eColorRW:
		case eDepthStencilRead:
		case eDepthStencilRW:
		case eFragmentRead:
		case eFragmentSampled:
		case eTransferRead:
		case eComputeRead:
		case eComputeSampled:
		case eComputeRW:
		case eHostRead:
		case eHostRW:
		case eMemoryRead:
		case eMemoryRW:
			return true;
		default:
			return false;
		}
	}

	inline ResourceUse to_use(Access ia) {
		switch (ia) {
		case eColorResolveWrite:
		case eColorWrite: return { vuk::PipelineStageFlagBits::eColorAttachmentOutput, vuk::AccessFlagBits::eColorAttachmentWrite, vuk::ImageLayout::eColorAttachmentOptimal };
		case eColorRW: return { vuk::PipelineStageFlagBits::eColorAttachmentOutput, vuk::AccessFlagBits::eColorAttachmentWrite | vuk::AccessFlagBits::eColorAttachmentRead, vuk::ImageLayout::eColorAttachmentOptimal };
		case eColorResolveRead:
		case eColorRead: return { vuk::PipelineStageFlagBits::eColorAttachmentOutput, vuk::AccessFlagBits::eColorAttachmentRead, vuk::ImageLayout::eColorAttachmentOptimal };
		case eDepthStencilRW: return { vuk::PipelineStageFlagBits::eEarlyFragmentTests | vuk::PipelineStageFlagBits::eLateFragmentTests, vuk::AccessFlagBits::eDepthStencilAttachmentRead | vuk::AccessFlagBits::eDepthStencilAttachmentWrite, vuk::ImageLayout::eDepthStencilAttachmentOptimal };

		case eFragmentSampled: return { vuk::PipelineStageFlagBits::eFragmentShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eShaderReadOnlyOptimal };
		case eFragmentRead: return { vuk::PipelineStageFlagBits::eFragmentShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eShaderReadOnlyOptimal };

		case eTransferRead: return { vuk::PipelineStageFlagBits::eTransfer, vuk::AccessFlagBits::eTransferRead, vuk::ImageLayout::eTransferSrcOptimal };
		case eTransferWrite: return { vuk::PipelineStageFlagBits::eTransfer, vuk::AccessFlagBits::eTransferWrite, vuk::ImageLayout::eTransferDstOptimal };

		case eComputeRead: return { vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eGeneral };
		case eComputeWrite: return { vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderWrite, vuk::ImageLayout::eGeneral };
		case eComputeRW: return { vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderRead | vuk::AccessFlagBits::eShaderWrite, vuk::ImageLayout::eGeneral };
		case eComputeSampled: return { vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eShaderReadOnlyOptimal };

		case eAttributeRead: return { vuk::PipelineStageFlagBits::eVertexInput, vuk::AccessFlagBits::eVertexAttributeRead, vuk::ImageLayout::eGeneral /* ignored */ };
		case eVertexRead:
			return { vuk::PipelineStageFlagBits::eVertexShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eGeneral /* ignored */ };
		case eIndexRead: return { vuk::PipelineStageFlagBits::eVertexInput, vuk::AccessFlagBits::eIndexRead, vuk::ImageLayout::eGeneral /* ignored */ };
		case eIndirectRead: return { vuk::PipelineStageFlagBits::eDrawIndirect, vuk::AccessFlagBits::eIndirectCommandRead, vuk::ImageLayout::eGeneral /* ignored */ };

		case eHostRead:
			return { vuk::PipelineStageFlagBits::eHost, vuk::AccessFlagBits::eHostRead, vuk::ImageLayout::eGeneral };
		case eHostWrite:
			return { vuk::PipelineStageFlagBits::eHost, vuk::AccessFlagBits::eHostWrite, vuk::ImageLayout::eGeneral };
		case eHostRW:
			return { vuk::PipelineStageFlagBits::eHost, vuk::AccessFlagBits::eHostRead | vuk::AccessFlagBits::eHostWrite, vuk::ImageLayout::eGeneral };

		case eMemoryRead: return { vuk::PipelineStageFlagBits::eBottomOfPipe, vuk::AccessFlagBits::eMemoryRead, vuk::ImageLayout::eGeneral };
		case eMemoryWrite: return { vuk::PipelineStageFlagBits::eBottomOfPipe, vuk::AccessFlagBits::eMemoryWrite, vuk::ImageLayout::eGeneral };
		case eMemoryRW: return { vuk::PipelineStageFlagBits::eBottomOfPipe, vuk::AccessFlagBits::eMemoryRead | vuk::AccessFlagBits::eMemoryWrite, vuk::ImageLayout::eGeneral };

		case eNone:
			return { vuk::PipelineStageFlagBits::eTopOfPipe, vuk::AccessFlagBits{}, vuk::ImageLayout::eUndefined };
		case eClear:
			return { vuk::PipelineStageFlagBits::eColorAttachmentOutput, vuk::AccessFlagBits::eColorAttachmentWrite, vuk::ImageLayout::ePreinitialized };
		case eTransferClear:
			return { vuk::PipelineStageFlagBits::eTransfer, vuk::AccessFlagBits::eTransferWrite, vuk::ImageLayout::eTransferDstOptimal };
		default:
			assert(0 && "NYI");
			return {};
		}

	}

	inline bool is_framebuffer_attachment(const Resource& r) {
		if (r.type == Resource::Type::eBuffer) return false;
		switch (r.ia) {
		case eColorWrite:
		case eColorRW:
		case eDepthStencilRW:
		case eColorRead:
		case eDepthStencilRead:
		case eColorResolveRead:
		case eColorResolveWrite:
			return true;
		default:
			return false;
		}
	}

	inline bool is_framebuffer_attachment(ResourceUse u) {
		switch (u.layout) {
		case vuk::ImageLayout::eColorAttachmentOptimal:
		case vuk::ImageLayout::eDepthStencilAttachmentOptimal:
			return true;
		default:
			return false;
		}
	}

	inline bool is_write_access(ResourceUse u) {
		if (u.access == vuk::AccessFlagBits{}) return false;
		if (u.access & vuk::AccessFlagBits::eColorAttachmentWrite) return true;
		if (u.access & vuk::AccessFlagBits::eDepthStencilAttachmentWrite) return true;
		if (u.access & vuk::AccessFlagBits::eShaderWrite) return true;
		if (u.access & vuk::AccessFlagBits::eTransferWrite) return true;
		if (u.access & vuk::AccessFlagBits::eHostWrite) return true;
		if (u.access & vuk::AccessFlagBits::eMemoryWrite) return true;
		if (u.access & vuk::AccessFlagBits::eAccelerationStructureWriteKHR) return true;
		return false;
	}

	inline bool is_read_access(ResourceUse u) {
		if (u.access == vuk::AccessFlagBits{}) return false;
		return !is_write_access(u);
	}

	struct UseRef {
		ResourceUse use;
		PassInfo* pass = nullptr;
		Domain domain;
	};

	struct PassInfo {
		PassInfo(arena&, Pass&&);

		Pass pass;

		size_t render_pass_index;
		uint32_t subpass;

		std::vector<std::pair<Domain, uint32_t>> relative_waits;
		std::vector<Resource, short_alloc<Resource, 16>> inputs;
		uint32_t bloom_resolved_inputs = 0;
		std::vector<uint32_t, short_alloc<uint32_t, 16>> resolved_input_name_hashes;
		uint32_t bloom_outputs = 0;
		std::vector<Resource, short_alloc<Resource, 16>> outputs;
		std::vector<uint32_t, short_alloc<uint32_t, 16>> output_name_hashes;
		std::vector<uint32_t> signals;

		std::vector<Future<Buffer>*> buffer_future_signals;

		bool is_head_pass = false;
		bool is_tail_pass = false;
	};

	struct AttachmentSInfo {
		vuk::ImageLayout layout;
		vuk::AccessFlags access;
		vuk::PipelineStageFlags stage;
	};

	struct ImageBarrier {
		Name image;
		VkImageMemoryBarrier barrier = {};
		vuk::PipelineStageFlags src;
		vuk::PipelineStageFlags dst;
	};

	struct MemoryBarrier {
		VkMemoryBarrier barrier = {};
		vuk::PipelineStageFlags src;
		vuk::PipelineStageFlags dst;
	};

	struct SubpassInfo {
		SubpassInfo(arena&);
		bool use_secondary_command_buffers;
		std::vector<PassInfo*, short_alloc<PassInfo*, 16>> passes;
		std::vector<ImageBarrier> pre_barriers, post_barriers;
		std::vector<MemoryBarrier> pre_mem_barriers, post_mem_barriers;
	};

	struct RenderPassInfo {
		RenderPassInfo(arena&);
		std::vector<SubpassInfo, short_alloc<SubpassInfo, 64>> subpasses;
		std::vector<AttachmentRPInfo, short_alloc<AttachmentRPInfo, 16>> attachments;
		vuk::RenderPassCreateInfo rpci;
		vuk::FramebufferCreateInfo fbci;
		bool framebufferless = false;
		VkRenderPass handle = {};
		VkFramebuffer framebuffer;
		std::vector<ImageBarrier> pre_barriers, post_barriers;
		std::vector<MemoryBarrier> pre_mem_barriers, post_mem_barriers;
	};

} // namespace vuk
