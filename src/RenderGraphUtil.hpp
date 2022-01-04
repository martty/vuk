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
		case eColorWrite: return { ia, vuk::PipelineStageFlagBits::eColorAttachmentOutput, vuk::AccessFlagBits::eColorAttachmentWrite, vuk::ImageLayout::eColorAttachmentOptimal };
		case eColorRW: return { ia, vuk::PipelineStageFlagBits::eColorAttachmentOutput, vuk::AccessFlagBits::eColorAttachmentWrite | vuk::AccessFlagBits::eColorAttachmentRead, vuk::ImageLayout::eColorAttachmentOptimal };
		case eColorResolveRead:
		case eColorRead: return { ia, vuk::PipelineStageFlagBits::eColorAttachmentOutput, vuk::AccessFlagBits::eColorAttachmentRead, vuk::ImageLayout::eColorAttachmentOptimal };
		case eDepthStencilRW: return { ia, vuk::PipelineStageFlagBits::eEarlyFragmentTests | vuk::PipelineStageFlagBits::eLateFragmentTests, vuk::AccessFlagBits::eDepthStencilAttachmentRead | vuk::AccessFlagBits::eDepthStencilAttachmentWrite, vuk::ImageLayout::eDepthStencilAttachmentOptimal };

		case eFragmentSampled: return { ia, vuk::PipelineStageFlagBits::eFragmentShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eShaderReadOnlyOptimal };
		case eFragmentRead: return { ia, vuk::PipelineStageFlagBits::eFragmentShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eShaderReadOnlyOptimal };

		case eTransferRead: return { ia, vuk::PipelineStageFlagBits::eTransfer, vuk::AccessFlagBits::eTransferRead, vuk::ImageLayout::eTransferSrcOptimal };
		case eTransferWrite: return { ia, vuk::PipelineStageFlagBits::eTransfer, vuk::AccessFlagBits::eTransferWrite, vuk::ImageLayout::eTransferDstOptimal };

		case eComputeRead: return { ia, vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eGeneral };
		case eComputeWrite: return { ia, vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderWrite, vuk::ImageLayout::eGeneral };
		case eComputeRW: return { ia, vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderRead | vuk::AccessFlagBits::eShaderWrite, vuk::ImageLayout::eGeneral };
		case eComputeSampled: return { ia, vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eShaderReadOnlyOptimal };

		case eAttributeRead: return { ia, vuk::PipelineStageFlagBits::eVertexInput, vuk::AccessFlagBits::eVertexAttributeRead, vuk::ImageLayout::eGeneral /* ignored */ };
		case eVertexRead:
			return { ia, vuk::PipelineStageFlagBits::eVertexShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eGeneral /* ignored */ };
		case eIndexRead: return { ia, vuk::PipelineStageFlagBits::eVertexInput, vuk::AccessFlagBits::eIndexRead, vuk::ImageLayout::eGeneral /* ignored */ };
		case eIndirectRead: return { ia, vuk::PipelineStageFlagBits::eDrawIndirect, vuk::AccessFlagBits::eIndirectCommandRead, vuk::ImageLayout::eGeneral /* ignored */ };

		case eHostRead:
			return { ia, vuk::PipelineStageFlagBits::eHost, vuk::AccessFlagBits::eHostRead, vuk::ImageLayout::eGeneral };
		case eHostWrite:
			return { ia, vuk::PipelineStageFlagBits::eHost, vuk::AccessFlagBits::eHostWrite, vuk::ImageLayout::eGeneral };
		case eHostRW:
			return { ia, vuk::PipelineStageFlagBits::eHost, vuk::AccessFlagBits::eHostRead | vuk::AccessFlagBits::eHostWrite, vuk::ImageLayout::eGeneral };

		case eMemoryRead: return { ia, vuk::PipelineStageFlagBits::eBottomOfPipe, vuk::AccessFlagBits::eMemoryRead, vuk::ImageLayout::eGeneral };
		case eMemoryWrite: return { ia, vuk::PipelineStageFlagBits::eBottomOfPipe, vuk::AccessFlagBits::eMemoryWrite, vuk::ImageLayout::eGeneral };
		case eMemoryRW: return { ia, vuk::PipelineStageFlagBits::eBottomOfPipe, vuk::AccessFlagBits::eMemoryRead | vuk::AccessFlagBits::eMemoryWrite, vuk::ImageLayout::eGeneral };

		case eNone:
			return { ia, vuk::PipelineStageFlagBits::eTopOfPipe, vuk::AccessFlagBits{}, vuk::ImageLayout::eUndefined };
		case eClear:
			return { ia, vuk::PipelineStageFlagBits::eColorAttachmentOutput, vuk::AccessFlagBits::eColorAttachmentWrite, vuk::ImageLayout::ePreinitialized };
		case eTransferClear:
			return { ia, vuk::PipelineStageFlagBits::eTransfer, vuk::AccessFlagBits::eTransferWrite, vuk::ImageLayout::eTransferDstOptimal };
		case eRelease:
		case eAcquire:
			return { ia, vuk::PipelineStageFlagBits::eTopOfPipe, vuk::AccessFlagBits{}, vuk::ImageLayout::eGeneral }; // ignored
		default:
			assert(0 && "NYI");
			return {};
		}

	}

	// not all domains can support all stages, this function corrects stage flags
	inline void scope_to_domain(PipelineStageFlags& src, PipelineStageFlags& dst, DomainFlags flags) {
		DomainFlags remove;
		// if no graphics in domain, remove all graphics
		if ((flags & DomainFlagBits::eGraphicsQueue) == DomainFlags{}) {
			remove |= DomainFlagBits::eGraphicsQueue;

			// if no graphics & compute in domain, remove all compute
			if ((flags & DomainFlagBits::eComputeQueue) == DomainFlags{}) {
				remove |= DomainFlagBits::eComputeQueue;
			}
		}
		
		if (remove & DomainFlagBits::eGraphicsQueue) {
			src &= (PipelineStageFlags)~0b11111111110;
			dst &= (PipelineStageFlags)~0b11111111110;
		}
		if (remove & DomainFlagBits::eComputeQueue) {
			src &= (PipelineStageFlags)~0b100000000000;
			dst &= (PipelineStageFlags)~0b100000000000;
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
		DomainFlagBits domain;
	};

	struct PassInfo {
		PassInfo(arena&, Pass&&);

		Pass pass;

		size_t render_pass_index;
		uint32_t subpass;
		DomainFlags domain;

		Name prefix;

		std::vector<std::pair<DomainFlagBits, PassInfo*>> waits;
		bool is_waited_on = false;
		std::vector<Resource, short_alloc<Resource, 16>> inputs;
		uint32_t bloom_resolved_inputs = 0;
		std::vector<Name, short_alloc<Name, 16>> input_names;
		uint32_t bloom_outputs = 0;
		uint32_t bloom_write_inputs = 0;
		std::vector<Resource, short_alloc<Resource, 16>> outputs;
		std::vector<Name, short_alloc<Name, 16>> output_names;
		std::vector<Name, short_alloc<Name, 16>> write_input_names;

		std::vector<FutureBase*> future_signals;

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
		uint64_t command_buffer_index;
		uint64_t batch_index;
		std::vector<SubpassInfo, short_alloc<SubpassInfo, 64>> subpasses;
		std::vector<AttachmentRPInfo, short_alloc<AttachmentRPInfo, 16>> attachments;
		vuk::RenderPassCreateInfo rpci;
		vuk::FramebufferCreateInfo fbci;
		bool framebufferless = false;
		VkRenderPass handle = {};
		VkFramebuffer framebuffer;
		std::vector<ImageBarrier> pre_barriers, post_barriers;
		std::vector<MemoryBarrier> pre_mem_barriers, post_mem_barriers;
		std::vector<std::pair<DomainFlagBits, uint32_t>> waits;
	};

} // namespace vuk
