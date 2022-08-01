#pragma once

#include "vuk/RenderGraph.hpp"
#include "vuk/ShortAlloc.hpp"

namespace std {
	template<>
	struct hash<vuk::Resource> {
		std::size_t operator()(vuk::Resource const& s) const noexcept {
			return (size_t)((uintptr_t)s.name.c_str());
		}
	};
} // namespace std

namespace vuk {
	inline bool is_write_access(Access ia) {
		switch (ia) {
		case eColorResolveWrite:
		case eColorWrite:
		case eColorRW:
		case eDepthStencilRW:
		case eFragmentWrite:
		case eFragmentRW:
		case eTransferWrite:
		case eComputeWrite:
		case eComputeRW:
		case eHostWrite:
		case eHostRW:
		case eMemoryWrite:
		case eMemoryRW:
		case eRayTracingWrite:
		case eRayTracingRW:
		case eAccelerationStructureBuildWrite:
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
		case eFragmentRW:
		case eFragmentSampled:
		case eTransferRead:
		case eComputeRead:
		case eComputeSampled:
		case eComputeRW:
		case eHostRead:
		case eHostRW:
		case eMemoryRead:
		case eMemoryRW:
		case eRayTracingRead:
		case eRayTracingSampled:
		case eAccelerationStructureBuildRead:
			return true;
		default:
			return false;
		}
	}

	inline QueueResourceUse to_use(Access ia, DomainFlags domain) {
		switch (ia) {
		case eColorResolveWrite:
		case eColorWrite:
			return {
				vuk::PipelineStageFlagBits::eColorAttachmentOutput, vuk::AccessFlagBits::eColorAttachmentWrite, vuk::ImageLayout::eColorAttachmentOptimal, domain
			};
		case eColorRW:
			return { vuk::PipelineStageFlagBits::eColorAttachmentOutput,
				       vuk::AccessFlagBits::eColorAttachmentWrite | vuk::AccessFlagBits::eColorAttachmentRead,
				       vuk::ImageLayout::eColorAttachmentOptimal,
				       domain };
		case eColorResolveRead:
		case eColorRead:
			return {
				vuk::PipelineStageFlagBits::eColorAttachmentOutput, vuk::AccessFlagBits::eColorAttachmentRead, vuk::ImageLayout::eColorAttachmentOptimal, domain
			};
		case eDepthStencilRead:
			return { vuk::PipelineStageFlagBits::eEarlyFragmentTests | vuk::PipelineStageFlagBits::eLateFragmentTests,
				       vuk::AccessFlagBits::eDepthStencilAttachmentRead,
				       vuk::ImageLayout::eDepthStencilAttachmentOptimal,
				       domain };
		case eDepthStencilRW:
			return { vuk::PipelineStageFlagBits::eEarlyFragmentTests | vuk::PipelineStageFlagBits::eLateFragmentTests,
				       vuk::AccessFlagBits::eDepthStencilAttachmentRead | vuk::AccessFlagBits::eDepthStencilAttachmentWrite,
				       vuk::ImageLayout::eDepthStencilAttachmentOptimal,
				       domain };

		case eFragmentSampled:
			return { vuk::PipelineStageFlagBits::eFragmentShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eShaderReadOnlyOptimal, domain };
		case eFragmentRead:
			return { vuk::PipelineStageFlagBits::eFragmentShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eGeneral, domain };
		case eFragmentWrite:
			return { vuk::PipelineStageFlagBits::eFragmentShader, vuk::AccessFlagBits::eShaderWrite, vuk::ImageLayout::eGeneral, domain };
		case eFragmentRW:
			return {
				vuk::PipelineStageFlagBits::eFragmentShader, vuk::AccessFlagBits::eShaderRead | vuk::AccessFlagBits::eShaderWrite, vuk::ImageLayout::eGeneral, domain
			};

		case eTransferRead:
			return { vuk::PipelineStageFlagBits::eTransfer, vuk::AccessFlagBits::eTransferRead, vuk::ImageLayout::eTransferSrcOptimal, domain };
		case eTransferWrite:
			return { vuk::PipelineStageFlagBits::eTransfer, vuk::AccessFlagBits::eTransferWrite, vuk::ImageLayout::eTransferDstOptimal, domain };

		case eComputeRead:
			return { vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eGeneral, domain };
		case eComputeWrite:
			return { vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderWrite, vuk::ImageLayout::eGeneral, domain };
		case eComputeRW:
			return {
				vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderRead | vuk::AccessFlagBits::eShaderWrite, vuk::ImageLayout::eGeneral, domain
			};
		case eComputeSampled:
			return { vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eShaderReadOnlyOptimal, domain };

		case eAttributeRead:
			return { vuk::PipelineStageFlagBits::eVertexInput, vuk::AccessFlagBits::eVertexAttributeRead, vuk::ImageLayout::eGeneral /* ignored */, domain };
		case eVertexRead:
			return { vuk::PipelineStageFlagBits::eVertexShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eGeneral /* ignored */, domain };
		case eIndexRead:
			return { vuk::PipelineStageFlagBits::eVertexInput, vuk::AccessFlagBits::eIndexRead, vuk::ImageLayout::eGeneral /* ignored */, domain };
		case eIndirectRead:
			return { vuk::PipelineStageFlagBits::eDrawIndirect, vuk::AccessFlagBits::eIndirectCommandRead, vuk::ImageLayout::eGeneral /* ignored */, domain };

		case eRayTracingRead:
			return { vuk::PipelineStageFlagBits::eRayTracingShaderKHR,
				       vuk::AccessFlagBits::eShaderRead | vuk::AccessFlagBits::eAccelerationStructureReadKHR,
				       vuk::ImageLayout::eGeneral,
				       domain };
		case eRayTracingWrite:
			return { vuk::PipelineStageFlagBits::eRayTracingShaderKHR, vuk::AccessFlagBits::eShaderWrite, vuk::ImageLayout::eGeneral, domain };
		case eRayTracingRW:
			return { vuk::PipelineStageFlagBits::eRayTracingShaderKHR,
				       vuk::AccessFlagBits::eAccelerationStructureReadKHR | vuk::AccessFlagBits::eShaderRead | vuk::AccessFlagBits::eShaderWrite,
				       vuk::ImageLayout::eGeneral,
				       domain };
		case eRayTracingSampled:
			return { vuk::PipelineStageFlagBits::eRayTracingShaderKHR, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eShaderReadOnlyOptimal, domain };

		case eAccelerationStructureBuildRead:
			return { vuk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
				       vuk::AccessFlagBits::eShaderRead | vuk::AccessFlagBits::eAccelerationStructureReadKHR,
				       vuk::ImageLayout::eGeneral /* ignored */,
				       domain };
		case eAccelerationStructureBuildWrite:
			return { vuk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
				       vuk::AccessFlagBits::eAccelerationStructureWriteKHR,
				       vuk::ImageLayout::eGeneral /* ignored */,
				       domain };
		case eHostRead:
			return { vuk::PipelineStageFlagBits::eHost, vuk::AccessFlagBits::eHostRead, vuk::ImageLayout::eGeneral, domain };
		case eHostWrite:
			return { vuk::PipelineStageFlagBits::eHost, vuk::AccessFlagBits::eHostWrite, vuk::ImageLayout::eGeneral, domain };
		case eHostRW:
			return { vuk::PipelineStageFlagBits::eHost, vuk::AccessFlagBits::eHostRead | vuk::AccessFlagBits::eHostWrite, vuk::ImageLayout::eGeneral, domain };

		case eMemoryRead:
			return { vuk::PipelineStageFlagBits::eAllCommands, vuk::AccessFlagBits::eMemoryRead, vuk::ImageLayout::eGeneral, domain };
		case eMemoryWrite:
			return { vuk::PipelineStageFlagBits::eAllCommands, vuk::AccessFlagBits::eMemoryWrite, vuk::ImageLayout::eGeneral, domain };
		case eMemoryRW:
			return {
				vuk::PipelineStageFlagBits::eAllCommands, vuk::AccessFlagBits::eMemoryRead | vuk::AccessFlagBits::eMemoryWrite, vuk::ImageLayout::eGeneral, domain
			};

		case eNone:
			return { vuk::PipelineStageFlagBits::eTopOfPipe, vuk::AccessFlagBits{}, vuk::ImageLayout::eUndefined, domain };
		case eClear:
			return { vuk::PipelineStageFlagBits::eTransfer, vuk::AccessFlagBits::eTransferWrite, vuk::ImageLayout::eTransferDstOptimal, domain };
		case eRelease:
		case eReleaseToGraphics:
		case eReleaseToCompute:
		case eReleaseToTransfer:
		case eAcquire:
		case eAcquireFromGraphics:
		case eAcquireFromCompute:
		case eAcquireFromTransfer:
			return { vuk::PipelineStageFlagBits::eTopOfPipe, vuk::AccessFlagBits{}, vuk::ImageLayout::eGeneral, domain }; // ignored
		default:
			assert(0 && "NYI");
			return {};
		}
	}

	inline bool is_acquire(Access a) {
		switch (a) {
		case eAcquire:
		case eAcquireFromGraphics:
		case eAcquireFromCompute:
		case eAcquireFromTransfer:
			return true;
		default:
			return false;
		}
	}

	inline bool is_release(Access a) {
		switch (a) {
		case eRelease:
		case eReleaseToGraphics:
		case eReleaseToCompute:
		case eReleaseToTransfer:
			return true;
		default:
			return false;
		}
	}

	inline Access domain_to_release_access(DomainFlags dst) {
		auto queue = (DomainFlagBits)(dst & DomainFlagBits::eQueueMask).m_mask;
		switch (queue) {
		case DomainFlagBits::eGraphicsQueue:
			return Access::eReleaseToGraphics;
		case DomainFlagBits::eComputeQueue:
			return Access::eReleaseToCompute;
		case DomainFlagBits::eTransferQueue:
			return Access::eReleaseToTransfer;
		default:
			return Access::eRelease;
		}
	}

	inline Access domain_to_acquire_access(DomainFlags dst) {
		auto queue = (DomainFlagBits)(dst & DomainFlagBits::eQueueMask).m_mask;
		switch (queue) {
		case DomainFlagBits::eGraphicsQueue:
			return Access::eAcquireFromGraphics;
		case DomainFlagBits::eComputeQueue:
			return Access::eAcquireFromCompute;
		case DomainFlagBits::eTransferQueue:
			return Access::eAcquireFromTransfer;
		default:
			return Access::eAcquire;
		}
	}

	// not all domains can support all stages, this function corrects stage flags
	inline void scope_to_domain(PipelineStageFlags& src, DomainFlags flags) {
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
		}
		if (remove & DomainFlagBits::eComputeQueue) {
			src &= (PipelineStageFlags)~0b100000000000;
		}
	}

	inline bool is_framebuffer_attachment(const Resource& r) {
		if (r.type == Resource::Type::eBuffer)
			return false;
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

	inline bool is_framebuffer_attachment(QueueResourceUse u) {
		switch (u.layout) {
		case vuk::ImageLayout::eColorAttachmentOptimal:
		case vuk::ImageLayout::eDepthStencilAttachmentOptimal:
			return true;
		default:
			return false;
		}
	}

	inline bool is_write_access(QueueResourceUse u) {
		if (u.access == vuk::AccessFlagBits{})
			return false;
		if (u.access & vuk::AccessFlagBits::eColorAttachmentWrite)
			return true;
		if (u.access & vuk::AccessFlagBits::eDepthStencilAttachmentWrite)
			return true;
		if (u.access & vuk::AccessFlagBits::eShaderWrite)
			return true;
		if (u.access & vuk::AccessFlagBits::eTransferWrite)
			return true;
		if (u.access & vuk::AccessFlagBits::eHostWrite)
			return true;
		if (u.access & vuk::AccessFlagBits::eMemoryWrite)
			return true;
		if (u.access & vuk::AccessFlagBits::eAccelerationStructureWriteKHR)
			return true;
		return false;
	}

	inline bool is_read_access(QueueResourceUse u) {
		if (u.access == vuk::AccessFlagBits{})
			return false;
		return !is_write_access(u);
	}

	struct UseRef {
		Name name;
		Name out_name;
		vuk::Access original = vuk::eNone;
		vuk::Access high_level_access;
		QueueResourceUse use;
		Resource::Type type;
		Subrange subrange;
		PassInfo* pass = nullptr;
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
		std::vector<ImageBarrier, short_alloc<ImageBarrier, 16>> pre_barriers, post_barriers;
		std::vector<MemoryBarrier, short_alloc<MemoryBarrier, 16>> pre_mem_barriers, post_mem_barriers;
	};

	struct BufferInfo {
		Name name;

		QueueResourceUse initial;
		QueueResourceUse final;

		Buffer buffer;
		FutureBase* attached_future = nullptr;

		std::vector<void*> use_chains;
	};

	struct AttachmentInfo {
		Name name;

		ImageAttachment attachment = {};

		QueueResourceUse initial, final;

		enum class Type { eInternal, eExternal, eSwapchain } type;

		// swapchain for swapchain
		Swapchain* swapchain = nullptr;

		std::vector<struct RenderPassInfo*> rp_uses;

		FutureBase* attached_future = nullptr;

		std::vector<void*> use_chains;
	};

	struct AttachmentRPInfo {
		AttachmentInfo* attachment_info;

		VkAttachmentDescription description = {};

		QueueResourceUse initial, final;

		std::optional<Clear> clear_value;

		bool is_resolve_dst = false;
		AttachmentInfo* resolve_src = nullptr;
	};
} // namespace vuk
