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
		constexpr uint64_t write_mask = eColorResolveWrite | eColorWrite | eDepthStencilWrite | eFragmentWrite | eTransferWrite | eComputeWrite | eHostWrite |
		                                eMemoryWrite | eRayTracingWrite | eAccelerationStructureBuildWrite;
		return ia & write_mask;
	}

	inline bool is_read_access(Access ia) {
		constexpr uint64_t read_mask = eColorResolveRead | eColorRead | eDepthStencilRead | eFragmentRead | eFragmentSampled | eTransferRead | eComputeRead |
		                               eComputeSampled | eHostRead | eMemoryRead | eRayTracingRead | eRayTracingSampled | eAccelerationStructureBuildRead;
		return ia & read_mask;
	}

	inline ImageLayout combine_layout(ImageLayout a, ImageLayout b) {
		if (a == ImageLayout::eUndefined) {
			return b;
		}
		if (b == ImageLayout::eUndefined) {
			return a;
		}
		if (a == b) {
			return a;
		}
		if (a == ImageLayout::eDepthStencilReadOnlyOptimal && b == ImageLayout::eDepthStencilAttachmentOptimal ||
		    b == ImageLayout::eDepthStencilReadOnlyOptimal && a == ImageLayout::eDepthStencilAttachmentOptimal) {
			return ImageLayout::eDepthStencilAttachmentOptimal;
		}
		assert(a != ImageLayout::ePresentSrcKHR && b != ImageLayout::ePresentSrcKHR);
		return ImageLayout::eGeneral;
	}

	inline QueueResourceUse to_use(Access ia, DomainFlags domain) {
		constexpr uint64_t color_read = eColorResolveRead | eColorRead;
		constexpr uint64_t color_write = eColorResolveWrite | eColorWrite;
		constexpr uint64_t color_rw = color_read | color_write;

		QueueResourceUse qr{};
		if (ia & color_read) {
			qr.access |= AccessFlagBits::eColorAttachmentRead;
		}
		if (ia & color_write) {
			qr.access |= AccessFlagBits::eColorAttachmentWrite;
		}
		if (ia & color_rw) {
			qr.stages |= PipelineStageFlagBits::eColorAttachmentOutput;
			qr.layout = combine_layout(qr.layout, ImageLayout::eColorAttachmentOptimal);
		}
		if (ia & eDepthStencilRead) {
			qr.access |= AccessFlagBits::eDepthStencilAttachmentRead;
			qr.layout = combine_layout(qr.layout, ImageLayout::eDepthStencilReadOnlyOptimal);
		}
		if (ia & eDepthStencilWrite) {
			qr.access |= AccessFlagBits::eDepthStencilAttachmentWrite;
			qr.layout = combine_layout(qr.layout, ImageLayout::eDepthStencilAttachmentOptimal);
		}
		if (ia & eDepthStencilRW) {
			qr.stages |= PipelineStageFlagBits::eEarlyFragmentTests | PipelineStageFlagBits::eLateFragmentTests;
		}
		if (ia & (eFragmentRead | eComputeRead | eVertexRead | eRayTracingRead)) {
			qr.access |= AccessFlagBits::eShaderRead | AccessFlagBits::eAccelerationStructureReadKHR;
			qr.layout = combine_layout(qr.layout, ImageLayout::eGeneral);
		}
		if (ia & (eFragmentWrite | eComputeWrite | eRayTracingWrite)) {
			qr.access |= AccessFlagBits::eShaderWrite;
			qr.layout = combine_layout(qr.layout, ImageLayout::eGeneral);
		}
		if (ia & (eFragmentSampled | eComputeSampled | eRayTracingSampled)) {
			qr.access |= AccessFlagBits::eShaderRead;
			qr.layout = combine_layout(qr.layout, ImageLayout::eShaderReadOnlyOptimal);
		}

		if (ia & (eFragmentRW | eFragmentSampled)) {
			qr.stages |= PipelineStageFlagBits::eFragmentShader;
		}
		if (ia & (eComputeRW | eComputeSampled)) {
			qr.stages |= PipelineStageFlagBits::eComputeShader;
		}
		if (ia & (eRayTracingRW | eRayTracingSampled)) {
			qr.stages |= PipelineStageFlagBits::eRayTracingShaderKHR;
		}

		if (ia & eTransferRead) {
			qr.access |= AccessFlagBits::eTransferRead;
			qr.layout = combine_layout(qr.layout, ImageLayout::eTransferSrcOptimal);
		}
		if (ia & eTransferWrite) {
			qr.access |= AccessFlagBits::eTransferWrite;
			qr.layout = combine_layout(qr.layout, ImageLayout::eTransferDstOptimal);
		}
		if (ia & eTransferRW) {
			qr.stages |= PipelineStageFlagBits::eTransfer;
		}

		if (ia & eAttributeRead) {
			qr.access |= AccessFlagBits::eVertexAttributeRead;
			qr.stages |= PipelineStageFlagBits::eVertexInput;
		}
		if (ia & eIndexRead) {
			qr.access |= AccessFlagBits::eIndexRead;
			qr.stages |= PipelineStageFlagBits::eVertexInput;
		}
		if (ia & eIndirectRead) {
			qr.access |= AccessFlagBits::eIndirectCommandRead;
			qr.stages |= PipelineStageFlagBits::eDrawIndirect;
		}

		if (ia & eAccelerationStructureBuildRead) {
			qr.stages |= PipelineStageFlagBits::eAccelerationStructureBuildKHR;
			qr.access |= AccessFlagBits::eShaderRead | AccessFlagBits::eAccelerationStructureReadKHR;
		}
		if (ia & eAccelerationStructureBuildWrite) {
			qr.stages |= PipelineStageFlagBits::eAccelerationStructureBuildKHR;
			qr.access |= AccessFlagBits::eAccelerationStructureWriteKHR;
		}

		if (ia & eHostRead) {
			qr.access |= AccessFlagBits::eHostRead;
			qr.layout = combine_layout(qr.layout, ImageLayout::eGeneral);
		}
		if (ia & eHostWrite) {
			qr.access |= AccessFlagBits::eHostWrite;
			qr.layout = combine_layout(qr.layout, ImageLayout::eGeneral);
		}
		if (ia & eHostRW) {
			qr.stages |= PipelineStageFlagBits::eHost;
		}

		if (ia & eMemoryRead) {
			qr.access |= AccessFlagBits::eMemoryRead;
			qr.layout = combine_layout(qr.layout, ImageLayout::eGeneral);
		}
		if (ia & eMemoryWrite) {
			qr.access |= AccessFlagBits::eMemoryWrite;
			qr.layout = combine_layout(qr.layout, ImageLayout::eGeneral);
		}
		if (ia & eMemoryRW) {
			qr.stages |= PipelineStageFlagBits::eAllCommands;
		}

		if (ia & eClear) {
			qr.stages |= PipelineStageFlagBits::eTransfer;
			qr.access |= AccessFlagBits::eTransferWrite;
			qr.layout = combine_layout(qr.layout, ImageLayout::eTransferDstOptimal);
		}

		qr.domain = domain;
		return qr;
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
