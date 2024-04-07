#pragma once

#include "vuk/RelSpan.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/ShortAlloc.hpp"
#include <optional>

namespace vuk {
	inline bool is_write_access(Access ia) {
		constexpr uint64_t write_mask = eColorResolveWrite | eColorWrite | eDepthStencilWrite | eFragmentWrite | eTransferWrite | eComputeWrite | eHostWrite |
		                                eMemoryWrite | eRayTracingWrite | eAccelerationStructureBuildWrite | eClear;
		return ia & write_mask;
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
		if ((a == ImageLayout::eDepthStencilReadOnlyOptimal && b == ImageLayout::eDepthStencilAttachmentOptimal) ||
		    (b == ImageLayout::eDepthStencilReadOnlyOptimal && a == ImageLayout::eDepthStencilAttachmentOptimal)) {
			return ImageLayout::eAttachmentOptimalKHR;
		}
		if ((a == ImageLayout::eReadOnlyOptimalKHR && b == ImageLayout::eAttachmentOptimalKHR) ||
		    (b == ImageLayout::eReadOnlyOptimalKHR && a == ImageLayout::eAttachmentOptimalKHR)) {
			return ImageLayout::eAttachmentOptimalKHR;
		}
		assert(a != ImageLayout::ePresentSrcKHR && b != ImageLayout::ePresentSrcKHR);
		return ImageLayout::eGeneral;
	}

	inline ResourceUse to_use(Access ia) {
		constexpr uint64_t color_read = eColorResolveRead | eColorRead;
		constexpr uint64_t color_write = eColorResolveWrite | eColorWrite;
		constexpr uint64_t color_rw = color_read | color_write;

		ResourceUse qr{};
		if (ia & color_read) {
			qr.access |= AccessFlagBits::eColorAttachmentRead;
		}
		if (ia & color_write) {
			qr.access |= AccessFlagBits::eColorAttachmentWrite;
		}
		if (ia & color_rw) {
			qr.stages |= PipelineStageFlagBits::eColorAttachmentOutput;
			qr.layout = combine_layout(qr.layout, ImageLayout::eAttachmentOptimalKHR);
		}
		if (ia & eDepthStencilRead) {
			qr.access |= AccessFlagBits::eDepthStencilAttachmentRead;
			qr.layout = combine_layout(qr.layout, ImageLayout::eAttachmentOptimalKHR);
		}
		if (ia & eDepthStencilWrite) {
			qr.access |= AccessFlagBits::eDepthStencilAttachmentWrite;
			qr.layout = combine_layout(qr.layout, ImageLayout::eAttachmentOptimalKHR);
		}
		if (ia & eDepthStencilRW) {
			qr.stages |= PipelineStageFlagBits::eEarlyFragmentTests | PipelineStageFlagBits::eLateFragmentTests;
		}
		if (ia & (eFragmentRead | eComputeRead | eVertexRead | eRayTracingRead)) {
			qr.access |= AccessFlagBits::eShaderRead;
			qr.layout = combine_layout(qr.layout, ImageLayout::eGeneral);
		}
		if (ia & eRayTracingRead) {
			qr.access |= AccessFlagBits::eAccelerationStructureReadKHR;
			qr.layout = combine_layout(qr.layout, ImageLayout::eGeneral);
		}
		if (ia & (eFragmentWrite | eComputeWrite | eRayTracingWrite)) {
			qr.access |= AccessFlagBits::eShaderWrite;
			qr.layout = combine_layout(qr.layout, ImageLayout::eGeneral);
		}
		if (ia & (eFragmentSampled | eComputeSampled | eRayTracingSampled)) {
			qr.access |= AccessFlagBits::eShaderRead;
			qr.layout = combine_layout(qr.layout, ImageLayout::eReadOnlyOptimalKHR);
		}

		if (ia & (eVertexRead | eVertexSampled)) {
			qr.stages |= PipelineStageFlagBits::eVertexShader;
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
			qr.access |= AccessFlagBits::eAccelerationStructureReadKHR;
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

		if (ia & ePresent) {
			qr.stages = PipelineStageFlagBits::eNone;
			qr.access = {};
			qr.layout = ImageLayout::ePresentSrcKHR;
		}

		return qr;
	}

	// not all domains can support all stages, this function corrects stage flags
	inline void scope_to_domain(VkPipelineStageFlags2KHR& src, DomainFlags flags) {
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
			src &= (VkPipelineStageFlags2KHR)~0b11111111110;
		}
		if (remove & DomainFlagBits::eComputeQueue) {
			src &= (VkPipelineStageFlags2KHR)~0b100000000000;
		}
	}

	inline bool is_framebuffer_attachment(const Access acc) {
		switch (acc) {
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
		case vuk::ImageLayout::eAttachmentOptimal:
			return true;
		default:
			return false;
		}
	}

	inline bool is_write_access(ResourceUse u) {
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

	inline bool is_readonly_access(ResourceUse u) {
		if (u.access == vuk::AccessFlagBits{})
			return false;
		return !is_write_access(u);
	}

	inline bool is_transfer_access(Access a) {
		return (a & eTransferRW);
	}

	inline bool is_storage_access(Access a) {
		return (a & (eComputeRW | eVertexRead | eFragmentRW | eRayTracingRW | eHostRW));
	}

	inline bool is_readonly_access(Access a) {
		return a & ~(eTransferRW | eComputeRW | eVertexRead | eFragmentRW | eRayTracingRW | eHostRW);
	}

	struct Acquire {
		ResourceUse src_use;
		DomainFlagBits initial_domain = DomainFlagBits::eAny;
		uint64_t initial_visibility;
		bool unsynchronized = false;
	};
} // namespace vuk
