#pragma once

#include "Cache.hpp"
#include "CreateInfo.hpp"
#include "vuk/Image.hpp"
#include "vuk/ShortAlloc.hpp"
#include "vuk/Types.hpp"
#include "vuk/vuk_fwd.hpp"

#include <optional>
#include <tuple>
#include <vector>

inline bool operator==(VkAttachmentDescription const& lhs, VkAttachmentDescription const& rhs) noexcept {
	return (lhs.flags == rhs.flags) && (lhs.format == rhs.format) && (lhs.samples == rhs.samples) && (lhs.loadOp == rhs.loadOp) && (lhs.storeOp == rhs.storeOp) &&
	       (lhs.stencilLoadOp == rhs.stencilLoadOp) && (lhs.stencilStoreOp == rhs.stencilStoreOp) && (lhs.initialLayout == rhs.initialLayout) &&
	       (lhs.finalLayout == rhs.finalLayout);
}

inline bool operator==(VkSubpassDependency const& lhs, VkSubpassDependency const& rhs) noexcept {
	return (lhs.srcSubpass == rhs.srcSubpass) && (lhs.dstSubpass == rhs.dstSubpass) && (lhs.srcStageMask == rhs.srcStageMask) &&
	       (lhs.dstStageMask == rhs.dstStageMask) && (lhs.srcAccessMask == rhs.srcAccessMask) && (lhs.dstAccessMask == rhs.dstAccessMask) &&
	       (lhs.dependencyFlags == rhs.dependencyFlags);
}

inline bool operator==(VkAttachmentReference const& lhs, VkAttachmentReference const& rhs) noexcept {
	return (lhs.attachment == rhs.attachment) && (lhs.layout == rhs.layout);
}

namespace vuk {
	struct SubpassDescription : public VkSubpassDescription {
		SubpassDescription() : VkSubpassDescription{} {}
		bool operator==(const SubpassDescription& o) const noexcept {
			return std::tie(flags, pipelineBindPoint) == std::tie(o.flags, o.pipelineBindPoint);
		}
	};

	struct RenderPassCreateInfo : public VkRenderPassCreateInfo {
		RenderPassCreateInfo() : VkRenderPassCreateInfo{ .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO } {}
		std::vector<VkAttachmentDescription> attachments;
		std::vector<vuk::SubpassDescription> subpass_descriptions;
		std::vector<VkSubpassDependency> subpass_dependencies;
		std::vector<VkAttachmentReference> color_refs;
		std::vector<VkAttachmentReference> resolve_refs;
		std::optional<VkAttachmentReference> ds_ref;

		bool operator==(const RenderPassCreateInfo& o) const noexcept {
			return std::forward_as_tuple(flags, attachments, subpass_descriptions, subpass_dependencies, color_refs, ds_ref, resolve_refs) ==
			       std::forward_as_tuple(
			           o.flags, o.attachments, o.subpass_descriptions, o.subpass_dependencies, o.color_refs, o.ds_ref, o.resolve_refs);
		}
	};

	template<>
	struct create_info<VkRenderPass> {
		using type = vuk::RenderPassCreateInfo;
	};

	struct FramebufferCreateInfo : public VkFramebufferCreateInfo {
		FramebufferCreateInfo() : VkFramebufferCreateInfo{ .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO } {}
		std::vector<vuk::ImageView> attachments;
		vuk::Samples sample_count = vuk::Samples::eInfer;

		bool operator==(const FramebufferCreateInfo& o) const noexcept {
			return std::tie(flags, attachments, width, height, renderPass, layers, sample_count) ==
			       std::tie(o.flags, o.attachments, o.width, o.height, o.renderPass, o.layers, o.sample_count);
		}
	};

	template<>
	struct create_info<VkFramebuffer> {
		using type = vuk::FramebufferCreateInfo;
	};
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::SubpassDescription> {
		size_t operator()(vuk::SubpassDescription const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.pipelineBindPoint);
			return h;
		}
	};

	template<>
	struct hash<vuk::RenderPassCreateInfo> {
		size_t operator()(vuk::RenderPassCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.attachments, x.color_refs, x.ds_ref, x.subpass_dependencies, x.subpass_descriptions);
			return h;
		}
	};

	template<>
	struct hash<vuk::FramebufferCreateInfo> {
		size_t operator()(vuk::FramebufferCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.attachments, x.width, x.height, x.layers);
			return h;
		}
	};
} // namespace std
