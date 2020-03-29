#pragma once

#include <vulkan/vulkan.hpp>
#include "vuk_fwd.hpp"
#include "CreateInfo.hpp"
#include "Types.hpp"

namespace vuk {
	struct SubpassDescription : public vk::SubpassDescription {
		bool operator==(const SubpassDescription& o) const {
			return std::tie(flags, pipelineBindPoint) ==
				std::tie(o.flags, o.pipelineBindPoint);
		}
	};

	struct RenderPassCreateInfo : public vk::RenderPassCreateInfo {
		std::vector<vk::AttachmentDescription> attachments;
		std::vector<vuk::SubpassDescription> subpass_descriptions;
		std::vector<vk::SubpassDependency> subpass_dependencies;
		std::vector<vk::AttachmentReference> color_refs;
		std::vector<vk::AttachmentReference> resolve_refs;
		std::vector<std::optional<vk::AttachmentReference>> ds_refs;
		std::vector<size_t> color_ref_offsets;

		bool operator==(const RenderPassCreateInfo& o) const {
			return std::forward_as_tuple(flags, attachments, subpass_descriptions, subpass_dependencies, color_refs, color_ref_offsets, ds_refs, resolve_refs) ==
				std::forward_as_tuple(o.flags, o.attachments, o.subpass_descriptions, o.subpass_dependencies, o.color_refs, o.color_ref_offsets, o.ds_refs, o.resolve_refs);
		}
	};

	template<> struct create_info<vk::RenderPass> {
		using type = vuk::RenderPassCreateInfo;
	};

	struct FramebufferCreateInfo : public vk::FramebufferCreateInfo {
		std::vector<vuk::ImageView> attachments;

		bool operator==(const FramebufferCreateInfo& o) const {
			return std::tie(flags, attachments, width, height, renderPass, layers) ==
				std::tie(o.flags, o.attachments, o.width, o.height, o.renderPass, o.layers);
		}
	};

	template<> struct create_info<vk::Framebuffer> {
		using type = vuk::FramebufferCreateInfo;
	};

	template<> struct create_info<vk::Sampler> {
		using type = vk::SamplerCreateInfo;
	};
}

namespace std {
	template <>
	struct hash<vuk::SubpassDescription> {
		size_t operator()(vuk::SubpassDescription const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.pipelineBindPoint);
			return h;
		}
	};

	template <>
	struct hash<vuk::RenderPassCreateInfo> {
		size_t operator()(vuk::RenderPassCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.attachments, x.color_refs, x.color_ref_offsets, x.ds_refs, x.subpass_dependencies, x.subpass_descriptions);
			return h;
		}
	};

	template <>
	struct hash<vuk::FramebufferCreateInfo> {
		size_t operator()(vuk::FramebufferCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.attachments, x.width, x.height, x.layers);
			return h;
		}
	};
}