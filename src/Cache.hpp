#pragma once
#include <shared_mutex>
#include <unordered_map>
#include <plf_colony.h>
#include <vulkan/vulkan.hpp>
#include "Hash.hpp"
#include "Handle.hpp"
#include "Pipeline.hpp"
#include "Program.hpp"
#include <gsl/span>

class Context;

namespace std {
	template <class BitType, class MaskType>
	struct hash<vk::Flags<BitType, MaskType>> {
		size_t operator()(vk::Flags<BitType, MaskType> const& x) const noexcept {
			size_t h = 0;
			return std::hash<MaskType>()((MaskType)x);
		}
	};
};

namespace std {
	template <class T, ptrdiff_t E>
	struct hash<gsl::span<T, E>> {
		size_t operator()(gsl::span<T, E> const& x) const noexcept {
			size_t h = 0;
			for (auto& e : x) {
				hash_combine(h, e);
			}
			return h;
		}
	};
};

namespace std {
	template <>
	struct hash<vk::VertexInputBindingDescription> {
		size_t operator()(vk::VertexInputBindingDescription const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.binding, x.inputRate, x.stride);
			return h;
		}
	};
};

namespace std {
	template <>
	struct hash<vk::VertexInputAttributeDescription> {
		size_t operator()(vk::VertexInputAttributeDescription const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.binding, x.format, x.location, x.offset);
			return h;
		}
	};
};


namespace std {
	template <>
	struct hash<vk::PipelineVertexInputStateCreateInfo> {
		size_t operator()(vk::PipelineVertexInputStateCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, gsl::span(x.pVertexBindingDescriptions, x.vertexBindingDescriptionCount), gsl::span(x.pVertexAttributeDescriptions, x.vertexAttributeDescriptionCount));
			return h;
		}
	};
};

namespace std {
	template <>
	struct hash<vk::SpecializationMapEntry> {
		size_t operator()(vk::SpecializationMapEntry const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.constantID, x.offset, x.size);
			return h;
		}
	};
};


namespace std {
	template <>
	struct hash<vk::SpecializationInfo> {
		size_t operator()(vk::SpecializationInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, gsl::span(x.pMapEntries, x.mapEntryCount), gsl::span((std::byte*)x.pData, x.dataSize));
			return h;
		}
	};
};

namespace std {
	template <>
	struct hash<vk::PipelineShaderStageCreateInfo> {
		size_t operator()(vk::PipelineShaderStageCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.pName, to_integral(x.stage), reinterpret_cast<uint64_t>((VkShaderModule)x.module));
			if (x.pSpecializationInfo) hash_combine(h, *x.pSpecializationInfo);
			return h;
		}
	};
};


namespace std {
	template <>
	struct hash<vk::PipelineInputAssemblyStateCreateInfo> {
		size_t operator()(vk::PipelineInputAssemblyStateCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.primitiveRestartEnable, to_integral(x.topology));
			return h;
		}
	};
};

namespace std {
	template <>
	struct hash<vk::PipelineTessellationStateCreateInfo> {
		size_t operator()(vk::PipelineTessellationStateCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.patchControlPoints);
			return h;
		}
	};
};


namespace std {
	template <>
	struct hash<vk::Extent2D> {
		size_t operator()(vk::Extent2D const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.width, x.height);
			return h;
		}
	};
};

namespace std {
	template <>
	struct hash<vk::Extent3D> {
		size_t operator()(vk::Extent3D const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.width, x.height, x.depth);
			return h;
		}
	};
};


namespace std {
	template <>
	struct hash<vk::Offset2D> {
		size_t operator()(vk::Offset2D const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.x, x.y);
			return h;
		}
	};
};

namespace std {
	template <>
	struct hash<vk::Rect2D> {
		size_t operator()(vk::Rect2D const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.extent, x.offset);
			return h;
		}
	};
};

namespace std {
	template <>
	struct hash<vk::Viewport> {
		size_t operator()(vk::Viewport const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.x, x.y, x.width, x.height, x.minDepth, x.maxDepth);
			return h;
		}
	};
};

namespace std {
	template <>
	struct hash<vk::PipelineViewportStateCreateInfo> {
		size_t operator()(vk::PipelineViewportStateCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags);
			if (x.pScissors) hash_combine(h, gsl::span(x.pScissors, x.scissorCount));
			else hash_combine(h, x.scissorCount);
			if (x.pViewports) hash_combine(h, gsl::span(x.pViewports, x.viewportCount));
			else hash_combine(h, x.viewportCount);
			return h;
		}
	};
};

namespace std {
	template <>
	struct hash<vk::PipelineMultisampleStateCreateInfo> {
		size_t operator()(vk::PipelineMultisampleStateCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.alphaToCoverageEnable, x.alphaToOneEnable, x.minSampleShading, x.rasterizationSamples, x.sampleShadingEnable);
			if (x.pSampleMask) hash_combine(h, *x.pSampleMask);
			return h;
		}
	};
};

namespace std {
	template <>
	struct hash<vk::StencilOpState> {
		size_t operator()(vk::StencilOpState const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.compareMask, to_integral(x.compareOp), to_integral(x.failOp), to_integral(x.depthFailOp), to_integral(x.passOp), x.reference, x.writeMask);
			return h;
		}
	};
};


namespace std {
	template <>
	struct hash<vk::PipelineDepthStencilStateCreateInfo> {
		size_t operator()(vk::PipelineDepthStencilStateCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.back, x.front, x.depthBoundsTestEnable, to_integral(x.depthCompareOp), x.depthTestEnable, x.depthWriteEnable, x.maxDepthBounds, x.minDepthBounds, x.stencilTestEnable);
			return h;
		}
	};
};

namespace std {
	template <>
	struct hash<vk::PipelineColorBlendAttachmentState> {
		size_t operator()(vk::PipelineColorBlendAttachmentState const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, to_integral(x.alphaBlendOp), x.blendEnable, to_integral(x.colorBlendOp), to_integral(x.dstAlphaBlendFactor), to_integral(x.srcAlphaBlendFactor), to_integral(x.dstColorBlendFactor), to_integral(x.srcColorBlendFactor));
			return h;
		}
	};
};


namespace std {
	template <>
	struct hash<vk::PipelineColorBlendStateCreateInfo> {
		size_t operator()(vk::PipelineColorBlendStateCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, gsl::span(x.pAttachments, x.attachmentCount), x.blendConstants, to_integral(x.logicOp), x.logicOpEnable);
			return h;
		}
	};
};

namespace std {
	template <>
	struct hash<vk::DynamicState> {
		size_t operator()(vk::DynamicState const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, to_integral(x));
			return h;
		}
	};
};


namespace std {
	template <>
	struct hash<vk::PipelineDynamicStateCreateInfo> {
		size_t operator()(vk::PipelineDynamicStateCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, gsl::span(x.pDynamicStates, x.dynamicStateCount));
			return h;
		}
	};
};



namespace std {
	template <>
	struct hash<vk::GraphicsPipelineCreateInfo> {
		size_t operator()(vk::GraphicsPipelineCreateInfo const & x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, gsl::span(x.pStages, x.stageCount));
			if (x.pVertexInputState) hash_combine(h, *x.pVertexInputState);
			if (x.pInputAssemblyState) hash_combine(h, *x.pInputAssemblyState);
			if (x.pTessellationState) hash_combine(h, *x.pTessellationState);
			if (x.pViewportState) hash_combine(h, *x.pViewportState);
			if (x.pMultisampleState) hash_combine(h, *x.pMultisampleState);
			if (x.pDepthStencilState) hash_combine(h, *x.pDepthStencilState);
			if (x.pColorBlendState) hash_combine(h, *x.pColorBlendState);
			if (x.pDynamicState) hash_combine(h, *x.pDynamicState);
			hash_combine(h,
				reinterpret_cast<uint64_t>((VkPipelineLayout)x.layout),
				reinterpret_cast<uint64_t>((VkRenderPass)x.renderPass),
				x.subpass,
				reinterpret_cast<uint64_t>((VkPipeline)x.basePipelineHandle),
				x.basePipelineIndex);
			return h;
		}
	};
};

#include <optional>

namespace vuk {
	template<class T>
	struct deep_equal_span : public gsl::span<T> {
		using size_type = typename gsl::span<T>::size_type;

		using gsl::span<T>::span;
		using gsl::span<T>::size;

		bool operator==(const deep_equal_span& o) const {
			if (size() != o.size()) return false;
			for (size_type i = 0; i < size(); i++) {
				if ((*this)[i] != o[i]) return false;
			}
			return true;
		}
	};

	template <class Type>
	deep_equal_span(Type*, size_t)->deep_equal_span<Type>;

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

	struct FramebufferCreateInfo : public vk::FramebufferCreateInfo {
		std::vector<vuk::ImageView> attachments;

		bool operator==(const FramebufferCreateInfo& o) const {
			return std::tie(flags, attachments, width, height, renderPass, layers) ==
				std::tie(o.flags, o.attachments, o.width, o.height, o.renderPass, o.layers);
		}
	};
}

namespace std {
	template <>
	struct hash<vk::AttachmentDescription> {
		size_t operator()(vk::AttachmentDescription const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.initialLayout, x.finalLayout, x.format, x.loadOp, x.stencilLoadOp, x.storeOp, x.stencilStoreOp, x.samples);
			return h;
		}
	};

	template <>
	struct hash<vk::AttachmentReference> {
		size_t operator()(vk::AttachmentReference const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.attachment, x.layout);
			return h;
		}
	};

	template <>
	struct hash<vk::SubpassDependency> {
		size_t operator()(vk::SubpassDependency const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.dependencyFlags, x.srcAccessMask, x.srcStageMask, x.srcSubpass, x.dstAccessMask, x.dstStageMask, x.dstSubpass);
			return h;
		}
	};

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
};

namespace std {
	template <>
	struct hash<vk::ImageCreateInfo> {
		size_t operator()(vk::ImageCreateInfo const & x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.arrayLayers, x.extent, to_integral(x.format), to_integral(x.imageType), to_integral(x.initialLayout), x.mipLevels, gsl::span(x.pQueueFamilyIndices, x.queueFamilyIndexCount), to_integral(x.samples), to_integral(x.sharingMode), to_integral(x.tiling), x.usage);
			return h;
		}
	};
	
	template <>
	struct hash<vk::ImageSubresourceRange> {
		size_t operator()(vk::ImageSubresourceRange const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.aspectMask, x.baseArrayLayer, x.baseMipLevel, x.layerCount, x.levelCount);
			return h;
		}
	};

	template <>
	struct hash<vk::ComponentMapping> {
		size_t operator()(vk::ComponentMapping const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, to_integral(x.r), to_integral(x.g), to_integral(x.b), to_integral(x.a));
			return h;
		}
	};

	template <>
	struct hash<vk::ImageViewCreateInfo> {
		size_t operator()(vk::ImageViewCreateInfo const & x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.components, to_integral(x.format), 
				reinterpret_cast<uint64_t>((VkImage)x.image),
				x.subresourceRange, to_integral(x.viewType));
			return h;
		}
	};
};

#define VUK_MAX_BINDINGS 16
#include <bitset>

namespace vuk {
	struct DescriptorSetLayoutAllocInfo {
		std::array<size_t, VkDescriptorType::VK_DESCRIPTOR_TYPE_END_RANGE> descriptor_counts = {};
		vk::DescriptorSetLayout layout;

		bool operator==(const DescriptorSetLayoutAllocInfo& o) const {
			// TODO!
			return layout == o.layout;
		}
	};

	struct DescriptorImageInfo {
		vuk::Sampler sampler;
		vuk::ImageView image_view;
		vk::ImageLayout image_layout;

		bool operator==(const DescriptorImageInfo& o) const {
			return std::tie(sampler, image_view, image_layout) == std::tie(o.sampler, o.image_view, o.image_layout);
		}

		operator vk::DescriptorImageInfo() const {
			return {sampler.payload, image_view.payload, image_layout};
		}
	};

	// use hand rolled variant to control bits
	// memset to clear out the union
#pragma pack(push, 1)
	struct DescriptorBinding {
		DescriptorBinding() {}

		vk::DescriptorType type;
		union {
			struct Unbound {} unbound;
			vk::DescriptorBufferInfo buffer;
			vuk::DescriptorImageInfo image;
		};

		bool operator==(const DescriptorBinding& o) const {
			if (type != o.type) return false;
			switch (type) {
			case vk::DescriptorType::eUniformBuffer:
			case vk::DescriptorType::eStorageBuffer:
				return buffer == o.buffer;
				break;
			case vk::DescriptorType::eSampledImage:
			case vk::DescriptorType::eSampler:
			case vk::DescriptorType::eCombinedImageSampler:
				return image == o.image;
				break;
			default:
				printf("%d\n", to_integral(type));
				assert(0);
			}

		}
	};
#pragma pack(pop)
	struct SetBinding {
		std::bitset<VUK_MAX_BINDINGS> used = {};
		std::array<DescriptorBinding, VUK_MAX_BINDINGS> bindings;
		DescriptorSetLayoutAllocInfo layout_info = {};

		bool operator==(const SetBinding& o) const {
			if (layout_info != o.layout_info) return false;
			for (size_t i = 0; i < VUK_MAX_BINDINGS; i++) {
				if (!used[i]) continue;
				if (bindings[i] != o.bindings[i]) return false;
			}
			return true;
		}
	};
}

namespace std {
	template <>
	struct hash<vuk::SetBinding> {
		size_t operator()(vuk::SetBinding const & x) const noexcept {
			// TODO: should we hash in layout too?
			unsigned long leading_zero = 0;
			auto mask = x.used.to_ulong();
			auto is_null = _BitScanReverse(&leading_zero, mask);
			leading_zero++;
			return ::hash::fnv1a::hash(reinterpret_cast<const char*>(&x.bindings[0]), leading_zero * sizeof(vuk::DescriptorBinding));
		}
	};
};

namespace std {
	template <>
	struct hash<vk::SamplerCreateInfo> {
		size_t operator()(vk::SamplerCreateInfo const & x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.addressModeU, x.addressModeV, x.addressModeW, x.anisotropyEnable, x.borderColor, x.compareEnable, x.compareOp, x.magFilter, x.maxAnisotropy, x.maxLod, x.minFilter, x.minLod, x.mipLodBias, x.mipmapMode, x.unnormalizedCoordinates); 
			return h;
		}
	};
};

#define VUK_MAX_SETS 8

namespace vuk {
	class Context;
	class InflightContext;
	class PerThreadContext;

	template<class T>
	struct create_info;

	template<class T>
	using create_info_t = typename create_info<T>::type;

	struct PipelineInfo {
		vk::Pipeline pipeline;
		vk::PipelineLayout pipeline_layout;
		std::array<DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> layout_info;
	};

	template<> struct create_info<PipelineInfo> {
		using type = vuk::PipelineCreateInfo;
	};

	template<> struct create_info<vk::RenderPass> {
		using type = vuk::RenderPassCreateInfo;
	};

	struct DescriptorPool {
		std::vector<vk::DescriptorPool> pools;
		size_t pool_needle = 0;
		size_t sets_allocated = 0;
		std::vector<vk::DescriptorSet> free_sets;

		vk::DescriptorPool get_pool(PerThreadContext& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info);
		vk::DescriptorSet acquire(PerThreadContext& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info);
	};
}

namespace std {
	template <>
	struct hash<vuk::DescriptorSetLayoutAllocInfo> {
		size_t operator()(vuk::DescriptorSetLayoutAllocInfo const & x) const noexcept {
			size_t h = 0;
			// TODO: should use vuk::DescriptorSetLayout here
			hash_combine(h, ::hash::fnv1a::hash((const char *)&x.descriptor_counts[0], x.descriptor_counts.size() * sizeof(x.descriptor_counts[0])), (VkDescriptorSetLayout)x.layout); 
			return h;
		}
	};
};

namespace vuk {
	struct ShaderModuleCreateInfo {
		std::string source;
		std::string filename;

		bool operator==(const ShaderModuleCreateInfo& o) const {
			return source == o.source;
		}
	};

	struct ShaderModule {
		vk::ShaderModule shader_module;
		vuk::Program reflection_info;
		vk::ShaderStageFlagBits stage;
	};
}

namespace std {
	template <>
	struct hash<vuk::ShaderModuleCreateInfo> {
		size_t operator()(vuk::ShaderModuleCreateInfo const & x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.source); // filename is intentionally not hashed in
			return h;
		}
	};
};


namespace vuk {
	template<> struct create_info<vuk::DescriptorPool> {
		using type = vuk::DescriptorSetLayoutAllocInfo;
	};

	struct DescriptorSet {
		vk::DescriptorSet descriptor_set;
		DescriptorSetLayoutAllocInfo layout_info;
	};

	template<> struct create_info<vuk::DescriptorSet> {
		using type = vuk::SetBinding;
	};

	template<> struct create_info<vk::Framebuffer> {
		using type = vuk::FramebufferCreateInfo;
	};

	template<> struct create_info<vk::Sampler> {
		using type = vk::SamplerCreateInfo;
	};

	template<> struct create_info<vuk::ShaderModule> {
		using type = vuk::ShaderModuleCreateInfo;
	};

	template<> struct create_info<vuk::DescriptorSetLayoutAllocInfo> {
		using type = vuk::DescriptorSetLayoutCreateInfo;
	};

	template<> struct create_info<vk::PipelineLayout> {
		using type = vuk::PipelineLayoutCreateInfo;
	};


	template<class T>
	struct Cache {
		struct LRUEntry {
			T* ptr;
			unsigned last_use_frame;
		};

		Context& ctx;
		plf::colony<T> pool;
		std::unordered_map<create_info_t<T>, LRUEntry> lru_map; // possibly vector_map or an intrusive map
		std::shared_mutex cache_mtx;

		Cache(Context& ctx) : ctx(ctx) {}
		~Cache();

		struct PFView {
			InflightContext& ifc;
			Cache& cache;

			PFView(InflightContext& ifc, Cache<T>& cache) : ifc(ifc), cache(cache) {}
		};

		struct PFPTView {
			PerThreadContext& ptc;
			PFView& view;

			PFPTView(PerThreadContext& ptc, PFView& view) : ptc(ptc), view(view) {}
			T& acquire(const create_info_t<T>& ci);
			void collect(size_t threshold);
		};
	};

	template<class T, size_t FC>
	struct PerFrameCache {
		struct LRUEntry {
			T* ptr;
			unsigned last_use_frame;
		};

		Context& ctx;
		struct PerFrame {
			plf::colony<T> pool;
			// possibly vector_map or an intrusive map
			std::unordered_map<create_info_t<T>, LRUEntry> lru_map;

			std::shared_mutex cache_mtx;
		};
		std::array<PerFrame, FC> data;

		PerFrameCache(Context& ctx) : ctx(ctx) {}
		~PerFrameCache();

		struct PFView {
			InflightContext& ifc;
			PerFrameCache& cache;

			PFView(InflightContext& ifc, PerFrameCache& cache) : ifc(ifc), cache(cache) {}
		};

		struct PFPTView {
			PerThreadContext& ptc;
			PFView& view;

			PFPTView(PerThreadContext& ptc, PFView& view) : ptc(ptc), view(view) {}
			T& acquire(const create_info_t<T>& ci);
			void collect(size_t threshold);
		};

	};
}
