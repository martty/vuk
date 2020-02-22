#pragma once
#include <shared_mutex>
#include <unordered_map>
#include <plf_colony.h>
#include <vulkan/vulkan.hpp>
#include "Hash.hpp"
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

namespace std {
	template <>
	struct hash<vk::RenderPassCreateInfo> {
		size_t operator()(vk::RenderPassCreateInfo const & x) const noexcept {
			return x.attachmentCount; // TODO: ...
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
};

namespace std {
	template <>
	struct hash<vk::ImageSubresourceRange> {
		size_t operator()(vk::ImageSubresourceRange const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.aspectMask, x.baseArrayLayer, x.baseMipLevel, x.layerCount, x.levelCount);
			return h;
		}
	};
};

namespace std {
	template <>
	struct hash<vk::ComponentMapping> {
		size_t operator()(vk::ComponentMapping const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, to_integral(x.r), to_integral(x.g), to_integral(x.b), to_integral(x.a));
			return h;
		}
	};
};


namespace std {
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


namespace vuk {
	class Context;
	class InflightContext;

	template<class T>
	struct create_info;
	
	template<class T>
	using create_info_t = typename create_info<T>::type;

	template<class T>
	T create(Context& ctx, create_info_t<T> cinfo);

	template<> struct create_info<vk::Pipeline> {
		using type = vk::GraphicsPipelineCreateInfo;
	};

	template<> struct create_info<vk::RenderPass> {
		using type = vk::RenderPassCreateInfo;
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

		struct View {
			InflightContext& ifc;
			Cache& cache;

			View(InflightContext& ifc, Cache<T>& cache) : ifc(ifc), cache(cache) {}
			T acquire(create_info_t<T> ci);
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

		struct View {
			InflightContext& ifc;
			PerFrameCache& cache;

			View(InflightContext& ifc, PerFrameCache& cache) : ifc(ifc), cache(cache) {}
			T acquire(create_info_t<T> ci);
			void collect(size_t threshold);
		};
	};
}
