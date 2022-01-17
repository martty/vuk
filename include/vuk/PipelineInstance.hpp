#pragma once

#include "../src/CreateInfo.hpp"
#include "Pipeline.hpp"
#include "vuk/Config.hpp"
#include "vuk/FixedVector.hpp"
#include "vuk/Hash.hpp"
#include <bit>

inline bool operator==(VkSpecializationMapEntry const& lhs, VkSpecializationMapEntry const& rhs) noexcept {
	return (lhs.constantID == rhs.constantID) && (lhs.offset == rhs.offset) && (lhs.size == rhs.size);
}

namespace vuk {
	struct PipelineInstanceCreateInfo {
		PipelineBaseInfo* base;
		VkRenderPass render_pass;
		DynamicStateFlags dynamic_state_flags;
		uint16_t extended_size = 0;
		struct RecordsExist {
			uint32_t nonzero_subpass : 1;
			uint32_t vertex_input : 1;
			uint32_t color_blend_attachments : 1;
			uint32_t broadcast_color_blend_attachment_0 : 1;
			uint32_t logic_op : 1;
			uint32_t blend_constants : 1;
			uint32_t specialization_constants : 1;
			uint32_t viewports : 1;
			uint32_t scissors : 1;
			uint32_t non_trivial_raster_state : 1;
			uint32_t depth_stencil : 1;
			uint32_t depth_bias : 1;
			uint32_t depth_bounds : 1;
			uint32_t stencil_state : 1;
			uint32_t line_width_not_1 : 1;
			uint32_t more_than_one_sample : 1;
		} records = {};
		uint8_t attachmentCount : std::bit_width(VUK_MAX_COLOR_ATTACHMENTS); // up to VUK_MAX_COLOR_ATTACHMENTS attachments
		// input assembly state
		VkPrimitiveTopology topology : std::bit_width(10u);
		bool primitive_restart_enable : 1;
		VkCullModeFlags cullMode : 2;
		union {
			std::byte inline_data[80];
			std::byte* extended_data;
		};

#pragma pack(push, 1)
		struct VertexInputBindingDescription {
			uint32_t stride : 31;
			uint32_t inputRate : 1;
			uint8_t binding;
		};
		struct VertexInputAttributeDescription {
			Format format;
			uint32_t offset;
			uint8_t location;
			uint8_t binding;
		};

		struct PipelineColorBlendAttachmentState {
			Bool32 blendEnable : 1;
			BlendFactor srcColorBlendFactor : std::bit_width(18u);
			BlendFactor dstColorBlendFactor : std::bit_width(18u);
			BlendOp colorBlendOp : std::bit_width(5u); // not supporting blend op zoo yet
			BlendFactor srcAlphaBlendFactor : std::bit_width(18u);
			BlendFactor dstAlphaBlendFactor : std::bit_width(18u);
			BlendOp alphaBlendOp : std::bit_width(5u); // not supporting blend op zoo yet
			uint32_t colorWriteMask : 4;
		};

		// blend state
		struct BlendStateLogicOp {
			VkLogicOp logic_op : std::bit_width(16u);
		};

		struct RasterizationState {
			uint8_t depthClampEnable : 1;
			uint8_t rasterizerDiscardEnable : 1;
			uint8_t polygonMode : 2;
			uint8_t frontFace : 1;
		};

		struct DepthBias {
			float depthBiasConstantFactor;
			float depthBiasClamp;
			float depthBiasSlopeFactor;
		};

		struct Depth {
			uint8_t depthTestEnable : 1;
			uint8_t depthWriteEnable : 1;
			uint8_t depthCompareOp : std::bit_width(7u);
		};
		struct DepthBounds {
			float minDepthBounds;
			float maxDepthBounds;
		};
		struct Stencil {
			VkStencilOpState front;
			VkStencilOpState back;
		};

		struct Multisample {
			VkSampleCountFlagBits rasterization_samples : 7;
			bool sample_shading_enable : 1;
			// pSampleMask not yet supported
			bool alpha_to_coverage_enable : 1;
			bool alpha_to_one_enable : 1;
			float min_sample_shading;
		};

#pragma pack(pop)

		bool operator==(const PipelineInstanceCreateInfo& o) const noexcept {
			return base == o.base && render_pass == o.render_pass && extended_size == o.extended_size &&
			       (is_inline() ? (memcmp(inline_data, o.inline_data, extended_size) == 0) : (memcmp(extended_data, o.extended_data, extended_size) == 0));
		}

		bool is_inline() const noexcept {
			return extended_size <= sizeof(inline_data);
		}
	};

	struct PipelineInfo {
		VkPipeline pipeline;
		VkPipelineLayout pipeline_layout;
		std::array<DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> layout_info;
	};

	template<>
	struct create_info<PipelineInfo> {
		using type = vuk::PipelineInstanceCreateInfo;
	};

	struct ComputePipelineInstanceCreateInfo {
		PipelineBaseInfo* base;

		std::array<std::byte, VUK_MAX_SPECIALIZATIONCONSTANT_DATA> specialization_constant_data;
		vuk::fixed_vector<VkSpecializationMapEntry, VUK_MAX_SPECIALIZATIONCONSTANT_RANGES> specialization_map_entries;
		VkSpecializationInfo specialization_info = {};

		bool operator==(const ComputePipelineInstanceCreateInfo& o) const noexcept {
			return base == o.base && specialization_map_entries == o.specialization_map_entries && specialization_info.dataSize == o.specialization_info.dataSize &&
			       memcmp(specialization_constant_data.data(), o.specialization_constant_data.data(), specialization_info.dataSize) == 0;
		}
	};

	struct ComputePipelineInfo : PipelineInfo {
		std::array<unsigned, 3> local_size;
	};

	template<>
	struct create_info<ComputePipelineInfo> {
		using type = vuk::ComputePipelineInstanceCreateInfo;
	};
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::PipelineInstanceCreateInfo> {
		size_t operator()(vuk::PipelineInstanceCreateInfo const& x) const noexcept {
			size_t h = 0;
			auto ext_hash = x.is_inline() ? robin_hood::hash_bytes(x.inline_data, x.extended_size) : robin_hood::hash_bytes(x.extended_data, x.extended_size);
			hash_combine(h, x.base, reinterpret_cast<uint64_t>((VkRenderPass)x.render_pass), x.extended_size, ext_hash);
			return h;
		}
	};

	template<>
	struct hash<VkSpecializationMapEntry> {
		size_t operator()(VkSpecializationMapEntry const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.constantID, x.offset, x.size);
			return h;
		}
	};

	template<>
	struct hash<vuk::ComputePipelineInstanceCreateInfo> {
		size_t operator()(vuk::ComputePipelineInstanceCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.base, robin_hood::hash_bytes(x.specialization_constant_data.data(), x.specialization_info.dataSize), x.specialization_map_entries);
			return h;
		}
	};

	template<>
	struct hash<VkPushConstantRange> {
		size_t operator()(VkPushConstantRange const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.offset, x.size, (VkShaderStageFlags)x.stageFlags);
			return h;
		}
	};

	template<>
	struct hash<vuk::PipelineLayoutCreateInfo> {
		size_t operator()(vuk::PipelineLayoutCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.pcrs, x.dslcis);
			return h;
		}
	};
}; // namespace std
