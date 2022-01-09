#pragma once

#include <vuk/Config.hpp>

namespace vuk {
	enum class PrimitiveTopology {
		ePointList = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
		eLineList = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
		eLineStrip = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
		eTriangleList = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		eTriangleStrip = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
		eTriangleFan = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
		eLineListWithAdjacency = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
		eLineStripWithAdjacency = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY,
		eTriangleListWithAdjacency = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
		eTriangleStripWithAdjacency = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY,
		ePatchList = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST
	};

	enum class BlendFactor {
		eZero = VK_BLEND_FACTOR_ZERO,
		eOne = VK_BLEND_FACTOR_ONE,
		eSrcColor = VK_BLEND_FACTOR_SRC_COLOR,
		eOneMinusSrcColor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
		eDstColor = VK_BLEND_FACTOR_DST_COLOR,
		eOneMinusDstColor = VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
		eSrcAlpha = VK_BLEND_FACTOR_SRC_ALPHA,
		eOneMinusSrcAlpha = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		eDstAlpha = VK_BLEND_FACTOR_DST_ALPHA,
		eOneMinusDstAlpha = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
		eConstantColor = VK_BLEND_FACTOR_CONSTANT_COLOR,
		eOneMinusConstantColor = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
		eConstantAlpha = VK_BLEND_FACTOR_CONSTANT_ALPHA,
		eOneMinusConstantAlpha = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
		eSrcAlphaSaturate = VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
		eSrc1Color = VK_BLEND_FACTOR_SRC1_COLOR,
		eOneMinusSrc1Color = VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
		eSrc1Alpha = VK_BLEND_FACTOR_SRC1_ALPHA,
		eOneMinusSrc1Alpha = VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA
	};

	enum class BlendOp {
		eAdd = VK_BLEND_OP_ADD,
		eSubtract = VK_BLEND_OP_SUBTRACT,
		eReverseSubtract = VK_BLEND_OP_REVERSE_SUBTRACT,
		eMin = VK_BLEND_OP_MIN,
		eMax = VK_BLEND_OP_MAX,
		/*eZeroEXT = VK_BLEND_OP_ZERO_EXT,
		eSrcEXT = VK_BLEND_OP_SRC_EXT,
		eDstEXT = VK_BLEND_OP_DST_EXT,
		eSrcOverEXT = VK_BLEND_OP_SRC_OVER_EXT,
		eDstOverEXT = VK_BLEND_OP_DST_OVER_EXT,
		eSrcInEXT = VK_BLEND_OP_SRC_IN_EXT,
		eDstInEXT = VK_BLEND_OP_DST_IN_EXT,
		eSrcOutEXT = VK_BLEND_OP_SRC_OUT_EXT,
		eDstOutEXT = VK_BLEND_OP_DST_OUT_EXT,
		eSrcAtopEXT = VK_BLEND_OP_SRC_ATOP_EXT,
		eDstAtopEXT = VK_BLEND_OP_DST_ATOP_EXT,
		eXorEXT = VK_BLEND_OP_XOR_EXT,
		eMultiplyEXT = VK_BLEND_OP_MULTIPLY_EXT,
		eScreenEXT = VK_BLEND_OP_SCREEN_EXT,
		eOverlayEXT = VK_BLEND_OP_OVERLAY_EXT,
		eDarkenEXT = VK_BLEND_OP_DARKEN_EXT,
		eLightenEXT = VK_BLEND_OP_LIGHTEN_EXT,
		eColordodgeEXT = VK_BLEND_OP_COLORDODGE_EXT,
		eColorburnEXT = VK_BLEND_OP_COLORBURN_EXT,
		eHardlightEXT = VK_BLEND_OP_HARDLIGHT_EXT,
		eSoftlightEXT = VK_BLEND_OP_SOFTLIGHT_EXT,
		eDifferenceEXT = VK_BLEND_OP_DIFFERENCE_EXT,
		eExclusionEXT = VK_BLEND_OP_EXCLUSION_EXT,
		eInvertEXT = VK_BLEND_OP_INVERT_EXT,
		eInvertRgbEXT = VK_BLEND_OP_INVERT_RGB_EXT,
		eLineardodgeEXT = VK_BLEND_OP_LINEARDODGE_EXT,
		eLinearburnEXT = VK_BLEND_OP_LINEARBURN_EXT,
		eVividlightEXT = VK_BLEND_OP_VIVIDLIGHT_EXT,
		eLinearlightEXT = VK_BLEND_OP_LINEARLIGHT_EXT,
		ePinlightEXT = VK_BLEND_OP_PINLIGHT_EXT,
		eHardmixEXT = VK_BLEND_OP_HARDMIX_EXT,
		eHslHueEXT = VK_BLEND_OP_HSL_HUE_EXT,
		eHslSaturationEXT = VK_BLEND_OP_HSL_SATURATION_EXT,
		eHslColorEXT = VK_BLEND_OP_HSL_COLOR_EXT,
		eHslLuminosityEXT = VK_BLEND_OP_HSL_LUMINOSITY_EXT,
		ePlusEXT = VK_BLEND_OP_PLUS_EXT,
		ePlusClampedEXT = VK_BLEND_OP_PLUS_CLAMPED_EXT,
		ePlusClampedAlphaEXT = VK_BLEND_OP_PLUS_CLAMPED_ALPHA_EXT,
		ePlusDarkerEXT = VK_BLEND_OP_PLUS_DARKER_EXT,
		eMinusEXT = VK_BLEND_OP_MINUS_EXT,
		eMinusClampedEXT = VK_BLEND_OP_MINUS_CLAMPED_EXT,
		eContrastEXT = VK_BLEND_OP_CONTRAST_EXT,
		eInvertOvgEXT = VK_BLEND_OP_INVERT_OVG_EXT,
		eRedEXT = VK_BLEND_OP_RED_EXT,
		eGreenEXT = VK_BLEND_OP_GREEN_EXT,
		eBlueEXT = VK_BLEND_OP_BLUE_EXT*/
	};

	enum class BlendPreset { eOff, eAlphaBlend, ePremultipliedAlphaBlend };

	enum class PolygonMode {
		eFill = VK_POLYGON_MODE_FILL,
		eLine = VK_POLYGON_MODE_LINE,
		ePoint = VK_POLYGON_MODE_POINT,
		// eFillRectangleNV = VK_POLYGON_MODE_FILL_RECTANGLE_NV
	};

	enum class CullModeFlagBits : VkCullModeFlags {
		eNone = VK_CULL_MODE_NONE,
		eFront = VK_CULL_MODE_FRONT_BIT,
		eBack = VK_CULL_MODE_BACK_BIT,
		eFrontAndBack = VK_CULL_MODE_FRONT_AND_BACK
	};

	enum class FrontFace { eCounterClockwise = VK_FRONT_FACE_COUNTER_CLOCKWISE, eClockwise = VK_FRONT_FACE_CLOCKWISE };

	using CullModeFlags = Flags<CullModeFlagBits>;

	struct PipelineRasterizationStateCreateInfo {
		bool operator==(PipelineRasterizationStateCreateInfo const& rhs) const noexcept {
			return (depthClampEnable == rhs.depthClampEnable) && (rasterizerDiscardEnable == rhs.rasterizerDiscardEnable) && (polygonMode == rhs.polygonMode) &&
			       (cullMode == rhs.cullMode) && (frontFace == rhs.frontFace) && (depthBiasEnable == rhs.depthBiasEnable) &&
			       (depthBiasConstantFactor == rhs.depthBiasConstantFactor) && (depthBiasClamp == rhs.depthBiasClamp) &&
			       (depthBiasSlopeFactor == rhs.depthBiasSlopeFactor) && (lineWidth == rhs.lineWidth);
		}

		bool operator!=(PipelineRasterizationStateCreateInfo const& rhs) const noexcept {
			return !operator==(rhs);
		}

		Bool32 depthClampEnable = {};
		Bool32 rasterizerDiscardEnable = {};
		PolygonMode polygonMode = PolygonMode::eFill;
		CullModeFlags cullMode = {};
		FrontFace frontFace = FrontFace::eCounterClockwise;
		Bool32 depthBiasEnable = {};
		float depthBiasConstantFactor = {};
		float depthBiasClamp = {};
		float depthBiasSlopeFactor = {};
		float lineWidth = {};
	};

	enum class ColorComponentFlagBits : VkColorComponentFlags {
		eR = VK_COLOR_COMPONENT_R_BIT,
		eG = VK_COLOR_COMPONENT_G_BIT,
		eB = VK_COLOR_COMPONENT_B_BIT,
		eA = VK_COLOR_COMPONENT_A_BIT
	};

	using ColorComponentFlags = Flags<ColorComponentFlagBits>;

	inline constexpr ColorComponentFlags operator|(ColorComponentFlagBits bit0, ColorComponentFlagBits bit1) noexcept {
		return (ColorComponentFlags)bit0 | bit1;
	}

	inline constexpr ColorComponentFlags operator&(ColorComponentFlagBits bit0, ColorComponentFlagBits bit1) noexcept {
		return (ColorComponentFlags)bit0 & bit1;
	}

	inline constexpr ColorComponentFlags operator^(ColorComponentFlagBits bit0, ColorComponentFlagBits bit1) noexcept {
		return (ColorComponentFlags)bit0 ^ bit1;
	}

	struct PipelineColorBlendAttachmentState {
		bool operator==(PipelineColorBlendAttachmentState const& rhs) const noexcept {
			return (blendEnable == rhs.blendEnable) && (srcColorBlendFactor == rhs.srcColorBlendFactor) && (dstColorBlendFactor == rhs.dstColorBlendFactor) &&
			       (colorBlendOp == rhs.colorBlendOp) && (srcAlphaBlendFactor == rhs.srcAlphaBlendFactor) && (dstAlphaBlendFactor == rhs.dstAlphaBlendFactor) &&
			       (alphaBlendOp == rhs.alphaBlendOp) && (colorWriteMask == rhs.colorWriteMask);
		}

		bool operator!=(PipelineColorBlendAttachmentState const& rhs) const noexcept {
			return !operator==(rhs);
		}

		Bool32 blendEnable = {};
		BlendFactor srcColorBlendFactor = BlendFactor::eZero;
		BlendFactor dstColorBlendFactor = BlendFactor::eZero;
		BlendOp colorBlendOp = BlendOp::eAdd;
		BlendFactor srcAlphaBlendFactor = BlendFactor::eZero;
		BlendFactor dstAlphaBlendFactor = BlendFactor::eZero;
		BlendOp alphaBlendOp = BlendOp::eAdd;
		ColorComponentFlags colorWriteMask =
		    vuk::ColorComponentFlagBits::eR | vuk::ColorComponentFlagBits::eG | vuk::ColorComponentFlagBits::eB | vuk::ColorComponentFlagBits::eA;
	};

	enum class LogicOp {
		eClear = VK_LOGIC_OP_CLEAR,
		eAnd = VK_LOGIC_OP_AND,
		eAndReverse = VK_LOGIC_OP_AND_REVERSE,
		eCopy = VK_LOGIC_OP_COPY,
		eAndInverted = VK_LOGIC_OP_AND_INVERTED,
		eNoOp = VK_LOGIC_OP_NO_OP,
		eXor = VK_LOGIC_OP_XOR,
		eOr = VK_LOGIC_OP_OR,
		eNor = VK_LOGIC_OP_NOR,
		eEquivalent = VK_LOGIC_OP_EQUIVALENT,
		eInvert = VK_LOGIC_OP_INVERT,
		eOrReverse = VK_LOGIC_OP_OR_REVERSE,
		eCopyInverted = VK_LOGIC_OP_COPY_INVERTED,
		eOrInverted = VK_LOGIC_OP_OR_INVERTED,
		eNand = VK_LOGIC_OP_NAND,
		eSet = VK_LOGIC_OP_SET
	};

	struct PipelineColorBlendStateCreateInfo {
		bool operator==(PipelineColorBlendStateCreateInfo const& rhs) const noexcept {
			return (logicOpEnable == rhs.logicOpEnable) && (logicOp == rhs.logicOp) && (attachmentCount == rhs.attachmentCount) &&
			       (pAttachments == rhs.pAttachments) && (blendConstants == rhs.blendConstants);
		}

		bool operator!=(PipelineColorBlendStateCreateInfo const& rhs) const noexcept {
			return !operator==(rhs);
		}

		Bool32 logicOpEnable = {};
		LogicOp logicOp = LogicOp::eClear;
		uint32_t attachmentCount = {};
		const vuk::PipelineColorBlendAttachmentState* pAttachments = {};
		std::array<float, 4> blendConstants = {};
	};

	enum class StencilOp {
		eKeep = VK_STENCIL_OP_KEEP,
		eZero = VK_STENCIL_OP_ZERO,
		eReplace = VK_STENCIL_OP_REPLACE,
		eIncrementAndClamp = VK_STENCIL_OP_INCREMENT_AND_CLAMP,
		eDecrementAndClamp = VK_STENCIL_OP_DECREMENT_AND_CLAMP,
		eInvert = VK_STENCIL_OP_INVERT,
		eIncrementAndWrap = VK_STENCIL_OP_INCREMENT_AND_WRAP,
		eDecrementAndWrap = VK_STENCIL_OP_DECREMENT_AND_WRAP
	};

	struct StencilOpState {
		operator VkStencilOpState const&() const noexcept {
			return *reinterpret_cast<const VkStencilOpState*>(this);
		}

		operator VkStencilOpState&() noexcept {
			return *reinterpret_cast<VkStencilOpState*>(this);
		}

		bool operator==(StencilOpState const& rhs) const noexcept {
			return (failOp == rhs.failOp) && (passOp == rhs.passOp) && (depthFailOp == rhs.depthFailOp) && (compareOp == rhs.compareOp) &&
			       (compareMask == rhs.compareMask) && (writeMask == rhs.writeMask) && (reference == rhs.reference);
		}

		bool operator!=(StencilOpState const& rhs) const noexcept {
			return !operator==(rhs);
		}

		StencilOp failOp = StencilOp::eKeep;
		StencilOp passOp = StencilOp::eKeep;
		StencilOp depthFailOp = StencilOp::eKeep;
		CompareOp compareOp = CompareOp::eNever;
		uint32_t compareMask = {};
		uint32_t writeMask = {};
		uint32_t reference = {};
	};
	static_assert(sizeof(StencilOpState) == sizeof(VkStencilOpState), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<StencilOpState>::value, "struct wrapper is not a standard layout!");

	struct PipelineDepthStencilStateCreateInfo {
		bool operator==(PipelineDepthStencilStateCreateInfo const& rhs) const noexcept {
			return (depthTestEnable == rhs.depthTestEnable) && (depthWriteEnable == rhs.depthWriteEnable) && (depthCompareOp == rhs.depthCompareOp) &&
			       (depthBoundsTestEnable == rhs.depthBoundsTestEnable) && (stencilTestEnable == rhs.stencilTestEnable) && (front == rhs.front) &&
			       (back == rhs.back) && (minDepthBounds == rhs.minDepthBounds) && (maxDepthBounds == rhs.maxDepthBounds);
		}

		bool operator!=(PipelineDepthStencilStateCreateInfo const& rhs) const noexcept {
			return !operator==(rhs);
		}

		Bool32 depthTestEnable = {};
		Bool32 depthWriteEnable = {};
		CompareOp depthCompareOp = CompareOp::eNever;
		Bool32 depthBoundsTestEnable = {};
		Bool32 stencilTestEnable = {};
		StencilOpState front = {};
		StencilOpState back = {};
		float minDepthBounds = {};
		float maxDepthBounds = {};
	};

	struct VertexInputAttributeDescription {
		bool operator==(VertexInputAttributeDescription const& rhs) const noexcept {
			return (location == rhs.location) && (binding == rhs.binding) && (format == rhs.format) && (offset == rhs.offset);
		}

		bool operator!=(VertexInputAttributeDescription const& rhs) const noexcept {
			return !operator==(rhs);
		}

		uint32_t location = {};
		uint32_t binding = {};
		Format format = Format::eUndefined;
		uint32_t offset = {};
	};

	enum class DynamicStateFlagBits : uint64_t {
		eNone = 0,
		eViewport = 1 << 0,
		eScissor = 1 << 1,
		eLineWidth = 1 << 2,
		eDepthBias = 1 << 3,
		eBlendConstants = 1 << 4,
		eDepthBounds = 1 << 5,
		// additional dynamic state to implement:
		/*eStencilCompareMask = 1 << 6,
		eStencilWriteMask = 1 << 7,
		eStencilReference = 1 << 8,
		eViewportWScalingNV = 1 << 9,
		eDiscardRectangleEXT = 1 << 10,
		eSampleLocationsEXT = 1 << 11,
		eRaytracingPipelineStackSizeKHR = 1 << 12,
		eViewportShadingRatePaletteNV = 1 << 13,
		eViewportCoarseSampleOrderNV = 1 << 14,
		eExclusiveScissorNV = 1 << 15,
		eFragmentShadingRateKHR = 1 << 16,
		eLineStippleEXT = 1 << 17,
		eCullModeEXT = 1 << 18,
		eFrontFaceEXT = 1 << 19,
		ePrimitiveTopologyEXT = 1 << 20,
		eViewportWithCountEXT = 1 << 21,
		eScissorWithCountEXT = 1 << 22,
		eVertexInputBindingStrideEXT = 1 << 23,
		eDepthTestEnableEXT = 1 << 24,
		eDepthWriteEnableEXT = 1 << 25,
		eDepthCompareOpEXT = 1 << 26,
		eStencilTestEnableEXT = 1 << 27,
		eStencilOpEXT = 1 << 28,
		eVertexInputEXT = 1 << 29,
		ePatchControlPointsEXT = 1 << 30,
		eRasterizerDiscardEnableEXT = 1 << 31,
		eDepthBiasEnableEXT = 1Ui64 << 32,
		eLogicOpEXT = 1Ui64 << 33,
		ePrimitiveRestartEnableEXT = 1Ui64 << 34,
		eColorWriteEnableEXT = 1Ui64 << 35*/
	};

	using DynamicStateFlags = Flags<DynamicStateFlagBits>;

	inline constexpr DynamicStateFlags operator|(DynamicStateFlagBits bit0, DynamicStateFlagBits bit1) noexcept {
		return (DynamicStateFlags)bit0 | bit1;
	}

	inline constexpr DynamicStateFlags operator&(DynamicStateFlagBits bit0, DynamicStateFlagBits bit1) noexcept {
		return (DynamicStateFlags)bit0 & bit1;
	}

	inline constexpr DynamicStateFlags operator^(DynamicStateFlagBits bit0, DynamicStateFlagBits bit1) noexcept {
		return (DynamicStateFlags)bit0 ^ bit1;
	}
}; // namespace vuk

inline bool operator==(VkPushConstantRange const& lhs, VkPushConstantRange const& rhs) noexcept {
	return (lhs.stageFlags == rhs.stageFlags) && (lhs.offset == rhs.offset) && (lhs.size == rhs.size);
}
