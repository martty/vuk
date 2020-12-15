#pragma once

#include <vector>
#include "Hash.hpp"
#include "CreateInfo.hpp"
#include "Descriptor.hpp"
#include "Program.hpp"
#include "FixedVector.hpp"
#include "Image.hpp"

#define VUK_MAX_SETS 8
#define VUK_MAX_ATTRIBUTES 8
#define VUK_MAX_COLOR_ATTACHMENTS 8
#define VUK_MAX_PUSHCONSTANT_RANGES 8
#define VUK_MAX_SPECIALIZATIONCONSTANT_RANGES 8

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
		eZeroEXT = VK_BLEND_OP_ZERO_EXT,
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
		eBlueEXT = VK_BLEND_OP_BLUE_EXT
	};

	enum class BlendPreset {
		eOff, eAlphaBlend, ePremultipliedAlphaBlend
	};

	enum class PolygonMode {
		eFill = VK_POLYGON_MODE_FILL,
		eLine = VK_POLYGON_MODE_LINE,
		ePoint = VK_POLYGON_MODE_POINT,
		eFillRectangleNV = VK_POLYGON_MODE_FILL_RECTANGLE_NV
	};

	enum class CullModeFlagBits : VkCullModeFlags {
		eNone = VK_CULL_MODE_NONE,
		eFront = VK_CULL_MODE_FRONT_BIT,
		eBack = VK_CULL_MODE_BACK_BIT,
		eFrontAndBack = VK_CULL_MODE_FRONT_AND_BACK
	};

	enum class FrontFace {
		eCounterClockwise = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		eClockwise = VK_FRONT_FACE_CLOCKWISE
	};

	using CullModeFlags = Flags<CullModeFlagBits>;

	struct PipelineRasterizationStateCreateInfo {
		static constexpr VkStructureType structureType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;

		operator VkPipelineRasterizationStateCreateInfo const& () const noexcept {
			return *reinterpret_cast<const VkPipelineRasterizationStateCreateInfo*>(this);
		}

		operator VkPipelineRasterizationStateCreateInfo& () noexcept {
			return *reinterpret_cast<VkPipelineRasterizationStateCreateInfo*>(this);
		}


		bool operator==(PipelineRasterizationStateCreateInfo const& rhs) const noexcept {
			return (sType == rhs.sType)
				&& (pNext == rhs.pNext)
				&& (flags == rhs.flags)
				&& (depthClampEnable == rhs.depthClampEnable)
				&& (rasterizerDiscardEnable == rhs.rasterizerDiscardEnable)
				&& (polygonMode == rhs.polygonMode)
				&& (cullMode == rhs.cullMode)
				&& (frontFace == rhs.frontFace)
				&& (depthBiasEnable == rhs.depthBiasEnable)
				&& (depthBiasConstantFactor == rhs.depthBiasConstantFactor)
				&& (depthBiasClamp == rhs.depthBiasClamp)
				&& (depthBiasSlopeFactor == rhs.depthBiasSlopeFactor)
				&& (lineWidth == rhs.lineWidth);
		}

		bool operator!=(PipelineRasterizationStateCreateInfo const& rhs) const noexcept {
			return !operator==(rhs);
		}

		VkStructureType sType = structureType;
		const void* pNext = {};
		uint32_t flags = {}; // unused
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
	static_assert(sizeof(PipelineRasterizationStateCreateInfo) == sizeof(VkPipelineRasterizationStateCreateInfo), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<PipelineRasterizationStateCreateInfo>::value, "struct wrapper is not a standard layout!");

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
		operator VkPipelineColorBlendAttachmentState const& () const noexcept {
			return *reinterpret_cast<const VkPipelineColorBlendAttachmentState*>(this);
		}

		operator VkPipelineColorBlendAttachmentState& () noexcept {
			return *reinterpret_cast<VkPipelineColorBlendAttachmentState*>(this);
		}

		bool operator==(PipelineColorBlendAttachmentState const& rhs) const noexcept {
			return (blendEnable == rhs.blendEnable)
				&& (srcColorBlendFactor == rhs.srcColorBlendFactor)
				&& (dstColorBlendFactor == rhs.dstColorBlendFactor)
				&& (colorBlendOp == rhs.colorBlendOp)
				&& (srcAlphaBlendFactor == rhs.srcAlphaBlendFactor)
				&& (dstAlphaBlendFactor == rhs.dstAlphaBlendFactor)
				&& (alphaBlendOp == rhs.alphaBlendOp)
				&& (colorWriteMask == rhs.colorWriteMask);
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
		ColorComponentFlags colorWriteMask = {};
	};
	static_assert(sizeof(PipelineColorBlendAttachmentState) == sizeof(VkPipelineColorBlendAttachmentState), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<PipelineColorBlendAttachmentState>::value, "struct wrapper is not a standard layout!");

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
		static constexpr VkStructureType structureType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

		operator VkPipelineColorBlendStateCreateInfo const& () const noexcept {
			return *reinterpret_cast<const VkPipelineColorBlendStateCreateInfo*>(this);
		}

		operator VkPipelineColorBlendStateCreateInfo& () noexcept {
			return *reinterpret_cast<VkPipelineColorBlendStateCreateInfo*>(this);
		}

		bool operator==(PipelineColorBlendStateCreateInfo const& rhs) const noexcept {
			return (sType == rhs.sType)
				&& (pNext == rhs.pNext)
				&& (flags == rhs.flags)
				&& (logicOpEnable == rhs.logicOpEnable)
				&& (logicOp == rhs.logicOp)
				&& (attachmentCount == rhs.attachmentCount)
				&& (pAttachments == rhs.pAttachments)
				&& (blendConstants == rhs.blendConstants);
		}

		bool operator!=(PipelineColorBlendStateCreateInfo const& rhs) const noexcept {
			return !operator==(rhs);
		}

		VkStructureType sType = structureType;
		const void* pNext = {};
		uint32_t flags = {}; // unused
		Bool32 logicOpEnable = {};
		LogicOp logicOp = LogicOp::eClear;
		uint32_t attachmentCount = {};
		const vuk::PipelineColorBlendAttachmentState* pAttachments = {};
		std::array<float, 4> blendConstants = {};
	};
	static_assert(sizeof(PipelineColorBlendStateCreateInfo) == sizeof(VkPipelineColorBlendStateCreateInfo), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<PipelineColorBlendStateCreateInfo>::value, "struct wrapper is not a standard layout!");

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

		operator VkStencilOpState const& () const noexcept {
			return *reinterpret_cast<const VkStencilOpState*>(this);
		}

		operator VkStencilOpState& () noexcept {
			return *reinterpret_cast<VkStencilOpState*>(this);
		}

		bool operator==(StencilOpState const& rhs) const noexcept {
			return (failOp == rhs.failOp)
				&& (passOp == rhs.passOp)
				&& (depthFailOp == rhs.depthFailOp)
				&& (compareOp == rhs.compareOp)
				&& (compareMask == rhs.compareMask)
				&& (writeMask == rhs.writeMask)
				&& (reference == rhs.reference);
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
		static constexpr VkStructureType structureType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

		operator VkPipelineDepthStencilStateCreateInfo const& () const noexcept {
			return *reinterpret_cast<const VkPipelineDepthStencilStateCreateInfo*>(this);
		}

		operator VkPipelineDepthStencilStateCreateInfo& () noexcept {
			return *reinterpret_cast<VkPipelineDepthStencilStateCreateInfo*>(this);
		}
		bool operator==(PipelineDepthStencilStateCreateInfo const& rhs) const noexcept {
			return (sType == rhs.sType)
				&& (pNext == rhs.pNext)
				&& (flags == rhs.flags)
				&& (depthTestEnable == rhs.depthTestEnable)
				&& (depthWriteEnable == rhs.depthWriteEnable)
				&& (depthCompareOp == rhs.depthCompareOp)
				&& (depthBoundsTestEnable == rhs.depthBoundsTestEnable)
				&& (stencilTestEnable == rhs.stencilTestEnable)
				&& (front == rhs.front)
				&& (back == rhs.back)
				&& (minDepthBounds == rhs.minDepthBounds)
				&& (maxDepthBounds == rhs.maxDepthBounds);
		}

		bool operator!=(PipelineDepthStencilStateCreateInfo const& rhs) const noexcept {
			return !operator==(rhs);
		}

		VkStructureType sType = structureType;
		const void* pNext = {};
		uint32_t flags = {}; // unused
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
	static_assert(sizeof(PipelineDepthStencilStateCreateInfo) == sizeof(VkPipelineDepthStencilStateCreateInfo), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<PipelineDepthStencilStateCreateInfo>::value, "struct wrapper is not a standard layout!");

	struct VertexInputAttributeDescription {
		operator VkVertexInputAttributeDescription const& () const noexcept {
			return *reinterpret_cast<const VkVertexInputAttributeDescription*>(this);
		}

		operator VkVertexInputAttributeDescription& () noexcept {
			return *reinterpret_cast<VkVertexInputAttributeDescription*>(this);
		}

		bool operator==(VertexInputAttributeDescription const& rhs) const noexcept {
			return (location == rhs.location)
				&& (binding == rhs.binding)
				&& (format == rhs.format)
				&& (offset == rhs.offset);
		}

		bool operator!=(VertexInputAttributeDescription const& rhs) const noexcept {
			return !operator==(rhs);
		}

	public:
		uint32_t location = {};
		uint32_t binding = {};
		Format format = Format::eUndefined;
		uint32_t offset = {};

	};
	static_assert(sizeof(VertexInputAttributeDescription) == sizeof(VkVertexInputAttributeDescription), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<VertexInputAttributeDescription>::value, "struct wrapper is not a standard layout!");



	struct PipelineLayoutCreateInfo {
		VkPipelineLayoutCreateInfo plci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
		vuk::fixed_vector<VkPushConstantRange, VUK_MAX_PUSHCONSTANT_RANGES> pcrs;
		vuk::fixed_vector<vuk::DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> dslcis;

		bool operator==(const PipelineLayoutCreateInfo& o) const {
			return std::tie(plci.flags, pcrs, dslcis) == std::tie(o.plci.flags, o.pcrs, o.dslcis);
		}
	};

	template<> struct create_info<VkPipelineLayout> {
		using type = vuk::PipelineLayoutCreateInfo;
	};

	struct Program;

	struct PipelineBaseCreateInfoBase {
		// 4 valid flags
		std::bitset<4 * VUK_MAX_SETS * VUK_MAX_BINDINGS> binding_flags = {};
		// set flags on specific descriptor in specific set
		void set_binding_flags(unsigned set, unsigned binding, vuk::DescriptorBindingFlags flags) {
			unsigned f = static_cast<unsigned>(flags);
			binding_flags.set(set * 4 * VUK_MAX_BINDINGS + binding * 4 + 0, f & 0b1);
			binding_flags.set(set * 4 * VUK_MAX_BINDINGS + binding * 4 + 1, f & 0b10);
			binding_flags.set(set * 4 * VUK_MAX_BINDINGS + binding * 4 + 2, f & 0b100);
			binding_flags.set(set * 4 * VUK_MAX_BINDINGS + binding * 4 + 3, f & 0b1000);
		}
		// if the set has a variable count binding, the maximum number of bindings possible
		std::array<uint32_t, VUK_MAX_SETS> variable_count_max = {};
		void set_variable_count_binding(unsigned set, unsigned binding, uint32_t max_descriptors) {
			// clear all variable count bits
			for (unsigned i = 0; i < VUK_MAX_BINDINGS; i++) {
				binding_flags.set(set * 4 * VUK_MAX_BINDINGS + i * 4 + 3, 0);
			}
			// set variable count (0x8)
			binding_flags.set(set * 4 * VUK_MAX_BINDINGS + binding * 4 + 3, 1);
			variable_count_max[set] = max_descriptors;
		}
	};

	/* filled out by the user */
	struct PipelineBaseCreateInfo : PipelineBaseCreateInfoBase {
		friend class CommandBuffer;
		friend class Context;
	public:
		void add_shader(std::string source, std::string filename) {
			shaders.emplace_back(std::move(source));
			shader_paths.emplace_back(std::move(filename));
		}

		vuk::PipelineRasterizationStateCreateInfo rasterization_state;
		vuk::PipelineColorBlendStateCreateInfo color_blend_state;
		vuk::fixed_vector<vuk::PipelineColorBlendAttachmentState, VUK_MAX_COLOR_ATTACHMENTS> color_blend_attachments;
		vuk::PipelineDepthStencilStateCreateInfo depth_stencil_state;

		vuk::fixed_vector<std::string, 5> shaders;
		vuk::fixed_vector<std::string, 5> shader_paths;

		void set_blend(size_t attachment_index, BlendPreset);
		void set_blend(BlendPreset);

		friend struct std::hash<PipelineBaseCreateInfo>;
		friend class PerThreadContext;
	public:
		PipelineBaseCreateInfo();

		static vuk::fixed_vector<vuk::DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> build_descriptor_layouts(const Program&, const PipelineBaseCreateInfoBase&);
		bool operator==(const PipelineBaseCreateInfo& o) const {
			return shaders == o.shaders && rasterization_state == o.rasterization_state && color_blend_state == o.color_blend_state &&
				color_blend_attachments == o.color_blend_attachments && depth_stencil_state == o.depth_stencil_state && binding_flags == o.binding_flags && variable_count_max == o.variable_count_max;
		}
	};

	struct PipelineBaseInfo {
		std::string pipeline_name;
		vuk::Program reflection_info;
		std::vector<VkPipelineShaderStageCreateInfo> psscis;
		VkPipelineLayout pipeline_layout;
		std::array<DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> layout_info;
		VkPipelineRasterizationStateCreateInfo rasterization_state;
		VkPipelineColorBlendStateCreateInfo color_blend_state;
		vuk::fixed_vector<vuk::PipelineColorBlendAttachmentState, VUK_MAX_COLOR_ATTACHMENTS> color_blend_attachments;
		VkPipelineDepthStencilStateCreateInfo depth_stencil_state;

		vuk::fixed_vector<VkDynamicState, 8> dynamic_states = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineViewportStateCreateInfo viewport_state{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr };

		// 4 valid flags
		std::bitset<4 * VUK_MAX_SETS * VUK_MAX_BINDINGS> binding_flags = {};
		// if the set has a variable count binding, the maximum number of bindings possible
		std::array<uint32_t, VUK_MAX_SETS> variable_count_max = {};
	};

	template<> struct create_info<PipelineBaseInfo> {
		using type = vuk::PipelineBaseCreateInfo;
	};

	struct ComputePipelineCreateInfo : PipelineBaseCreateInfoBase {
		friend class CommandBuffer;
		friend class Context;
	public:
		void add_shader(std::string source, std::string filename) {
			shader = std::move(source);
			shader_path = std::move(filename);
		}

		friend struct std::hash<ComputePipelineCreateInfo>;
		friend class PerThreadContext;
	private:
		std::string shader;
		std::string shader_path;

	public:
		bool operator==(const ComputePipelineCreateInfo& o) const {
			return shader == o.shader && binding_flags == o.binding_flags && variable_count_max == o.variable_count_max;
		}
	};
}

inline bool operator==(VkVertexInputBindingDescription const& lhs, VkVertexInputBindingDescription const& rhs) noexcept {
	return (lhs.binding == rhs.binding)
		&& (lhs.stride == rhs.stride)
		&& (lhs.inputRate == rhs.inputRate);
}

inline bool operator==(VkVertexInputAttributeDescription const& lhs, VkVertexInputAttributeDescription const& rhs) noexcept {
	return (lhs.location == rhs.location)
		&& (lhs.binding == rhs.binding)
		&& (lhs.format == rhs.format)
		&& (lhs.offset == rhs.offset);
}

inline bool operator==(VkPipelineColorBlendAttachmentState const& lhs, VkPipelineColorBlendAttachmentState const& rhs) noexcept {
	return (lhs.blendEnable == rhs.blendEnable)
		&& (lhs.srcColorBlendFactor == rhs.srcColorBlendFactor)
		&& (lhs.dstColorBlendFactor == rhs.dstColorBlendFactor)
		&& (lhs.colorBlendOp == rhs.colorBlendOp)
		&& (lhs.srcAlphaBlendFactor == rhs.srcAlphaBlendFactor)
		&& (lhs.dstAlphaBlendFactor == rhs.dstAlphaBlendFactor)
		&& (lhs.alphaBlendOp == rhs.alphaBlendOp)
		&& (lhs.colorWriteMask == rhs.colorWriteMask);
}

inline bool operator==(VkPipelineInputAssemblyStateCreateInfo const& lhs, VkPipelineInputAssemblyStateCreateInfo const& rhs) noexcept {
	return (lhs.flags == rhs.flags)
		&& (lhs.topology == rhs.topology)
		&& (lhs.primitiveRestartEnable == rhs.primitiveRestartEnable);
}

inline bool operator==(VkPipelineColorBlendStateCreateInfo const& lhs, VkPipelineColorBlendStateCreateInfo const& rhs) noexcept {
	return (lhs.flags == rhs.flags)
		&& (lhs.logicOpEnable == rhs.logicOpEnable)
		&& (lhs.logicOp == rhs.logicOp)
		&& (lhs.attachmentCount == rhs.attachmentCount)
		&& (lhs.pAttachments == rhs.pAttachments)
		&& (memcmp(lhs.blendConstants, rhs.blendConstants, sizeof(lhs.blendConstants)) == 0);
}

inline bool operator==(VkPipelineVertexInputStateCreateInfo const& lhs, VkPipelineVertexInputStateCreateInfo const& rhs) noexcept {
	return (lhs.flags == rhs.flags)
		&& (lhs.vertexBindingDescriptionCount == rhs.vertexBindingDescriptionCount)
		&& (lhs.pVertexBindingDescriptions == rhs.pVertexBindingDescriptions)
		&& (lhs.vertexAttributeDescriptionCount == rhs.vertexAttributeDescriptionCount)
		&& (lhs.pVertexAttributeDescriptions == rhs.pVertexAttributeDescriptions);
}

inline bool operator==(VkPipelineMultisampleStateCreateInfo const& lhs, VkPipelineMultisampleStateCreateInfo const& rhs) noexcept {
	return (lhs.flags == rhs.flags)
		&& (lhs.rasterizationSamples == rhs.rasterizationSamples)
		&& (lhs.sampleShadingEnable == rhs.sampleShadingEnable)
		&& (lhs.minSampleShading == rhs.minSampleShading)
		&& (lhs.pSampleMask == rhs.pSampleMask)
		&& (lhs.alphaToCoverageEnable == rhs.alphaToCoverageEnable)
		&& (lhs.alphaToOneEnable == rhs.alphaToOneEnable);
}

inline bool operator==(VkPipelineDynamicStateCreateInfo const& lhs, VkPipelineDynamicStateCreateInfo const& rhs) noexcept {
	return (lhs.flags == rhs.flags)
		&& (lhs.dynamicStateCount == rhs.dynamicStateCount)
		&& (lhs.pDynamicStates == rhs.pDynamicStates);
}

inline bool operator==(VkPushConstantRange const& lhs, VkPushConstantRange const& rhs) noexcept {
	return (lhs.stageFlags == rhs.stageFlags)
		&& (lhs.offset == rhs.offset)
		&& (lhs.size == rhs.size);
}

inline bool operator==(VkSpecializationMapEntry const& lhs, VkSpecializationMapEntry const& rhs) noexcept {
	return (lhs.constantID == rhs.constantID)
		&& (lhs.offset == rhs.offset)
		&& (lhs.size == rhs.size);
}

inline bool operator==(VkSpecializationInfo const& lhs, VkSpecializationInfo const& rhs) noexcept {
	return (lhs.dataSize == rhs.dataSize)
		&& (lhs.pData == rhs.pData)
		&& (lhs.mapEntryCount == rhs.mapEntryCount)
		&& (lhs.pMapEntries == rhs.pMapEntries);
}

namespace vuk {
	struct PipelineInstanceCreateInfo {
		PipelineBaseInfo* base;
		vuk::fixed_vector<VkVertexInputBindingDescription, VUK_MAX_ATTRIBUTES> binding_descriptions;
		vuk::fixed_vector<vuk::VertexInputAttributeDescription, VUK_MAX_ATTRIBUTES> attribute_descriptions;
		vuk::fixed_vector<vuk::PipelineColorBlendAttachmentState, VUK_MAX_COLOR_ATTACHMENTS> color_blend_attachments;
		VkPipelineInputAssemblyStateCreateInfo input_assembly_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
		VkPipelineColorBlendStateCreateInfo color_blend_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
		VkPipelineVertexInputStateCreateInfo vertex_input_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
		VkPipelineMultisampleStateCreateInfo multisample_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
		VkPipelineDynamicStateCreateInfo dynamic_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
		vuk::fixed_vector<VkSpecializationMapEntry, VUK_MAX_SPECIALIZATIONCONSTANT_RANGES> smes;
		vuk::fixed_vector<VkSpecializationInfo, 5> sis;
		VkRenderPass render_pass;
		uint32_t subpass;

		VkGraphicsPipelineCreateInfo to_vk() const;
		PipelineInstanceCreateInfo();

		bool operator==(const PipelineInstanceCreateInfo& o) const {
			return base == o.base && binding_descriptions == o.binding_descriptions && attribute_descriptions == o.attribute_descriptions &&
				color_blend_attachments == o.color_blend_attachments && color_blend_state == o.color_blend_state &&
				vertex_input_state == o.vertex_input_state && multisample_state == o.multisample_state && dynamic_state == o.dynamic_state &&
				render_pass == o.render_pass && subpass == o.subpass && smes == o.smes && sis == o.sis;
		}
	};

	struct PipelineInfo {
		VkPipeline pipeline;
		VkPipelineLayout pipeline_layout;
		std::array<DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> layout_info;
	};

	template<> struct create_info<PipelineInfo> {
		using type = vuk::PipelineInstanceCreateInfo;
	};

	struct ComputePipelineInfo : PipelineInfo {
		std::array<unsigned, 3> local_size;
	};

	template<> struct create_info<ComputePipelineInfo> {
		using type = vuk::ComputePipelineCreateInfo;
	};
}

namespace std {
	template <class BitType>
	struct hash<vuk::Flags<BitType>> {
		size_t operator()(vuk::Flags<BitType> const& x) const noexcept {
			return std::hash<typename vuk::Flags<BitType>::MaskType>()((typename vuk::Flags<BitType>::MaskType)x);
		}
	};
};

namespace std {
	template <class T>
	struct hash<std::vector<T>> {
		size_t operator()(std::vector<T> const& x) const noexcept {
			size_t h = 0;
			for (auto& e : x) {
				hash_combine(h, e);
			}
			return h;
		}
	};

	template <class T, size_t N>
	struct hash<vuk::fixed_vector<T, N>> {
		size_t operator()(vuk::fixed_vector<T, N> const& x) const noexcept {
			size_t h = 0;
			for (auto& e : x) {
				hash_combine(h, e);
			}
			return h;
		}
	};

	template <>
	struct hash<VkPipelineInputAssemblyStateCreateInfo> {
		size_t operator()(VkPipelineInputAssemblyStateCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.primitiveRestartEnable, to_integral(x.topology));
			return h;
		}
	};

	template <>
	struct hash<vuk::StencilOpState> {
		size_t operator()(vuk::StencilOpState const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.compareMask, to_integral(x.compareOp), to_integral(x.failOp), to_integral(x.depthFailOp), to_integral(x.passOp), x.reference, x.writeMask);
			return h;
		}
	};

	template <>
	struct hash<vuk::PipelineDepthStencilStateCreateInfo> {
		size_t operator()(vuk::PipelineDepthStencilStateCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.back, x.front, x.depthBoundsTestEnable, to_integral(x.depthCompareOp), x.depthTestEnable, x.depthWriteEnable, x.maxDepthBounds, x.minDepthBounds, x.stencilTestEnable);
			return h;
		}
	};

	template<>
	struct hash<vuk::PipelineRasterizationStateCreateInfo> {
		size_t operator()(vuk::PipelineRasterizationStateCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.depthClampEnable, x.rasterizerDiscardEnable, x.polygonMode,
				x.cullMode, x.frontFace, x.depthBiasEnable,
				x.depthBiasConstantFactor, x.depthBiasClamp, x.depthBiasSlopeFactor, x.lineWidth);
			return h;
		}
	};

	template <>
	struct hash<vuk::PipelineColorBlendAttachmentState> {
		size_t operator()(vuk::PipelineColorBlendAttachmentState const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, to_integral(x.alphaBlendOp), x.blendEnable, to_integral(x.colorBlendOp), to_integral(x.dstAlphaBlendFactor), to_integral(x.srcAlphaBlendFactor), to_integral(x.dstColorBlendFactor), to_integral(x.srcColorBlendFactor));
			return h;
		}
	};

	template <>
	struct hash<vuk::PipelineColorBlendStateCreateInfo> {
		size_t operator()(vuk::PipelineColorBlendStateCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, std::span(x.pAttachments, x.attachmentCount), x.blendConstants[0], x.blendConstants[1], x.blendConstants[2], x.blendConstants[3], to_integral(x.logicOp), x.logicOpEnable);
			return h;
		}
	};


	template <>
	struct hash<vuk::PipelineBaseCreateInfo> {
		size_t operator()(vuk::PipelineBaseCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.shaders, x.color_blend_state, x.color_blend_attachments, x.depth_stencil_state, x.rasterization_state);
			return h;
		}
	};

	template <>
	struct hash<vuk::ComputePipelineCreateInfo> {
		size_t operator()(vuk::ComputePipelineCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.shader);
			return h;
		}
	};

	template <>
	struct hash<vuk::PipelineInstanceCreateInfo> {
		size_t operator()(vuk::PipelineInstanceCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.base, reinterpret_cast<uint64_t>((VkRenderPass)x.render_pass), x.subpass);
			return h;
		}
	};

	template <>
	struct hash<VkPushConstantRange> {
		size_t operator()(VkPushConstantRange const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.offset, x.size, (VkShaderStageFlags)x.stageFlags);
			return h;
		}
	};


	template <>
	struct hash<vuk::PipelineLayoutCreateInfo> {
		size_t operator()(vuk::PipelineLayoutCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.pcrs, x.dslcis);
			return h;
		}
	};

};
