#pragma once

#include <vector>
#include <vuk/Config.hpp>
#include <vuk/Hash.hpp>
#include <../src/CreateInfo.hpp>
#include <vuk/Descriptor.hpp>
#include <vuk/Program.hpp>
#include <vuk/FixedVector.hpp>
#include <vuk/Image.hpp>
#include <bit>

namespace vuk {
	template<uint64_t Count>
	struct Bitset {
		static constexpr uint64_t bitmask(uint64_t const onecount) {
			return static_cast<uint64_t>(-(onecount != 0))
				& (static_cast<uint64_t>(-1) >> ((sizeof(uint64_t)) - onecount));
		}

		static constexpr uint64_t n_bits = sizeof(uint64_t) * 8;
		static constexpr uint64_t n_words = idivceil(Count, n_bits);
		static constexpr uint64_t remainder = Count - n_bits * (Count / n_bits);
		static constexpr uint64_t last_word_mask = bitmask(remainder);
		uint64_t words[n_words];

		Bitset& set(uint64_t pos, bool value = true) noexcept {
			auto word = pos / n_bits;
			if (value) {
				words[word] |= 1ULL << (pos - n_bits * word);
			} else {
				words[word] &= ~(1ULL << (pos - n_bits * word));
			}
			return *this;
		}

		uint64_t count() const noexcept {
			uint64_t accum = 0;
			for (uint64_t i = 0; i < (Count / n_bits); i++) {
				accum += std::popcount(words[i]);
			}
			if constexpr (remainder > 0) {
				accum += std::popcount(words[n_words - 1] & last_word_mask);
			}
			return accum;
		}

		bool test(uint64_t pos) const noexcept {
			auto word = pos / n_bits;
			return words[word] & 1ULL << (pos - n_bits * word);
		}

		void clear() noexcept {
			for (uint64_t i = 0; i < (Count / n_bits); i++) {
				words[i] = 0;
			}
		}

		bool operator==(const Bitset& other) const noexcept {
			for (uint64_t i = 0; i < (Count / n_bits); i++) {
				if (words[i] != other.words[i])
					return false;
			}
			if constexpr (remainder > 0) {
				return (words[n_words - 1] & last_word_mask) == (other.words[n_words - 1] & last_word_mask);
			}
			return true;
		}
	};

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

	enum class BlendPreset {
		eOff, eAlphaBlend, ePremultipliedAlphaBlend
	};

	enum class PolygonMode {
		eFill = VK_POLYGON_MODE_FILL,
		eLine = VK_POLYGON_MODE_LINE,
		ePoint = VK_POLYGON_MODE_POINT,
		//eFillRectangleNV = VK_POLYGON_MODE_FILL_RECTANGLE_NV
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

		bool operator==(PipelineRasterizationStateCreateInfo const& rhs) const noexcept {
			return (depthClampEnable == rhs.depthClampEnable)
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
			return (logicOpEnable == rhs.logicOpEnable)
				&& (logicOp == rhs.logicOp)
				&& (attachmentCount == rhs.attachmentCount)
				&& (pAttachments == rhs.pAttachments)
				&& (blendConstants == rhs.blendConstants);
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
		bool operator==(PipelineDepthStencilStateCreateInfo const& rhs) const noexcept {
			return (depthTestEnable == rhs.depthTestEnable)
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
			return (location == rhs.location)
				&& (binding == rhs.binding)
				&& (format == rhs.format)
				&& (offset == rhs.offset);
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

	static constexpr uint32_t graphics_stage_count = 5;

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
		Bitset<4 * VUK_MAX_SETS * VUK_MAX_BINDINGS> binding_flags = {};
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

		vuk::fixed_vector<DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> explicit_set_layouts = {};
	};

	/* filled out by the user */
	struct PipelineBaseCreateInfo : PipelineBaseCreateInfoBase {
		friend class CommandBuffer;
		friend class Context;
	public:
		void add_shader(ShaderSource source, std::string filename) {
			shaders.emplace_back(std::move(source));
			shader_paths.emplace_back(std::move(filename));
		}

		void add_glsl(std::string_view source, std::string filename) {
			shaders.emplace_back(ShaderSource::glsl(source));
			shader_paths.emplace_back(std::move(filename));
		}

		void add_spirv(std::vector<uint32_t> source, std::string filename) {
			shaders.emplace_back(ShaderSource::spirv(std::move(source)));
			shader_paths.emplace_back(std::move(filename));
		}

		vuk::fixed_vector<ShaderSource, graphics_stage_count> shaders;
		vuk::fixed_vector<std::string, graphics_stage_count> shader_paths;

		friend struct std::hash<PipelineBaseCreateInfo>;
		friend class PerThreadContext;
	public:

		static vuk::fixed_vector<vuk::DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> build_descriptor_layouts(const Program&, const PipelineBaseCreateInfoBase&);
		bool operator==(const PipelineBaseCreateInfo& o) const {
			return shaders == o.shaders && binding_flags == o.binding_flags && variable_count_max == o.variable_count_max;
		}
	};

	struct PipelineBaseInfo {
		Name pipeline_name;
		vuk::Program reflection_info;
		vuk::fixed_vector<VkPipelineShaderStageCreateInfo, vuk::graphics_stage_count> psscis;
		VkPipelineLayout pipeline_layout;
		std::array<DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> layout_info;

		// 4 valid flags
		Bitset<4 * VUK_MAX_SETS * VUK_MAX_BINDINGS> binding_flags = {};
		// if the set has a variable count binding, the maximum number of bindings possible
		std::array<uint32_t, VUK_MAX_SETS> variable_count_max = {};
	};

	template<> struct create_info<PipelineBaseInfo> {
		using type = vuk::PipelineBaseCreateInfo;
	};

	struct ComputePipelineBaseCreateInfo : PipelineBaseCreateInfoBase {
		friend class CommandBuffer;
		friend class Context;
	public:
		void add_shader(ShaderSource source, std::string filename) {
			shader = std::move(source);
			shader_path = std::move(filename);
		}

		void add_glsl(std::string source, std::string filename) {
			shader = ShaderSource::glsl(std::move(source));
			shader_path = std::move(filename);
		}

		void add_spirv(std::vector<uint32_t> source, std::string filename) {
			shader = ShaderSource::spirv(std::move(source));
			shader_path = std::move(filename);
		}

		friend struct std::hash<ComputePipelineBaseCreateInfo>;
		friend class PerThreadContext;
	private:
		ShaderSource shader;
		std::string shader_path;

	public:
		bool operator==(const ComputePipelineBaseCreateInfo& o) const {
			return shader == o.shader && binding_flags == o.binding_flags && variable_count_max == o.variable_count_max;
		}
	};

	struct ComputePipelineBaseInfo {
		Name pipeline_name;
		vuk::Program reflection_info;
		VkPipelineShaderStageCreateInfo pssci;
		VkPipelineLayout pipeline_layout;
		std::array<DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> layout_info;

		// 4 valid flags
		Bitset<4 * VUK_MAX_SETS * VUK_MAX_BINDINGS> binding_flags = {};
		// if the set has a variable count binding, the maximum number of bindings possible
		std::array<uint32_t, VUK_MAX_SETS> variable_count_max = {};
	};

	template<> struct create_info<ComputePipelineBaseInfo> {
		using type = vuk::ComputePipelineBaseCreateInfo;
	};
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

		struct SpecializationMapEntry {
			uint32_t shader_stage;
			uint16_t constantID : std::bit_width(VUK_MAX_SPECIALIZATIONCONSTANT_RANGES);
			uint16_t offset : std::bit_width(VUK_MAX_SPECIALIZATIONCONSTANT_DATA);
			uint16_t size : std::bit_width(VUK_MAX_SPECIALIZATIONCONSTANT_DATA);
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
			return base == o.base && render_pass == o.render_pass && extended_size == o.extended_size && (is_inline() ? (memcmp(inline_data, o.inline_data, extended_size) == 0) : (memcmp(extended_data, o.extended_data, extended_size) == 0));
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

	template<> struct create_info<PipelineInfo> {
		using type = vuk::PipelineInstanceCreateInfo;
	};

	struct ComputePipelineInstanceCreateInfo {
		ComputePipelineBaseInfo* base;

		std::array<std::byte, VUK_MAX_SPECIALIZATIONCONSTANT_DATA> specialization_constant_data;
		vuk::fixed_vector<VkSpecializationMapEntry, VUK_MAX_SPECIALIZATIONCONSTANT_RANGES> specialization_map_entries;
		VkSpecializationInfo specialization_info = {};

		VkComputePipelineCreateInfo to_vk() const;

		bool operator==(const ComputePipelineInstanceCreateInfo& o) const noexcept {
			return base == o.base && specialization_map_entries == o.specialization_map_entries && specialization_info.dataSize == o.specialization_info.dataSize && memcmp(specialization_constant_data.data(), o.specialization_constant_data.data(), specialization_info.dataSize) == 0;
		}
	};

	struct ComputePipelineInfo : PipelineInfo {
		std::array<unsigned, 3> local_size;
	};

	template<> struct create_info<ComputePipelineInfo> {
		using type = vuk::ComputePipelineInstanceCreateInfo;
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
	struct hash<vuk::ShaderSource> {
		size_t operator()(vuk::ShaderSource const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.is_spirv, x.data);
			return h;
		}
	};

	template <>
	struct hash<vuk::PipelineBaseCreateInfo> {
		size_t operator()(vuk::PipelineBaseCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.shaders);
			return h;
		}
	};

	template <>
	struct hash<vuk::ComputePipelineBaseCreateInfo> {
		size_t operator()(vuk::ComputePipelineBaseCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.shader);
			return h;
		}
	};

	template <>
	struct hash<vuk::PipelineInstanceCreateInfo> {
		size_t operator()(vuk::PipelineInstanceCreateInfo const& x) const noexcept {
			size_t h = 0;
			auto ext_hash = x.is_inline() ? robin_hood::hash_bytes(x.inline_data, x.extended_size) : robin_hood::hash_bytes(x.extended_data, x.extended_size);
			hash_combine(h, x.base, reinterpret_cast<uint64_t>((VkRenderPass)x.render_pass), x.extended_size, ext_hash);
			return h;
		}
	};

	template <>
	struct hash<VkSpecializationMapEntry> {
		size_t operator()(VkSpecializationMapEntry const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.constantID, x.offset, x.size);
			return h;
		}
	};

	template <>
	struct hash<vuk::ComputePipelineInstanceCreateInfo> {
		size_t operator()(vuk::ComputePipelineInstanceCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.base, robin_hood::hash_bytes(x.specialization_constant_data.data(), x.specialization_info.dataSize), x.specialization_map_entries);
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
