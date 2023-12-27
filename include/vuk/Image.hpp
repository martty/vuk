#pragma once

#include <cassert>

#include "../src/CreateInfo.hpp"
#include "Types.hpp"

namespace vuk {
	struct Image {
		VkImage image = VK_NULL_HANDLE;
		void* allocation = nullptr;

		constexpr explicit operator bool() const noexcept {
			return image != VK_NULL_HANDLE;
		}

		constexpr bool operator==(const Image&) const = default;
	};

	using Sampler = Handle<VkSampler>;

	enum class ImageTiling {
		eOptimal = VK_IMAGE_TILING_OPTIMAL,
		eLinear = VK_IMAGE_TILING_LINEAR,
		eDrmFormatModifierEXT = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT
	};

	enum class ImageType { e1D = VK_IMAGE_TYPE_1D, e2D = VK_IMAGE_TYPE_2D, e3D = VK_IMAGE_TYPE_3D, eInfer };

	enum class ImageUsageFlagBits : VkImageUsageFlags {
		eTransferSrc = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		eTransferDst = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		eSampled = VK_IMAGE_USAGE_SAMPLED_BIT,
		eStorage = VK_IMAGE_USAGE_STORAGE_BIT,
		eColorAttachment = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		eDepthStencilAttachment = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		eTransientAttachment = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
		eInputAttachment = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
		eShadingRateImageNV = VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV,
		eFragmentDensityMapEXT = VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT,
		eInfer = 1024
	};

	using ImageUsageFlags = Flags<ImageUsageFlagBits>;
	inline constexpr ImageUsageFlags operator|(ImageUsageFlagBits bit0, ImageUsageFlagBits bit1) noexcept {
		return ImageUsageFlags(bit0) | bit1;
	}

	inline constexpr ImageUsageFlags operator&(ImageUsageFlagBits bit0, ImageUsageFlagBits bit1) noexcept {
		return ImageUsageFlags(bit0) & bit1;
	}

	inline constexpr ImageUsageFlags operator^(ImageUsageFlagBits bit0, ImageUsageFlagBits bit1) noexcept {
		return ImageUsageFlags(bit0) ^ bit1;
	}

	enum class ImageCreateFlagBits : VkImageCreateFlags {
		eSparseBinding = VK_IMAGE_CREATE_SPARSE_BINDING_BIT,
		eSparseResidency = VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
		eSparseAliased = VK_IMAGE_CREATE_SPARSE_ALIASED_BIT,
		eMutableFormat = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
		eCubeCompatible = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
		eAlias = VK_IMAGE_CREATE_ALIAS_BIT,
		eSplitInstanceBindRegions = VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT,
		e2DArrayCompatible = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT,
		eBlockTexelViewCompatible = VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT,
		eExtendedUsage = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT,
		eProtected = VK_IMAGE_CREATE_PROTECTED_BIT,
		eDisjoint = VK_IMAGE_CREATE_DISJOINT_BIT,
		eCornerSampledNV = VK_IMAGE_CREATE_CORNER_SAMPLED_BIT_NV,
		eSampleLocationsCompatibleDepthEXT = VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT,
		eSubsampledEXT = VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT,
		e2DArrayCompatibleKHR = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT_KHR,
		eAliasKHR = VK_IMAGE_CREATE_ALIAS_BIT_KHR,
		eBlockTexelViewCompatibleKHR = VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT_KHR,
		eDisjointKHR = VK_IMAGE_CREATE_DISJOINT_BIT_KHR,
		eExtendedUsageKHR = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT_KHR,
		eSplitInstanceBindRegionsKHR = VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT_KHR
	};

	using ImageCreateFlags = Flags<ImageCreateFlagBits>;
	inline constexpr ImageCreateFlags operator|(ImageCreateFlagBits bit0, ImageCreateFlagBits bit1) noexcept {
		return ImageCreateFlags(bit0) | bit1;
	}

	inline constexpr ImageCreateFlags operator&(ImageCreateFlagBits bit0, ImageCreateFlagBits bit1) noexcept {
		return ImageCreateFlags(bit0) & bit1;
	}

	inline constexpr ImageCreateFlags operator^(ImageCreateFlagBits bit0, ImageCreateFlagBits bit1) noexcept {
		return ImageCreateFlags(bit0) ^ bit1;
	}

	enum class ImageLayout {
		eUndefined = VK_IMAGE_LAYOUT_UNDEFINED,
		eGeneral = VK_IMAGE_LAYOUT_GENERAL,
		eColorAttachmentOptimal = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		eDepthStencilAttachmentOptimal = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		eDepthStencilReadOnlyOptimal = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
		eShaderReadOnlyOptimal = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		eTransferSrcOptimal = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		eTransferDstOptimal = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		ePreinitialized = VK_IMAGE_LAYOUT_PREINITIALIZED,
		eDepthReadOnlyStencilAttachmentOptimal = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
		eDepthAttachmentStencilReadOnlyOptimal = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
		eDepthAttachmentOptimal = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		eDepthReadOnlyOptimal = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
		eStencilAttachmentOptimal = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL,
		eStencilReadOnlyOptimal = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL,
		ePresentSrcKHR = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		eSharedPresentKHR = VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR,
		eShadingRateOptimalNV = VK_IMAGE_LAYOUT_SHADING_RATE_OPTIMAL_NV,
		eFragmentDensityMapOptimalEXT = VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT,
		eDepthAttachmentOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR,
		eDepthAttachmentStencilReadOnlyOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL_KHR,
		eDepthReadOnlyOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL_KHR,
		eDepthReadOnlyStencilAttachmentOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL_KHR,
		eStencilAttachmentOptimalKHR = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL_KHR,
		eStencilReadOnlyOptimalKHR = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL_KHR,
		eReadOnlyOptimalKHR = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR,
		eReadOnlyOptimal = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
		eAttachmentOptimalKHR = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
		eAttachmentOptimal = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL
	};

	enum class SharingMode { eExclusive = VK_SHARING_MODE_EXCLUSIVE, eConcurrent = VK_SHARING_MODE_CONCURRENT };

	struct ImageCreateInfo {
		static constexpr VkStructureType structureType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;

		VkStructureType sType = structureType;
		const void* pNext = {};
		ImageCreateFlags flags = {};
		ImageType imageType = ImageType::e2D;
		Format format = Format::eUndefined;
		Extent3D extent = {};
		uint32_t mipLevels = 1;
		uint32_t arrayLayers = 1;
		SampleCountFlagBits samples = SampleCountFlagBits::e1;
		ImageTiling tiling = ImageTiling::eOptimal;
		ImageUsageFlags usage = {};
		SharingMode sharingMode = SharingMode::eExclusive;
		uint32_t queueFamilyIndexCount = {};
		const uint32_t* pQueueFamilyIndices = {};
		ImageLayout initialLayout = ImageLayout::eUndefined;

		operator VkImageCreateInfo const&() const noexcept {
			return *reinterpret_cast<const VkImageCreateInfo*>(this);
		}

		operator VkImageCreateInfo&() noexcept {
			return *reinterpret_cast<VkImageCreateInfo*>(this);
		}

		bool operator==(ImageCreateInfo const& rhs) const noexcept {
			return (sType == rhs.sType) && (pNext == rhs.pNext) && (flags == rhs.flags) && (imageType == rhs.imageType) && (format == rhs.format) &&
			       (extent == rhs.extent) && (mipLevels == rhs.mipLevels) && (arrayLayers == rhs.arrayLayers) && (samples == rhs.samples) && (tiling == rhs.tiling) &&
			       (usage == rhs.usage) && (sharingMode == rhs.sharingMode) && (queueFamilyIndexCount == rhs.queueFamilyIndexCount) &&
			       (pQueueFamilyIndices == rhs.pQueueFamilyIndices) && (initialLayout == rhs.initialLayout);
		}
	};
	static_assert(sizeof(ImageCreateInfo) == sizeof(VkImageCreateInfo), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<ImageCreateInfo>::value, "struct wrapper is not a standard layout!");

	template<>
	struct create_info<Image> {
		using type = ImageCreateInfo;
	};

	struct ImageWithIdentity {
		Image image;
	};

	struct CachedImageIdentifier {
		ImageCreateInfo ici;
		uint32_t id;
		uint32_t multi_frame_index;

		bool operator==(const CachedImageIdentifier&) const = default;
	};

	template<>
	struct create_info<ImageWithIdentity> {
		using type = CachedImageIdentifier;
	};

	enum class ImageViewCreateFlagBits : VkImageViewCreateFlags { eFragmentDensityMapDynamicEXT = VK_IMAGE_VIEW_CREATE_FRAGMENT_DENSITY_MAP_DYNAMIC_BIT_EXT };

	enum class ImageViewType : uint32_t {
		e1D = VK_IMAGE_VIEW_TYPE_1D,
		e2D = VK_IMAGE_VIEW_TYPE_2D,
		e3D = VK_IMAGE_VIEW_TYPE_3D,
		eCube = VK_IMAGE_VIEW_TYPE_CUBE,
		e1DArray = VK_IMAGE_VIEW_TYPE_1D_ARRAY,
		e2DArray = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
		eCubeArray = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY,
		eInfer
	};

	using ImageViewCreateFlags = Flags<ImageViewCreateFlagBits>;
	enum class ComponentSwizzle : uint32_t {
		eIdentity = VK_COMPONENT_SWIZZLE_IDENTITY,
		eZero = VK_COMPONENT_SWIZZLE_ZERO,
		eOne = VK_COMPONENT_SWIZZLE_ONE,
		eR = VK_COMPONENT_SWIZZLE_R,
		eG = VK_COMPONENT_SWIZZLE_G,
		eB = VK_COMPONENT_SWIZZLE_B,
		eA = VK_COMPONENT_SWIZZLE_A,
		eInfer
	};
	struct ComponentMapping {
		ComponentSwizzle r = ComponentSwizzle::eIdentity;
		ComponentSwizzle g = ComponentSwizzle::eIdentity;
		ComponentSwizzle b = ComponentSwizzle::eIdentity;
		ComponentSwizzle a = ComponentSwizzle::eIdentity;

		operator VkComponentMapping const&() const noexcept {
			return *reinterpret_cast<const VkComponentMapping*>(this);
		}

		operator VkComponentMapping&() noexcept {
			return *reinterpret_cast<VkComponentMapping*>(this);
		}

		constexpr bool operator==(const ComponentMapping&) const = default;
	};
	static_assert(sizeof(ComponentMapping) == sizeof(VkComponentMapping), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<ComponentMapping>::value, "struct wrapper is not a standard layout!");

	enum class ImageAspectFlagBits : VkImageAspectFlags {
		eColor = VK_IMAGE_ASPECT_COLOR_BIT,
		eDepth = VK_IMAGE_ASPECT_DEPTH_BIT,
		eStencil = VK_IMAGE_ASPECT_STENCIL_BIT,
		eMetadata = VK_IMAGE_ASPECT_METADATA_BIT,
		ePlane0 = VK_IMAGE_ASPECT_PLANE_0_BIT,
		ePlane1 = VK_IMAGE_ASPECT_PLANE_1_BIT,
		ePlane2 = VK_IMAGE_ASPECT_PLANE_2_BIT,
		eMemoryPlane0EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
		eMemoryPlane1EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT,
		eMemoryPlane2EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT,
		eMemoryPlane3EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT,
		ePlane0KHR = VK_IMAGE_ASPECT_PLANE_0_BIT_KHR,
		ePlane1KHR = VK_IMAGE_ASPECT_PLANE_1_BIT_KHR,
		ePlane2KHR = VK_IMAGE_ASPECT_PLANE_2_BIT_KHR
	};

	using ImageAspectFlags = Flags<ImageAspectFlagBits>;
	inline constexpr ImageAspectFlags operator|(ImageAspectFlagBits bit0, ImageAspectFlagBits bit1) noexcept {
		return ImageAspectFlags(bit0) | bit1;
	}

	inline constexpr ImageAspectFlags operator&(ImageAspectFlagBits bit0, ImageAspectFlagBits bit1) noexcept {
		return ImageAspectFlags(bit0) & bit1;
	}

	inline constexpr ImageAspectFlags operator^(ImageAspectFlagBits bit0, ImageAspectFlagBits bit1) noexcept {
		return ImageAspectFlags(bit0) ^ bit1;
	}

	struct ImageSubresourceRange {
		ImageAspectFlags aspectMask = {};
		uint32_t baseMipLevel = 0;
		uint32_t levelCount = 1;
		uint32_t baseArrayLayer = 0;
		uint32_t layerCount = 1;

		operator VkImageSubresourceRange const&() const noexcept {
			return *reinterpret_cast<const VkImageSubresourceRange*>(this);
		}

		operator VkImageSubresourceRange&() noexcept {
			return *reinterpret_cast<VkImageSubresourceRange*>(this);
		}

		bool operator==(ImageSubresourceRange const& rhs) const noexcept {
			return (aspectMask == rhs.aspectMask) && (baseMipLevel == rhs.baseMipLevel) && (levelCount == rhs.levelCount) && (baseArrayLayer == rhs.baseArrayLayer) &&
			       (layerCount == rhs.layerCount);
		}

		bool operator!=(ImageSubresourceRange const& rhs) const noexcept {
			return !operator==(rhs);
		}
	};
	static_assert(sizeof(ImageSubresourceRange) == sizeof(VkImageSubresourceRange), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<ImageSubresourceRange>::value, "struct wrapper is not a standard layout!");

	struct ImageViewCreateInfo {
		static constexpr VkStructureType structureType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;

		VkStructureType sType = structureType;
		const void* pNext = {};
		ImageViewCreateFlags flags = {};
		VkImage image = {};
		ImageViewType viewType = ImageViewType::e2D;
		Format format = Format::eUndefined;
		ComponentMapping components = {};
		ImageSubresourceRange subresourceRange = {};
		vuk::ImageUsageFlags view_usage = {}; // added for vuk

		operator VkImageViewCreateInfo const&() const noexcept {
			return *reinterpret_cast<const VkImageViewCreateInfo*>(this);
		}

		operator VkImageViewCreateInfo&() noexcept {
			return *reinterpret_cast<VkImageViewCreateInfo*>(this);
		}

		bool operator==(ImageViewCreateInfo const& rhs) const noexcept {
			return (sType == rhs.sType) && (pNext == rhs.pNext) && (flags == rhs.flags) && (image == rhs.image) && (viewType == rhs.viewType) &&
			       (format == rhs.format) && (components == rhs.components) && (subresourceRange == rhs.subresourceRange);
		}

		bool operator!=(ImageViewCreateInfo const& rhs) const noexcept {
			return !operator==(rhs);
		}
	};
	static_assert(std::is_standard_layout<ImageViewCreateInfo>::value, "struct wrapper is not a standard layout!");

#pragma pack(push, 1)
	struct CompressedImageViewCreateInfo {
		uint32_t flags : 2;
		ImageViewType viewType : 3 = ImageViewType::e2D;
		ComponentSwizzle r_swizzle : 3 = ComponentSwizzle::eIdentity;
		ComponentSwizzle g_swizzle : 3 = ComponentSwizzle::eIdentity;
		ComponentSwizzle b_swizzle : 3 = ComponentSwizzle::eIdentity;
		ComponentSwizzle a_swizzle : 3 = ComponentSwizzle::eIdentity;
		uint32_t padding : 4 = 0;
		uint32_t aspectMask : 11 = {}; // 27 bits ~ pad to 4 bytes
		uint32_t baseMipLevel : 16 = 0;
		uint32_t levelCount : 16 = 1;
		uint32_t baseArrayLayer : 16 = 0; 
		uint32_t layerCount : 16 = 1;
		VkImageUsageFlags view_usage : 10; // 8 bytes
		VkImage image = {};                 // 16 bytes
		Format format = Format::eUndefined; // 32 bytes in total

		CompressedImageViewCreateInfo(ImageViewCreateInfo ivci) {
			assert(ivci.pNext == nullptr && "Compression does not support pNextended IVCIs");
			flags = ivci.flags.m_mask;
			viewType = ivci.viewType;
			r_swizzle = ivci.components.r;
			g_swizzle = ivci.components.g;
			b_swizzle = ivci.components.b;
			a_swizzle = ivci.components.a;
			aspectMask = ivci.subresourceRange.aspectMask.m_mask;
			baseMipLevel = ivci.subresourceRange.baseMipLevel;
			levelCount = ivci.subresourceRange.levelCount;
			baseArrayLayer = ivci.subresourceRange.baseArrayLayer;
			layerCount = ivci.subresourceRange.layerCount;
			image = ivci.image;
			format = ivci.format;
			view_usage = (VkImageUsageFlags)ivci.view_usage;
		}

		explicit operator ImageViewCreateInfo() const noexcept {
			ImageViewCreateInfo ivci;
			ivci.flags = (ImageViewCreateFlags)flags;
			ivci.viewType = (ImageViewType)viewType;
			ivci.components.r = (ComponentSwizzle)r_swizzle;
			ivci.components.g = (ComponentSwizzle)g_swizzle;
			ivci.components.b = (ComponentSwizzle)b_swizzle;
			ivci.components.a = (ComponentSwizzle)a_swizzle;
			ivci.subresourceRange = { .aspectMask = (ImageAspectFlags)aspectMask,
				                        .baseMipLevel = baseMipLevel,
				                        .levelCount = levelCount,
				                        .baseArrayLayer = baseArrayLayer,
				                        .layerCount = layerCount };
			ivci.image = image;
			ivci.format = (Format)format;
			ivci.view_usage = (ImageUsageFlags)view_usage;
			return ivci;
		}

		constexpr bool operator==(CompressedImageViewCreateInfo const& rhs) const noexcept = default;
	};
#pragma pack(pop)

	using ImageView = Handle<VkImageView>;

	template<>
	struct create_info<ImageView> {
		using type = CompressedImageViewCreateInfo;
	};

	enum class SamplerCreateFlagBits : VkSamplerCreateFlags {
		eSubsampledEXT = VK_SAMPLER_CREATE_SUBSAMPLED_BIT_EXT,
		eSubsampledCoarseReconstructionEXT = VK_SAMPLER_CREATE_SUBSAMPLED_COARSE_RECONSTRUCTION_BIT_EXT
	};

	using SamplerCreateFlags = Flags<SamplerCreateFlagBits>;
	enum class Filter { eNearest = VK_FILTER_NEAREST, eLinear = VK_FILTER_LINEAR, eCubicIMG = VK_FILTER_CUBIC_IMG, eCubicEXT = VK_FILTER_CUBIC_EXT };

	enum class SamplerMipmapMode { eNearest = VK_SAMPLER_MIPMAP_MODE_NEAREST, eLinear = VK_SAMPLER_MIPMAP_MODE_LINEAR };

	enum class SamplerAddressMode {
		eRepeat = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		eMirroredRepeat = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
		eClampToEdge = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		eClampToBorder = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		eMirrorClampToEdge = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
		eMirrorClampToEdgeKHR = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE_KHR
	};

	enum class CompareOp {
		eNever = VK_COMPARE_OP_NEVER,
		eLess = VK_COMPARE_OP_LESS,
		eEqual = VK_COMPARE_OP_EQUAL,
		eLessOrEqual = VK_COMPARE_OP_LESS_OR_EQUAL,
		eGreater = VK_COMPARE_OP_GREATER,
		eNotEqual = VK_COMPARE_OP_NOT_EQUAL,
		eGreaterOrEqual = VK_COMPARE_OP_GREATER_OR_EQUAL,
		eAlways = VK_COMPARE_OP_ALWAYS
	};

	enum class BorderColor {
		eFloatTransparentBlack = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
		eIntTransparentBlack = VK_BORDER_COLOR_INT_TRANSPARENT_BLACK,
		eFloatOpaqueBlack = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
		eIntOpaqueBlack = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		eFloatOpaqueWhite = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
		eIntOpaqueWhite = VK_BORDER_COLOR_INT_OPAQUE_WHITE,
		eFloatCustomEXT = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT,
		eIntCustomEXT = VK_BORDER_COLOR_INT_CUSTOM_EXT
	};

	struct SamplerCreateInfo {
		static constexpr VkStructureType structureType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

		VkStructureType sType = structureType;
		const void* pNext = {};
		SamplerCreateFlags flags = {};
		Filter magFilter = Filter::eNearest;
		Filter minFilter = Filter::eNearest;
		SamplerMipmapMode mipmapMode = SamplerMipmapMode::eNearest;
		SamplerAddressMode addressModeU = SamplerAddressMode::eRepeat;
		SamplerAddressMode addressModeV = SamplerAddressMode::eRepeat;
		SamplerAddressMode addressModeW = SamplerAddressMode::eRepeat;
		float mipLodBias = {};
		Bool32 anisotropyEnable = {};
		float maxAnisotropy = {};
		Bool32 compareEnable = {};
		CompareOp compareOp = CompareOp::eNever;
		float minLod = 0.f;
		float maxLod = VK_LOD_CLAMP_NONE;
		BorderColor borderColor = BorderColor::eFloatTransparentBlack;
		Bool32 unnormalizedCoordinates = {};

		operator VkSamplerCreateInfo const&() const noexcept {
			return *reinterpret_cast<const VkSamplerCreateInfo*>(this);
		}

		operator VkSamplerCreateInfo&() noexcept {
			return *reinterpret_cast<VkSamplerCreateInfo*>(this);
		}

		bool operator==(SamplerCreateInfo const& rhs) const noexcept {
			return (sType == rhs.sType) && (pNext == rhs.pNext) && (flags == rhs.flags) && (magFilter == rhs.magFilter) && (minFilter == rhs.minFilter) &&
			       (mipmapMode == rhs.mipmapMode) && (addressModeU == rhs.addressModeU) && (addressModeV == rhs.addressModeV) && (addressModeW == rhs.addressModeW) &&
			       (mipLodBias == rhs.mipLodBias) && (anisotropyEnable == rhs.anisotropyEnable) && (maxAnisotropy == rhs.maxAnisotropy) &&
			       (compareEnable == rhs.compareEnable) && (compareOp == rhs.compareOp) && (minLod == rhs.minLod) && (maxLod == rhs.maxLod) &&
			       (borderColor == rhs.borderColor) && (unnormalizedCoordinates == rhs.unnormalizedCoordinates);
		}

		bool operator!=(SamplerCreateInfo const& rhs) const noexcept {
			return !operator==(rhs);
		}
	};
	static_assert(sizeof(SamplerCreateInfo) == sizeof(VkSamplerCreateInfo), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<SamplerCreateInfo>::value, "struct wrapper is not a standard layout!");

	template<>
	struct create_info<Sampler> {
		using type = vuk::SamplerCreateInfo;
	};

	ImageAspectFlags format_to_aspect(Format format) noexcept;
}; // namespace vuk

namespace std {
	template<>
	struct hash<vuk::ImageView> {
		size_t operator()(vuk::ImageView const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.id, reinterpret_cast<uint64_t>(x.payload));
			return h;
		}
	};
} // namespace std
