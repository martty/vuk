#include <cassert>
#include "vuk/Image.hpp"

namespace vuk {
	uint32_t format_to_texel_block_size(Format format) noexcept {
		switch (format) {
		case Format::eR4G4UnormPack8:
		case Format::eR8Unorm:
		case Format::eR8Snorm:
		case Format::eR8Uscaled:
		case Format::eR8Sscaled:
		case Format::eR8Uint:
		case Format::eR8Sint:
		case Format::eR8Srgb:
			return 1;
		case Format::eR4G4B4A4UnormPack16:
		case Format::eB4G4R4A4UnormPack16:
		case Format::eR5G6B5UnormPack16:
		case Format::eB5G6R5UnormPack16:
		case Format::eR5G5B5A1UnormPack16:
		case Format::eB5G5R5A1UnormPack16:
		case Format::eA1R5G5B5UnormPack16:
		case Format::eR8G8Unorm:
		case Format::eR8G8Snorm:
		case Format::eR8G8Uscaled:
		case Format::eR8G8Sscaled:
		case Format::eR8G8Uint:
		case Format::eR8G8Sint:
		case Format::eR8G8Srgb:
		case Format::eR16Unorm:
		case Format::eR16Snorm:
		case Format::eR16Uscaled:
		case Format::eR16Sscaled:
		case Format::eR16Uint:
		case Format::eR16Sint:
		case Format::eR16Sfloat:
		case Format::eR10X6UnormPack16:
		case Format::eR12X4UnormPack16:
			return 2;
		case Format::eR8G8B8Unorm:
		case Format::eR8G8B8Snorm:
		case Format::eR8G8B8Uscaled:
		case Format::eR8G8B8Sscaled:
		case Format::eR8G8B8Uint:
		case Format::eR8G8B8Sint:
		case Format::eR8G8B8Srgb:
		case Format::eB8G8R8Unorm:
		case Format::eB8G8R8Snorm:
		case Format::eB8G8R8Uscaled:
		case Format::eB8G8R8Sscaled:
		case Format::eB8G8R8Uint:
		case Format::eB8G8R8Sint:
		case Format::eB8G8R8Srgb:
			return 3;
		case Format::eR8G8B8A8Unorm:
		case Format::eR8G8B8A8Snorm:
		case Format::eR8G8B8A8Uscaled:
		case Format::eR8G8B8A8Sscaled:
		case Format::eR8G8B8A8Uint:
		case Format::eR8G8B8A8Sint:
		case Format::eR8G8B8A8Srgb:
		case Format::eB8G8R8A8Unorm:
		case Format::eB8G8R8A8Snorm:
		case Format::eB8G8R8A8Uscaled:
		case Format::eB8G8R8A8Sscaled:
		case Format::eB8G8R8A8Uint:
		case Format::eB8G8R8A8Sint:
		case Format::eB8G8R8A8Srgb:
		case Format::eA8B8G8R8UnormPack32:
		case Format::eA8B8G8R8SnormPack32:
		case Format::eA8B8G8R8UscaledPack32:
		case Format::eA8B8G8R8SscaledPack32:
		case Format::eA8B8G8R8UintPack32:
		case Format::eA8B8G8R8SintPack32:
		case Format::eA8B8G8R8SrgbPack32:
		case Format::eA2R10G10B10UnormPack32:
		case Format::eA2R10G10B10SnormPack32:
		case Format::eA2R10G10B10UscaledPack32:
		case Format::eA2R10G10B10SscaledPack32:
		case Format::eA2R10G10B10UintPack32:
		case Format::eA2R10G10B10SintPack32:
		case Format::eA2B10G10R10UnormPack32:
		case Format::eA2B10G10R10SnormPack32:
		case Format::eA2B10G10R10UscaledPack32:
		case Format::eA2B10G10R10SscaledPack32:
		case Format::eA2B10G10R10UintPack32:
		case Format::eA2B10G10R10SintPack32:
		case Format::eR16G16Unorm:
		case Format::eR16G16Snorm:
		case Format::eR16G16Uscaled:
		case Format::eR16G16Sscaled:
		case Format::eR16G16Uint:
		case Format::eR16G16Sint:
		case Format::eR16G16Sfloat:
		case Format::eR32Uint:
		case Format::eR32Sint:
		case Format::eR32Sfloat:
		case Format::eB10G11R11UfloatPack32:
		case Format::eE5B9G9R9UfloatPack32:
		case Format::eR10X6G10X6Unorm2Pack16:
		case Format::eR12X4G12X4Unorm2Pack16:
			return 4;
		case Format::eG8B8G8R8422Unorm:
			return 4;
		case Format::eB8G8R8G8422Unorm:
			return 4;
		case Format::eR16G16B16Unorm:
		case Format::eR16G16B16Snorm:
		case Format::eR16G16B16Uscaled:
		case Format::eR16G16B16Sscaled:
		case Format::eR16G16B16Uint:
		case Format::eR16G16B16Sint:
		case Format::eR16G16B16Sfloat:
			return 6;
		case Format::eR16G16B16A16Unorm:
		case Format::eR16G16B16A16Snorm:
		case Format::eR16G16B16A16Uscaled:
		case Format::eR16G16B16A16Sscaled:
		case Format::eR16G16B16A16Uint:
		case Format::eR16G16B16A16Sint:
		case Format::eR16G16B16A16Sfloat:
		case Format::eR32G32Uint:
		case Format::eR32G32Sint:
		case Format::eR32G32Sfloat:
		case Format::eR64Uint:
		case Format::eR64Sint:
		case Format::eR64Sfloat:
			return 8;
		case Format::eR10X6G10X6B10X6A10X6Unorm4Pack16:
			return 8;
		case Format::eG10X6B10X6G10X6R10X6422Unorm4Pack16:
			return 8;
		case Format::eB10X6G10X6R10X6G10X6422Unorm4Pack16:
			return 8;
		case Format::eR12X4G12X4B12X4A12X4Unorm4Pack16:
			return 8;
		case Format::eG12X4B12X4G12X4R12X4422Unorm4Pack16:
			return 8;
		case Format::eB12X4G12X4R12X4G12X4422Unorm4Pack16:
			return 8;
		case Format::eG16B16G16R16422Unorm:
			return 8;
		case Format::eB16G16R16G16422Unorm:
			return 8;
		case Format::eR32G32B32Uint:
		case Format::eR32G32B32Sint:
		case Format::eR32G32B32Sfloat:
			return 12;
		case Format::eR32G32B32A32Uint:
		case Format::eR32G32B32A32Sint:
		case Format::eR32G32B32A32Sfloat:
		case Format::eR64G64Uint:
		case Format::eR64G64Sint:
		case Format::eR64G64Sfloat:
			return 16;
		case Format::eR64G64B64Uint:
		case Format::eR64G64B64Sint:
		case Format::eR64G64B64Sfloat:
			return 24;
		case Format::eR64G64B64A64Uint:
		case Format::eR64G64B64A64Sint:
		case Format::eR64G64B64A64Sfloat:
			return 32;
		case Format::eBc1RgbUnormBlock:
		case Format::eBc1RgbSrgbBlock:
			return 8;
		case Format::eBc1RgbaUnormBlock:
		case Format::eBc1RgbaSrgbBlock:
			return 8;
		case Format::eBc2UnormBlock:
		case Format::eBc2SrgbBlock:
			return 16;
		case Format::eBc3UnormBlock:
		case Format::eBc3SrgbBlock:
			return 16;
		case Format::eBc4UnormBlock:
		case Format::eBc4SnormBlock:
			return 8;
		case Format::eBc5UnormBlock:
		case Format::eBc5SnormBlock:
			return 16;
		case Format::eBc6HUfloatBlock:
		case Format::eBc6HSfloatBlock:
			return 16;
		case Format::eBc7UnormBlock:
		case Format::eBc7SrgbBlock:
			return 16;
		case Format::eEtc2R8G8B8UnormBlock:
		case Format::eEtc2R8G8B8SrgbBlock:
			return 8;
		case Format::eEtc2R8G8B8A1UnormBlock:
		case Format::eEtc2R8G8B8A1SrgbBlock:
			return 8;
		case Format::eEtc2R8G8B8A8UnormBlock:
		case Format::eEtc2R8G8B8A8SrgbBlock:
			return 8;
		case Format::eEacR11UnormBlock:
		case Format::eEacR11SnormBlock:
			return 8;
		case Format::eEacR11G11UnormBlock:
		case Format::eEacR11G11SnormBlock:
			return 16;
		case Format::eAstc4x4UnormBlock:
		case Format::eAstc4x4SfloatBlockEXT:
		case Format::eAstc4x4SrgbBlock:
			return 16;
		case Format::eAstc5x4UnormBlock:
		case Format::eAstc5x4SfloatBlockEXT:
		case Format::eAstc5x4SrgbBlock:
			return 16;
		case Format::eAstc5x5UnormBlock:
		case Format::eAstc5x5SfloatBlockEXT:
		case Format::eAstc5x5SrgbBlock:
			return 16;
		case Format::eAstc6x5UnormBlock:
		case Format::eAstc6x5SfloatBlockEXT:
		case Format::eAstc6x5SrgbBlock:
			return 16;
		case Format::eAstc6x6UnormBlock:
		case Format::eAstc6x6SfloatBlockEXT:
		case Format::eAstc6x6SrgbBlock:
			return 16;
		case Format::eAstc8x5UnormBlock:
		case Format::eAstc8x5SfloatBlockEXT:
		case Format::eAstc8x5SrgbBlock:
			return 16;
		case Format::eAstc8x6UnormBlock:
		case Format::eAstc8x6SfloatBlockEXT:
		case Format::eAstc8x6SrgbBlock:
			return 16;
		case Format::eAstc8x8UnormBlock:
		case Format::eAstc8x8SfloatBlockEXT:
		case Format::eAstc8x8SrgbBlock:
			return 16;
		case Format::eAstc10x5UnormBlock:
		case Format::eAstc10x5SfloatBlockEXT:
		case Format::eAstc10x5SrgbBlock:
			return 16;
		case Format::eAstc10x6UnormBlock:
		case Format::eAstc10x6SfloatBlockEXT:
		case Format::eAstc10x6SrgbBlock:
			return 16;
		case Format::eAstc10x8UnormBlock:
		case Format::eAstc10x8SfloatBlockEXT:
		case Format::eAstc10x8SrgbBlock:
			return 16;
		case Format::eAstc10x10UnormBlock:
		case Format::eAstc10x10SfloatBlockEXT:
		case Format::eAstc10x10SrgbBlock:
			return 16;
		case Format::eAstc12x10UnormBlock:
		case Format::eAstc12x10SfloatBlockEXT:
		case Format::eAstc12x10SrgbBlock:
			return 16;
		case Format::eAstc12x12UnormBlock:
		case Format::eAstc12x12SfloatBlockEXT:
		case Format::eAstc12x12SrgbBlock:
			return 16;
		case Format::eD16Unorm:
			return 2;
		case Format::eX8D24UnormPack32:
			return 4;
		case Format::eD32Sfloat:
			return 4;
		case Format::eS8Uint:
			return 1;
		case Format::eD16UnormS8Uint:
			return 3;
		case Format::eD24UnormS8Uint:
			return 4;
		case Format::eD32SfloatS8Uint:
			return 5;
		default:
			assert(0 && "format cannot be used with this function.");
			return 0;
		}
	}

	Extent3D format_to_texel_block_extent(Format format) noexcept {
		switch (format) {
		case Format::eR4G4UnormPack8:
		case Format::eR8Unorm:
		case Format::eR8Snorm:
		case Format::eR8Uscaled:
		case Format::eR8Sscaled:
		case Format::eR8Uint:
		case Format::eR8Sint:
		case Format::eR8Srgb:
		case Format::eR4G4B4A4UnormPack16:
		case Format::eB4G4R4A4UnormPack16:
		case Format::eR5G6B5UnormPack16:
		case Format::eB5G6R5UnormPack16:
		case Format::eR5G5B5A1UnormPack16:
		case Format::eB5G5R5A1UnormPack16:
		case Format::eA1R5G5B5UnormPack16:
		case Format::eR8G8Unorm:
		case Format::eR8G8Snorm:
		case Format::eR8G8Uscaled:
		case Format::eR8G8Sscaled:
		case Format::eR8G8Uint:
		case Format::eR8G8Sint:
		case Format::eR8G8Srgb:
		case Format::eR16Unorm:
		case Format::eR16Snorm:
		case Format::eR16Uscaled:
		case Format::eR16Sscaled:
		case Format::eR16Uint:
		case Format::eR16Sint:
		case Format::eR16Sfloat:
		case Format::eR10X6UnormPack16:
		case Format::eR12X4UnormPack16:
		case Format::eR8G8B8Unorm:
		case Format::eR8G8B8Snorm:
		case Format::eR8G8B8Uscaled:
		case Format::eR8G8B8Sscaled:
		case Format::eR8G8B8Uint:
		case Format::eR8G8B8Sint:
		case Format::eR8G8B8Srgb:
		case Format::eB8G8R8Unorm:
		case Format::eB8G8R8Snorm:
		case Format::eB8G8R8Uscaled:
		case Format::eB8G8R8Sscaled:
		case Format::eB8G8R8Uint:
		case Format::eB8G8R8Sint:
		case Format::eB8G8R8Srgb:
		case Format::eR8G8B8A8Unorm:
		case Format::eR8G8B8A8Snorm:
		case Format::eR8G8B8A8Uscaled:
		case Format::eR8G8B8A8Sscaled:
		case Format::eR8G8B8A8Uint:
		case Format::eR8G8B8A8Sint:
		case Format::eR8G8B8A8Srgb:
		case Format::eB8G8R8A8Unorm:
		case Format::eB8G8R8A8Snorm:
		case Format::eB8G8R8A8Uscaled:
		case Format::eB8G8R8A8Sscaled:
		case Format::eB8G8R8A8Uint:
		case Format::eB8G8R8A8Sint:
		case Format::eB8G8R8A8Srgb:
		case Format::eA8B8G8R8UnormPack32:
		case Format::eA8B8G8R8SnormPack32:
		case Format::eA8B8G8R8UscaledPack32:
		case Format::eA8B8G8R8SscaledPack32:
		case Format::eA8B8G8R8UintPack32:
		case Format::eA8B8G8R8SintPack32:
		case Format::eA8B8G8R8SrgbPack32:
		case Format::eA2R10G10B10UnormPack32:
		case Format::eA2R10G10B10SnormPack32:
		case Format::eA2R10G10B10UscaledPack32:
		case Format::eA2R10G10B10SscaledPack32:
		case Format::eA2R10G10B10UintPack32:
		case Format::eA2R10G10B10SintPack32:
		case Format::eA2B10G10R10UnormPack32:
		case Format::eA2B10G10R10SnormPack32:
		case Format::eA2B10G10R10UscaledPack32:
		case Format::eA2B10G10R10SscaledPack32:
		case Format::eA2B10G10R10UintPack32:
		case Format::eA2B10G10R10SintPack32:
		case Format::eR16G16Unorm:
		case Format::eR16G16Snorm:
		case Format::eR16G16Uscaled:
		case Format::eR16G16Sscaled:
		case Format::eR16G16Uint:
		case Format::eR16G16Sint:
		case Format::eR16G16Sfloat:
		case Format::eR32Uint:
		case Format::eR32Sint:
		case Format::eR32Sfloat:
		case Format::eB10G11R11UfloatPack32:
		case Format::eE5B9G9R9UfloatPack32:
		case Format::eR10X6G10X6Unorm2Pack16:
		case Format::eR12X4G12X4Unorm2Pack16:
		case Format::eG8B8G8R8422Unorm:
		case Format::eB8G8R8G8422Unorm:
		case Format::eR16G16B16Unorm:
		case Format::eR16G16B16Snorm:
		case Format::eR16G16B16Uscaled:
		case Format::eR16G16B16Sscaled:
		case Format::eR16G16B16Uint:
		case Format::eR16G16B16Sint:
		case Format::eR16G16B16Sfloat:
		case Format::eR16G16B16A16Unorm:
		case Format::eR16G16B16A16Snorm:
		case Format::eR16G16B16A16Uscaled:
		case Format::eR16G16B16A16Sscaled:
		case Format::eR16G16B16A16Uint:
		case Format::eR16G16B16A16Sint:
		case Format::eR16G16B16A16Sfloat:
		case Format::eR32G32Uint:
		case Format::eR32G32Sint:
		case Format::eR32G32Sfloat:
		case Format::eR64Uint:
		case Format::eR64Sint:
		case Format::eR64Sfloat:
		case Format::eR10X6G10X6B10X6A10X6Unorm4Pack16:
		case Format::eG10X6B10X6G10X6R10X6422Unorm4Pack16:
		case Format::eB10X6G10X6R10X6G10X6422Unorm4Pack16:
		case Format::eR12X4G12X4B12X4A12X4Unorm4Pack16:
		case Format::eG12X4B12X4G12X4R12X4422Unorm4Pack16:
		case Format::eB12X4G12X4R12X4G12X4422Unorm4Pack16:
		case Format::eG16B16G16R16422Unorm:
		case Format::eB16G16R16G16422Unorm:
		case Format::eR32G32B32Uint:
		case Format::eR32G32B32Sint:
		case Format::eR32G32B32Sfloat:
		case Format::eR32G32B32A32Uint:
		case Format::eR32G32B32A32Sint:
		case Format::eR32G32B32A32Sfloat:
		case Format::eR64G64Uint:
		case Format::eR64G64Sint:
		case Format::eR64G64Sfloat:
		case Format::eR64G64B64Uint:
		case Format::eR64G64B64Sint:
		case Format::eR64G64B64Sfloat:
		case Format::eR64G64B64A64Uint:
		case Format::eR64G64B64A64Sint:
		case Format::eR64G64B64A64Sfloat:
			return { 1, 1, 1 };
		case Format::eBc1RgbUnormBlock:
		case Format::eBc1RgbSrgbBlock:
			return { 4, 4, 1 };
		case Format::eBc1RgbaUnormBlock:
		case Format::eBc1RgbaSrgbBlock:
			return { 4, 4, 1 };
		case Format::eBc2UnormBlock:
		case Format::eBc2SrgbBlock:
			return { 4, 4, 1 };
		case Format::eBc3UnormBlock:
		case Format::eBc3SrgbBlock:
			return { 4, 4, 1 };
		case Format::eBc4UnormBlock:
		case Format::eBc4SnormBlock:
			return { 4, 4, 1 };
		case Format::eBc5UnormBlock:
		case Format::eBc5SnormBlock:
			return { 4, 4, 1 };
		case Format::eBc6HUfloatBlock:
		case Format::eBc6HSfloatBlock:
			return { 4, 4, 1 };
		case Format::eBc7UnormBlock:
		case Format::eBc7SrgbBlock:
			return { 4, 4, 1 };
		case Format::eEtc2R8G8B8UnormBlock:
		case Format::eEtc2R8G8B8SrgbBlock:
			return { 4, 4, 1 };
		case Format::eEtc2R8G8B8A1UnormBlock:
		case Format::eEtc2R8G8B8A1SrgbBlock:
			return { 4, 4, 1 };
		case Format::eEtc2R8G8B8A8UnormBlock:
		case Format::eEtc2R8G8B8A8SrgbBlock:
			return { 4, 4, 1 };
		case Format::eEacR11UnormBlock:
		case Format::eEacR11SnormBlock:
			return { 4, 4, 1 };
		case Format::eEacR11G11UnormBlock:
		case Format::eEacR11G11SnormBlock:
			return { 4, 4, 1 };
		case Format::eAstc4x4UnormBlock:
		case Format::eAstc4x4SfloatBlockEXT:
		case Format::eAstc4x4SrgbBlock:
			return { 4, 4, 1 };
		case Format::eAstc5x4UnormBlock:
		case Format::eAstc5x4SfloatBlockEXT:
		case Format::eAstc5x4SrgbBlock:
			return { 5, 4, 1 };
		case Format::eAstc5x5UnormBlock:
		case Format::eAstc5x5SfloatBlockEXT:
		case Format::eAstc5x5SrgbBlock:
			return { 5, 5, 1 };
		case Format::eAstc6x5UnormBlock:
		case Format::eAstc6x5SfloatBlockEXT:
		case Format::eAstc6x5SrgbBlock:
			return { 6, 5, 1 };
		case Format::eAstc6x6UnormBlock:
		case Format::eAstc6x6SfloatBlockEXT:
		case Format::eAstc6x6SrgbBlock:
			return { 6, 6, 1 };
		case Format::eAstc8x5UnormBlock:
		case Format::eAstc8x5SfloatBlockEXT:
		case Format::eAstc8x5SrgbBlock:
			return { 8, 5, 1 };
		case Format::eAstc8x6UnormBlock:
		case Format::eAstc8x6SfloatBlockEXT:
		case Format::eAstc8x6SrgbBlock:
			return { 8, 6, 1 };
		case Format::eAstc8x8UnormBlock:
		case Format::eAstc8x8SfloatBlockEXT:
		case Format::eAstc8x8SrgbBlock:
			return { 8, 8, 1 };
		case Format::eAstc10x5UnormBlock:
		case Format::eAstc10x5SfloatBlockEXT:
		case Format::eAstc10x5SrgbBlock:
			return { 10, 5, 1 };
		case Format::eAstc10x6UnormBlock:
		case Format::eAstc10x6SfloatBlockEXT:
		case Format::eAstc10x6SrgbBlock:
			return { 10, 6, 1 };
		case Format::eAstc10x8UnormBlock:
		case Format::eAstc10x8SfloatBlockEXT:
		case Format::eAstc10x8SrgbBlock:
			return { 10, 8, 1 };
		case Format::eAstc10x10UnormBlock:
		case Format::eAstc10x10SfloatBlockEXT:
		case Format::eAstc10x10SrgbBlock:
			return { 10, 10, 1 };
		case Format::eAstc12x10UnormBlock:
		case Format::eAstc12x10SfloatBlockEXT:
		case Format::eAstc12x10SrgbBlock:
			return { 12, 10, 1 };
		case Format::eAstc12x12UnormBlock:
		case Format::eAstc12x12SfloatBlockEXT:
		case Format::eAstc12x12SrgbBlock:
			return { 12, 12, 1 };
		case Format::eD16Unorm:
		case Format::eX8D24UnormPack32:
		case Format::eD32Sfloat:
		case Format::eS8Uint:
		case Format::eD16UnormS8Uint:
		case Format::eD24UnormS8Uint:
		case Format::eD32SfloatS8Uint:
			return { 1, 1, 1 };
		default:
			assert(0 && "format cannot be used with this function.");
			return { 0, 0, 0 };
		}
	}

	uint32_t compute_image_size(Format format, Extent3D extent) noexcept {
		auto block_extent = format_to_texel_block_extent(format);
		auto extent_in_blocks = Extent3D{ (extent.width + block_extent.width - 1) / block_extent.width,
			                                (extent.height + block_extent.height - 1) / block_extent.height,
			                                (extent.depth + block_extent.depth - 1) / block_extent.depth };
		return extent_in_blocks.width * extent_in_blocks.height * extent_in_blocks.depth * format_to_texel_block_size(format);
	}

	ImageAspectFlags format_to_aspect(Format format) noexcept {
		switch (format) {
		case Format::eD16Unorm:
		case Format::eD32Sfloat:
		case Format::eX8D24UnormPack32:
			return ImageAspectFlagBits::eDepth;
		case Format::eD16UnormS8Uint:
		case Format::eD24UnormS8Uint:
		case Format::eD32SfloatS8Uint:
			return ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil;
		case Format::eS8Uint:
			return ImageAspectFlagBits::eStencil;
		default:
			return ImageAspectFlagBits::eColor;
		}
	}
} // namespace vuk