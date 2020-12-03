#include <vuk/Image.hpp>
#include <cassert>

namespace vuk {
	uint32_t format_to_texel_block_size(vuk::Format format) noexcept {
		switch (format) {
		case vuk::Format::eR4G4UnormPack8:
		case vuk::Format::eR8Unorm:
		case vuk::Format::eR8Snorm:
		case vuk::Format::eR8Uscaled:
		case vuk::Format::eR8Sscaled:
		case vuk::Format::eR8Uint:
		case vuk::Format::eR8Sint:
		case vuk::Format::eR8Srgb:
			return 1;
		case vuk::Format::eR4G4B4A4UnormPack16:
		case vuk::Format::eB4G4R4A4UnormPack16:
		case vuk::Format::eR5G6B5UnormPack16:
		case vuk::Format::eB5G6R5UnormPack16:
		case vuk::Format::eR5G5B5A1UnormPack16:
		case vuk::Format::eB5G5R5A1UnormPack16:
		case vuk::Format::eA1R5G5B5UnormPack16:
		case vuk::Format::eR8G8Unorm:
		case vuk::Format::eR8G8Snorm:
		case vuk::Format::eR8G8Uscaled:
		case vuk::Format::eR8G8Sscaled:
		case vuk::Format::eR8G8Uint:
		case vuk::Format::eR8G8Sint:
		case vuk::Format::eR8G8Srgb:
		case vuk::Format::eR16Unorm:
		case vuk::Format::eR16Snorm:
		case vuk::Format::eR16Uscaled:
		case vuk::Format::eR16Sscaled:
		case vuk::Format::eR16Uint:
		case vuk::Format::eR16Sint:
		case vuk::Format::eR16Sfloat:
		case vuk::Format::eR10X6UnormPack16:
		case vuk::Format::eR12X4UnormPack16:
			return 2;
		case vuk::Format::eR8G8B8Unorm:
		case vuk::Format::eR8G8B8Snorm:
		case vuk::Format::eR8G8B8Uscaled:
		case vuk::Format::eR8G8B8Sscaled:
		case vuk::Format::eR8G8B8Uint:
		case vuk::Format::eR8G8B8Sint:
		case vuk::Format::eR8G8B8Srgb:
		case vuk::Format::eB8G8R8Unorm:
		case vuk::Format::eB8G8R8Snorm:
		case vuk::Format::eB8G8R8Uscaled:
		case vuk::Format::eB8G8R8Sscaled:
		case vuk::Format::eB8G8R8Uint:
		case vuk::Format::eB8G8R8Sint:
		case vuk::Format::eB8G8R8Srgb:
			return 3;
		case vuk::Format::eR8G8B8A8Unorm:
		case vuk::Format::eR8G8B8A8Snorm:
		case vuk::Format::eR8G8B8A8Uscaled:
		case vuk::Format::eR8G8B8A8Sscaled:
		case vuk::Format::eR8G8B8A8Uint:
		case vuk::Format::eR8G8B8A8Sint:
		case vuk::Format::eR8G8B8A8Srgb:
		case vuk::Format::eB8G8R8A8Unorm:
		case vuk::Format::eB8G8R8A8Snorm:
		case vuk::Format::eB8G8R8A8Uscaled:
		case vuk::Format::eB8G8R8A8Sscaled:
		case vuk::Format::eB8G8R8A8Uint:
		case vuk::Format::eB8G8R8A8Sint:
		case vuk::Format::eB8G8R8A8Srgb:
		case vuk::Format::eA8B8G8R8UnormPack32:
		case vuk::Format::eA8B8G8R8SnormPack32:
		case vuk::Format::eA8B8G8R8UscaledPack32:
		case vuk::Format::eA8B8G8R8SscaledPack32:
		case vuk::Format::eA8B8G8R8UintPack32:
		case vuk::Format::eA8B8G8R8SintPack32:
		case vuk::Format::eA8B8G8R8SrgbPack32:
		case vuk::Format::eA2R10G10B10UnormPack32:
		case vuk::Format::eA2R10G10B10SnormPack32:
		case vuk::Format::eA2R10G10B10UscaledPack32:
		case vuk::Format::eA2R10G10B10SscaledPack32:
		case vuk::Format::eA2R10G10B10UintPack32:
		case vuk::Format::eA2R10G10B10SintPack32:
		case vuk::Format::eA2B10G10R10UnormPack32:
		case vuk::Format::eA2B10G10R10SnormPack32:
		case vuk::Format::eA2B10G10R10UscaledPack32:
		case vuk::Format::eA2B10G10R10SscaledPack32:
		case vuk::Format::eA2B10G10R10UintPack32:
		case vuk::Format::eA2B10G10R10SintPack32:
		case vuk::Format::eR16G16Unorm:
		case vuk::Format::eR16G16Snorm:
		case vuk::Format::eR16G16Uscaled:
		case vuk::Format::eR16G16Sscaled:
		case vuk::Format::eR16G16Uint:
		case vuk::Format::eR16G16Sint:
		case vuk::Format::eR16G16Sfloat:
		case vuk::Format::eR32Uint:
		case vuk::Format::eR32Sint:
		case vuk::Format::eR32Sfloat:
		case vuk::Format::eB10G11R11UfloatPack32:
		case vuk::Format::eE5B9G9R9UfloatPack32:
		case vuk::Format::eR10X6G10X6Unorm2Pack16:
		case vuk::Format::eR12X4G12X4Unorm2Pack16:
			return 4;
		case vuk::Format::eG8B8G8R8422Unorm:
			return 4;
		case vuk::Format::eB8G8R8G8422Unorm:
			return 4;
		case vuk::Format::eR16G16B16Unorm:
		case vuk::Format::eR16G16B16Snorm:
		case vuk::Format::eR16G16B16Uscaled:
		case vuk::Format::eR16G16B16Sscaled:
		case vuk::Format::eR16G16B16Uint:
		case vuk::Format::eR16G16B16Sint:
		case vuk::Format::eR16G16B16Sfloat:
			return 6;
		case vuk::Format::eR16G16B16A16Unorm:
		case vuk::Format::eR16G16B16A16Snorm:
		case vuk::Format::eR16G16B16A16Uscaled:
		case vuk::Format::eR16G16B16A16Sscaled:
		case vuk::Format::eR16G16B16A16Uint:
		case vuk::Format::eR16G16B16A16Sint:
		case vuk::Format::eR16G16B16A16Sfloat:
		case vuk::Format::eR32G32Uint:
		case vuk::Format::eR32G32Sint:
		case vuk::Format::eR32G32Sfloat:
		case vuk::Format::eR64Uint:
		case vuk::Format::eR64Sint:
		case vuk::Format::eR64Sfloat:
			return 8;
		case vuk::Format::eR10X6G10X6B10X6A10X6Unorm4Pack16:
			return 8;
		case vuk::Format::eG10X6B10X6G10X6R10X6422Unorm4Pack16:
			return 8;
		case vuk::Format::eB10X6G10X6R10X6G10X6422Unorm4Pack16:
			return 8;
		case vuk::Format::eR12X4G12X4B12X4A12X4Unorm4Pack16:
			return 8;
		case vuk::Format::eG12X4B12X4G12X4R12X4422Unorm4Pack16:
			return 8;
		case vuk::Format::eB12X4G12X4R12X4G12X4422Unorm4Pack16:
			return 8;
		case vuk::Format::eG16B16G16R16422Unorm:
			return 8;
		case vuk::Format::eB16G16R16G16422Unorm:
			return 8;
		case vuk::Format::eR32G32B32Uint:
		case vuk::Format::eR32G32B32Sint:
		case vuk::Format::eR32G32B32Sfloat:
			return 12;
		case vuk::Format::eR32G32B32A32Uint:
		case vuk::Format::eR32G32B32A32Sint:
		case vuk::Format::eR32G32B32A32Sfloat:
		case vuk::Format::eR64G64Uint:
		case vuk::Format::eR64G64Sint:
		case vuk::Format::eR64G64Sfloat:
			return 16;
		case vuk::Format::eR64G64B64Uint:
		case vuk::Format::eR64G64B64Sint:
		case vuk::Format::eR64G64B64Sfloat:
			return 24;
		case vuk::Format::eR64G64B64A64Uint:
		case vuk::Format::eR64G64B64A64Sint:
		case vuk::Format::eR64G64B64A64Sfloat:
			return 32;
		case vuk::Format::eBc1RgbUnormBlock:
		case vuk::Format::eBc1RgbSrgbBlock:
			return 8;
		case vuk::Format::eBc1RgbaUnormBlock:
		case vuk::Format::eBc1RgbaSrgbBlock:
			return 8;
		case vuk::Format::eBc2UnormBlock:
		case vuk::Format::eBc2SrgbBlock:
			return 16;
		case vuk::Format::eBc3UnormBlock:
		case vuk::Format::eBc3SrgbBlock:
			return 16;
		case vuk::Format::eBc4UnormBlock:
		case vuk::Format::eBc4SnormBlock:
			return 8;
		case vuk::Format::eBc5UnormBlock:
		case vuk::Format::eBc5SnormBlock:
			return 16;
		case vuk::Format::eBc6HUfloatBlock:
		case vuk::Format::eBc6HSfloatBlock:
			return 16;
		case vuk::Format::eBc7UnormBlock:
		case vuk::Format::eBc7SrgbBlock:
			return 16;
		case vuk::Format::eEtc2R8G8B8UnormBlock:
		case vuk::Format::eEtc2R8G8B8SrgbBlock:
			return 8;
		case vuk::Format::eEtc2R8G8B8A1UnormBlock:
		case vuk::Format::eEtc2R8G8B8A1SrgbBlock:
			return 8;
		case vuk::Format::eEtc2R8G8B8A8UnormBlock:
		case vuk::Format::eEtc2R8G8B8A8SrgbBlock:
			return 8;
		case vuk::Format::eEacR11UnormBlock:
		case vuk::Format::eEacR11SnormBlock:
			return 8;
		case vuk::Format::eEacR11G11UnormBlock:
		case vuk::Format::eEacR11G11SnormBlock:
			return 16;
		case vuk::Format::eAstc4x4UnormBlock:
		case vuk::Format::eAstc4x4SfloatBlockEXT:
		case vuk::Format::eAstc4x4SrgbBlock:
			return 16;
		case vuk::Format::eAstc5x4UnormBlock:
		case vuk::Format::eAstc5x4SfloatBlockEXT:
		case vuk::Format::eAstc5x4SrgbBlock:
			return 16;
		case vuk::Format::eAstc5x5UnormBlock:
		case vuk::Format::eAstc5x5SfloatBlockEXT:
		case vuk::Format::eAstc5x5SrgbBlock:
			return 16;
		case vuk::Format::eAstc6x5UnormBlock:
		case vuk::Format::eAstc6x5SfloatBlockEXT:
		case vuk::Format::eAstc6x5SrgbBlock:
			return 16;
		case vuk::Format::eAstc6x6UnormBlock:
		case vuk::Format::eAstc6x6SfloatBlockEXT:
		case vuk::Format::eAstc6x6SrgbBlock:
			return 16;
		case vuk::Format::eAstc8x5UnormBlock:
		case vuk::Format::eAstc8x5SfloatBlockEXT:
		case vuk::Format::eAstc8x5SrgbBlock:
			return 16;
		case vuk::Format::eAstc8x6UnormBlock:
		case vuk::Format::eAstc8x6SfloatBlockEXT:
		case vuk::Format::eAstc8x6SrgbBlock:
			return 16;
		case vuk::Format::eAstc8x8UnormBlock:
		case vuk::Format::eAstc8x8SfloatBlockEXT:
		case vuk::Format::eAstc8x8SrgbBlock:
			return 16;
		case vuk::Format::eAstc10x5UnormBlock:
		case vuk::Format::eAstc10x5SfloatBlockEXT:
		case vuk::Format::eAstc10x5SrgbBlock:
			return 16;
		case vuk::Format::eAstc10x6UnormBlock:
		case vuk::Format::eAstc10x6SfloatBlockEXT:
		case vuk::Format::eAstc10x6SrgbBlock:
			return 16;
		case vuk::Format::eAstc10x8UnormBlock:
		case vuk::Format::eAstc10x8SfloatBlockEXT:
		case vuk::Format::eAstc10x8SrgbBlock:
			return 16;
		case vuk::Format::eAstc10x10UnormBlock:
		case vuk::Format::eAstc10x10SfloatBlockEXT:
		case vuk::Format::eAstc10x10SrgbBlock:
			return 16;
		case vuk::Format::eAstc12x10UnormBlock:
		case vuk::Format::eAstc12x10SfloatBlockEXT:
		case vuk::Format::eAstc12x10SrgbBlock:
			return 16;
		case vuk::Format::eAstc12x12UnormBlock:
		case vuk::Format::eAstc12x12SfloatBlockEXT:
		case vuk::Format::eAstc12x12SrgbBlock:
			return 16;
		case vuk::Format::eD16Unorm:
			return 2;
		case vuk::Format::eX8D24UnormPack32:
			return 4;
		case vuk::Format::eD32Sfloat:
			return 4;
		case vuk::Format::eS8Uint:
			return 1;
		case vuk::Format::eD16UnormS8Uint:
			return 3;
		case vuk::Format::eD24UnormS8Uint:
			return 4;
		case vuk::Format::eD32SfloatS8Uint:
			return 5;
		default:
			assert(0 && "format cannot be used with this function.");
			return 0;
		}
	}

	Extent3D format_to_texel_block_extent(vuk::Format format) noexcept {
		switch (format) {
		case vuk::Format::eR4G4UnormPack8:
		case vuk::Format::eR8Unorm:
		case vuk::Format::eR8Snorm:
		case vuk::Format::eR8Uscaled:
		case vuk::Format::eR8Sscaled:
		case vuk::Format::eR8Uint:
		case vuk::Format::eR8Sint:
		case vuk::Format::eR8Srgb:
		case vuk::Format::eR4G4B4A4UnormPack16:
		case vuk::Format::eB4G4R4A4UnormPack16:
		case vuk::Format::eR5G6B5UnormPack16:
		case vuk::Format::eB5G6R5UnormPack16:
		case vuk::Format::eR5G5B5A1UnormPack16:
		case vuk::Format::eB5G5R5A1UnormPack16:
		case vuk::Format::eA1R5G5B5UnormPack16:
		case vuk::Format::eR8G8Unorm:
		case vuk::Format::eR8G8Snorm:
		case vuk::Format::eR8G8Uscaled:
		case vuk::Format::eR8G8Sscaled:
		case vuk::Format::eR8G8Uint:
		case vuk::Format::eR8G8Sint:
		case vuk::Format::eR8G8Srgb:
		case vuk::Format::eR16Unorm:
		case vuk::Format::eR16Snorm:
		case vuk::Format::eR16Uscaled:
		case vuk::Format::eR16Sscaled:
		case vuk::Format::eR16Uint:
		case vuk::Format::eR16Sint:
		case vuk::Format::eR16Sfloat:
		case vuk::Format::eR10X6UnormPack16:
		case vuk::Format::eR12X4UnormPack16:
		case vuk::Format::eR8G8B8Unorm:
		case vuk::Format::eR8G8B8Snorm:
		case vuk::Format::eR8G8B8Uscaled:
		case vuk::Format::eR8G8B8Sscaled:
		case vuk::Format::eR8G8B8Uint:
		case vuk::Format::eR8G8B8Sint:
		case vuk::Format::eR8G8B8Srgb:
		case vuk::Format::eB8G8R8Unorm:
		case vuk::Format::eB8G8R8Snorm:
		case vuk::Format::eB8G8R8Uscaled:
		case vuk::Format::eB8G8R8Sscaled:
		case vuk::Format::eB8G8R8Uint:
		case vuk::Format::eB8G8R8Sint:
		case vuk::Format::eB8G8R8Srgb:
		case vuk::Format::eR8G8B8A8Unorm:
		case vuk::Format::eR8G8B8A8Snorm:
		case vuk::Format::eR8G8B8A8Uscaled:
		case vuk::Format::eR8G8B8A8Sscaled:
		case vuk::Format::eR8G8B8A8Uint:
		case vuk::Format::eR8G8B8A8Sint:
		case vuk::Format::eR8G8B8A8Srgb:
		case vuk::Format::eB8G8R8A8Unorm:
		case vuk::Format::eB8G8R8A8Snorm:
		case vuk::Format::eB8G8R8A8Uscaled:
		case vuk::Format::eB8G8R8A8Sscaled:
		case vuk::Format::eB8G8R8A8Uint:
		case vuk::Format::eB8G8R8A8Sint:
		case vuk::Format::eB8G8R8A8Srgb:
		case vuk::Format::eA8B8G8R8UnormPack32:
		case vuk::Format::eA8B8G8R8SnormPack32:
		case vuk::Format::eA8B8G8R8UscaledPack32:
		case vuk::Format::eA8B8G8R8SscaledPack32:
		case vuk::Format::eA8B8G8R8UintPack32:
		case vuk::Format::eA8B8G8R8SintPack32:
		case vuk::Format::eA8B8G8R8SrgbPack32:
		case vuk::Format::eA2R10G10B10UnormPack32:
		case vuk::Format::eA2R10G10B10SnormPack32:
		case vuk::Format::eA2R10G10B10UscaledPack32:
		case vuk::Format::eA2R10G10B10SscaledPack32:
		case vuk::Format::eA2R10G10B10UintPack32:
		case vuk::Format::eA2R10G10B10SintPack32:
		case vuk::Format::eA2B10G10R10UnormPack32:
		case vuk::Format::eA2B10G10R10SnormPack32:
		case vuk::Format::eA2B10G10R10UscaledPack32:
		case vuk::Format::eA2B10G10R10SscaledPack32:
		case vuk::Format::eA2B10G10R10UintPack32:
		case vuk::Format::eA2B10G10R10SintPack32:
		case vuk::Format::eR16G16Unorm:
		case vuk::Format::eR16G16Snorm:
		case vuk::Format::eR16G16Uscaled:
		case vuk::Format::eR16G16Sscaled:
		case vuk::Format::eR16G16Uint:
		case vuk::Format::eR16G16Sint:
		case vuk::Format::eR16G16Sfloat:
		case vuk::Format::eR32Uint:
		case vuk::Format::eR32Sint:
		case vuk::Format::eR32Sfloat:
		case vuk::Format::eB10G11R11UfloatPack32:
		case vuk::Format::eE5B9G9R9UfloatPack32:
		case vuk::Format::eR10X6G10X6Unorm2Pack16:
		case vuk::Format::eR12X4G12X4Unorm2Pack16:
		case vuk::Format::eG8B8G8R8422Unorm:
		case vuk::Format::eB8G8R8G8422Unorm:
		case vuk::Format::eR16G16B16Unorm:
		case vuk::Format::eR16G16B16Snorm:
		case vuk::Format::eR16G16B16Uscaled:
		case vuk::Format::eR16G16B16Sscaled:
		case vuk::Format::eR16G16B16Uint:
		case vuk::Format::eR16G16B16Sint:
		case vuk::Format::eR16G16B16Sfloat:
		case vuk::Format::eR16G16B16A16Unorm:
		case vuk::Format::eR16G16B16A16Snorm:
		case vuk::Format::eR16G16B16A16Uscaled:
		case vuk::Format::eR16G16B16A16Sscaled:
		case vuk::Format::eR16G16B16A16Uint:
		case vuk::Format::eR16G16B16A16Sint:
		case vuk::Format::eR16G16B16A16Sfloat:
		case vuk::Format::eR32G32Uint:
		case vuk::Format::eR32G32Sint:
		case vuk::Format::eR32G32Sfloat:
		case vuk::Format::eR64Uint:
		case vuk::Format::eR64Sint:
		case vuk::Format::eR64Sfloat:
		case vuk::Format::eR10X6G10X6B10X6A10X6Unorm4Pack16:
		case vuk::Format::eG10X6B10X6G10X6R10X6422Unorm4Pack16:
		case vuk::Format::eB10X6G10X6R10X6G10X6422Unorm4Pack16:
		case vuk::Format::eR12X4G12X4B12X4A12X4Unorm4Pack16:
		case vuk::Format::eG12X4B12X4G12X4R12X4422Unorm4Pack16:
		case vuk::Format::eB12X4G12X4R12X4G12X4422Unorm4Pack16:
		case vuk::Format::eG16B16G16R16422Unorm:
		case vuk::Format::eB16G16R16G16422Unorm:
		case vuk::Format::eR32G32B32Uint:
		case vuk::Format::eR32G32B32Sint:
		case vuk::Format::eR32G32B32Sfloat:
		case vuk::Format::eR32G32B32A32Uint:
		case vuk::Format::eR32G32B32A32Sint:
		case vuk::Format::eR32G32B32A32Sfloat:
		case vuk::Format::eR64G64Uint:
		case vuk::Format::eR64G64Sint:
		case vuk::Format::eR64G64Sfloat:
		case vuk::Format::eR64G64B64Uint:
		case vuk::Format::eR64G64B64Sint:
		case vuk::Format::eR64G64B64Sfloat:
		case vuk::Format::eR64G64B64A64Uint:
		case vuk::Format::eR64G64B64A64Sint:
		case vuk::Format::eR64G64B64A64Sfloat:
			return { 1,1,1 };
		case vuk::Format::eBc1RgbUnormBlock:
		case vuk::Format::eBc1RgbSrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eBc1RgbaUnormBlock:
		case vuk::Format::eBc1RgbaSrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eBc2UnormBlock:
		case vuk::Format::eBc2SrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eBc3UnormBlock:
		case vuk::Format::eBc3SrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eBc4UnormBlock:
		case vuk::Format::eBc4SnormBlock:
			return { 4,4,1 };
		case vuk::Format::eBc5UnormBlock:
		case vuk::Format::eBc5SnormBlock:
			return{ 4,4,1 };
		case vuk::Format::eBc6HUfloatBlock:
		case vuk::Format::eBc6HSfloatBlock:
			return{ 4,4,1 };
		case vuk::Format::eBc7UnormBlock:
		case vuk::Format::eBc7SrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eEtc2R8G8B8UnormBlock:
		case vuk::Format::eEtc2R8G8B8SrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eEtc2R8G8B8A1UnormBlock:
		case vuk::Format::eEtc2R8G8B8A1SrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eEtc2R8G8B8A8UnormBlock:
		case vuk::Format::eEtc2R8G8B8A8SrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eEacR11UnormBlock:
		case vuk::Format::eEacR11SnormBlock:
			return { 4,4,1 };
		case vuk::Format::eEacR11G11UnormBlock:
		case vuk::Format::eEacR11G11SnormBlock:
			return { 4,4,1 };
		case vuk::Format::eAstc4x4UnormBlock:
		case vuk::Format::eAstc4x4SfloatBlockEXT:
		case vuk::Format::eAstc4x4SrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eAstc5x4UnormBlock:
		case vuk::Format::eAstc5x4SfloatBlockEXT:
		case vuk::Format::eAstc5x4SrgbBlock:
			return { 5,4,1 };
		case vuk::Format::eAstc5x5UnormBlock:
		case vuk::Format::eAstc5x5SfloatBlockEXT:
		case vuk::Format::eAstc5x5SrgbBlock:
			return { 5,5,1 };
		case vuk::Format::eAstc6x5UnormBlock:
		case vuk::Format::eAstc6x5SfloatBlockEXT:
		case vuk::Format::eAstc6x5SrgbBlock:
			return { 6,5,1 };
		case vuk::Format::eAstc6x6UnormBlock:
		case vuk::Format::eAstc6x6SfloatBlockEXT:
		case vuk::Format::eAstc6x6SrgbBlock:
			return { 6,6,1 };
		case vuk::Format::eAstc8x5UnormBlock:
		case vuk::Format::eAstc8x5SfloatBlockEXT:
		case vuk::Format::eAstc8x5SrgbBlock:
			return { 8,5,1 };
		case vuk::Format::eAstc8x6UnormBlock:
		case vuk::Format::eAstc8x6SfloatBlockEXT:
		case vuk::Format::eAstc8x6SrgbBlock:
			return { 8,6,1 };
		case vuk::Format::eAstc8x8UnormBlock:
		case vuk::Format::eAstc8x8SfloatBlockEXT:
		case vuk::Format::eAstc8x8SrgbBlock:
			return { 8,8,1 };
		case vuk::Format::eAstc10x5UnormBlock:
		case vuk::Format::eAstc10x5SfloatBlockEXT:
		case vuk::Format::eAstc10x5SrgbBlock:
			return { 10, 5, 1 };
		case vuk::Format::eAstc10x6UnormBlock:
		case vuk::Format::eAstc10x6SfloatBlockEXT:
		case vuk::Format::eAstc10x6SrgbBlock:
			return { 10, 6, 1 };
		case vuk::Format::eAstc10x8UnormBlock:
		case vuk::Format::eAstc10x8SfloatBlockEXT:
		case vuk::Format::eAstc10x8SrgbBlock:
			return { 10, 8, 1 };
		case vuk::Format::eAstc10x10UnormBlock:
		case vuk::Format::eAstc10x10SfloatBlockEXT:
		case vuk::Format::eAstc10x10SrgbBlock:
			return { 10, 10, 1 };
		case vuk::Format::eAstc12x10UnormBlock:
		case vuk::Format::eAstc12x10SfloatBlockEXT:
		case vuk::Format::eAstc12x10SrgbBlock:
			return { 12, 10, 1 };
		case vuk::Format::eAstc12x12UnormBlock:
		case vuk::Format::eAstc12x12SfloatBlockEXT:
		case vuk::Format::eAstc12x12SrgbBlock:
			return { 12, 12, 1 };
		case vuk::Format::eD16Unorm:
		case vuk::Format::eX8D24UnormPack32:
		case vuk::Format::eD32Sfloat:
		case vuk::Format::eS8Uint:
		case vuk::Format::eD16UnormS8Uint:
		case vuk::Format::eD24UnormS8Uint:
		case vuk::Format::eD32SfloatS8Uint:
			return { 1,1,1 };
		default:
			assert(0 && "format cannot be used with this function.");
			return { 0,0,0 };
		}
	}

	uint32_t compute_image_size(vuk::Format format, vuk::Extent3D extent) noexcept {
		auto block_extent = format_to_texel_block_extent(format);
		auto extent_in_blocks = Extent3D{
			(extent.width + block_extent.width - 1) / block_extent.width,
			(extent.height + block_extent.height - 1) / block_extent.height,
			(extent.depth + block_extent.depth - 1) / block_extent.depth
		};
		return extent_in_blocks.width * extent_in_blocks.height * extent_in_blocks.depth * format_to_texel_block_size(format);
	}

}