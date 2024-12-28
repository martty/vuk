#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/runtime/vk/Image.hpp"
#include "vuk/vuk_fwd.hpp"
#include <cmath>
#include <compare>

namespace vuk {
	struct ImageAttachment {
		Image image = {};
		ImageView image_view = {};

		ImageCreateFlags image_flags = {};
		ImageType image_type = ImageType::e2D;
		ImageTiling tiling = ImageTiling::eOptimal;
		ImageUsageFlags usage = {};
		Extent3D extent = {};
		Format format = Format::eUndefined;
		Samples sample_count = Samples::eInfer;
		bool allow_srgb_unorm_mutable = false;
		ImageViewCreateFlags image_view_flags = {};
		ImageViewType view_type = ImageViewType::eInfer;
		ComponentMapping components = {};
		ImageLayout layout = ImageLayout::eUndefined;

		uint32_t base_level = VK_REMAINING_MIP_LEVELS;
		uint32_t level_count = VK_REMAINING_MIP_LEVELS;

		uint32_t base_layer = VK_REMAINING_ARRAY_LAYERS;
		uint32_t layer_count = VK_REMAINING_ARRAY_LAYERS;

		bool operator==(const ImageAttachment&) const = default;

		constexpr bool has_concrete_image() const noexcept {
			return image != Image{};
		}

		constexpr bool has_concrete_image_view() const noexcept {
			return image_view != ImageView{};
		}

		constexpr bool may_require_image_view() const noexcept {
			return usage == ImageUsageFlagBits::eInfer ||
			       (usage & (ImageUsageFlagBits::eColorAttachment | ImageUsageFlagBits::eDepthStencilAttachment | ImageUsageFlagBits::eSampled |
			                 ImageUsageFlagBits::eStorage | ImageUsageFlagBits::eInputAttachment)) != ImageUsageFlags{};
		}

		constexpr bool is_fully_known() const noexcept {
			return image_type != ImageType::eInfer && usage != ImageUsageFlagBits::eInfer && extent.width != 0 && extent.height != 0 && extent.depth != 0 &&
			       format != Format::eUndefined && sample_count != Samples::eInfer && base_level != VK_REMAINING_MIP_LEVELS &&
			       level_count != VK_REMAINING_MIP_LEVELS && base_layer != VK_REMAINING_ARRAY_LAYERS && layer_count != VK_REMAINING_ARRAY_LAYERS &&
			       (!may_require_image_view() || view_type != ImageViewType::eInfer);
		}

		enum class MipPreset { eNoMips, eFullMips };

		enum class UsagePreset { eUpload, eDownload, eCopy, eRender, eStore };

		enum class Preset {
			eMap1D,         // 1D image with upload, sampled, never rendered to. Full mip chain. No arraying.
			eMap2D,         // 2D image with upload, sampled, never rendered to. Full mip chain. No arraying.
			eMap3D,         // 3D image with upload, sampled, never rendered to. Full mip chain. No arraying.
			eMapCube,       // Cubemap with upload, sampled, never rendered to. Full mip chain. No arraying.
			eRTT2D,         // 2D image sampled and rendered to. Full mip chain. No arraying.
			eRTTCube,       // Cubemap sampled and rendered to. Full mip chain. No arraying.
			eRTT2DUnmipped, // 2D image sampled and rendered to. No mip chain. No arraying.
			eSTT2D,         // 2D image sampled and stored to. Full mip chain. No arraying.
			eSTT2DUnmipped, // 2D image sampled and stored to. No mip chain. No arraying.
			eGeneric2D,     // 2D image with upload, download, sampling, rendering and storing. Full mip chain. No arraying.
		};

		static ImageAttachment from_preset(Preset preset, Format format, Extent3D extent, Samples sample_count) {
			ImageAttachment ia = {};
			ia.usage = {};
			ia.format = format;
			ia.extent = extent;
			ia.sample_count = sample_count;
			ia.allow_srgb_unorm_mutable = true;
			ImageAspectFlags aspect = format_to_aspect(format);
			switch (preset) {
			case Preset::eMap1D:
			case Preset::eMap2D:
			case Preset::eMap3D:
			case Preset::eMapCube:
				ia.usage |= ImageUsageFlagBits::eTransferDst | ImageUsageFlagBits::eSampled;
				break;
			case Preset::eRTT2D:
			case Preset::eRTTCube:
			case Preset::eRTT2DUnmipped:
				if (aspect & ImageAspectFlagBits::eColor)
					ia.usage |= ImageUsageFlagBits::eColorAttachment;
				if (aspect & (ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil))
					ia.usage |= ImageUsageFlagBits::eDepthStencilAttachment;
				ia.usage |= ImageUsageFlagBits::eSampled;
				break;
			case Preset::eSTT2D:
			case Preset::eSTT2DUnmipped:
				ia.usage |= ImageUsageFlagBits::eStorage | ImageUsageFlagBits::eSampled;
				break;
			case Preset::eGeneric2D:
				ia.usage |= ImageUsageFlagBits::eStorage | ImageUsageFlagBits::eTransferDst | ImageUsageFlagBits::eTransferSrc | ImageUsageFlagBits::eSampled;
				if (aspect & ImageAspectFlagBits::eColor)
					ia.usage |= ImageUsageFlagBits::eColorAttachment;
				if (aspect & (ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil))
					ia.usage |= ImageUsageFlagBits::eDepthStencilAttachment;
				break;
			default:
				assert(0);
			}

			const uint32_t max_mips = (uint32_t)log2f((float)std::max(std::max(extent.width, extent.height), extent.depth)) + 1;
			ia.base_level = 0;
			if (preset == Preset::eRTT2DUnmipped || preset == Preset::eSTT2DUnmipped) {
				ia.level_count = 1;
			} else {
				ia.level_count = max_mips;
			}
			ia.base_layer = 0;
			if (preset == Preset::eRTTCube || preset == Preset::eMapCube) {
				ia.layer_count = 6;
				ia.image_flags = ImageCreateFlagBits::eCubeCompatible;
			} else {
				ia.layer_count = 1;
			}

			switch (preset) {
			case Preset::eMap1D:
				ia.view_type = ImageViewType::e1D;
				break;
			case Preset::eMap2D:
			case Preset::eRTT2D:
			case Preset::eRTT2DUnmipped:
			case Preset::eSTT2D:
			case Preset::eSTT2DUnmipped:
			case Preset::eGeneric2D:
				ia.view_type = ImageViewType::e2D;
				break;
			case Preset::eMap3D:
				ia.view_type = ImageViewType::e3D;
				break;
			case Preset::eMapCube:
			case Preset::eRTTCube:
				ia.view_type = ImageViewType::eCube;
				break;
			default:
				assert(0);
			}

			return ia;
		}

		ImageAttachment mip(uint32_t mip) const noexcept {
			ImageAttachment a = *this;
			a.base_level = (a.base_level == VK_REMAINING_MIP_LEVELS ? 0 : a.base_level) + mip;
			a.level_count = 1;
			a.image_view = {};
			return a;
		}

		ImageAttachment mip_range(uint32_t mip_base, uint32_t mip_count) const noexcept {
			ImageAttachment a = *this;
			a.base_level = (a.base_level == VK_REMAINING_MIP_LEVELS ? 0 : a.base_level) + mip_base;
			a.level_count = mip_count;
			a.image_view = {};
			return a;
		}

		ImageAttachment layer(uint32_t layer) const {
			ImageAttachment a = *this;
			a.base_layer = (a.base_layer == VK_REMAINING_ARRAY_LAYERS ? 0 : a.base_layer) + layer;
			a.layer_count = 1;
			a.image_view = {};
			return a;
		}

		ImageAttachment layer_range(uint32_t layer_base, uint32_t layer_count) const noexcept {
			ImageAttachment a = *this;
			a.base_layer = (a.base_layer == VK_REMAINING_ARRAY_LAYERS ? 0 : a.base_layer) + layer_base;
			a.layer_count = layer_count;
			a.image_view = {};
			return a;
		}

		Extent3D base_mip_extent() const noexcept {
			return { std::max(1u, extent.width >> base_level), std::max(1u, extent.height >> base_level), std::max(1u, extent.depth >> base_level) };
		}
	};

	union Subrange {
		struct Image {
			uint32_t base_level = 0;
			uint32_t level_count = VK_REMAINING_MIP_LEVELS;

			uint32_t base_layer = 0;
			uint32_t layer_count = VK_REMAINING_ARRAY_LAYERS;

			constexpr auto operator<=>(const Image& o) const noexcept = default;
		} image = {};
		struct Buffer {
			uint64_t offset = 0;
			uint64_t size = VK_WHOLE_SIZE;

			constexpr bool operator==(const Buffer& o) const noexcept {
				return offset == o.offset && size == o.size;
			}
		} buffer;
	};

	// high level type around binding a sampled image with a sampler
	struct SampledImage {
		ImageAttachment ia;
		SamplerCreateInfo sci = {};
	};
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::ImageAttachment> {
		size_t operator()(vuk::ImageAttachment const& x) const noexcept;
	};
} // namespace std
