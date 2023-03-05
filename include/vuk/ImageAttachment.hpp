#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/Image.hpp"
#include "vuk/vuk_fwd.hpp"
#include <compare>

namespace vuk {
	struct ImageAttachment {
		Image image = {};
		ImageView image_view = {};

		ImageCreateFlags image_flags = {};
		ImageType image_type = ImageType::e2D;
		ImageTiling tiling = ImageTiling::eOptimal;
		ImageUsageFlags usage = ImageUsageFlagBits::eInfer;
		Dimension3D extent = Dimension3D::framebuffer();
		Format format = Format::eUndefined;
		Samples sample_count = Samples::eInfer;
		ImageViewCreateFlags image_view_flags = {};
		ImageViewType view_type = ImageViewType::eInfer;
		ComponentMapping components;

		uint32_t base_level = VK_REMAINING_MIP_LEVELS;
		uint32_t level_count = VK_REMAINING_MIP_LEVELS;

		uint32_t base_layer = VK_REMAINING_ARRAY_LAYERS;
		uint32_t layer_count = VK_REMAINING_ARRAY_LAYERS;

		bool operator==(const ImageAttachment&) const = default;

		static ImageAttachment from_texture(const vuk::Texture& t) {
			return ImageAttachment{ .image = t.image.get(),
				                      .image_view = t.view.get(),
				                      .extent = { Sizing::eAbsolute, { t.extent.width, t.extent.height, t.extent.depth } },
				                      .format = t.format,
				                      .sample_count = { t.sample_count },
				                      .base_level = 0,
				                      .level_count = t.level_count,
				                      .base_layer = 0,
				                      .layer_count = t.layer_count };
		}

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
			return image_type != ImageType::eInfer && usage != ImageUsageFlagBits::eInfer && extent.sizing != Sizing::eRelative && extent.extent.width != 0 &&
			       extent.extent.height != 0 && extent.extent.depth != 0 && format != Format::eUndefined && sample_count != Samples::eInfer &&
			       base_level != VK_REMAINING_MIP_LEVELS && level_count != VK_REMAINING_MIP_LEVELS && base_layer != VK_REMAINING_ARRAY_LAYERS &&
			       layer_count != VK_REMAINING_ARRAY_LAYERS && (!may_require_image_view() || view_type != ImageViewType::eInfer);
		}
	};

	struct QueueResourceUse {
		PipelineStageFlags stages;
		AccessFlags access;
		ImageLayout layout; // ignored for buffers
		DomainFlags domain = DomainFlagBits::eAny;
	};

	union Subrange {
		struct Image {
			uint32_t base_layer = 0;
			uint32_t base_level = 0;

			uint32_t layer_count = VK_REMAINING_ARRAY_LAYERS;
			uint32_t level_count = VK_REMAINING_MIP_LEVELS;

			constexpr bool operator==(const Image& o) const noexcept {
				return base_level == o.base_level && level_count == o.level_count && base_layer == o.base_layer && layer_count == o.layer_count;
			}

			constexpr bool operator<(const Image& o) const noexcept {
				return std::tie(base_layer, base_level, layer_count, level_count) < std::tie(o.base_layer, o.base_level, o.layer_count, o.level_count);
			}
		} image = {};
		struct Buffer {
			uint64_t offset = 0;
			uint64_t size = VK_WHOLE_SIZE;

			constexpr bool operator==(const Buffer& o) const noexcept {
				return offset == o.offset && size == o.size;
			}
		} buffer;
	};
} // namespace vuk