#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/Image.hpp"
#include "vuk/vuk_fwd.hpp"

namespace vuk {
	struct ImageAttachment {
		Image image = {};
		ImageView image_view = {};

		ImageCreateFlags image_flags = {};
		ImageType imageType = ImageType::eInfer;
		ImageTiling tiling = ImageTiling::eInfer;
		ImageUsageFlags usage = ImageUsageFlagBits::eInfer;
		Dimension2D extent = Dimension2D::framebuffer();
		Format format = Format::eUndefined;
		Samples sample_count = Samples::eInfer;
		ImageViewCreateFlags image_view_flags = {};
		ImageViewType viewType = ImageViewType::eInfer;
		ComponentMapping components;

		uint32_t base_level = 0;
		uint32_t level_count = VK_REMAINING_MIP_LEVELS;

		uint32_t base_layer = 0;
		uint32_t layer_count = VK_REMAINING_ARRAY_LAYERS;

		static ImageAttachment from_texture(const vuk::Texture& t) {
			return ImageAttachment{ .image = t.image.get(),
				                      .image_view = t.view.get(),
				                      .extent = { Sizing::eAbsolute, { t.extent.width, t.extent.height } },
				                      .format = t.format,
				                      .sample_count = { t.sample_count },
				                      .level_count = t.view->level_count,
				                      .layer_count = t.view->layer_count };
		}

		constexpr bool has_concrete_image() const noexcept {
			return image != Image{};
		}

		constexpr bool has_concrete_image_view() const noexcept {
			return image_view != ImageView{};
		}
	};
} // namespace vuk