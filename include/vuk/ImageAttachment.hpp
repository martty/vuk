#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/Image.hpp"
#include "vuk/vuk_fwd.hpp"

namespace vuk {
	struct ImageAttachment {
		vuk::Image image = {};
		vuk::ImageView image_view = {};

		vuk::Dimension2D extent;
		vuk::Format format;
		vuk::Samples sample_count = vuk::Samples::e1;
		Clear clear_value;

		uint32_t base_level = 0;
		uint32_t level_count = VK_REMAINING_MIP_LEVELS;

		uint32_t base_layer = 0;
		uint32_t layer_count = VK_REMAINING_ARRAY_LAYERS;

		static ImageAttachment from_texture(const vuk::Texture& t, Clear clear_value) {
			return ImageAttachment{ .image = t.image.get(),
				                      .image_view = t.view.get(),
				                      .extent = { Sizing::eAbsolute, { t.extent.width, t.extent.height } },
				                      .format = t.format,
				                      .sample_count = { t.sample_count },
				                      .clear_value = clear_value };
		}
		static ImageAttachment from_texture(const vuk::Texture& t) {
			return ImageAttachment{ .image = t.image.get(),
				                      .image_view = t.view.get(),
				                      .extent = { Sizing::eAbsolute, { t.extent.width, t.extent.height } },
				                      .format = t.format,
				                      .sample_count = { t.sample_count },
				                      .layer_count = t.view->layer_count };
		}
	};

} // namespace vuk