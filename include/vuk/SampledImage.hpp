#pragma once

#include "vuk/Image.hpp"
#include "vuk/Types.hpp"
#include <optional>

namespace vuk {
	// high level type around binding a sampled image with a sampler
	struct SampledImage {
		struct Global {
			vuk::ImageView iv;
			vuk::SamplerCreateInfo sci = {};
			vuk::ImageLayout image_layout;
		};

		struct RenderGraphAttachment {
			NameReference reference;
			vuk::SamplerCreateInfo sci = {};
			std::optional<vuk::ImageViewCreateInfo> ivci = {};
			vuk::ImageLayout image_layout;
		};

		union {
			Global global = {};
			RenderGraphAttachment rg_attachment;
		};
		bool is_global;

		SampledImage(Global g) : global(g), is_global(true) {}
		SampledImage(RenderGraphAttachment g) : rg_attachment(g), is_global(false) {}

		SampledImage(const SampledImage& o) {
			*this = o;
		}

		SampledImage& operator=(const SampledImage& o) {
			if (o.is_global) {
				global = {};
				global = o.global;
			} else {
				rg_attachment = {};
				rg_attachment = o.rg_attachment;
			}
			is_global = o.is_global;
			return *this;
		}
	};
} // namespace vuk
