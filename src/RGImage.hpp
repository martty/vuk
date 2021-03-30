#pragma once

#include "vuk/Image.hpp"

namespace vuk {
	struct RGImage {
		vuk::Image image;
		vuk::ImageView image_view;
	};
	struct RGCI {
		Name name;
		vuk::ImageCreateInfo ici;
		vuk::ImageViewCreateInfo ivci;

		bool operator==(const RGCI& other) const {
			return std::tie(name, ici, ivci) == std::tie(other.name, other.ici, other.ivci);
		}
	};
}

namespace std {
	template <>
	struct hash<vuk::RGCI> {
		size_t operator()(vuk::RGCI const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.name, x.ici, x.ivci);
			return h;
		}
	};
};
