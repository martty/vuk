#pragma once

#include "vuk/Image.hpp"

#include <tuple>

namespace vuk {
	struct RGImage {
		vuk::Image image;
	};
	struct RGCI {
		Name name;
		vuk::ImageCreateInfo ici;

		bool operator==(const RGCI& other) const noexcept {
			return std::tie(name, ici) == std::tie(other.name, other.ici);
		}
	};
	template<>
	struct create_info<RGImage> {
		using type = RGCI;
	};
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::RGCI> {
		size_t operator()(vuk::RGCI const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.name, x.ici);
			return h;
		}
	};
}; // namespace std
