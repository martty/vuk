#pragma once

#include "vuk/runtime/vk/Allocation.hpp"
#include "vuk/vuk_fwd.hpp"
#include <cmath>
#include <compare>

namespace vuk {
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

	struct Range {
		static constexpr uint64_t REMAINING = (~0ULL);

		uint64_t offset = 0;
		uint64_t count = REMAINING;

		constexpr bool operator==(const Range& o) const noexcept {
			return offset == o.offset && count == o.count;
		}

		constexpr bool operator<=(const Range& o) const noexcept {
			return offset >= o.offset && (offset + count) <= (o.offset + o.count);
		}

		constexpr bool intersect(const Range& o) const noexcept {
			if (count == REMAINING) {
				return o.offset >= offset;
			} else if (o.count == REMAINING) {
				return offset >= o.offset;
			}
			return (offset >= o.offset && offset < (o.offset + o.count)) || (o.offset >= offset && o.offset < (offset + count));
		}
	};

	// high level type around binding a sampled image with a sampler
	struct SampledImage {
		ImageView<> ia;
		SamplerCreateInfo sci = {};
	};

	inline void synchronize(struct SampledImage, struct SyncHelper&) {}
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::ImageView<>> {
		size_t operator()(vuk::ImageView<> const& x) const noexcept;
	};
} // namespace std
