#pragma once

#include "vuk/Types.hpp"
#include "vuk/Image.hpp"
#include "Pool.hpp"
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
			Name attachment_name;
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

	// the returned values are pointer stable until the frame gets recycled
	template<>
	struct PooledType<vuk::SampledImage> {
		plf::colony<vuk::SampledImage> values;
		size_t needle = 0;

		PooledType(GlobalAllocator&) {}
		vuk::SampledImage& acquire(GlobalAllocator& ptc, vuk::SampledImage si);
		void reset(GlobalAllocator&) { needle = 0; }
		void free(GlobalAllocator&) {} // nothing to free, this is non-owning
	};

	inline vuk::SampledImage& PooledType<vuk::SampledImage>::acquire(GlobalAllocator&, vuk::SampledImage si) {
		if (values.size() < (needle + 1)) {
			needle++;
			return *values.emplace(std::move(si));
		} else {
			auto it = values.begin();
			values.advance(it, needle++);
			*it = si;
			return *it;
		}
	}
}
