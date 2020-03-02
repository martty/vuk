#pragma once

#include <vulkan/vulkan.hpp>

namespace vuk {
	struct HandleBase {
		size_t id = UINT64_MAX;
	};

	template<class T>
	struct Handle : public HandleBase {
		T payload;

		bool operator==(const Handle& o) const noexcept {
			return id == o.id;
		}
	};
}

namespace std {
	template<class T>
	struct hash<vuk::Handle<T>> {
		size_t operator()(vuk::Handle<T> const & x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.id, T::objectType);
			return h;
		}
	};

}

namespace vuk {
	using ImageView = Handle<vk::ImageView>;
	using Sampler = Handle<vk::Sampler>;
}
