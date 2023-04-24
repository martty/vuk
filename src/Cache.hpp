#pragma once
#include "../src/ToIntegral.hpp"
#include "CreateInfo.hpp"
#include "RenderPass.hpp"
#include "vuk/Hash.hpp"
#include "vuk/Pipeline.hpp"
#include "vuk/Program.hpp"
#include "vuk/Types.hpp"

#include <atomic>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>
#include <robin_hood.h>

namespace std {
	template<>
	struct hash<VkVertexInputBindingDescription> {
		size_t operator()(VkVertexInputBindingDescription const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.binding, x.inputRate, x.stride);
			return h;
		}
	};

	template<>
	struct hash<VkVertexInputAttributeDescription> {
		size_t operator()(VkVertexInputAttributeDescription const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.binding, x.format, x.location, x.offset);
			return h;
		}
	};

	template<>
	struct hash<VkPipelineTessellationStateCreateInfo> {
		size_t operator()(VkPipelineTessellationStateCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.patchControlPoints);
			return h;
		}
	};

	template<>
	struct hash<vuk::Extent2D> {
		size_t operator()(vuk::Extent2D const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.width, x.height);
			return h;
		}
	};

	template<>
	struct hash<vuk::Extent3D> {
		size_t operator()(vuk::Extent3D const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.width, x.height, x.depth);
			return h;
		}
	};

	template<>
	struct hash<vuk::Offset2D> {
		size_t operator()(vuk::Offset2D const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.x, x.y);
			return h;
		}
	};

	template<>
	struct hash<VkRect2D> {
		size_t operator()(VkRect2D const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.extent, x.offset);
			return h;
		}
	};

	template<>
	struct hash<VkExtent2D> {
		size_t operator()(VkExtent2D const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.width, x.height);
			return h;
		}
	};

	template<>
	struct hash<VkExtent3D> {
		size_t operator()(VkExtent3D const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.width, x.height, x.depth);
			return h;
		}
	};

	template<>
	struct hash<VkOffset2D> {
		size_t operator()(VkOffset2D const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.x, x.y);
			return h;
		}
	};

	template<>
	struct hash<VkViewport> {
		size_t operator()(VkViewport const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.x, x.y, x.width, x.height, x.minDepth, x.maxDepth);
			return h;
		}
	};

	template<>
	struct hash<VkAttachmentDescription> {
		size_t operator()(VkAttachmentDescription const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.initialLayout, x.finalLayout, x.format, x.loadOp, x.stencilLoadOp, x.storeOp, x.stencilStoreOp, x.samples);
			return h;
		}
	};

	template<>
	struct hash<VkAttachmentReference> {
		size_t operator()(VkAttachmentReference const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.attachment, x.layout);
			return h;
		}
	};

	template<>
	struct hash<VkSubpassDependency> {
		size_t operator()(VkSubpassDependency const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.dependencyFlags, x.srcAccessMask, x.srcStageMask, x.srcSubpass, x.dstAccessMask, x.dstStageMask, x.dstSubpass);
			return h;
		}
	};

	template<>
	struct hash<vuk::ImageCreateInfo> {
		size_t operator()(vuk::ImageCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h,
			             x.flags,
			             x.arrayLayers,
			             x.extent,
			             to_integral(x.format),
			             to_integral(x.imageType),
			             to_integral(x.initialLayout),
			             x.mipLevels,
			             std::span(x.pQueueFamilyIndices, x.queueFamilyIndexCount),
			             to_integral(x.samples),
			             to_integral(x.sharingMode),
			             to_integral(x.tiling),
			             x.usage);
			return h;
		}
	};

	template<>
	struct hash<vuk::CachedImageIdentifier> {
		size_t operator()(vuk::CachedImageIdentifier const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.ici, x.id, x.multi_frame_index);
			return h;
		}
	};

	template<>
	struct hash<vuk::ImageSubresourceRange> {
		size_t operator()(vuk::ImageSubresourceRange const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.aspectMask, x.baseArrayLayer, x.baseMipLevel, x.layerCount, x.levelCount);
			return h;
		}
	};

	template<>
	struct hash<vuk::ComponentMapping> {
		size_t operator()(vuk::ComponentMapping const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, to_integral(x.r), to_integral(x.g), to_integral(x.b), to_integral(x.a));
			return h;
		}
	};

	template<>
	struct hash<vuk::ImageViewCreateInfo> {
		size_t operator()(vuk::ImageViewCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.flags, x.components, to_integral(x.format), reinterpret_cast<uint64_t>((VkImage)x.image), x.subresourceRange, to_integral(x.viewType));
			return h;
		}
	};

	template<>
	struct hash<vuk::CompressedImageViewCreateInfo> {
		size_t operator()(vuk::CompressedImageViewCreateInfo const& x) const noexcept {
			return robin_hood::hash_bytes((const char*)&x, sizeof(vuk::CompressedImageViewCreateInfo));
		}
	};

	template<>
	struct hash<vuk::SamplerCreateInfo> {
		size_t operator()(vuk::SamplerCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h,
			             x.flags,
			             x.addressModeU,
			             x.addressModeV,
			             x.addressModeW,
			             x.anisotropyEnable,
			             x.borderColor,
			             x.compareEnable,
			             x.compareOp,
			             x.magFilter,
			             x.maxAnisotropy,
			             x.maxLod,
			             x.minFilter,
			             x.minLod,
			             x.mipLodBias,
			             x.mipmapMode,
			             x.unnormalizedCoordinates);
			return h;
		}
	};
}; // namespace std

namespace vuk {
	template<class U>
	struct CacheImpl;

	template<class T>
	class Cache {
	private:
		CacheImpl<T>* impl = nullptr;

	public:
		using create_fn = T (*)(void*, const create_info_t<T>&);
		using destroy_fn = void (*)(void*, const T&);

		Cache(void* allocator, create_fn create, destroy_fn destroy);
		~Cache();

		struct LRUEntry {
			T* ptr;
			size_t last_use_frame;
			std::atomic<uint8_t> load_cnt;

			LRUEntry(T* ptr, size_t last_use_frame) : ptr(ptr), last_use_frame(last_use_frame), load_cnt(0) {}
			LRUEntry(const LRUEntry& other) : ptr(other.ptr), last_use_frame(other.last_use_frame), load_cnt(other.load_cnt.load()) {}
		};

		std::optional<T> remove(const create_info_t<T>& ci);

		void remove_ptr(const T* ptr);

		T& acquire(const create_info_t<T>& ci);
		T& acquire(const create_info_t<T>& ci, uint64_t current_frame);
		void collect(uint64_t current_frame, size_t threshold);
		void clear();

		create_fn create;
		destroy_fn destroy;

		void* allocator;
	};
} // namespace vuk
