#pragma once

#define VUK_MAX_BINDINGS 16
#include <bitset>
#include <vulkan/vulkan.hpp>
#include <vector>
#include "vuk_fwd.hpp"
#include "Types.hpp"
#include <concurrentqueue.h>
#include <mutex>

namespace vuk {
	struct DescriptorSetLayoutAllocInfo {
		std::array<uint32_t, VkDescriptorType::VK_DESCRIPTOR_TYPE_END_RANGE> descriptor_counts = {};
		vk::DescriptorSetLayout layout;

		bool operator==(const DescriptorSetLayoutAllocInfo& o) const {
			return layout == o.layout && descriptor_counts == o.descriptor_counts;
		}
	};

	struct DescriptorImageInfo {
		vuk::Sampler sampler;
		vuk::ImageView image_view;
		vk::ImageLayout image_layout;

		bool operator==(const DescriptorImageInfo& o) const {
			return std::tie(sampler, image_view, image_layout) == std::tie(o.sampler, o.image_view, o.image_layout);
		}

		operator vk::DescriptorImageInfo() const {
			return {sampler.payload, image_view.payload, image_layout};
		}
	};

	// use hand rolled variant to control bits
	// memset to clear out the union
#pragma pack(push, 1)
	struct DescriptorBinding {
		DescriptorBinding() {}

		vk::DescriptorType type;
		union {
			VkDescriptorBufferInfo buffer;
			vuk::DescriptorImageInfo image;
		};

		bool operator==(const DescriptorBinding& o) const {
			if (type != o.type) return false;
			switch (type) {
			case vk::DescriptorType::eUniformBuffer:
			case vk::DescriptorType::eStorageBuffer:
				return memcmp(&buffer, &o.buffer, sizeof(VkDescriptorBufferInfo)) == 0;
			case vk::DescriptorType::eSampledImage:
			case vk::DescriptorType::eSampler:
			case vk::DescriptorType::eCombinedImageSampler:
				return image == o.image;
			default:
				assert(0);
				return false;
			}
		}
	};
#pragma pack(pop)
	struct SetBinding {
		std::bitset<VUK_MAX_BINDINGS> used = {};
		std::array<DescriptorBinding, VUK_MAX_BINDINGS> bindings;
		DescriptorSetLayoutAllocInfo layout_info = {};

		bool operator==(const SetBinding& o) const {
			if (layout_info != o.layout_info) return false;
			for (size_t i = 0; i < VUK_MAX_BINDINGS; i++) {
				if (!used[i]) continue;
				if (bindings[i] != o.bindings[i]) return false;
			}
			return true;
		}
	};

	struct DescriptorSetLayoutCreateInfo {
		vk::DescriptorSetLayoutCreateInfo dslci;
		std::vector<vk::DescriptorSetLayoutBinding> bindings;
		size_t index;

		bool operator==(const DescriptorSetLayoutCreateInfo& o) const {
			return std::tie(dslci.flags, bindings) == std::tie(o.dslci.flags, o.bindings);
		}
	};

	template<> struct create_info<vuk::DescriptorSetLayoutAllocInfo> {
		using type = vuk::DescriptorSetLayoutCreateInfo;
	};

	struct DescriptorSet {
		vk::DescriptorSet descriptor_set;
		DescriptorSetLayoutAllocInfo layout_info;
	};

	template<> struct create_info<vuk::DescriptorSet> {
		using type = vuk::SetBinding;
	};

	struct DescriptorPool {
        std::mutex grow_mutex;
		std::vector<vk::DescriptorPool> pools;
        size_t sets_allocated = 0;
        moodycamel::ConcurrentQueue<vk::DescriptorSet> free_sets{1024};

		void grow(PerThreadContext& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info);
		vk::DescriptorSet acquire(PerThreadContext& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info);

		DescriptorPool() = default;
		DescriptorPool(DescriptorPool&& o) {
            pools = o.pools;
            sets_allocated = o.sets_allocated;
		}
	};

	template<> struct create_info<vuk::DescriptorPool> {
		using type = vuk::DescriptorSetLayoutAllocInfo;
	};
}

namespace std {
	template <>
	struct hash<vuk::SetBinding> {
		size_t operator()(vuk::SetBinding const & x) const noexcept {
			// TODO: should we hash in layout too?
			auto mask = x.used.to_ulong();
			unsigned long leading_ones = vuk::num_leading_ones(mask);
			return ::hash::fnv1a::hash(reinterpret_cast<const char*>(&x.bindings[0]), leading_ones * sizeof(vuk::DescriptorBinding), ::hash::fnv1a::default_offset_basis);
		}
	};

	template <>
	struct hash<vuk::DescriptorSetLayoutAllocInfo> {
		size_t operator()(vuk::DescriptorSetLayoutAllocInfo const & x) const noexcept {
			size_t h = 0;
			// TODO: should use vuk::DescriptorSetLayout here
			hash_combine(h, ::hash::fnv1a::hash((const char *)&x.descriptor_counts[0], x.descriptor_counts.size() * sizeof(x.descriptor_counts[0]), ::hash::fnv1a::default_offset_basis), (VkDescriptorSetLayout)x.layout); 
			return h;
		}
	};

	template <>
	struct hash<vk::DescriptorSetLayoutBinding> {
		size_t operator()(vk::DescriptorSetLayoutBinding const & x) const noexcept {
			size_t h = 0;
			// TODO: immutable samplers
			hash_combine(h, x.binding, x.descriptorCount, x.descriptorType, (VkShaderStageFlags)x.stageFlags);
			return h;
		}
	};


	template <>
	struct hash<vuk::DescriptorSetLayoutCreateInfo> {
		size_t operator()(vuk::DescriptorSetLayoutCreateInfo const & x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.bindings);
			return h;
		}
	};
};
