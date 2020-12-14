#pragma once

#define VUK_MAX_BINDINGS 16
#include <bitset>
#include <vector>
#include "vuk_fwd.hpp"
#include "Types.hpp"
#include <concurrentqueue.h>
#include <mutex>
#include "Image.hpp"

inline bool operator==(VkDescriptorSetLayoutBinding const& lhs, VkDescriptorSetLayoutBinding const& rhs) noexcept {
	return (lhs.binding == rhs.binding)
		&& (lhs.descriptorType == rhs.descriptorType)
		&& (lhs.descriptorCount == rhs.descriptorCount)
		&& (lhs.stageFlags == rhs.stageFlags)
		&& (lhs.pImmutableSamplers == rhs.pImmutableSamplers);
}

namespace vuk {
	enum class DescriptorType {
		eSampler = VK_DESCRIPTOR_TYPE_SAMPLER,
		eCombinedImageSampler = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		eSampledImage = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
		eStorageImage = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		eUniformTexelBuffer = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
		eStorageTexelBuffer = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
		eUniformBuffer = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		eStorageBuffer = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		eUniformBufferDynamic = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
		eStorageBufferDynamic = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
		eInputAttachment = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
		eInlineUniformBlockEXT = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT,
		eAccelerationStructureKHR = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
		eAccelerationStructureNV = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV
	};

	enum class DescriptorBindingFlagBits : VkDescriptorBindingFlags {
		eUpdateAfterBind = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
		eUpdateUnusedWhilePending = VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
		ePartiallyBound = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
		eVariableDescriptorCount = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT
	};

	using DescriptorBindingFlags = Flags<DescriptorBindingFlagBits>;

	inline constexpr DescriptorBindingFlags operator|(DescriptorBindingFlagBits bit0, DescriptorBindingFlagBits bit1) noexcept {
		return DescriptorBindingFlags(bit0) | bit1;
	}

	inline constexpr DescriptorBindingFlags operator&(DescriptorBindingFlagBits bit0, DescriptorBindingFlagBits bit1) noexcept {
		return DescriptorBindingFlags(bit0) & bit1;
	}

	inline constexpr DescriptorBindingFlags operator^(DescriptorBindingFlagBits bit0, DescriptorBindingFlagBits bit1) noexcept {
		return DescriptorBindingFlags(bit0) ^ bit1;
	}

	struct DescriptorSetLayoutAllocInfo {
		std::array<uint32_t, 12> descriptor_counts = {};
		VkDescriptorSetLayout layout;
		unsigned variable_count_binding = (unsigned)-1;
		vuk::DescriptorType variable_count_binding_type;
		unsigned variable_count_binding_max_size;

		bool operator==(const DescriptorSetLayoutAllocInfo& o) const {
			return layout == o.layout && descriptor_counts == o.descriptor_counts;
		}
	};

	struct DescriptorImageInfo {
		vuk::Sampler sampler;
		vuk::ImageView image_view;
		VkDescriptorImageInfo dii;

		DescriptorImageInfo(vuk::Sampler s, vuk::ImageView iv, vuk::ImageLayout il) : sampler(s), image_view(iv), dii{ s.payload, iv.payload, (VkImageLayout)il } {		}

		bool operator==(const DescriptorImageInfo& o) const {
			return std::tie(sampler, image_view, dii.imageLayout) == std::tie(o.sampler, o.image_view, o.dii.imageLayout);
		}

		operator VkDescriptorImageInfo() const {
			return dii;
		}
	};

	// use hand rolled variant to control bits
	// memset to clear out the union
#pragma pack(push, 1)
	struct DescriptorBinding {
		DescriptorBinding() {}

		vuk::DescriptorType type = vuk::DescriptorType(-1);
		union {
			VkDescriptorBufferInfo buffer;
			vuk::DescriptorImageInfo image;
		};

		bool operator==(const DescriptorBinding& o) const {
			if (type != o.type) return false;
			switch (type) {
			case vuk::DescriptorType::eUniformBuffer:
			case vuk::DescriptorType::eStorageBuffer:
				return memcmp(&buffer, &o.buffer, sizeof(VkDescriptorBufferInfo)) == 0;
			case vuk::DescriptorType::eStorageImage:
			case vuk::DescriptorType::eSampledImage:
			case vuk::DescriptorType::eSampler:
			case vuk::DescriptorType::eCombinedImageSampler:
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
		VkDescriptorSetLayoutCreateInfo dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		std::vector<VkDescriptorBindingFlags> flags;
		size_t index;

		bool operator==(const DescriptorSetLayoutCreateInfo& o) const {
			return std::tie(dslci.flags, bindings, flags) == std::tie(o.dslci.flags, o.bindings, o.flags);
		}
	};

	template<> struct create_info<vuk::DescriptorSetLayoutAllocInfo> {
		using type = vuk::DescriptorSetLayoutCreateInfo;
	};

	struct DescriptorSet {
		VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
		DescriptorSetLayoutAllocInfo layout_info;
	};

	template<> struct create_info<vuk::DescriptorSet> {
		using type = vuk::SetBinding;
	};

	struct DescriptorPool {
		std::mutex grow_mutex;
		std::vector<VkDescriptorPool> pools;
		uint32_t sets_allocated = 0;
		moodycamel::ConcurrentQueue<VkDescriptorSet> free_sets{ 1024 };

		void grow(PerThreadContext& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info);
		VkDescriptorSet acquire(PerThreadContext& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info);

		DescriptorPool() = default;
		DescriptorPool(DescriptorPool&& o) {
			pools = o.pools;
			sets_allocated = o.sets_allocated;
		}
	};

	template<> struct create_info<vuk::DescriptorPool> {
		using type = vuk::DescriptorSetLayoutAllocInfo;
	};

	struct PersistentDescriptorSet {
		VkDescriptorPool backing_pool;
		VkDescriptorSet backing_set;

		std::vector<DescriptorBinding> descriptor_bindings;

		std::vector<VkWriteDescriptorSet> pending_writes;

		bool operator==(const PersistentDescriptorSet& other) const {
			return backing_pool == other.backing_pool;
		}

		void update_combined_image_sampler(PerThreadContext& ptc, unsigned binding, unsigned array_index, vuk::ImageView iv, vuk::SamplerCreateInfo sampler_create_info, vuk::ImageLayout layout);
		void update_storage_image(PerThreadContext& ptc, unsigned binding, unsigned array_index, vuk::ImageView iv);
	};
}

namespace std {
	template <>
	struct hash<vuk::SetBinding> {
		size_t operator()(vuk::SetBinding const& x) const noexcept {
			// TODO: should we hash in layout too?
			auto mask = x.used.to_ulong();
			unsigned long leading_ones = vuk::num_leading_ones(mask);
			return ::hash::fnv1a::hash(reinterpret_cast<const char*>(&x.bindings[0]), leading_ones * sizeof(vuk::DescriptorBinding), ::hash::fnv1a::default_offset_basis);
		}
	};

	template <>
	struct hash<vuk::DescriptorSetLayoutAllocInfo> {
		size_t operator()(vuk::DescriptorSetLayoutAllocInfo const& x) const noexcept {
			size_t h = 0;
			// TODO: should use vuk::DescriptorSetLayout here
			hash_combine(h, ::hash::fnv1a::hash((const char*)&x.descriptor_counts[0], x.descriptor_counts.size() * sizeof(x.descriptor_counts[0]), ::hash::fnv1a::default_offset_basis), (VkDescriptorSetLayout)x.layout);
			return h;
		}
	};

	template <>
	struct hash<VkDescriptorSetLayoutBinding> {
		size_t operator()(VkDescriptorSetLayoutBinding const& x) const noexcept {
			size_t h = 0;
			// TODO: immutable samplers
			hash_combine(h, x.binding, x.descriptorCount, x.descriptorType, x.stageFlags);
			return h;
		}
	};


	template <>
	struct hash<vuk::DescriptorSetLayoutCreateInfo> {
		size_t operator()(vuk::DescriptorSetLayoutCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.bindings);
			return h;
		}
	};
};
