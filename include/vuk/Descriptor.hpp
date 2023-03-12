#pragma once

#include "vuk/Config.hpp"
#include "vuk/Hash.hpp"
#include "vuk/Image.hpp"
#include "vuk/Types.hpp"
#include "vuk/vuk_fwd.hpp"

#include <string.h>
#include <array>
#include <bitset>
#include <cassert>
#include <span>
#include <vector>
#include <tuple>

inline bool operator==(VkDescriptorSetLayoutBinding const& lhs, VkDescriptorSetLayoutBinding const& rhs) noexcept {
	return (lhs.binding == rhs.binding) && (lhs.descriptorType == rhs.descriptorType) && (lhs.descriptorCount == rhs.descriptorCount) &&
	       (lhs.stageFlags == rhs.stageFlags) && (lhs.pImmutableSamplers == rhs.pImmutableSamplers);
}

namespace std {
	template<class T, size_t E>
	struct hash<std::span<T, E>> {
		size_t operator()(std::span<T, E> const& x) const noexcept {
			size_t h = 0;
			for (auto& e : x) {
				hash_combine(h, e);
			}
			return h;
		}
	};
}; // namespace std

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

		bool operator==(const DescriptorSetLayoutAllocInfo& o) const noexcept {
			return layout == o.layout && descriptor_counts == o.descriptor_counts;
		}
	};

	struct PersistentDescriptorSetCreateInfo {
		DescriptorSetLayoutAllocInfo dslai;
		uint32_t num_descriptors;
	};

	struct DescriptorImageInfo {
		VkDescriptorImageInfo dii;
		decltype(ImageView::id) image_view_id;
		decltype(Sampler::id) sampler_id;

		DescriptorImageInfo(Sampler s, ImageView iv, ImageLayout il) : dii{ s.payload, iv.payload, (VkImageLayout)il }, image_view_id(iv.id), sampler_id(s.id) {}

		void set_sampler(Sampler s) {
			dii.sampler = s.payload;
			sampler_id = s.id;
		}

		void set_image_view(ImageView iv) {
			dii.imageView = iv.payload;
			image_view_id = iv.id;
		}

		bool operator==(const DescriptorImageInfo& o) const noexcept {
			return std::tie(dii.sampler, dii.imageView, dii.imageLayout, image_view_id, sampler_id) ==
			       std::tie(o.dii.sampler, o.dii.imageView, o.dii.imageLayout, o.image_view_id, o.sampler_id);
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
			DescriptorImageInfo image;
			VkAccelerationStructureKHR as;
		};

		bool operator==(const DescriptorBinding& o) const noexcept {
			if (type != o.type)
				return false;
			switch (type) {
			case vuk::DescriptorType::eUniformBuffer:
			case vuk::DescriptorType::eStorageBuffer:
				return memcmp(&buffer, &o.buffer, sizeof(VkDescriptorBufferInfo)) == 0;
			case vuk::DescriptorType::eStorageImage:
			case vuk::DescriptorType::eSampledImage:
			case vuk::DescriptorType::eSampler:
			case vuk::DescriptorType::eCombinedImageSampler:
				return image == o.image;
			case vuk::DescriptorType::eAccelerationStructureKHR:
				return as == o.as;
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
		DescriptorSetLayoutAllocInfo* layout_info = nullptr;
		uint64_t hash = 0;

		SetBinding finalize(std::bitset<VUK_MAX_BINDINGS> used_mask);

		bool operator==(const SetBinding& o) const noexcept {
			if (layout_info != o.layout_info)
				return false;
			return memcmp(bindings.data(), o.bindings.data(), VUK_MAX_BINDINGS * sizeof(DescriptorBinding)) == 0;
		}
	};

	struct DescriptorSetLayoutCreateInfo {
		VkDescriptorSetLayoutCreateInfo dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		size_t index; // index of the descriptor set when used in a pipeline layout
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		std::bitset<VUK_MAX_BINDINGS> used_bindings = {}; // used for ephemeral desc sets
		std::bitset<VUK_MAX_BINDINGS> optional = {};
		std::vector<VkDescriptorBindingFlags> flags;

		bool operator==(const DescriptorSetLayoutCreateInfo& o) const noexcept {
			return std::tie(dslci.flags, bindings, flags) == std::tie(o.dslci.flags, o.bindings, o.flags);
		}
	};

	template<>
	struct create_info<vuk::DescriptorSetLayoutAllocInfo> {
		using type = vuk::DescriptorSetLayoutCreateInfo;
	};

	struct DescriptorSet {
		VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
		DescriptorSetLayoutAllocInfo layout_info;

		bool operator==(const DescriptorSet& o) const noexcept {
			return descriptor_set == o.descriptor_set;
		}
	};

	template<>
	struct create_info<vuk::DescriptorSet> {
		using type = vuk::SetBinding;
	};

	struct DescriptorPool {
		void grow(Context& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info);
		VkDescriptorSet acquire(Context& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info);
		void release(VkDescriptorSet ds);
		void destroy(Context& ctx, VkDevice) const;

		DescriptorPool();
		~DescriptorPool();
		DescriptorPool(DescriptorPool&& o) noexcept;

	private:
		struct DescriptorPoolImpl* impl = nullptr;
	};

	template<>
	struct create_info<vuk::DescriptorPool> {
		using type = vuk::DescriptorSetLayoutAllocInfo;
	};

	struct Buffer;

	struct PersistentDescriptorSet {
		VkDescriptorPool backing_pool;
		VkDescriptorSetLayout set_layout;
		VkDescriptorSet backing_set;

		std::array<std::vector<DescriptorBinding>, VUK_MAX_BINDINGS> descriptor_bindings;

		std::vector<VkWriteDescriptorSet> pending_writes;

		bool operator==(const PersistentDescriptorSet& other) const {
			return backing_pool == other.backing_pool;
		}

		void update_combined_image_sampler(Context& ctx,
		                                   unsigned binding,
		                                   unsigned array_index,
		                                   vuk::ImageView iv,
		                                   vuk::SamplerCreateInfo sampler_create_info,
		                                   vuk::ImageLayout layout);
		void update_storage_image(Context& ctx, unsigned binding, unsigned array_index, vuk::ImageView iv);
		void update_uniform_buffer(Context& ctx, unsigned binding, unsigned array_index, Buffer buf);
		void update_storage_buffer(Context& ctx, unsigned binding, unsigned array_index, Buffer buf);
	};
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::SetBinding> {
		size_t operator()(vuk::SetBinding const& x) const noexcept {
			return x.hash;
		}
	};

	template<>
	struct hash<vuk::DescriptorSetLayoutAllocInfo> {
		size_t operator()(vuk::DescriptorSetLayoutAllocInfo const& x) const noexcept {
			size_t h = 0;
			// TODO: should use vuk::DescriptorSetLayout here
			hash_combine(h,
			             ::hash::fnv1a::hash(
			                 (const char*)&x.descriptor_counts[0], x.descriptor_counts.size() * sizeof(x.descriptor_counts[0]), ::hash::fnv1a::default_offset_basis),
			             (VkDescriptorSetLayout)x.layout);
			return h;
		}
	};

	template<>
	struct hash<VkDescriptorSetLayoutBinding> {
		size_t operator()(VkDescriptorSetLayoutBinding const& x) const noexcept {
			size_t h = 0;
			// TODO: immutable samplers
			hash_combine(h, x.binding, x.descriptorCount, x.descriptorType, x.stageFlags);
			return h;
		}
	};

	template<>
	struct hash<vuk::DescriptorSetLayoutCreateInfo> {
		size_t operator()(vuk::DescriptorSetLayoutCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, std::span(x.bindings));
			return h;
		}
	};
}; // namespace std
