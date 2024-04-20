#pragma once

#include "vuk/Bitset.hpp"
#include "vuk/Config.hpp"
#include "vuk/Hash.hpp"
#include "vuk/runtime/vk/Image.hpp"
#include "vuk/Types.hpp"
#include "vuk/vuk_fwd.hpp"

#include <array>
#include <cassert>
#include <span>
#include <string.h>
#include <tuple>
#include <vector>

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
	enum class DescriptorType : uint8_t {
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
		eInlineUniformBlockEXT = 10,
		eAccelerationStructureKHR = 11,
		ePendingWrite = 128
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
		DescriptorType variable_count_binding_type;
		unsigned variable_count_binding_max_size;

		bool operator==(const DescriptorSetLayoutAllocInfo& o) const noexcept {
			return layout == o.layout && descriptor_counts == o.descriptor_counts;
		}
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
			return dii.sampler == o.dii.sampler && dii.imageView == o.dii.imageView && dii.imageLayout == o.dii.imageLayout && image_view_id == o.image_view_id &&
			       sampler_id == o.sampler_id;
		}

		operator VkDescriptorImageInfo() const {
			return dii;
		}
	};

	struct ASInfo {
		VkWriteDescriptorSetAccelerationStructureKHR wds{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
		VkAccelerationStructureKHR as;
	};

	// use hand rolled variant to control bits
	// memset to clear out the union
#pragma pack(push, 1)
	struct DescriptorBinding {
		DescriptorBinding() {}

		DescriptorType type = DescriptorType(127);
		union {
			VkDescriptorBufferInfo buffer;
			DescriptorImageInfo image;
			ASInfo as;
		};

		bool operator==(const DescriptorBinding& o) const noexcept {
			if (type != o.type)
				return false;
			switch (type) {
			case DescriptorType::eUniformBuffer:
			case DescriptorType::eStorageBuffer:
				return memcmp(&buffer, &o.buffer, sizeof(VkDescriptorBufferInfo)) == 0;
			case DescriptorType::eStorageImage:
			case DescriptorType::eSampledImage:
			case DescriptorType::eSampler:
			case DescriptorType::eCombinedImageSampler:
				return image == o.image;
			case DescriptorType::eAccelerationStructureKHR:
				return as.as == o.as.as;
			default:
				assert(0);
				return false;
			}
		}

		static VkDescriptorType vk_descriptor_type(DescriptorType type) {
			switch (type) {
			case DescriptorType::eUniformBuffer:
				return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			case DescriptorType::eStorageBuffer:
				return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			case DescriptorType::eStorageImage:
				return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			case DescriptorType::eSampledImage:
				return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
			case DescriptorType::eSampler:
				return VK_DESCRIPTOR_TYPE_SAMPLER;
			case DescriptorType::eCombinedImageSampler:
				return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			case DescriptorType::eAccelerationStructureKHR:
				return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
			default:
				assert(0);
				return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
			}
		}
	};
#pragma pack(pop)

	struct SetBinding {
		Bitset<VUK_MAX_BINDINGS> used = {};
		DescriptorBinding bindings[VUK_MAX_BINDINGS];
		DescriptorSetLayoutAllocInfo* layout_info = nullptr;
		uint64_t hash = 0;

		SetBinding finalize(Bitset<VUK_MAX_BINDINGS> used_mask);

		bool operator==(const SetBinding& o) const noexcept {
			if (layout_info != o.layout_info)
				return false;
			return memcmp(bindings, o.bindings, VUK_MAX_BINDINGS * sizeof(DescriptorBinding)) == 0;
		}
	};

	struct DescriptorSetLayoutCreateInfo {
		VkDescriptorSetLayoutCreateInfo dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		size_t index;                                     // index of the descriptor set when used in a pipeline layout
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		Bitset<VUK_MAX_BINDINGS> used_bindings = {}; // used for ephemeral desc sets
		Bitset<VUK_MAX_BINDINGS> optional = {};
		std::vector<VkDescriptorBindingFlags> flags;

		bool operator==(const DescriptorSetLayoutCreateInfo& o) const noexcept {
			return dslci.flags == o.dslci.flags && bindings == o.bindings && flags == o.flags;
		}
	};

	template<>
	struct create_info<DescriptorSetLayoutAllocInfo> {
		using type = DescriptorSetLayoutCreateInfo;
	};

	struct DescriptorSet {
		VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
		DescriptorSetLayoutAllocInfo layout_info;

		bool operator==(const DescriptorSet& o) const noexcept {
			return descriptor_set == o.descriptor_set;
		}
	};

	template<>
	struct create_info<DescriptorSet> {
		using type = SetBinding;
	};

	struct DescriptorPool {
		void grow(Runtime& ptc, DescriptorSetLayoutAllocInfo layout_alloc_info);
		VkDescriptorSet acquire(Runtime& ptc, DescriptorSetLayoutAllocInfo layout_alloc_info);
		void release(VkDescriptorSet ds);
		void destroy(Runtime& ctx, VkDevice) const;

		DescriptorPool();
		~DescriptorPool();
		DescriptorPool(DescriptorPool&& o) noexcept;

	private:
		struct DescriptorPoolImpl* impl = nullptr;
	};

	template<>
	struct create_info<DescriptorPool> {
		using type = DescriptorSetLayoutAllocInfo;
	};

	struct PersistentDescriptorSetCreateInfo {
		DescriptorSetLayoutAllocInfo dslai;
		DescriptorSetLayoutCreateInfo dslci;
		uint32_t num_descriptors;
	};

	struct Buffer;

	struct PersistentDescriptorSet {
		VkDescriptorPool backing_pool;
		DescriptorSetLayoutCreateInfo set_layout_create_info;
		VkDescriptorSetLayout set_layout;
		VkDescriptorSet backing_set;

		std::vector<VkWriteDescriptorSet> wdss;

		std::array<std::vector<DescriptorBinding>, VUK_MAX_BINDINGS> descriptor_bindings;

		bool operator==(const PersistentDescriptorSet& other) const {
			return backing_pool == other.backing_pool;
		}

		// all of the update_ functions are thread safe

		void update_combined_image_sampler(unsigned binding, unsigned array_index, ImageView iv, Sampler sampler, ImageLayout layout);
		void update_storage_image(unsigned binding, unsigned array_index, ImageView iv);
		void update_uniform_buffer(unsigned binding, unsigned array_index, Buffer buf);
		void update_storage_buffer(unsigned binding, unsigned array_index, Buffer buf);
		void update_sampler(unsigned binding, unsigned array_index, Sampler sampler);
		void update_sampled_image(unsigned binding, unsigned array_index, ImageView iv, ImageLayout layout);
		void update_acceleration_structure(unsigned binding, unsigned array_index, VkAccelerationStructureKHR as);

		// non-thread safe
		void commit(Runtime& ctx);
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
