#pragma once

#include <bitset>
#include <vector>
#include <vuk/Config.hpp>
#include <vuk/vuk_fwd.hpp>
#include <vuk/Types.hpp>
#include <concurrentqueue.h>
#include <mutex>
#include <vuk/Image.hpp>
#include <robin_hood.h>

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

	// from robin_hood
	inline size_t hash_bytes(void const* ptr, size_t len) noexcept {
		static constexpr uint64_t m = UINT64_C(0xc6a4a7935bd1e995);
		static constexpr uint64_t seed = UINT64_C(0xe17a1465);
		static constexpr unsigned int r = 47;

		auto const* const data64 = static_cast<uint64_t const*>(ptr);
		uint64_t h = seed ^ (len * m);

		size_t const n_blocks = len / 8;
		for (size_t i = 0; i < n_blocks; ++i) {
			auto k = *(data64 + i);

			k *= m;
			k ^= k >> r;
			k *= m;

			h ^= k;
			h *= m;
		}

		auto const* const data8 = reinterpret_cast<uint8_t const*>(data64 + n_blocks);
		switch (len & 7U) {
		case 7:
			h ^= static_cast<uint64_t>(data8[6]) << 48U;
			ROBIN_HOOD(FALLTHROUGH); // FALLTHROUGH
		case 6:
			h ^= static_cast<uint64_t>(data8[5]) << 40U;
			ROBIN_HOOD(FALLTHROUGH); // FALLTHROUGH
		case 5:
			h ^= static_cast<uint64_t>(data8[4]) << 32U;
			ROBIN_HOOD(FALLTHROUGH); // FALLTHROUGH
		case 4:
			h ^= static_cast<uint64_t>(data8[3]) << 24U;
			ROBIN_HOOD(FALLTHROUGH); // FALLTHROUGH
		case 3:
			h ^= static_cast<uint64_t>(data8[2]) << 16U;
			ROBIN_HOOD(FALLTHROUGH); // FALLTHROUGH
		case 2:
			h ^= static_cast<uint64_t>(data8[1]) << 8U;
			ROBIN_HOOD(FALLTHROUGH); // FALLTHROUGH
		case 1:
			h ^= static_cast<uint64_t>(data8[0]);
			h *= m;
			ROBIN_HOOD(FALLTHROUGH); // FALLTHROUGH
		default:
			break;
		}

		h ^= h >> r;
		h *= m;
		h ^= h >> r;
		return static_cast<size_t>(h);
	}

	struct SetBinding {
		uint32_t mask = 0;
		uint32_t count = 0;
		std::bitset<VUK_MAX_BINDINGS> used = {};
		std::array<DescriptorBinding, VUK_MAX_BINDINGS> bindings;
		std::array<DescriptorBinding, VUK_MAX_BINDINGS> compressed_bindings; // TODO: remove bindings and uncompress on use
		DescriptorSetLayoutAllocInfo layout_info = {};
		uint64_t hash = 0;

		void calculate_hash() {
			mask = used.to_ulong();
			unsigned long leading_ones = vuk::num_leading_ones(mask);
			hash = hash_bytes(reinterpret_cast<const char*>(&bindings[0]), leading_ones * sizeof(DescriptorBinding));
			hash_combine(hash, layout_info.layout);
			count = 0;
			for (size_t i = 0; i < VUK_MAX_BINDINGS; i++) {
				if ((mask & (1 << i)) == 0) {
					continue;
				} else {
					*(compressed_bindings.data() + count) = *(bindings.data() + i); // ugly, but fast in debug
					count++;
				}
			}
		}

		bool operator==(const SetBinding& o) const noexcept {
			if (layout_info != o.layout_info) return false;
			return memcmp(compressed_bindings.data(), o.compressed_bindings.data(), count * sizeof(DescriptorBinding)) == 0;
		}
	};

	struct DescriptorSetLayoutCreateInfo {
		VkDescriptorSetLayoutCreateInfo dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        size_t index; // index of the descriptor set when used in a pipeline layout
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		std::vector<VkDescriptorBindingFlags> flags;

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

		void grow(Context& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info);
		VkDescriptorSet acquire(Context& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info);

		DescriptorPool() = default;
		DescriptorPool(DescriptorPool&& o) {
			pools = o.pools;
			sets_allocated = o.sets_allocated;
		}
	};

	template<> struct create_info<vuk::DescriptorPool> {
		using type = vuk::DescriptorSetLayoutAllocInfo;
	};

	struct Buffer;

	struct PersistentDescriptorSet {
		VkDescriptorPool backing_pool;
		VkDescriptorSet backing_set;

        std::array<std::vector<DescriptorBinding>, VUK_MAX_BINDINGS> descriptor_bindings;

		std::vector<VkWriteDescriptorSet> pending_writes;

		bool operator==(const PersistentDescriptorSet& other) const {
			return backing_pool == other.backing_pool;
		}

		void update_combined_image_sampler(Context& ctx, unsigned binding, unsigned array_index, vuk::ImageView iv, vuk::SamplerCreateInfo sampler_create_info, vuk::ImageLayout layout);
		void update_storage_image(Context& ctx, unsigned binding, unsigned array_index, vuk::ImageView iv);
        void update_uniform_buffer(Context& ctx, unsigned binding, unsigned array_index, Buffer buf);
        void update_storage_buffer(Context& ctx, unsigned binding, unsigned array_index, Buffer buf);
	};
}

namespace robin_hood {
	template<>
	struct hash<vuk::SetBinding, std::true_type> {
		size_t operator()(vuk::SetBinding const& x) const noexcept {
			return x.hash;
		}
	};
}

namespace std {
	template <>
	struct hash<vuk::SetBinding> {
		size_t operator()(vuk::SetBinding const& x) const noexcept {
			return x.hash;
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
