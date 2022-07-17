#include "vuk/Pipeline.hpp"
#include "vuk/PipelineInstance.hpp"
#include "vuk/Program.hpp"

#include <robin_hood.h>

namespace vuk {
	vuk::fixed_vector<vuk::DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> PipelineBaseCreateInfo::build_descriptor_layouts(const Program& program,
	                                                                                                                     const PipelineBaseCreateInfoBase& bci) {
		vuk::fixed_vector<vuk::DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> dslcis;

		for (const auto& [index, set] : program.sets) {
			// fill up unused sets, if there are holes in descriptor set order
			dslcis.resize(std::max(dslcis.size(), index + 1), {});

			vuk::DescriptorSetLayoutCreateInfo dslci;
			dslci.index = index;
			auto& bindings = dslci.bindings;

			for (auto& ub : set.uniform_buffers) {
				VkDescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = ub.binding;
				layoutBinding.descriptorType = (VkDescriptorType)vuk::DescriptorType::eUniformBuffer;
				layoutBinding.descriptorCount = 1;
				layoutBinding.stageFlags = ub.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				bindings.push_back(layoutBinding);
			}

			for (auto& sb : set.storage_buffers) {
				VkDescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = sb.binding;
				layoutBinding.descriptorType = (VkDescriptorType)vuk::DescriptorType::eStorageBuffer;
				layoutBinding.descriptorCount = 1;
				layoutBinding.stageFlags = sb.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				bindings.push_back(layoutBinding);
			}

			for (auto& tb : set.texel_buffers) {
				VkDescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = tb.binding;
				layoutBinding.descriptorType = (VkDescriptorType)vuk::DescriptorType::eUniformTexelBuffer;
				layoutBinding.descriptorCount = 1;
				layoutBinding.stageFlags = tb.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				bindings.push_back(layoutBinding);
			}

			for (auto& si : set.combined_image_samplers) {
				VkDescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = si.binding;
				layoutBinding.descriptorType = (VkDescriptorType)vuk::DescriptorType::eCombinedImageSampler;
				layoutBinding.descriptorCount = si.array_size == (unsigned)-1 ? 1 : si.array_size;
				layoutBinding.stageFlags = si.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				if (si.array_size == 0) {
					assert(bci.variable_count_max[index] > 0); // forgot to mark this descriptor as variable count
					layoutBinding.descriptorCount = bci.variable_count_max[index];
				}
				bindings.push_back(layoutBinding);
			}

			for (auto& si : set.samplers) {
				VkDescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = si.binding;
				layoutBinding.descriptorType = (VkDescriptorType)vuk::DescriptorType::eSampler;
				layoutBinding.descriptorCount = si.array_size == (unsigned)-1 ? 1 : si.array_size;
				layoutBinding.stageFlags = si.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				if (si.array_size == 0) {
					assert(bci.variable_count_max[index] > 0); // forgot to mark this descriptor as variable count
					layoutBinding.descriptorCount = bci.variable_count_max[index];
				}
				bindings.push_back(layoutBinding);
			}

			for (auto& si : set.sampled_images) {
				VkDescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = si.binding;
				layoutBinding.descriptorType = (VkDescriptorType)vuk::DescriptorType::eSampledImage;
				layoutBinding.descriptorCount = si.array_size == (unsigned)-1 ? 1 : si.array_size;
				layoutBinding.stageFlags = si.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				if (si.array_size == 0) {
					assert(bci.variable_count_max[index] > 0); // forgot to mark this descriptor as variable count
					layoutBinding.descriptorCount = bci.variable_count_max[index];
				}
				bindings.push_back(layoutBinding);
			}

			for (auto& si : set.storage_images) {
				VkDescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = si.binding;
				layoutBinding.descriptorType = (VkDescriptorType)vuk::DescriptorType::eStorageImage;
				layoutBinding.descriptorCount = si.array_size == (unsigned)-1 ? 1 : si.array_size;
				layoutBinding.stageFlags = si.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				if (si.array_size == 0) {
					assert(bci.variable_count_max[index] > 0); // forgot to mark this descriptor as variable count
					layoutBinding.descriptorCount = bci.variable_count_max[index];
				}
				bindings.push_back(layoutBinding);
			}

			for (auto& si : set.subpass_inputs) {
				VkDescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = si.binding;
				layoutBinding.descriptorType = (VkDescriptorType)vuk::DescriptorType::eInputAttachment;
				layoutBinding.descriptorCount = 1;
				layoutBinding.stageFlags = si.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				bindings.push_back(layoutBinding);
			}

			for (auto& si : set.acceleration_structures) {
				VkDescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = si.binding;
				layoutBinding.descriptorType = (VkDescriptorType)vuk::DescriptorType::eAccelerationStructureKHR;
				layoutBinding.descriptorCount = si.array_size == (unsigned)-1 ? 1 : si.array_size;
				layoutBinding.stageFlags = si.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				if (si.array_size == 0) {
					assert(bci.variable_count_max[index] > 0); // forgot to mark this descriptor as variable count
					layoutBinding.descriptorCount = bci.variable_count_max[index];
				}
				bindings.push_back(layoutBinding);
			}

			// extract flags from the packed bitset
			auto set_word_offset = index * VUK_MAX_BINDINGS * 4 / (sizeof(unsigned long long) * 8);
			for (unsigned i = 0; i <= set.highest_descriptor_binding; i++) {
				auto word = bci.binding_flags.words[set_word_offset + i * 4 / (sizeof(unsigned long long) * 8)];
				if (word & ((0b1111) << i)) {
					VkDescriptorBindingFlags f((word >> i) & 0b1111);
					dslci.flags.resize(i + 1);
					dslci.flags[i] = f;
				}
			}

			dslcis[index] = std::move(dslci);
		}
		return dslcis;
	}
} // namespace vuk

namespace std {
	size_t hash<vuk::PipelineInstanceCreateInfo>::operator()(vuk::PipelineInstanceCreateInfo const& x) const noexcept {
		size_t h = 0;
		auto ext_hash = x.is_inline() ? robin_hood::hash_bytes(x.inline_data, x.extended_size) : robin_hood::hash_bytes(x.extended_data, x.extended_size);
		hash_combine(h, x.base, reinterpret_cast<uint64_t>((VkRenderPass)x.render_pass), x.extended_size, ext_hash);
		return h;
	}

	size_t hash<VkSpecializationMapEntry>::operator()(VkSpecializationMapEntry const& x) const noexcept {
		size_t h = 0;
		hash_combine(h, x.constantID, x.offset, x.size);
		return h;
	}

	size_t hash<vuk::ComputePipelineInstanceCreateInfo>::operator()(vuk::ComputePipelineInstanceCreateInfo const& x) const noexcept {
		size_t h = 0;
		hash_combine(h, x.base, robin_hood::hash_bytes(x.specialization_constant_data.data(), x.specialization_info.dataSize), x.specialization_map_entries);
		return h;
	}

	size_t hash<vuk::RayTracingPipelineInstanceCreateInfo>::operator()(vuk::RayTracingPipelineInstanceCreateInfo const& x) const noexcept {
		size_t h = 0;
		hash_combine(h, x.base, robin_hood::hash_bytes(x.specialization_constant_data.data(), x.specialization_info.dataSize), x.specialization_map_entries);
		return h;
	}

	size_t hash<VkPushConstantRange>::operator()(VkPushConstantRange const& x) const noexcept {
		size_t h = 0;
		hash_combine(h, x.offset, x.size, (VkShaderStageFlags)x.stageFlags);
		return h;
	}

	size_t hash<vuk::PipelineLayoutCreateInfo>::operator()(vuk::PipelineLayoutCreateInfo const& x) const noexcept {
		size_t h = 0;
		hash_combine(h, x.pcrs, x.dslcis);
		return h;
	}
}; // namespace std
