#include "vuk/runtime/vk/Pipeline.hpp"
#include "vuk/SourceLocation.hpp"
#include "vuk/runtime/vk/PipelineInstance.hpp"
#include "vuk/runtime/vk/Program.hpp"

#include <robin_hood.h>

namespace vuk {
#if VUK_USE_SHADERC
	PipelineBaseCreateInfo PipelineBaseCreateInfo::from_inline_glsl(std::string_view source, SourceLocationAtFrame _pscope) {
		PipelineBaseCreateInfo pbci;
		pbci.shaders.emplace_back(ShaderSource::glsl(source, {}, "main"));
		pbci.shader_paths.emplace_back(format_source_location(_pscope));

		return pbci;
	}
#endif

	fixed_vector<DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> PipelineBaseCreateInfo::build_descriptor_layouts(const Program& program,
	                                                                                                           const PipelineBaseCreateInfoBase& bci) {
		fixed_vector<DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> dslcis(program.sets.size());

		for (size_t index = 0; index < VUK_MAX_SETS; index++) {
			dslcis[index] = DescriptorSetLayoutCreateInfo();
			dslcis[index].index = index;
		}

		for (size_t index = 0; index < program.sets.size(); index++) {
			const auto& set = program.sets[index];
			if (!set) {
				continue;
			}
			DescriptorSetLayoutCreateInfo dslci;
			dslci.index = index;
			auto& bindings = dslci.bindings;

			for (auto& ub : set->bindings) {
				VkDescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = ub.binding;
				layoutBinding.descriptorType = (VkDescriptorType)ub.type;
				layoutBinding.descriptorCount = ub.array_size == (unsigned)-1 ? 1 : ub.array_size;
				if (ub.array_size == 0) {
					layoutBinding.descriptorCount = bci.variable_count_max[index];
				}
				layoutBinding.stageFlags = ub.stage;
				layoutBinding.pImmutableSamplers = nullptr;

				if (layoutBinding.binding < VUK_MAX_BINDINGS) {
					dslci.used_bindings.set(layoutBinding.binding, true);
					if (ub.type == DescriptorType::eStorageBuffer) {
						dslci.optional.set(layoutBinding.binding, ub.is_hlsl_counter_buffer);
					}
				}
				bindings.push_back(layoutBinding);
			}

			// extract flags from the packed bitset
			auto set_word_offset = index * VUK_MAX_BINDINGS * 4 / (sizeof(unsigned long long) * 8);
			for (unsigned i = 0; i <= set->highest_descriptor_binding; i++) {
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
	size_t hash<vuk::GraphicsPipelineInstanceCreateInfo>::operator()(vuk::GraphicsPipelineInstanceCreateInfo const& x) const noexcept {
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

	size_t hash<vuk::ShaderSource>::operator()(vuk::ShaderSource const& x) const noexcept {
		size_t h = 0;
		hash_combine(h, x.language, robin_hood::hash_bytes(x.data_ptr, x.size));
		return h;
	}
}; // namespace std
