#include "Program.hpp"
#include <gsl/gsl_util>
#include <spirv_cross.hpp>
#include <regex>
#include "Hash.hpp"

vk::ShaderStageFlagBits vuk::Program::introspect(const spirv_cross::Compiler& refl) {
	auto resources = refl.get_shader_resources();
	auto entry_name = refl.get_entry_points_and_stages()[0];
	auto entry_point = refl.get_entry_point(entry_name.name, entry_name.execution_model);
	auto model = entry_point.model;
	auto stage = [=]() {
		switch (model) {
			case spv::ExecutionModel::ExecutionModelVertex: return vk::ShaderStageFlagBits::eVertex;
			case spv::ExecutionModel::ExecutionModelTessellationControl: return vk::ShaderStageFlagBits::eTessellationControl;
			case spv::ExecutionModel::ExecutionModelTessellationEvaluation: return vk::ShaderStageFlagBits::eTessellationEvaluation;
			case spv::ExecutionModel::ExecutionModelGeometry: return vk::ShaderStageFlagBits::eGeometry;
			case spv::ExecutionModel::ExecutionModelFragment: return vk::ShaderStageFlagBits::eFragment;
			case spv::ExecutionModel::ExecutionModelGLCompute: return vk::ShaderStageFlagBits::eCompute;
			default: return vk::ShaderStageFlagBits::eVertex;
		}
	}();
	// uniform buffers
	for (auto& ub : resources.uniform_buffers) {
		auto type = refl.get_type(ub.type_id);
		auto binding = refl.get_decoration(ub.id, spv::DecorationBinding);
		auto set = refl.get_decoration(ub.id, spv::DecorationDescriptorSet);
		UniformBuffer un;
		un.binding = binding;
		un.stage = stage;
		un.name = std::string(ub.name.c_str());
		if (type.array.size() > 0)
			un.array_size = type.array[0];
		else
			un.array_size = 1;

		un.size = refl.get_declared_struct_size(type);
		if (type.member_types.size() == 1 && refl.get_type(type.member_types[0]).array.size() > 0 && un.name == "Materials") { // we process only Materials
			auto arr_t_id = type.member_types[0];
			auto arr_t = refl.get_type(arr_t_id);

			un.array_size = arr_t.array[0];
			un.size /= un.array_size;
		}
		sets[set].uniform_buffers.push_back(un);
	}
	for (auto& sb : resources.storage_buffers) {
		auto type = refl.get_type(sb.type_id);
		auto binding = refl.get_decoration(sb.id, spv::DecorationBinding);
		auto set = refl.get_decoration(sb.id, spv::DecorationDescriptorSet);
		StorageBuffer un;
		un.binding = binding;
		un.stage = stage;
		un.name = sb.name.c_str();
		un.min_size = refl.get_declared_struct_size(refl.get_type(sb.type_id));
		sets[set].storage_buffers.push_back(un);
	}

	for (auto& si : resources.sampled_images) {
		auto type = refl.get_type(si.type_id);
		auto binding = refl.get_decoration(si.id, spv::DecorationBinding);
		auto set = refl.get_decoration(si.id, spv::DecorationDescriptorSet);
		Sampler t;
		t.binding = binding;
		t.name = std::string(si.name.c_str());
		t.stage = stage;
		t.array_size = type.array[0];
		sets[set].samplers.push_back(t);
	}
	
	// subpass inputs
	for (auto& si : resources.subpass_inputs) {
		auto type = refl.get_type(si.type_id);
		auto binding = refl.get_decoration(si.id, spv::DecorationBinding);
		auto set = refl.get_decoration(si.id, spv::DecorationDescriptorSet);
		SubpassInput s;
		s.name = std::string(si.name.c_str());
		s.binding = binding;
		s.stage = stage;
		sets[set].subpass_inputs.push_back(s);
	}

	// push constants
	for (auto& si : resources.push_constant_buffers) {
		auto type = refl.get_type(si.base_type_id);
		vk::PushConstantRange pcr;
		pcr.offset = 0;
		pcr.size = (uint32_t)refl.get_declared_struct_size(type);
		pcr.stageFlags = stage;
		push_constant_ranges.push_back(pcr);
	}

	return stage;
}

void vuk::Program::append(const Program& o) {
	attributes.insert(attributes.end(), o.attributes.begin(), o.attributes.end());
	push_constant_ranges.insert(push_constant_ranges.end(), o.push_constant_ranges.begin(), o.push_constant_ranges.end());
	for (auto& [index, os] : o.sets) {
		auto& s = sets[index];
		s.samplers.insert(s.samplers.end(), os.samplers.begin(), os.samplers.end());
		s.uniform_buffers.insert(s.uniform_buffers.end(), os.uniform_buffers.begin(), os.uniform_buffers.end());
		s.storage_buffers.insert(s.storage_buffers.end(), os.storage_buffers.begin(), os.storage_buffers.end());
		s.texel_buffers.insert(s.texel_buffers.end(), os.texel_buffers.begin(), os.texel_buffers.end());
		s.subpass_inputs.insert(s.subpass_inputs.end(), os.subpass_inputs.begin(), os.subpass_inputs.end());
	}
}

size_t std::hash<vuk::ShaderModuleCreateInfo>::operator()(vuk::ShaderModuleCreateInfo const& x) const noexcept {
	size_t h = 0;
	hash_combine(h, x.filename);
	return h;
}
