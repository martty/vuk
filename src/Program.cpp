#include "Program.hpp"
#include <spirv_cross.hpp>
#include <regex>
#include "Hash.hpp"

vuk::Program::Type to_type(spirv_cross::SPIRType s) {
	using namespace spirv_cross;
	using namespace vuk;

	switch (s.basetype) {
	case SPIRType::Float:
		switch (s.columns) {
		case 1:
			switch (s.vecsize) {
			case 1:	return Program::Type::efloat; break;
			case 2: return Program::Type::evec2; break;
			case 3: return Program::Type::evec3; break;
			case 4: return Program::Type::evec4; break;
			default: assert("NYI" && 0);
			}
		case 4:
			return Program::Type::emat4; break;
		}
	case SPIRType::Int:
		switch (s.vecsize) {
		case 1:	return Program::Type::eint; break;
		case 2: return Program::Type::eivec2; break;
		case 3: return Program::Type::eivec3; break;
		case 4: return Program::Type::eivec4; break;
		default: assert("NYI" && 0);
		}
	case SPIRType::UInt:
		switch (s.vecsize) {
		case 1:	return Program::Type::euint; break;
		case 2: return Program::Type::euvec2; break;
		case 3: return Program::Type::euvec3; break;
		case 4: return Program::Type::euvec4; break;
		default: assert("NYI" && 0);
		}
	case SPIRType::Struct: return Program::Type::estruct;
    default:
        assert("NYI" && 0);
        return Program::Type::estruct;
	}
}

void reflect_members(const spirv_cross::Compiler& refl, const spirv_cross::SPIRType& type, std::vector<vuk::Program::Member>& members) {
	for (size_t i = 0; i < type.member_types.size(); i++) {
		auto& t = type.member_types[i];
		vuk::Program::Member m;
		auto spirtype = refl.get_type(t);
		m.type = to_type(spirtype);
		m.name = refl.get_member_name(type.self, i);
		m.size = refl.get_declared_struct_member_size(type, i);
		m.offset = refl.type_struct_member_offset(type, i);
		if (m.type == vuk::Program::Type::estruct) {
			reflect_members(refl, spirtype, m.members);
		}
		if (spirtype.array.size() > 0)
			m.array_size = type.array[0];
		else
			m.array_size = 1;
		members.push_back(m);
	}
}

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
	stages = stage;
	if (stage == vk::ShaderStageFlagBits::eVertex) {
		for (auto& sb : resources.stage_inputs) {
			auto type = refl.get_type(sb.type_id);
			auto location = refl.get_decoration(sb.id, spv::DecorationLocation);
			Attribute a;
			a.location = location;
			a.name = sb.name.c_str();
			a.type = to_type(type);
			attributes.push_back(a);
		}
	}
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
		if (type.basetype == spirv_cross::SPIRType::Struct) {
			reflect_members(refl, refl.get_type(ub.type_id), un.members);
		}
		un.size = refl.get_declared_struct_size(type);
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
        t.shadow = type.image.depth;
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

static auto binding_cmp = [](auto& s1, auto& s2) { return s1.binding < s2.binding; };
static auto binding_eq = [](auto& s1, auto& s2) { return s1.binding == s2.binding; };

template<class T>
void unq(T& s) {
	std::sort(s.begin(), s.end(), binding_cmp);
	for (auto it = s.begin(); it != s.end();) {
		vk::ShaderStageFlags stages = it->stage;
		for (auto it2 = it; it2 != s.end(); it2++) {
			if (it->binding == it2->binding) {
				stages |= it2->stage;
			} else
				break;
		}
		it->stage = stages;
		auto it2 = it;
		for (; it2 != s.end(); it2++) {
			it2->stage = stages;
			if (it->binding != it2->binding) break;
		}
		it = it2;

	}
	s.erase(std::unique(s.begin(), s.end(), binding_eq), s.end());
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

		unq(s.samplers);
		unq(s.uniform_buffers);
		unq(s.storage_buffers);
		unq(s.texel_buffers);
		unq(s.subpass_inputs);
	}

	stages |= o.stages;
}

size_t std::hash<vuk::ShaderModuleCreateInfo>::operator()(vuk::ShaderModuleCreateInfo const& x) const noexcept {
	size_t h = 0;
	hash_combine(h, x.filename);
	return h;
}
