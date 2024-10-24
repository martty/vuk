#include "vuk/runtime/vk/Program.hpp"
#include "vuk/Hash.hpp"
#include "vuk/ShaderSource.hpp"

#include <spirv_cross.hpp>

static auto binding_cmp = [](auto& s1, auto& s2) {
	return s1.binding < s2.binding;
};
static auto binding_eq = [](auto& s1, auto& s2) {
	return s1.binding == s2.binding;
};

template<class T>
void unq(T& s) {
	std::sort(s.begin(), s.end(), binding_cmp);
	for (auto it = s.begin(); it != s.end();) {
		VkShaderStageFlags stages = it->stage;
		for (auto it2 = it; it2 != s.end(); it2++) {
			if (it->binding == it2->binding) {
				stages |= it2->stage;
			} else
				break;
		}
		it->stage = stages;
		auto it2 = it;
		for (; it2 != s.end(); it2++) {
			if (it->binding != it2->binding)
				break;
			it2->stage = stages;
		}
		it = it2;
	}
	s.erase(std::unique(s.begin(), s.end(), binding_eq), s.end());
}

namespace vuk {
	Program::Type to_type(spirv_cross::SPIRType s) {
		using namespace spirv_cross;

		switch (s.basetype) {
		case SPIRType::Float:
			switch (s.columns) {
			case 1:
				switch (s.vecsize) {
				case 1:
					return Program::Type::efloat;
				case 2:
					return Program::Type::evec2;
				case 3:
					return Program::Type::evec3;
				case 4:
					return Program::Type::evec4;
				default:
					assert("NYI" && 0);
				}
				break;
			case 3:
				return Program::Type::emat3;
			case 4:
				return Program::Type::emat4;
			}
			break;
		case SPIRType::Double:
			switch (s.columns) {
			case 1:
				switch (s.vecsize) {
				case 1:
					return Program::Type::edouble;
				case 2:
					return Program::Type::edvec2;
				case 3:
					return Program::Type::edvec3;
				case 4:
					return Program::Type::edvec4;
				default:
					assert("NYI" && 0);
				}
				break;
			case 3:
				return Program::Type::edmat3;
			case 4:
				return Program::Type::edmat4;
			}
			break;
		case SPIRType::Int:
			switch (s.vecsize) {
			case 1:
				return Program::Type::eint;
			case 2:
				return Program::Type::eivec2;
			case 3:
				return Program::Type::eivec3;
			case 4:
				return Program::Type::eivec4;
			default:
				assert("NYI" && 0);
			}
			break;
		case SPIRType::UInt:
			switch (s.vecsize) {
			case 1:
				return Program::Type::euint;
			case 2:
				return Program::Type::euvec2;
			case 3:
				return Program::Type::euvec3;
			case 4:
				return Program::Type::euvec4;
			default:
				assert("NYI" && 0);
			}
			break;
		case SPIRType::UInt64:
			switch (s.vecsize) {
			case 1:
				return Program::Type::euint64_t;
			case 2:
				return Program::Type::eu64vec2;
			case 3:
				return Program::Type::eu64vec3;
			case 4:
				return Program::Type::eu64vec4;
			default:
				assert("NYI" && 0);
			}
			break;
		case SPIRType::Int64:
			switch (s.vecsize) {
			case 1:
				return Program::Type::eint64_t;
			case 2:
				return Program::Type::ei64vec2;
			case 3:
				return Program::Type::ei64vec3;
			case 4:
				return Program::Type::ei64vec4;
			default:
				assert("NYI" && 0);
			}
			break;
		case SPIRType::Struct:
			return Program::Type::estruct;
		default:
			assert("NYI" && 0);
		}
		return Program::Type::einvalid;
	}

	void reflect_members(const spirv_cross::Compiler& refl, const spirv_cross::SPIRType& type, std::vector<Program::Member>& members) {
		for (uint32_t i = 0; i < type.member_types.size(); i++) {
			auto& t = type.member_types[i];
			Program::Member m;
			auto spirtype = refl.get_type(t);
			m.type = to_type(spirtype);
			if (m.type == Program::Type::estruct) {
				m.type_name = refl.get_name(t);
				if (m.type_name == "") {
					m.type_name = refl.get_name(spirtype.parent_type);
				}
			}
			m.name = refl.get_member_name(type.self, i);
			m.size = refl.get_declared_struct_member_size(type, i);
			m.offset = refl.type_struct_member_offset(type, i);

			if (m.type == Program::Type::estruct) {
				m.size = refl.get_declared_struct_size(spirtype);
				reflect_members(refl, spirtype, m.members);
			}

			if (spirtype.array.size() > 0) {
				m.array_size = spirtype.array[0];
			} else {
				m.array_size = 1;
			}

			members.push_back(m);
		}
	}

	VkShaderStageFlagBits Program::introspect(const uint32_t* ir, size_t word_count) {
		spirv_cross::Compiler refl(ir, word_count);
		auto resources = refl.get_shader_resources();
		auto entry_name = refl.get_entry_points_and_stages()[0];
		auto entry_point = refl.get_entry_point(entry_name.name, entry_name.execution_model);
		auto model = entry_point.model;
		auto stage = [=]() {
			switch (model) {
			case spv::ExecutionModel::ExecutionModelVertex:
				return VK_SHADER_STAGE_VERTEX_BIT;
			case spv::ExecutionModel::ExecutionModelTessellationControl:
				return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
			case spv::ExecutionModel::ExecutionModelTessellationEvaluation:
				return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
			case spv::ExecutionModel::ExecutionModelGeometry:
				return VK_SHADER_STAGE_GEOMETRY_BIT;
			case spv::ExecutionModel::ExecutionModelFragment:
				return VK_SHADER_STAGE_FRAGMENT_BIT;
			case spv::ExecutionModel::ExecutionModelGLCompute:
				return VK_SHADER_STAGE_COMPUTE_BIT;
			case spv::ExecutionModel::ExecutionModelAnyHitKHR:
				return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
			case spv::ExecutionModel::ExecutionModelCallableKHR:
				return VK_SHADER_STAGE_CALLABLE_BIT_KHR;
			case spv::ExecutionModel::ExecutionModelClosestHitKHR:
				return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
			case spv::ExecutionModel::ExecutionModelIntersectionKHR:
				return VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
			case spv::ExecutionModel::ExecutionModelMissKHR:
				return VK_SHADER_STAGE_MISS_BIT_KHR;
			case spv::ExecutionModel::ExecutionModelRayGenerationKHR:
				return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
			default:
				return VK_SHADER_STAGE_VERTEX_BIT;
			}
		}();
		stages = stage;
		if (stage == VK_SHADER_STAGE_VERTEX_BIT) {
			for (auto& sb : resources.stage_inputs) {
				auto type = refl.get_type(sb.type_id);
				auto location = refl.get_decoration(sb.id, spv::DecorationLocation);
				unsigned count = 1;
				if (type.array.size() > 0) {
					count = type.array[0];
				}
				for (uint32_t i = 0; i < count; i++) {
					Attribute a;
					a.location = location + i;
					a.name = sb.name.c_str();
					a.type = to_type(type);
					attributes.push_back(a);
				}
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
			if (set >= sets.size()) {
				sets.resize(set + 1, std::nullopt);
				sets[set] = Descriptors{};
			}
			sets[set]->uniform_buffers.push_back(un);
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
			if (type.basetype == spirv_cross::SPIRType::Struct) {
				reflect_members(refl, refl.get_type(sb.type_id), un.members);
			}
			un.is_hlsl_counter_buffer = refl.buffer_is_hlsl_counter_buffer(sb.id);
			if (set >= sets.size()) {
				sets.resize(set + 1, std::nullopt);
				sets[set] = Descriptors{};
			}
			sets[set]->storage_buffers.push_back(un);
		}

		for (auto& si : resources.sampled_images) {
			auto type = refl.get_type(si.type_id);
			auto binding = refl.get_decoration(si.id, spv::DecorationBinding);
			auto set = refl.get_decoration(si.id, spv::DecorationDescriptorSet);
			CombinedImageSampler t;
			t.binding = binding;
			t.name = std::string(si.name.c_str());
			t.stage = stage;
			// maybe spirv cross bug?
			t.array_size = type.array.size() == 1 ? (type.array[0] == 1 ? 0 : type.array[0]) : -1;
			t.shadow = type.image.depth;
			if (set >= sets.size()) {
				sets.resize(set + 1, std::nullopt);
				sets[set] = Descriptors{};
			}
			sets[set]->combined_image_samplers.push_back(t);
		}

		for (auto& sa : resources.separate_samplers) {
			auto type = refl.get_type(sa.type_id);
			auto binding = refl.get_decoration(sa.id, spv::DecorationBinding);
			auto set = refl.get_decoration(sa.id, spv::DecorationDescriptorSet);
			Sampler t;
			t.binding = binding;
			t.name = std::string(sa.name.c_str());
			t.stage = stage;
			// maybe spirv cross bug?
			t.array_size = type.array.size() == 1 ? (type.array[0] == 1 ? 0 : type.array[0]) : -1;
			t.shadow = type.image.depth;
			if (set >= sets.size()) {
				sets.resize(set + 1, std::nullopt);
				sets[set] = Descriptors{};
			}
			sets[set]->samplers.push_back(t);
		}

		for (auto& si : resources.separate_images) {
			auto type = refl.get_type(si.type_id);
			auto binding = refl.get_decoration(si.id, spv::DecorationBinding);
			auto set = refl.get_decoration(si.id, spv::DecorationDescriptorSet);
			SampledImage t;
			t.binding = binding;
			t.name = std::string(si.name.c_str());
			t.stage = stage;
			// maybe spirv cross bug?
			t.array_size = type.array.size() == 1 ? (type.array[0] == 1 ? 0 : type.array[0]) : -1;
			if (set >= sets.size()) {
				sets.resize(set + 1, std::nullopt);
				sets[set] = Descriptors{};
			}
			sets[set]->sampled_images.push_back(t);
		}

		for (auto& sb : resources.storage_images) {
			auto type = refl.get_type(sb.type_id);
			auto binding = refl.get_decoration(sb.id, spv::DecorationBinding);
			auto set = refl.get_decoration(sb.id, spv::DecorationDescriptorSet);
			StorageImage un;
			un.binding = binding;
			un.stage = stage;
			un.name = sb.name.c_str();
			// maybe spirv cross bug?
			un.array_size = type.array.size() == 1 ? (type.array[0] == 1 ? 0 : type.array[0]) : -1;
			if (set >= sets.size()) {
				sets.resize(set + 1, std::nullopt);
				sets[set] = Descriptors{};
			}
			sets[set]->storage_images.push_back(un);
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
			if (set >= sets.size()) {
				sets.resize(set + 1, std::nullopt);
				sets[set] = Descriptors{};
			}
			sets[set]->subpass_inputs.push_back(s);
		}

		// ASs
		for (auto& as : resources.acceleration_structures) {
			auto type = refl.get_type(as.type_id);
			auto binding = refl.get_decoration(as.id, spv::DecorationBinding);
			auto set = refl.get_decoration(as.id, spv::DecorationDescriptorSet);
			AccelerationStructure s;
			s.name = std::string(as.name.c_str());
			s.binding = binding;
			s.stage = stage;
			s.array_size = type.array.size() == 1 ? (type.array[0] == 1 ? 0 : type.array[0]) : -1;
			if (set >= sets.size()) {
				sets.resize(set + 1, std::nullopt);
				sets[set] = Descriptors{};
			}
			sets[set]->acceleration_structures.push_back(s);
		}

		for (auto& sc : refl.get_specialization_constants()) {
			spec_constants.emplace_back(SpecConstant{ sc.constant_id, to_type(refl.get_type(refl.get_constant(sc.id).constant_type)), (VkShaderStageFlags)stage });
		}

		// remove duplicated bindings (aliased bindings)
		// TODO: we need to preserve this information somewhere
		for (auto& set : sets) {
			if (!set) {
				continue;
			}
			unq(set->samplers);
			unq(set->sampled_images);
			unq(set->combined_image_samplers);
			unq(set->uniform_buffers);
			unq(set->storage_buffers);
			unq(set->texel_buffers);
			unq(set->subpass_inputs);
			unq(set->storage_images);
			unq(set->acceleration_structures);
		}

		std::sort(spec_constants.begin(), spec_constants.end(), binding_cmp);

		for (auto& set : sets) {
			if (!set) {
				continue;
			}
			unsigned max_binding = 0;
			for (auto& ub : set->uniform_buffers) {
				max_binding = std::max(max_binding, ub.binding);
			}
			for (auto& ub : set->storage_buffers) {
				max_binding = std::max(max_binding, ub.binding);
			}
			for (auto& ub : set->samplers) {
				max_binding = std::max(max_binding, ub.binding);
			}
			for (auto& ub : set->sampled_images) {
				max_binding = std::max(max_binding, ub.binding);
			}
			for (auto& ub : set->combined_image_samplers) {
				max_binding = std::max(max_binding, ub.binding);
			}
			for (auto& ub : set->subpass_inputs) {
				max_binding = std::max(max_binding, ub.binding);
			}
			for (auto& ub : set->storage_buffers) {
				max_binding = std::max(max_binding, ub.binding);
			}
			for (auto& ub : set->acceleration_structures) {
				max_binding = std::max(max_binding, ub.binding);
			}
			set->highest_descriptor_binding = max_binding;
		}

		// push constants
		for (auto& si : resources.push_constant_buffers) {
			auto type = refl.get_type(si.base_type_id);
			VkPushConstantRange pcr;
			pcr.offset = 0;
			pcr.size = (uint32_t)refl.get_declared_struct_size(type);
			pcr.stageFlags = stage;
			push_constant_ranges.push_back(pcr);
		}

		if (stage == VK_SHADER_STAGE_COMPUTE_BIT) {
			local_size = { refl.get_execution_mode_argument(spv::ExecutionMode::ExecutionModeLocalSize, 0),
				             refl.get_execution_mode_argument(spv::ExecutionMode::ExecutionModeLocalSize, 1),
				             refl.get_execution_mode_argument(spv::ExecutionMode::ExecutionModeLocalSize, 2) };
		}

		return stage;
	}

	void Program::append(const Program& o) {
		attributes.insert(attributes.end(), o.attributes.begin(), o.attributes.end());
		push_constant_ranges.insert(push_constant_ranges.end(), o.push_constant_ranges.begin(), o.push_constant_ranges.end());
		spec_constants.insert(spec_constants.end(), o.spec_constants.begin(), o.spec_constants.end());
		unq(spec_constants);
		if (o.sets.size() > sets.size()) {
			sets.resize(o.sets.size());
		}
		for (auto index = 0; index < o.sets.size(); index++) {
			auto& os = o.sets[index];
			if (!os) {
				continue;
			}
			auto& s = sets[index];
			if (!s) {
				s = os;
			}
			s->samplers.insert(s->samplers.end(), os->samplers.begin(), os->samplers.end());
			s->sampled_images.insert(s->sampled_images.end(), os->sampled_images.begin(), os->sampled_images.end());
			s->combined_image_samplers.insert(s->combined_image_samplers.end(), os->combined_image_samplers.begin(), os->combined_image_samplers.end());
			s->uniform_buffers.insert(s->uniform_buffers.end(), os->uniform_buffers.begin(), os->uniform_buffers.end());
			s->storage_buffers.insert(s->storage_buffers.end(), os->storage_buffers.begin(), os->storage_buffers.end());
			s->texel_buffers.insert(s->texel_buffers.end(), os->texel_buffers.begin(), os->texel_buffers.end());
			s->subpass_inputs.insert(s->subpass_inputs.end(), os->subpass_inputs.begin(), os->subpass_inputs.end());
			s->storage_images.insert(s->storage_images.end(), os->storage_images.begin(), os->storage_images.end());
			s->acceleration_structures.insert(s->acceleration_structures.end(), os->acceleration_structures.begin(), os->acceleration_structures.end());

			unq(s->samplers);
			unq(s->sampled_images);
			unq(s->combined_image_samplers);
			unq(s->uniform_buffers);
			unq(s->storage_buffers);
			unq(s->texel_buffers);
			unq(s->subpass_inputs);
			unq(s->storage_images);
			unq(s->acceleration_structures);
			s->highest_descriptor_binding = std::max(s->highest_descriptor_binding, os->highest_descriptor_binding);
		}

		stages |= o.stages;
		local_size = o.local_size;
	}
} // namespace vuk

size_t std::hash<vuk::ShaderModuleCreateInfo>::operator()(vuk::ShaderModuleCreateInfo const& x) const noexcept {
	size_t h = 0;
	hash_combine(h, x.filename);
	return h;
}
