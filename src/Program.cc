#include "Program.hpp"
#include <gsl/gsl_util>
#include <spirv_cross.hpp>
#include <regex>
#include <smolog.hpp>
#include <fstream>
#include <sstream>

std::string slurp(const std::string& path) {
	std::ostringstream buf;
	std::ifstream input(path.c_str());
	buf << input.rdbuf();
	return buf.str();
}

void burp(const std::string& in, const std::string& path) {
	std::ofstream output(path.c_str(), std::ios::trunc);
	if (!output.is_open()) {
	}
	output << in;
	output.close();
}

void Program::compile(const std::string& per_pass_glsl) {
	load_source(per_pass_glsl);
}

bool Program::load_source(const std::string& per_pass_glsl) {
	shaderc::Compiler compiler;
	// Like -DMY_DEFINE=1

	shaderc::CompileOptions options;
	options.AddMacroDefinition("B_ATTRIBUTE", "0");
	options.AddMacroDefinition("B_ATTRIBUTE_DYNAMIC", "1");
	options.AddMacroDefinition("B_PER_PASS", "2");
	options.AddMacroDefinition("B_PER_DRAW", "3");
	options.AddMacroDefinition("B_TEXTURE_2D", "4");
	options.AddMacroDefinition("B_TEXTURE_2D_ADDRESSES", "5");
	options.AddMacroDefinition("B_MATERIAL", "6");
	options.AddMacroDefinition("B_DECAL", "7");
	options.AddMacroDefinition("B_USER0", "8");
	//options.SetWarningsAsErrors();



	for (size_t i = 0; i < shaders.size(); i++) {
		auto& shader = shaders[i];
		auto source = slurp(shader);
		shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(source, shaderc_glsl_infer_from_source, shader.c_str(), options);
		if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
			//Platform::log->error("%s", module.GetErrorMessage().c_str());
			return false;
		} else {
			std::vector<unsigned> spirv(module.cbegin(), module.cend());

			spirv_cross::Compiler refl(spirv.data(), spirv.size());
			auto stage = introspect(refl);
			spirv_data[stage] = spirv;

			stage_to_shader_mapping[stage] = i;
		}
	}

	return true;
}

void Program::link(vk::Device device) {
	this->device = device;
	// TODO: some recycling
	pipeline_shader_stage_CIs.clear();
	modules.clear();

	for (auto& spirv_it : spirv_data) {
		auto stage = spirv_it.first;
		auto& spirv = spirv_it.second;
		vk::ShaderModuleCreateInfo moduleCreateInfo;
		moduleCreateInfo.codeSize = spirv.size() * sizeof(uint32_t);
		moduleCreateInfo.pCode = (uint32_t*)spirv.data();
		auto module = device.createShaderModule(moduleCreateInfo);
		modules[stage] = module;
		vk::PipelineShaderStageCreateInfo shaderStage;
		shaderStage.stage = stage;
		shaderStage.module = module;
		shaderStage.pName = "main"; // todo : make param
		pipeline_shader_stage_CIs.push_back(shaderStage);
	}
}

Program::~Program() {
	/*if (pipeline_shader_stage_CIs.size() > 0 && shaders.size() > 1)
		printf("tearing down program %s %s..\n", shaders[0].uid.c_str(), shaders[1].uid.c_str());
	for (auto& ci : pipeline_shader_stage_CIs) {
		device.destroyShaderModule(ci.module);
		ci.module = vk::ShaderModule{};
	}*/
}

vk::ShaderStageFlagBits Program::introspect(const spirv_cross::Compiler& refl) {
	auto resources = refl.get_shader_resources();
	//auto cross_stage = refl.
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

		auto ub_it = std::find_if(uniform_buffers.begin(), uniform_buffers.end(), [=](auto& a) {return a.name == ub.name.c_str(); });
		if (ub_it == uniform_buffers.end()) {
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
				
				material_size = un.size;// refl.get_declared_struct_size(arr_t);

				for (unsigned i = 0; i < arr_t.member_types.size(); i++) {
					auto name = refl.get_member_name(arr_t.self, i);
					auto sfield_type = refl.get_type(arr_t.member_types[i]);
					auto offset = refl.type_struct_member_offset(arr_t, i);
				}
			}
			uniform_buffers.push_back(un);
		} else {
			ub_it->stage |= stage;
		}
	}
	for (auto& sb : resources.storage_buffers) {
		auto type = refl.get_type(sb.type_id);
		auto binding = refl.get_decoration(sb.id, spv::DecorationBinding);

		auto sb_it = std::find_if(storage_buffers.begin(), storage_buffers.end(), [=](auto& a) {return a.name == sb.name.c_str(); });
		if (sb_it == storage_buffers.end()) {
			StorageBuffer un;
			un.binding = binding;
			un.stage = stage;
			un.name = sb.name.c_str();
			un.min_size = refl.get_declared_struct_size(refl.get_type(sb.type_id));
			/*if (type.array.size() > 0)
				un.array_size = type.array[0];
			else
				un.array_size = 1;*/
			storage_buffers.push_back(un);
		} else {
			sb_it->stage |= stage;
		}
	}

	for (auto& si : resources.sampled_images) {
		auto type = refl.get_type(si.type_id);
		auto binding = refl.get_decoration(si.id, spv::DecorationBinding);
		auto si_it = std::find_if(samplers.begin(), samplers.end(), [=](auto& a) {return a.name == si.name.c_str(); });
		if (si_it == samplers.end()) {
			Sampler t;
			t.binding = binding;
			t.name = std::string(si.name.c_str());
			t.stage = stage;
			t.array_size = type.array[0];
			samplers.push_back(t);
		} else {
			si_it->stage |= stage;
		}
	}
	
	// subpass inputs
	for (auto& si : resources.subpass_inputs) {
		auto type = refl.get_type(si.type_id);
		auto binding = refl.get_decoration(si.id, spv::DecorationBinding);
		SubpassInput s;
		s.name = std::string(si.name.c_str());
		s.binding = binding;
		s.stage = stage;
		subpass_inputs.push_back(s);
	}

	return stage;
}
