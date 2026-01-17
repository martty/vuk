#include "../src/ShadercIncluder.hpp"
#include "vuk/Exception.hpp"
#include "vuk/Result.hpp"
#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/ShaderSource.hpp"

#include <shaderc/shaderc.hpp>
#include <unordered_map>

namespace vuk {
	Result<std::vector<uint32_t>> compile_glsl(const ShaderModuleCreateInfo& cinfo, uint32_t shader_compiler_target_version) {
		shaderc::CompileOptions options;

		static const std::unordered_map<uint32_t, uint32_t> target_version = {
			{ VK_API_VERSION_1_0, shaderc_env_version_vulkan_1_0 },
			{ VK_API_VERSION_1_1, shaderc_env_version_vulkan_1_1 },
			{ VK_API_VERSION_1_2, shaderc_env_version_vulkan_1_2 },
			{ VK_API_VERSION_1_3, shaderc_env_version_vulkan_1_3 },
		};

		options.SetTargetEnvironment(shaderc_target_env_vulkan, target_version.at(shader_compiler_target_version));

		static const std::unordered_map<ShaderCompileOptions::OptimizationLevel, shaderc_optimization_level> optimization_level = {
			{ ShaderCompileOptions::OptimizationLevel::O0, shaderc_optimization_level_zero },
			{ ShaderCompileOptions::OptimizationLevel::O1, shaderc_optimization_level_performance },
			{ ShaderCompileOptions::OptimizationLevel::O2, shaderc_optimization_level_performance },
			{ ShaderCompileOptions::OptimizationLevel::O3, shaderc_optimization_level_performance },
		};

		options.SetOptimizationLevel(optimization_level.at(cinfo.compile_options.optimization_level));

		options.SetIncluder(std::make_unique<ShadercDefaultIncluder>());
		for (auto& [k, v] : cinfo.defines) {
			options.AddMacroDefinition(k, v);
		}

		if (cinfo.compile_options.compiler_flags & ShaderCompilerFlagBits::eNoWarnings)
			options.SetSuppressWarnings();
		else if (cinfo.compile_options.compiler_flags & ShaderCompilerFlagBits::eWarningsAsErrors)
			options.SetWarningsAsErrors();

		if (cinfo.compile_options.compiler_flags & ShaderCompilerFlagBits::eInvertY)
			options.SetInvertY(true);

		const shaderc::Compiler compiler;
		const auto result = compiler.CompileGlslToSpv(cinfo.source.as_c_str(), shaderc_glsl_infer_from_source, cinfo.filename.c_str(), options);
		if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
			std::string message = result.GetErrorMessage();
			return { expected_error, ShaderCompilationException{ message } };
		}

		return { expected_value, std::vector<uint32_t>{ result.cbegin(), result.cend() } };
	}
} // namespace vuk