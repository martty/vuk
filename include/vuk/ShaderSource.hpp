#pragma once
#include <unordered_map>
#include <vector>
#include <array>
#include <../src/CreateInfo.hpp>
#include <vuk/Config.hpp>
#include <vuk/vuk_fwd.hpp>

namespace spirv_cross {
	struct SPIRType;
	class Compiler;
};
namespace vuk {
	enum class ShaderSourceLanguage {
#if VUK_USE_SHADERC
		eGlsl,
#endif
#if VUK_USE_DXC
		eHlsl,
#endif
		eSpirv
	};

	/// @brief Specifies the HLSL Shader Stage for a given HLSL shader.
	enum class HlslShaderStage {
		/// @brief Will infer the Shader Stage from the filename.
		eInferred,
		eVertex,
		ePixel,
		eCompute,
		eGeometry,
		eMesh,
		eHull,
		eDomain,
		eAmplification
	};

	/// @brief Wrapper over either a GLSL, HLSL, or SPIR-V source
	struct ShaderSource {
#if VUK_USE_SHADERC
		static ShaderSource glsl(std::string_view source) {
			ShaderSource shader;
			shader.data.resize(idivceil(source.size() + 1, sizeof(uint32_t)));
			memcpy(shader.data.data(), source.data(), source.size() * sizeof(std::string_view::value_type));
			shader.language = ShaderSourceLanguage::eGlsl;
			return shader;
		}
#endif

#if VUK_USE_DXC
		static ShaderSource hlsl(std::string_view source, HlslShaderStage stage = HlslShaderStage::eInferred) {
			ShaderSource shader;
			shader.data.resize(idivceil(source.size() + 1, sizeof(uint32_t)));
			memcpy(shader.data.data(), source.data(), source.size() * sizeof(std::string_view::value_type));
			shader.language = ShaderSourceLanguage::eHlsl;
			shader.hlsl_stage = HlslShaderStage::eInferred;
			return shader;
		}
#endif

		static ShaderSource spirv(std::vector<uint32_t> source) {
			ShaderSource shader;
			shader.data = std::move(source);
			shader.language = ShaderSourceLanguage::eSpirv;
			return shader;
		}

		const char* as_c_str() const {
			return reinterpret_cast<const char*>(data.data());
		}

		const uint32_t* as_spirv() const {
			return data.data();
		}

		std::vector<uint32_t> data;
		ShaderSourceLanguage language;
		HlslShaderStage hlsl_stage;
	};

	inline bool operator==(const ShaderSource& a, const ShaderSource& b) noexcept {
		return a.language == b.language && a.data == b.data;
	}

	struct ShaderModuleCreateInfo {
		ShaderSource source;
		std::string filename;

		bool operator==(const ShaderModuleCreateInfo& o) const {
			return source == o.source;
		}
	};
}
