#pragma once

#include "../src/CreateInfo.hpp"
#include "vuk/Config.hpp"
#include "vuk/vuk_fwd.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace spirv_cross {
	struct SPIRType;
	class Compiler;
}; // namespace spirv_cross
namespace vuk {
	enum class ShaderSourceLanguage { eGlsl, eHlsl, eSpirv };

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
		ShaderSource() = default;

		ShaderSource(const ShaderSource& o) noexcept {
			data = o.data;
			if (!data.empty()) {
				data_ptr = data.data();
			} else {
				data_ptr = o.data_ptr;
			}
			size = o.size;
			language = o.language;
			hlsl_stage = o.hlsl_stage;
		}
		ShaderSource& operator=(const ShaderSource& o) noexcept {
			data = o.data;
			if (!data.empty()) {
				data_ptr = data.data();
			} else {
				data_ptr = o.data_ptr;
			}
			language = o.language;
			size = o.size;
			hlsl_stage = o.hlsl_stage;

			return *this;
		}

		ShaderSource(ShaderSource&& o) noexcept {
			data = std::move(o.data);
			if (!data.empty()) {
				data_ptr = data.data();
			} else {
				data_ptr = o.data_ptr;
			}
			size = o.size;
			language = o.language;
			hlsl_stage = o.hlsl_stage;
		}
		ShaderSource& operator=(ShaderSource&& o) noexcept {
			data = std::move(o.data);
			if (!data.empty()) {
				data_ptr = data.data();
			} else {
				data_ptr = o.data_ptr;
			}
			size = o.size;
			language = o.language;
			hlsl_stage = o.hlsl_stage;
			
			return *this;
		}

#if VUK_USE_SHADERC
		static ShaderSource glsl(std::string_view source) {
			ShaderSource shader;
			shader.data.resize(idivceil(source.size() + 1, sizeof(uint32_t)));
			memcpy(shader.data.data(), source.data(), source.size() * sizeof(std::string_view::value_type));
			shader.data_ptr = shader.data.data();
			shader.size = shader.data.size();
			shader.language = ShaderSourceLanguage::eGlsl;
			return shader;
		}
#endif

#if VUK_USE_DXC
		static ShaderSource hlsl(std::string_view source, HlslShaderStage stage = HlslShaderStage::eInferred) {
			ShaderSource shader;
			shader.data.resize(idivceil(source.size() + 1, sizeof(uint32_t)));
			memcpy(shader.data.data(), source.data(), source.size() * sizeof(std::string_view::value_type));
			shader.data_ptr = shader.data.data();
			shader.language = ShaderSourceLanguage::eHlsl;
			shader.hlsl_stage = stage;
			return shader;
		}
#endif

		static ShaderSource spirv(std::vector<uint32_t> source) {
			ShaderSource shader;
			shader.data = std::move(source);
			shader.data_ptr = shader.data.data();
			shader.size = shader.data.size();
			shader.language = ShaderSourceLanguage::eSpirv;
			return shader;
		}

		static ShaderSource spirv(const uint32_t* source, size_t size) {
			ShaderSource shader;
			shader.data_ptr = source;
			shader.size = size;
			shader.language = ShaderSourceLanguage::eSpirv;
			return shader;
		}

		const char* as_c_str() const {
			return reinterpret_cast<const char*>(data_ptr);
		}

		const uint32_t* as_spirv() const {
			return data_ptr;
		}

		const uint32_t* data_ptr = nullptr;
		size_t size = 0;
		std::vector<uint32_t> data;
		ShaderSourceLanguage language;
		HlslShaderStage hlsl_stage;
	};

	inline bool operator==(const ShaderSource& a, const ShaderSource& b) noexcept {
		bool basics = a.language == b.language && a.size == b.size;
		if (!basics) {
			return false;
		}
		if (a.data_ptr == b.data_ptr) {
			return true;
		}
		return memcmp(a.data_ptr, b.data_ptr, a.size) == 0;
	}

	struct ShaderModuleCreateInfo {
		ShaderSource source;
		std::string filename;
		std::vector<std::pair<std::string, std::string>> defines;

		bool operator==(const ShaderModuleCreateInfo& o) const noexcept {
			return source == o.source && defines == o.defines;
		}
	};
} // namespace vuk
