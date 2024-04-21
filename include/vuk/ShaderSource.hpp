#pragma once

#include "vuk/Config.hpp"
#include "vuk/Util.hpp"
#include "vuk/runtime/CreateInfo.hpp"
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

	struct ShaderCompileOptions {
		enum class OptimizationLevel { O0, O1, O2, O3 } optimization_level = OptimizationLevel::O3;

		std::vector<std::wstring> dxc_extra_arguments = { L"-spirv", L"-fvk-use-gl-layout", L"-no-warnings" };
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
			entry_point = o.entry_point;
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
			entry_point = o.entry_point;

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
			entry_point = o.entry_point;
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
			entry_point = o.entry_point;

			return *this;
		}

#if VUK_USE_SHADERC
		static ShaderSource glsl(std::string_view source, const ShaderCompileOptions& compile_options, std::string entry_point = "main") {
			ShaderSource shader;
			shader.data.resize(idivceil(source.size() + 1, sizeof(uint32_t)));
			memcpy(shader.data.data(), source.data(), source.size() * sizeof(std::string_view::value_type));
			shader.data_ptr = shader.data.data();
			shader.size = shader.data.size();
			shader.language = ShaderSourceLanguage::eGlsl;
			shader.entry_point = std::move(entry_point);
			shader.opt_level = compile_options.optimization_level;
			return shader;
		}
#endif

#if VUK_USE_DXC
		static ShaderSource hlsl(std::string_view source,
		                         const ShaderCompileOptions& compile_options,
		                         HlslShaderStage stage = HlslShaderStage::eInferred,
		                         std::string entry_point = "main") {
			ShaderSource shader;
			shader.data.resize(idivceil(source.size() + 1, sizeof(uint32_t)));
			memcpy(shader.data.data(), source.data(), source.size() * sizeof(std::string_view::value_type));
			shader.data_ptr = shader.data.data();
			shader.size = shader.data.size();
			shader.language = ShaderSourceLanguage::eHlsl;
			shader.hlsl_stage = stage;
			shader.entry_point = std::move(entry_point);
			shader.opt_level = compile_options.optimization_level;
			return shader;
		}
#endif

		static ShaderSource spirv(std::vector<uint32_t> source, std::string entry_point = "main") {
			ShaderSource shader;
			shader.data = std::move(source);
			shader.data_ptr = shader.data.data();
			shader.size = shader.data.size();
			shader.language = ShaderSourceLanguage::eSpirv;
			shader.entry_point = std::move(entry_point);
			return shader;
		}

		static ShaderSource spirv(const uint32_t* source, size_t size, std::string entry_point = "main") {
			ShaderSource shader;
			shader.data_ptr = source;
			shader.size = size;
			shader.language = ShaderSourceLanguage::eSpirv;
			shader.entry_point = std::move(entry_point);
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
		std::string entry_point;
		ShaderCompileOptions::OptimizationLevel opt_level;
	};

	inline bool operator==(const ShaderSource& a, const ShaderSource& b) noexcept {
		bool basics = a.language == b.language && a.size == b.size && a.entry_point == b.entry_point && a.opt_level == b.opt_level;
		if (!basics) {
			return false;
		}
		if (a.data_ptr == b.data_ptr) {
			return true;
		}
		return memcmp(a.data_ptr, b.data_ptr, sizeof(uint32_t) * a.size) == 0;
	}
} // namespace vuk

namespace vuk {
	struct ShaderModuleCreateInfo {
		ShaderSource source;
		std::string filename;
		std::vector<std::pair<std::string, std::string>> defines;
		ShaderCompileOptions compile_options = {};

		bool operator==(const ShaderModuleCreateInfo& o) const noexcept {
			return source == o.source && defines == o.defines;
		}
	};
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::ShaderSource> {
		size_t operator()(vuk::ShaderSource const& x) const noexcept;
	};
} // namespace std
