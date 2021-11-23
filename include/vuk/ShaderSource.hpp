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
	/// @brief Wrapper over either a GLSL or SPIRV source
	struct ShaderSource {
		static ShaderSource glsl(std::string_view source) {
			ShaderSource shader;
			shader.data.resize(idivceil(source.size() + 1, sizeof(uint32_t)));
			memcpy(shader.data.data(), source.data(), source.size() * sizeof(std::string_view::value_type));
			shader.is_spirv = false;
			return shader;
		}

		static ShaderSource spirv(std::vector<uint32_t> source) {
			ShaderSource shader;
			shader.data = std::move(source);
			shader.is_spirv = true;
			return shader;
		}

		const char* as_glsl() const {
			return reinterpret_cast<const char*>(data.data());
		}

		const uint32_t* as_spirv() const {
			return data.data();
		}

		std::vector<uint32_t> data;
		bool is_spirv;
	};

	inline bool operator==(const ShaderSource& a, const ShaderSource& b) noexcept {
		return a.is_spirv == b.is_spirv && a.data == b.data;
	}

	struct ShaderModuleCreateInfo {
		ShaderSource source;
		std::string filename;

		bool operator==(const ShaderModuleCreateInfo& o) const {
			return source == o.source;
		}
	};
}
