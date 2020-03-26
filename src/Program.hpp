#pragma once
#include <vulkan/vulkan.hpp>
#include <unordered_map>
#include <vector>
#include "CreateInfo.hpp"

namespace spirv_cross {
	struct SPIRType;
	class Compiler;
};
namespace vuk {
	struct Program {
		struct Attribute {
			std::string name;
			std::string full_type;

			bool dynamic;
			size_t vector_size; // usually 2 for vec2, 3 for vec3..
			size_t size_of_elem; // sizeof(T) (12 for vec3)

			size_t offset;
		};

		struct TextureAddress {
			unsigned container;
			float page;
		};

		struct UniformBuffer {
			std::string name;
			std::string full_name;

			unsigned binding;
			size_t size;
			unsigned array_size;
			vk::ShaderStageFlags stage;
		};

		struct StorageBuffer {
			std::string name;
			std::string full_name;

			unsigned binding;
			size_t min_size;
			vk::ShaderStageFlags stage;

		};

		struct Sampler {
			std::string name;

			unsigned array_size;
			unsigned binding;

			vk::ShaderStageFlags stage;
		};

		struct TexelBuffer {
			std::string name;

			unsigned binding;
			vk::ShaderStageFlags stage;

		};

		struct SubpassInput {
			std::string name;

			unsigned binding;
			vk::ShaderStageFlags stage;
		};

		vk::ShaderStageFlagBits introspect(const spirv_cross::Compiler& refl);

		std::vector<Attribute> attributes;
		std::vector<vk::PushConstantRange> push_constant_ranges;
		struct Descriptors {
			std::vector<UniformBuffer> uniform_buffers;
			std::vector<StorageBuffer> storage_buffers;
			std::vector<TexelBuffer> texel_buffers;
			std::vector<Sampler> samplers;
			std::vector<SubpassInput> subpass_inputs;
		};
		std::unordered_map<size_t, Descriptors> sets;

		void append(const Program& o);
	};

	struct ShaderModuleCreateInfo {
		std::string source;
		std::string filename;

		bool operator==(const ShaderModuleCreateInfo& o) const {
			return source == o.source; // we check for content equality
		}
	};

	struct ShaderModule {
		vk::ShaderModule shader_module;
		vuk::Program reflection_info;
		vk::ShaderStageFlagBits stage;
	};

	template<> struct create_info<vuk::ShaderModule> {
		using type = vuk::ShaderModuleCreateInfo;
	};
}

namespace std {
	template <>
	struct hash<vuk::ShaderModuleCreateInfo> {
		size_t operator()(vuk::ShaderModuleCreateInfo const& x) const noexcept;
	};
};

