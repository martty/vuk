#pragma once
#include <vulkan/vulkan.hpp>
#include <shaderc/shaderc.hpp>
#include <unordered_map>
#include <vector>

namespace spirv_cross {
	struct SPIRType;
	class Compiler;
};

class Program {
	const static unsigned version = 2;
public:
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

public:
	void compile(const std::string& per_pass_glsl);

	bool load_source(const std::string& per_pass_glsl);

	void link(vk::Device device);

	~Program();
	vk::ShaderStageFlagBits introspect(const spirv_cross::Compiler& refl);

	std::vector<Attribute> attributes;
	std::vector<UniformBuffer> uniform_buffers;
	std::vector<StorageBuffer> storage_buffers;
	std::vector<TexelBuffer> texel_buffers;
	std::vector<Sampler> samplers;
	std::vector<SubpassInput> subpass_inputs;

	std::vector<std::string> shaders;

	size_t per_draw_size = 0;
	size_t material_size = 0;

	std::unordered_map<vk::ShaderStageFlagBits, std::vector<unsigned>> spirv_data;
	std::unordered_map<vk::ShaderStageFlagBits, vk::ShaderModule> modules;

	std::unordered_map<vk::ShaderStageFlagBits, size_t> stage_to_shader_mapping;

	std::vector<vk::PipelineShaderStageCreateInfo> pipeline_shader_stage_CIs;
	vk::Device device;

};