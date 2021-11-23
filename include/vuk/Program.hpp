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
	struct Program {
		enum class Type {
			euint, eint, efloat, edouble,
			euvec2, euvec3, euvec4,
			eivec2, eivec3, eivec4,
			evec2, evec3, evec4,
			edvec2, edvec3, edvec4,
			emat4, edmat4, estruct
		};

		struct Attribute {
			std::string name;

			size_t location;
			Type type;
		};

		struct TextureAddress {
			unsigned container;
			float page;
		};

		struct Member {
			std::string name;
			std::string type_name; // if this is a struct
			Type type;
			size_t size;
			size_t offset;
			unsigned array_size;
			std::vector<Member> members;
		};

		// always a struct
		struct UniformBuffer {
			std::string name;

			unsigned binding;
			size_t size;
			unsigned array_size;

			std::vector<Member> members;

			VkShaderStageFlags stage;
		};

		struct StorageBuffer {
			std::string name;

			unsigned binding;
			size_t min_size;

			std::vector<Member> members;

			VkShaderStageFlags stage;

		};

		struct StorageImage {
			std::string name;

			unsigned array_size;
			unsigned binding;
			VkShaderStageFlags stage;
		};

		struct Sampler {
			std::string name;

			unsigned array_size;
			unsigned binding;

			bool shadow; // if this is a samplerXXXShadow

			VkShaderStageFlags stage;
		};

		struct TexelBuffer {
			std::string name;

			unsigned binding;
			VkShaderStageFlags stage;
		};

		struct SubpassInput {
			std::string name;

			unsigned binding;
			VkShaderStageFlags stage;
		};

		struct SpecConstant {
			unsigned binding; // constant_id
			Type type;

			VkShaderStageFlags stage;
		};

		VkShaderStageFlagBits introspect(const spirv_cross::Compiler& refl);

		std::array<unsigned, 3> local_size;

		std::vector<Attribute> attributes;
		std::vector<VkPushConstantRange> push_constant_ranges;
		std::vector<SpecConstant> spec_constants;
		struct Descriptors {
			std::vector<UniformBuffer> uniform_buffers;
			std::vector<StorageBuffer> storage_buffers;
			std::vector<StorageImage> storage_images;
			std::vector<TexelBuffer> texel_buffers;
			std::vector<Sampler> samplers;
			std::vector<SubpassInput> subpass_inputs;

			unsigned highest_descriptor_binding = 0;
		};
		std::unordered_map<size_t, Descriptors> sets;
		VkShaderStageFlags stages = {};
		void append(const Program& o);
	};

	struct ShaderModule {
		VkShaderModule shader_module;
		vuk::Program reflection_info;
		VkShaderStageFlagBits stage;
	};

	struct ShaderModuleCreateInfo;

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
