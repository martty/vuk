#pragma once

#include "vuk/CreateInfo.hpp"
#include "vuk/Config.hpp"
#include "vuk/vuk_fwd.hpp"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace vuk {
	struct Program {
		enum class Type {
			einvalid,
			euint,
			euint64_t,
			eint,
			eint64_t,
			efloat,
			edouble,
			euvec2,
			euvec3,
			euvec4,
			eivec2,
			eivec3,
			eivec4,
			evec2,
			evec3,
			evec4,
			edvec2,
			edvec3,
			edvec4,
			emat3,
			emat4,
			edmat3,
			edmat4,
			eu64vec2,
			eu64vec3,
			eu64vec4,
			ei64vec2,
			ei64vec3,
			ei64vec4,
			estruct
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

			bool is_hlsl_counter_buffer = false;

			std::vector<Member> members;

			VkShaderStageFlags stage;
		};

		struct StorageImage {
			std::string name;

			unsigned array_size;
			unsigned binding;
			VkShaderStageFlags stage;
		};

		struct SampledImage {
			std::string name;

			unsigned array_size;
			unsigned binding;
			VkShaderStageFlags stage;
		};

		struct CombinedImageSampler {
			std::string name;

			unsigned array_size;
			unsigned binding;

			bool shadow; // if this is a samplerXXXShadow

			VkShaderStageFlags stage;
		};

		struct Sampler {
			std::string name;

			unsigned array_size;
			unsigned binding;

			bool shadow; // if this is a samplerShadow

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

		struct AccelerationStructure {
			std::string name;

			unsigned array_size;
			unsigned binding;
			VkShaderStageFlags stage;
		};

		struct SpecConstant {
			unsigned binding; // constant_id
			Type type;

			VkShaderStageFlags stage;
		};

		VkShaderStageFlagBits introspect(const uint32_t* ir, size_t word_count);

		std::array<unsigned, 3> local_size;

		std::vector<Attribute> attributes;
		std::vector<VkPushConstantRange> push_constant_ranges;
		std::vector<SpecConstant> spec_constants;
		struct Descriptors {
			std::vector<UniformBuffer> uniform_buffers;
			std::vector<StorageBuffer> storage_buffers;
			std::vector<StorageImage> storage_images;
			std::vector<TexelBuffer> texel_buffers;
			std::vector<CombinedImageSampler> combined_image_samplers;
			std::vector<SampledImage> sampled_images;
			std::vector<Sampler> samplers;
			std::vector<SubpassInput> subpass_inputs;
			std::vector<AccelerationStructure> acceleration_structures;

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

	template<>
	struct create_info<vuk::ShaderModule> {
		using type = vuk::ShaderModuleCreateInfo;
	};
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::ShaderModuleCreateInfo> {
		size_t operator()(vuk::ShaderModuleCreateInfo const& x) const noexcept;
	};
}; // namespace std
