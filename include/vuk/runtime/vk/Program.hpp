#pragma once

#include "vuk/Config.hpp"
#include "vuk/runtime/CreateInfo.hpp"
#include "vuk/runtime/vk/Descriptor.hpp"
#include "vuk/vuk_fwd.hpp"

#include <array>
#include <optional>
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

		struct Binding {
			DescriptorType type;

			std::string name;

			unsigned binding;
			size_t size;
			size_t min_size;
			unsigned array_size;

			std::vector<Member> members;

			bool is_hlsl_counter_buffer = false;
			bool shadow; // if this is a samplerXXXShadow / samplerShadow
			bool non_writable;
			bool non_readable;

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
			std::vector<Binding> bindings; // sorted by binding #

			unsigned highest_descriptor_binding = 0;
		};
		std::vector<std::optional<Descriptors>> sets;
		std::vector<std::pair<uint32_t, Binding*>> flat_bindings; // sorted by set, and then by binding
		VkShaderStageFlags stages = {};
		void append(const Program& o);

	private:
		void flatten_bindings();
		Descriptors& ensure_set(size_t set_index);
	};

	struct ShaderModule {
		VkShaderModule shader_module;
		Program reflection_info;
		VkShaderStageFlagBits stage;
		bool override_entry_point_name_to_main = false;
	};

	struct ShaderModuleCreateInfo;
} // namespace vuk

namespace vuk {
	template<>
	struct create_info<ShaderModule> {
		using type = ShaderModuleCreateInfo;
	};
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::ShaderModuleCreateInfo> {
		size_t operator()(vuk::ShaderModuleCreateInfo const& x) const noexcept;
	};
}; // namespace std
