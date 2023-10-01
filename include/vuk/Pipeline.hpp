#pragma once

#include "../src/CreateInfo.hpp"
#include "vuk/Bitset.hpp"
#include "vuk/Config.hpp"
#include "vuk/Descriptor.hpp"
#include "vuk/FixedVector.hpp"
#include "vuk/Hash.hpp"
#include "vuk/Image.hpp"
#include "vuk/PipelineTypes.hpp"
#include "vuk/Program.hpp"
#include "vuk/ShaderSource.hpp"

#include <bit>
#include <vector>

namespace vuk {
	static constexpr uint32_t graphics_stage_count = 5;

	struct PipelineLayoutCreateInfo {
		VkPipelineLayoutCreateInfo plci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
		vuk::fixed_vector<VkPushConstantRange, VUK_MAX_PUSHCONSTANT_RANGES> pcrs;
		vuk::fixed_vector<vuk::DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> dslcis;

		bool operator==(const PipelineLayoutCreateInfo& o) const noexcept {
			return plci.flags == o.plci.flags && pcrs == o.pcrs && dslcis == o.dslcis;
		}
	};

	template<>
	struct create_info<VkPipelineLayout> {
		using type = vuk::PipelineLayoutCreateInfo;
	};

	enum class HitGroupType { 
		eTriangles = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
		eProcedural = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
	};

	struct HitGroup {
		HitGroupType type;
		uint32_t closest_hit = VK_SHADER_UNUSED_KHR;
		uint32_t any_hit = VK_SHADER_UNUSED_KHR;
		uint32_t intersection = VK_SHADER_UNUSED_KHR;
	};

	struct PipelineBaseCreateInfoBase {
		// 4 valid flags
		Bitset<4 * VUK_MAX_SETS* VUK_MAX_BINDINGS> binding_flags = {};
		// set flags on specific descriptor in specific set
		void set_binding_flags(unsigned set, unsigned binding, vuk::DescriptorBindingFlags flags) noexcept {
			unsigned f = static_cast<unsigned>(flags);
			binding_flags.set(set * 4 * VUK_MAX_BINDINGS + binding * 4 + 0, f & 0b1);
			binding_flags.set(set * 4 * VUK_MAX_BINDINGS + binding * 4 + 1, f & 0b10);
			binding_flags.set(set * 4 * VUK_MAX_BINDINGS + binding * 4 + 2, f & 0b100);
			binding_flags.set(set * 4 * VUK_MAX_BINDINGS + binding * 4 + 3, f & 0b1000);
		}
		// if the set has a variable count binding, the maximum number of bindings possible
		std::array<uint32_t, VUK_MAX_SETS> variable_count_max = {};
		void set_variable_count_binding(unsigned set, unsigned binding, uint32_t max_descriptors) noexcept {
			// clear all variable count bits
			for (unsigned i = 0; i < VUK_MAX_BINDINGS; i++) {
				binding_flags.set(set * 4 * VUK_MAX_BINDINGS + i * 4 + 3, 0);
			}
			// set variable count (0x8)
			binding_flags.set(set * 4 * VUK_MAX_BINDINGS + binding * 4 + 3, 1);
			variable_count_max[set] = max_descriptors;
		}

		vuk::fixed_vector<DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> explicit_set_layouts = {};
	};

	/* filled out by the user */
	struct PipelineBaseCreateInfo : PipelineBaseCreateInfoBase {
		friend class CommandBuffer;
		friend class Context;

	public:
		void add_shader(ShaderSource source, std::string filename) {
			shaders.emplace_back(std::move(source));
			shader_paths.emplace_back(std::move(filename));
		}

#if VUK_USE_SHADERC
		void add_glsl(std::string_view source, std::string filename, std::string entry_point = "main") {
			shaders.emplace_back(ShaderSource::glsl(source, std::move(entry_point)));
			shader_paths.emplace_back(std::move(filename));
		}

		void define(std::string key, std::string value) {
			defines.emplace_back(std::move(key), std::move(value));
		}
#endif

#if VUK_USE_DXC
		void add_hlsl(std::string_view source, std::string filename, HlslShaderStage stage = HlslShaderStage::eInferred, std::string entry_point = "main") {
			shaders.emplace_back(ShaderSource::hlsl(source, stage, std::move(entry_point)));
			shader_paths.emplace_back(std::move(filename));
		}
#endif

		void add_spirv(std::vector<uint32_t> source, std::string filename, std::string entry_point = "main") {
			shaders.emplace_back(ShaderSource::spirv(std::move(source), std::move(entry_point)));
			shader_paths.emplace_back(std::move(filename));
		}

		void add_static_spirv(const uint32_t* source, size_t num_words, std::string identifier, std::string entry_point = "main") {
			shaders.emplace_back(ShaderSource::spirv(source, num_words, std::move(entry_point)));
			shader_paths.emplace_back(std::move(identifier));
		}

		void add_hit_group(HitGroup hit_group) {
			hit_groups.emplace_back(hit_group);
		}

		std::vector<ShaderSource> shaders;
		std::vector<std::string> shader_paths;
		std::vector<HitGroup> hit_groups;
		std::vector<std::pair<std::string, std::string>> defines;
		/// @brief Recursion depth for RT pipelines, corresponding to maxPipelineRayRecursionDepth
		uint32_t max_ray_recursion_depth = 1;

		friend struct std::hash<PipelineBaseCreateInfo>;

	public:
		static vuk::fixed_vector<vuk::DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> build_descriptor_layouts(const Program&, const PipelineBaseCreateInfoBase&);
		bool operator==(const PipelineBaseCreateInfo& o) const noexcept {
			return shaders == o.shaders && binding_flags == o.binding_flags && variable_count_max == o.variable_count_max && defines == o.defines;
		}
	};

	struct PipelineBaseInfo {
		Name pipeline_name;
		vuk::Program reflection_info;
		std::vector<VkPipelineShaderStageCreateInfo> psscis;
		std::vector<std::string> entry_point_names;
		VkPipelineLayout pipeline_layout;
		std::array<DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> layout_info = {};
		fixed_vector<DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> dslcis = {}; // saved for debug purposes
		std::vector<HitGroup> hit_groups;
		uint32_t max_ray_recursion_depth;

		// 4 valid flags
		Bitset<4 * VUK_MAX_SETS* VUK_MAX_BINDINGS> binding_flags = {};
		// if the set has a variable count binding, the maximum number of bindings possible
		std::array<uint32_t, VUK_MAX_SETS> variable_count_max = {};
	};

	template<>
	struct create_info<PipelineBaseInfo> {
		using type = vuk::PipelineBaseCreateInfo;
	};
} // namespace vuk

namespace std {
	template<class T>
	struct hash<std::vector<T>> {
		size_t operator()(std::vector<T> const& x) const noexcept {
			size_t h = 0;
			for (auto& e : x) {
				hash_combine(h, e);
			}
			return h;
		}
	};

	template<class T, size_t N>
	struct hash<vuk::fixed_vector<T, N>> {
		size_t operator()(vuk::fixed_vector<T, N> const& x) const noexcept {
			size_t h = 0;
			for (auto& e : x) {
				hash_combine(h, e);
			}
			return h;
		}
	};

	template<class T1, class T2>
	struct hash<std::pair<T1, T2>> {
		size_t operator()(std::pair<T1, T2> const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.first, x.second);
			return h;
		}
	};

	template<>
	struct hash<vuk::ShaderSource> {
		size_t operator()(vuk::ShaderSource const& x) const noexcept;
	};

	template<>
	struct hash<vuk::PipelineBaseCreateInfo> {
		size_t operator()(vuk::PipelineBaseCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.shaders, x.defines);
			return h;
		}
	};
}; // namespace std
