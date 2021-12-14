#pragma once

#include <vector>
#include <vuk/Config.hpp>
#include <vuk/Hash.hpp>
#include <../src/CreateInfo.hpp>
#include <vuk/Descriptor.hpp>
#include <vuk/ShaderSource.hpp>
#include <vuk/Program.hpp>
#include <vuk/FixedVector.hpp>
#include <vuk/Image.hpp>
#include <vuk/Bitset.hpp>
#include <vuk/PipelineTypes.hpp>
#include <bit>

namespace vuk {
	static constexpr uint32_t graphics_stage_count = 5;

	struct PipelineLayoutCreateInfo {
		VkPipelineLayoutCreateInfo plci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
		vuk::fixed_vector<VkPushConstantRange, VUK_MAX_PUSHCONSTANT_RANGES> pcrs;
		vuk::fixed_vector<vuk::DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> dslcis;

		bool operator==(const PipelineLayoutCreateInfo& o) const noexcept {
			return std::tie(plci.flags, pcrs, dslcis) == std::tie(o.plci.flags, o.pcrs, o.dslcis);
		}
	};

	template<> struct create_info<VkPipelineLayout> {
		using type = vuk::PipelineLayoutCreateInfo;
	};

	struct PipelineBaseCreateInfoBase {
		// 4 valid flags
		Bitset<4 * VUK_MAX_SETS * VUK_MAX_BINDINGS> binding_flags = {};
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

		void add_glsl(std::string_view source, std::string filename) {
			shaders.emplace_back(ShaderSource::glsl(source));
			shader_paths.emplace_back(std::move(filename));
		}

		void add_spirv(std::vector<uint32_t> source, std::string filename) {
			shaders.emplace_back(ShaderSource::spirv(std::move(source)));
			shader_paths.emplace_back(std::move(filename));
		}

		vuk::fixed_vector<ShaderSource, graphics_stage_count> shaders;
		vuk::fixed_vector<std::string, graphics_stage_count> shader_paths;

		friend struct std::hash<PipelineBaseCreateInfo>;
	public:

		static vuk::fixed_vector<vuk::DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> build_descriptor_layouts(const Program&, const PipelineBaseCreateInfoBase&);
		bool operator==(const PipelineBaseCreateInfo& o) const noexcept {
			return shaders == o.shaders && binding_flags == o.binding_flags && variable_count_max == o.variable_count_max;
		}
	};

	struct PipelineBaseInfo {
		Name pipeline_name;
		vuk::Program reflection_info;
		vuk::fixed_vector<VkPipelineShaderStageCreateInfo, vuk::graphics_stage_count> psscis;
		VkPipelineLayout pipeline_layout;
		std::array<DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> layout_info;

		// 4 valid flags
		Bitset<4 * VUK_MAX_SETS * VUK_MAX_BINDINGS> binding_flags = {};
		// if the set has a variable count binding, the maximum number of bindings possible
		std::array<uint32_t, VUK_MAX_SETS> variable_count_max = {};
	};

	template<> struct create_info<PipelineBaseInfo> {
		using type = vuk::PipelineBaseCreateInfo;
	};

	struct ComputePipelineBaseCreateInfo : PipelineBaseCreateInfoBase {
		friend class CommandBuffer;
		friend class Context;
	public:
		void add_shader(ShaderSource source, std::string filename) {
			shader = std::move(source);
			shader_path = std::move(filename);
		}

		void add_glsl(std::string source, std::string filename) {
			shader = ShaderSource::glsl(std::move(source));
			shader_path = std::move(filename);
		}

		void add_spirv(std::vector<uint32_t> source, std::string filename) {
			shader = ShaderSource::spirv(std::move(source));
			shader_path = std::move(filename);
		}

		friend struct std::hash<ComputePipelineBaseCreateInfo>;
		friend class PerThreadContext;
	private:
		ShaderSource shader;
		std::string shader_path;

	public:
		bool operator==(const ComputePipelineBaseCreateInfo& o) const noexcept {
			return shader == o.shader && binding_flags == o.binding_flags && variable_count_max == o.variable_count_max;
		}
	};

	struct ComputePipelineBaseInfo {
		Name pipeline_name;
		vuk::Program reflection_info;
		VkPipelineShaderStageCreateInfo pssci;
		VkPipelineLayout pipeline_layout;
		std::array<DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> layout_info;

		// 4 valid flags
		Bitset<4 * VUK_MAX_SETS * VUK_MAX_BINDINGS> binding_flags = {};
		// if the set has a variable count binding, the maximum number of bindings possible
		std::array<uint32_t, VUK_MAX_SETS> variable_count_max = {};
	};

	template<> struct create_info<ComputePipelineBaseInfo> {
		using type = vuk::ComputePipelineBaseCreateInfo;
	};
}


namespace std {
	template <class BitType>
	struct hash<vuk::Flags<BitType>> {
		size_t operator()(vuk::Flags<BitType> const& x) const noexcept {
			return std::hash<typename vuk::Flags<BitType>::MaskType>()((typename vuk::Flags<BitType>::MaskType)x);
		}
	};
};

namespace std {
	template <class T>
	struct hash<std::vector<T>> {
		size_t operator()(std::vector<T> const& x) const noexcept {
			size_t h = 0;
			for (auto& e : x) {
				hash_combine(h, e);
			}
			return h;
		}
	};

	template <class T, size_t N>
	struct hash<vuk::fixed_vector<T, N>> {
		size_t operator()(vuk::fixed_vector<T, N> const& x) const noexcept {
			size_t h = 0;
			for (auto& e : x) {
				hash_combine(h, e);
			}
			return h;
		}
	};

	template <>
	struct hash<vuk::ShaderSource> {
		size_t operator()(vuk::ShaderSource const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.is_spirv, x.data);
			return h;
		}
	};

	template <>
	struct hash<vuk::PipelineBaseCreateInfo> {
		size_t operator()(vuk::PipelineBaseCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.shaders);
			return h;
		}
	};

	template <>
	struct hash<vuk::ComputePipelineBaseCreateInfo> {
		size_t operator()(vuk::ComputePipelineBaseCreateInfo const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.shader);
			return h;
		}
	};
};
