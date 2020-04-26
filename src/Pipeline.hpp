#pragma once

#include <vector>
#include "vulkan/vulkan.hpp"
#include "Hash.hpp"
#include "CreateInfo.hpp"
#include "Descriptor.hpp"
#include "Program.hpp"
#include "FixedVector.hpp"

#define VUK_MAX_SETS 8
#define VUK_MAX_ATTRIBUTES 8
#define VUK_MAX_COLOR_ATTACHMENTS 8
#define VUK_MAX_PUSHCONSTANT_RANGES 8

namespace vuk {
	enum class BlendPreset {
		eOff, eAlphaBlend, ePremultipliedAlphaBlend
	};

	struct PipelineLayoutCreateInfo {
		vk::PipelineLayoutCreateInfo plci;
		vuk::fixed_vector<vk::PushConstantRange, VUK_MAX_PUSHCONSTANT_RANGES> pcrs;
		vuk::fixed_vector<vuk::DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> dslcis;

		bool operator==(const PipelineLayoutCreateInfo& o) const {
			return std::tie(plci.flags, pcrs, dslcis) == std::tie(o.plci.flags, o.pcrs, o.dslcis);
		}
	};

	template<> struct create_info<vk::PipelineLayout> {
		using type = vuk::PipelineLayoutCreateInfo;
	};

	struct Program;

	struct PipelineCreateInfo {
		friend class CommandBuffer;
		friend class Context;
	public:
        struct FString {
            char data[128];
		};

		/* filled out by the user */
        void add_shader(std::string_view filename) {
            auto& n = shaders.emplace_back();
            strncpy(n.data, filename.data(), filename.size());
            n.data[filename.size()] = '\0';
            hash_combine(shader_hash, filename);
		}

		vk::PipelineInputAssemblyStateCreateInfo input_assembly_state;
		vuk::fixed_vector<vk::VertexInputBindingDescription, VUK_MAX_ATTRIBUTES> binding_descriptions;
		vuk::fixed_vector<vk::VertexInputAttributeDescription, VUK_MAX_ATTRIBUTES> attribute_descriptions;
		vk::PipelineRasterizationStateCreateInfo rasterization_state;
		vk::PipelineColorBlendStateCreateInfo color_blend_state;
		vuk::fixed_vector<vk::PipelineColorBlendAttachmentState, VUK_MAX_COLOR_ATTACHMENTS> color_blend_attachments;
		vk::PipelineDepthStencilStateCreateInfo depth_stencil_state;

		void set_blend(size_t attachment_index, BlendPreset);
		void set_blend(BlendPreset);
	//private:
		/* filled out by vuk */
		vk::PipelineVertexInputStateCreateInfo vertex_input_state;
		vk::PipelineViewportStateCreateInfo viewport_state;
		vk::PipelineDynamicStateCreateInfo dynamic_state;
		vuk::fixed_vector<vk::DynamicState, 8> dynamic_states;
		vk::PipelineMultisampleStateCreateInfo multisample_state;
		vk::RenderPass render_pass;
		uint32_t subpass;
        uint64_t shader_hash = 0;
		vuk::fixed_vector<FString, 5> shaders;

		friend struct std::hash<PipelineCreateInfo>;
        friend class PerThreadContext;
	public:
		PipelineCreateInfo();

		PipelineCreateInfo(const PipelineCreateInfo& o) {
            memcpy(this, &o, offsetof(PipelineCreateInfo, shaders));
            shaders = o.shaders;
		}

		PipelineCreateInfo& operator=(const PipelineCreateInfo& o) {
			memcpy(this, &o, offsetof(PipelineCreateInfo, shaders));
            shaders = o.shaders;
            return *this;
		}

		PipelineCreateInfo(PipelineCreateInfo&& o) {
            memcpy(this, &o, offsetof(PipelineCreateInfo, shaders));
            shaders = std::move(o.shaders);
		}

		PipelineCreateInfo& operator=(PipelineCreateInfo&& o) {
			memcpy(this, &o, offsetof(PipelineCreateInfo, shaders));
            shaders = std::move(o.shaders);
            return *this;
		}
		
		vk::GraphicsPipelineCreateInfo to_vk() const;
		static vuk::fixed_vector<vuk::DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> build_descriptor_layouts(Program&);
		bool operator==(const PipelineCreateInfo& o) const {
            return memcmp(this, &o, offsetof(PipelineCreateInfo, shaders)) == 0;
		}
	};

	struct PipelineInfo {
		vk::Pipeline pipeline;
		vk::PipelineLayout pipeline_layout;
		std::array<DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> layout_info;
	};

	template<> struct create_info<PipelineInfo> {
		using type = vuk::PipelineCreateInfo;
	};
}

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
	struct hash<vuk::PipelineCreateInfo> {
		size_t operator()(vuk::PipelineCreateInfo const & x) const noexcept {
            return x.shader_hash;
		}
	};

	template <>
	struct hash<vk::PushConstantRange> {
		size_t operator()(vk::PushConstantRange const & x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.offset, x.size, (VkShaderStageFlags)x.stageFlags);
			return h;
		}
	};


	template <>
	struct hash<vuk::PipelineLayoutCreateInfo> {
		size_t operator()(vuk::PipelineLayoutCreateInfo const & x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.pcrs, x.dslcis);
			return h;
		}
	};

};
