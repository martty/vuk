#pragma once

#include <vector>
#include "vulkan/vulkan.hpp"
#include "Hash.hpp"
#include "CreateInfo.hpp"
#include "Descriptor.hpp"

#define VUK_MAX_SETS 8

namespace vuk {
	enum class BlendPreset {
		eOff, eAlphaBlend, ePremultipliedAlphaBlend
	};

	struct PipelineLayoutCreateInfo {
		vk::PipelineLayoutCreateInfo plci;
		std::vector<vk::PushConstantRange> pcrs;
		std::vector<vuk::DescriptorSetLayoutCreateInfo> dslcis;

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
		/* filled out by the user */
		std::vector<std::string> shaders;

		vk::PipelineInputAssemblyStateCreateInfo input_assembly_state;
		std::vector<vk::VertexInputBindingDescription> binding_descriptions;
		std::vector<vk::VertexInputAttributeDescription> attribute_descriptions;
		vk::PipelineRasterizationStateCreateInfo rasterization_state;
		vk::PipelineColorBlendStateCreateInfo color_blend_state;
		std::vector<vk::PipelineColorBlendAttachmentState> color_blend_attachments;
		vk::PipelineDepthStencilStateCreateInfo depth_stencil_state;

		void set_blend(size_t attachment_index, BlendPreset);
		void set_blend(BlendPreset);
	private:
		/* filled out by vuk */
		vk::PipelineVertexInputStateCreateInfo vertex_input_state;
		vk::PipelineViewportStateCreateInfo viewport_state;
		vk::PipelineDynamicStateCreateInfo dynamic_state;
		std::vector<vk::DynamicState> dynamic_states;
		vk::PipelineMultisampleStateCreateInfo multisample_state;
		vk::RenderPass render_pass;
		uint32_t subpass;

	public:
		PipelineCreateInfo();
		
		vk::GraphicsPipelineCreateInfo to_vk() const;
		static std::vector<vuk::DescriptorSetLayoutCreateInfo> build_descriptor_layouts(Program&);
		bool operator==(const PipelineCreateInfo& o) const {
			// TODO: colorblendstate, render_pass
			return std::tie(shaders, binding_descriptions, attribute_descriptions, input_assembly_state, rasterization_state, color_blend_attachments, viewport_state, dynamic_states, depth_stencil_state, multisample_state, render_pass, subpass) ==
				std::tie(o.shaders, o.binding_descriptions, o.attribute_descriptions, o.input_assembly_state, o.rasterization_state, o.color_blend_attachments, o.viewport_state, o.dynamic_states, o.depth_stencil_state, o.multisample_state, o.render_pass, o.subpass);
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

	template <>
	struct hash<vuk::PipelineCreateInfo> {
		size_t operator()(vuk::PipelineCreateInfo const & x) const noexcept {
			size_t h = 0;
			// TODO: better hash
			hash_combine(h, x.shaders);
			return h;
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
