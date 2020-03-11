#pragma once

#include <vector>
#include "vulkan/vulkan.hpp"
#include "Hash.hpp"

namespace vuk {
	struct DescriptorSetLayoutCreateInfo {
		vk::DescriptorSetLayoutCreateInfo dslci;
		std::vector<vk::DescriptorSetLayoutBinding> bindings;
		size_t index;

		bool operator==(const DescriptorSetLayoutCreateInfo& o) const {
			return std::tie(dslci.flags, bindings) == std::tie(o.dslci.flags, o.bindings);
		}
	};

	struct PipelineLayoutCreateInfo {
		vk::PipelineLayoutCreateInfo plci;
		std::vector<vk::PushConstantRange> pcrs;
		std::vector<vuk::DescriptorSetLayoutCreateInfo> dslcis;

		bool operator==(const PipelineLayoutCreateInfo& o) const {
			return std::tie(plci.flags, pcrs, dslcis) == std::tie(o.plci.flags, o.pcrs, o.dslcis);
		}
	};

	struct Program;

	struct PipelineCreateInfo {
		std::vector<std::string> shaders;

		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;

		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState;

		vk::PipelineRasterizationStateCreateInfo rasterizationState;
		vk::PipelineColorBlendStateCreateInfo colorBlendState;
		std::vector<vk::PipelineColorBlendAttachmentState> blendAttachmentState;

		vk::PipelineViewportStateCreateInfo viewportState;

		vk::PipelineDynamicStateCreateInfo dynamicState;
		std::vector<vk::DynamicState> dynamicStateEnables;

		vk::PipelineDepthStencilStateCreateInfo depthStencilState;
		vk::PipelineMultisampleStateCreateInfo multisampleState;

		static std::vector<vuk::DescriptorSetLayoutCreateInfo> build_descriptor_layouts(Program&);

		vk::RenderPass render_pass;
		uint32_t subpass;

		PipelineCreateInfo();
		vk::GraphicsPipelineCreateInfo to_vk() const;

		bool operator==(const PipelineCreateInfo& o) const {
			// TODO: colorblendstate, render_pass
			return std::tie(shaders, bindingDescriptions, attributeDescriptions, inputAssemblyState, rasterizationState, blendAttachmentState, viewportState, dynamicStateEnables, depthStencilState, multisampleState, render_pass, subpass) ==
				std::tie(o.shaders, o.bindingDescriptions, o.attributeDescriptions, o.inputAssemblyState, o.rasterizationState, o.blendAttachmentState, o.viewportState, o.dynamicStateEnables, o.depthStencilState, o.multisampleState, o.render_pass, o.subpass);
		}

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
			hash_combine(h, x.shaders);
			return h;
		}
	};

	template <>
	struct hash<vk::DescriptorSetLayoutBinding> {
		size_t operator()(vk::DescriptorSetLayoutBinding const & x) const noexcept {
			size_t h = 0;
			// TODO: immutable samplers
			hash_combine(h, x.binding, x.descriptorCount, x.descriptorType, (VkShaderStageFlags)x.stageFlags);
			return h;
		}
	};


	template <>
	struct hash<vuk::DescriptorSetLayoutCreateInfo> {
		size_t operator()(vuk::DescriptorSetLayoutCreateInfo const & x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.bindings);
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
