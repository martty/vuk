#pragma once

#include <vector>
#include "vulkan/vulkan.hpp"
class Program;

// PSO
struct Pipeline {
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


	vk::DescriptorSetLayoutCreateInfo descriptorLayout;
	vk::DescriptorSetLayout descriptorSetLayout;
	std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings;
	vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
	vk::PipelineLayout pipelineLayout;

	vk::Pipeline pipeline;

	Program* program;

	Pipeline();

	Pipeline(Program* p);

	void fill_layout_bindings();

	bool operator==(const Pipeline& b) const {
		return pipeline == b.pipeline;
	}
};


namespace std {
	template <>
	struct hash<Pipeline> {
		size_t operator()(Pipeline const & x) const noexcept {
			std::hash<std::uint64_t> hash_fn;
			return hash_fn(reinterpret_cast<uint64_t>(static_cast<VkPipeline>(x.pipeline)));
		}
	};
};