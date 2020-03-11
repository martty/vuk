#include "Pipeline.hpp"
#include "Program.hpp"
#include <GSL/gsl_util>

namespace vuk {
	// defaults
	PipelineCreateInfo::PipelineCreateInfo() {
		// One viewport
		viewportState.viewportCount = 1;
		// One scissor rectangle
		viewportState.scissorCount = 1;

		// The dynamic state properties themselves are stored in the command buffer
		dynamicStateEnables.push_back(vk::DynamicState::eViewport);
		dynamicStateEnables.push_back(vk::DynamicState::eScissor);
		dynamicStateEnables.push_back(vk::DynamicState::eDepthBias);
	
		multisampleState.pSampleMask = nullptr;
		multisampleState.rasterizationSamples = vk::SampleCountFlagBits::e1;

		inputState.vertexBindingDescriptionCount = 0;
		inputState.vertexAttributeDescriptionCount = 0;

		inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;
		rasterizationState.lineWidth = 1.f;

		depthStencilState.depthWriteEnable = true;
		depthStencilState.depthCompareOp = vk::CompareOp::eLessOrEqual;
		depthStencilState.depthTestEnable = true;

		blendAttachmentState.resize(1);
		auto& pcba = blendAttachmentState[0];
		pcba.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
	}

	std::vector<vuk::DescriptorSetLayoutCreateInfo> PipelineCreateInfo::build_descriptor_layouts(Program& program) {
		std::vector<vuk::DescriptorSetLayoutCreateInfo> dslcis;

		for (auto& [index, set] : program.sets) {
			vuk::DescriptorSetLayoutCreateInfo dslci;
			dslci.index = index;
			auto& bindings = dslci.bindings;

			for (auto& ub : set.uniform_buffers) {
				vk::DescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = ub.binding;
				layoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
				layoutBinding.descriptorCount = 1;
				layoutBinding.stageFlags = ub.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				bindings.push_back(layoutBinding);
			}

			for (auto& sb : set.storage_buffers) {
				vk::DescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = sb.binding;
				layoutBinding.descriptorType = vk::DescriptorType::eStorageBuffer;
				layoutBinding.descriptorCount = 1;
				layoutBinding.stageFlags = sb.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				bindings.push_back(layoutBinding);
			}

			for (auto& tb : set.texel_buffers) {
				vk::DescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = tb.binding;
				layoutBinding.descriptorType = vk::DescriptorType::eUniformTexelBuffer;
				layoutBinding.descriptorCount = 1;
				layoutBinding.stageFlags = tb.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				bindings.push_back(layoutBinding);
			}

			for (auto& si : set.samplers) {
				vk::DescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = si.binding;
				layoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
				layoutBinding.descriptorCount = 1;
				layoutBinding.stageFlags = si.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				bindings.push_back(layoutBinding);
			}

			for (auto& si : set.subpass_inputs) {
				vk::DescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = si.binding;
				layoutBinding.descriptorType = vk::DescriptorType::eInputAttachment;
				layoutBinding.descriptorCount = 1;
				layoutBinding.stageFlags = si.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				bindings.push_back(layoutBinding);
			}

			dslcis.push_back(dslci);
		}

		return dslcis;
	}

	vk::GraphicsPipelineCreateInfo vuk::PipelineCreateInfo::to_vk() const {
		vk::GraphicsPipelineCreateInfo gpci;
		gpci.pVertexInputState = &inputState;
		gpci.pInputAssemblyState = &inputAssemblyState;
		gpci.pRasterizationState = &rasterizationState;
		gpci.pColorBlendState = &colorBlendState;
		gpci.pMultisampleState = &multisampleState;
		gpci.pViewportState = &viewportState;
		gpci.pDepthStencilState = &depthStencilState;
		gpci.pDynamicState = &dynamicState;
		gpci.renderPass = render_pass;
		gpci.subpass = subpass;
		return gpci;
	}
}
