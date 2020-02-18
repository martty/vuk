#include "Pipeline.hpp"
#include "Program.hpp"
#include <GSL/gsl_util>

Pipeline::Pipeline() {
	// One viewport
	viewportState.viewportCount = 1;
	// One scissor rectangle
	viewportState.scissorCount = 1;

	// The dynamic state properties themselves are stored in the command buffer
	dynamicStateEnables.push_back(vk::DynamicState::eViewport);
	dynamicStateEnables.push_back(vk::DynamicState::eScissor);
	dynamicStateEnables.push_back(vk::DynamicState::eDepthBias);
	dynamicState.pDynamicStates = dynamicStateEnables.data();
	dynamicState.dynamicStateCount = gsl::narrow_cast<unsigned>(dynamicStateEnables.size());

	multisampleState.pSampleMask = NULL;
	// No multi sampling used in this example
	multisampleState.rasterizationSamples = vk::SampleCountFlagBits::e1;
}

Pipeline::Pipeline(Program* p) : Pipeline() {
	program = std::move(p);
	
	fill_layout_bindings();
}

void Pipeline::fill_layout_bindings() {
	descriptorSetLayoutBindings.clear();

	inputState.vertexBindingDescriptionCount = 0;
	inputState.vertexAttributeDescriptionCount = 0;

	// descriptorsetlayout binding
	for (auto& ub : program->uniform_buffers) {
		vk::DescriptorSetLayoutBinding layoutBinding;
		layoutBinding.binding = ub.binding;
		layoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
		layoutBinding.descriptorCount = 1;
		layoutBinding.stageFlags = ub.stage;
		layoutBinding.pImmutableSamplers = NULL;
		descriptorSetLayoutBindings.push_back(layoutBinding);
	}

	for (auto& sb : program->storage_buffers) {
		vk::DescriptorSetLayoutBinding layoutBinding;
		layoutBinding.binding = sb.binding;
		layoutBinding.descriptorType = vk::DescriptorType::eStorageBuffer;
		layoutBinding.descriptorCount = 1;
		layoutBinding.stageFlags = sb.stage;
		layoutBinding.pImmutableSamplers = NULL;
		descriptorSetLayoutBindings.push_back(layoutBinding);
	}

	for (auto& tb : program->texel_buffers) {
		vk::DescriptorSetLayoutBinding layoutBinding;
		layoutBinding.binding = tb.binding;
		layoutBinding.descriptorType = vk::DescriptorType::eUniformTexelBuffer;
		layoutBinding.descriptorCount = 1;
		layoutBinding.stageFlags = tb.stage;
		layoutBinding.pImmutableSamplers = NULL;
		descriptorSetLayoutBindings.push_back(layoutBinding);
	}

	for (auto& si : program->samplers) {
		vk::DescriptorSetLayoutBinding layoutBinding;
		layoutBinding.binding = si.binding;
		layoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
		layoutBinding.descriptorCount = si.array_size;
		layoutBinding.stageFlags = si.stage;
		layoutBinding.pImmutableSamplers = NULL;
		descriptorSetLayoutBindings.push_back(layoutBinding);
	}

	for (auto& si : program->subpass_inputs) {
		vk::DescriptorSetLayoutBinding layoutBinding;
		layoutBinding.binding = si.binding;
		layoutBinding.descriptorType = vk::DescriptorType::eInputAttachment;
		layoutBinding.descriptorCount = 1;
		layoutBinding.stageFlags = si.stage;
		layoutBinding.pImmutableSamplers = NULL;
		descriptorSetLayoutBindings.push_back(layoutBinding);
	}

	descriptorLayout.bindingCount = gsl::narrow_cast<unsigned>(descriptorSetLayoutBindings.size());
	descriptorLayout.pBindings = descriptorSetLayoutBindings.data();

	pipelineLayoutCreateInfo.setLayoutCount = 1;
}
