#include "Pipeline.hpp"
#include "Program.hpp"

namespace vuk {
	void PipelineBaseCreateInfo::set_blend(size_t attachment_index, BlendPreset preset) {
		if(color_blend_attachments.size() <= attachment_index)
			color_blend_attachments.resize(attachment_index + 1);
		auto& pcba = color_blend_attachments[attachment_index];

		switch (preset) {
		case BlendPreset::eAlphaBlend:
			pcba.blendEnable = true;
			pcba.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
			pcba.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
			pcba.colorBlendOp = vk::BlendOp::eAdd;
			pcba.srcAlphaBlendFactor = vk::BlendFactor::eOne;
			pcba.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
			pcba.alphaBlendOp = vk::BlendOp::eAdd;
			break;
		case BlendPreset::eOff:
			pcba.blendEnable = false;
			break;
		case BlendPreset::ePremultipliedAlphaBlend:
			assert(0 && "NYI");
		}
	}
	void PipelineBaseCreateInfo::set_blend(BlendPreset preset) {
		color_blend_attachments.resize(1);
		set_blend(0, preset);
	}
	// defaults
	PipelineBaseCreateInfo::PipelineBaseCreateInfo() {
		rasterization_state.lineWidth = 1.f;

		depth_stencil_state.depthWriteEnable = true;
		depth_stencil_state.depthCompareOp = vk::CompareOp::eLessOrEqual;
		depth_stencil_state.depthTestEnable = true;

		color_blend_attachments.resize(1);
		auto& pcba = color_blend_attachments[0];
        pcba = vk::PipelineColorBlendAttachmentState{};
		pcba.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
	}

	PipelineInstanceCreateInfo::PipelineInstanceCreateInfo() {	
		multisample_state.pSampleMask = nullptr;
		multisample_state.rasterizationSamples = vk::SampleCountFlagBits::e1;

		vertex_input_state.vertexBindingDescriptionCount = 0;
		vertex_input_state.vertexAttributeDescriptionCount = 0;

		input_assembly_state.topology = vk::PrimitiveTopology::eTriangleList;
	}

	vuk::fixed_vector<vuk::DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> PipelineBaseCreateInfo::build_descriptor_layouts(const Program& program, const PipelineBaseCreateInfoBase& bci) {
		vuk::fixed_vector<vuk::DescriptorSetLayoutCreateInfo, VUK_MAX_SETS> dslcis;

		for (const auto& [index, set] : program.sets) {
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
				layoutBinding.descriptorCount = si.array_size == (unsigned)-1 ? 1 : si.array_size;
				layoutBinding.stageFlags = si.stage;
				layoutBinding.pImmutableSamplers = nullptr;
				if (si.array_size == 0) {
					assert(bci.variable_count_max[index] > 0); // forgot to mark this descriptor as variable count
					layoutBinding.descriptorCount = bci.variable_count_max[index];
				}
				bindings.push_back(layoutBinding);
			}

			for (auto& si : set.storage_images) {
				vk::DescriptorSetLayoutBinding layoutBinding;
				layoutBinding.binding = si.binding;
				layoutBinding.descriptorType = vk::DescriptorType::eStorageImage;
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
			
			// extract flags from the packed bitset
			// TODO: rewrite this without _Getword
			auto set_word_offset = index * VUK_MAX_BINDINGS * 4 / (sizeof(unsigned long long) * 8);
			for (unsigned i = 0; i <= set.highest_descriptor_binding; i++) {
				auto word = bci.binding_flags._Getword(set_word_offset + i * 4 / (sizeof(unsigned long long) * 8));
				if (word & ((0b1111) << i)) {
					vk::DescriptorBindingFlags f((word >> i) & 0b1111);
					dslci.flags.resize(i + 1);
					dslci.flags[i] = f;
				}
			}

			dslcis.push_back(dslci);
		}
		return dslcis;
	}

	vk::GraphicsPipelineCreateInfo vuk::PipelineInstanceCreateInfo::to_vk() const {
		vk::GraphicsPipelineCreateInfo gpci;
		gpci.pVertexInputState = &vertex_input_state;
		gpci.pInputAssemblyState = &input_assembly_state;
		gpci.pRasterizationState = &base->rasterization_state;
		gpci.pColorBlendState = &color_blend_state;
		gpci.pDepthStencilState = &base->depth_stencil_state;
		gpci.pMultisampleState = &multisample_state;
		gpci.pViewportState = &base->viewport_state;
		gpci.pDynamicState = &dynamic_state;
		gpci.renderPass = render_pass;
		gpci.subpass = subpass;
		return gpci;
	}
}
