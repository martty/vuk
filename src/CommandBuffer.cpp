#include "CommandBuffer.hpp"
#include "Context.hpp"
#include "RenderGraph.hpp"

namespace vuk {
	const CommandBuffer::RenderPassInfo& CommandBuffer::get_ongoing_renderpass() const {
		return ongoing_renderpass.value();
	}

	CommandBuffer& CommandBuffer::set_viewport(unsigned index, vk::Viewport vp) {
		command_buffer.setViewport(index, vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_viewport(unsigned index, Area area) {
		vk::Viewport vp;
		vp.x = (float)area.offset.x;
		vp.y = (float)area.offset.y;
		vp.width = (float)area.extent.width;
		vp.height = (float)area.extent.height;
		vp.minDepth = 0.f;
		vp.maxDepth = 1.f;
		command_buffer.setViewport(index, vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_viewport(unsigned index, Area::Framebuffer area) {
		assert(ongoing_renderpass);
		auto fb_dimensions = ongoing_renderpass->extent;
		vk::Viewport vp;
		vp.x = area.x * fb_dimensions.width;
		vp.height = -area.height * fb_dimensions.height;
		vp.y = area.y * fb_dimensions.height - vp.height;
		vp.width = area.width * fb_dimensions.width;
		vp.minDepth = 0.f;
		vp.maxDepth = 1.f;
		command_buffer.setViewport(index, vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_scissor(unsigned index, vk::Rect2D vp) {
		command_buffer.setScissor(index, vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_scissor(unsigned index, Area area) {
		command_buffer.setScissor(index, vk::Rect2D{ area.offset, area.extent });
		return *this;
	}

	CommandBuffer& CommandBuffer::set_scissor(unsigned index, Area::Framebuffer area) {
		assert(ongoing_renderpass);
		auto fb_dimensions = ongoing_renderpass->extent;
		vk::Rect2D vp;
		vp.offset.x = static_cast<int32_t>(area.x * fb_dimensions.width);
		vp.offset.y = static_cast<int32_t>(area.y * fb_dimensions.height);
		vp.extent.width = static_cast<int32_t>(area.width * fb_dimensions.width);
		vp.extent.height = static_cast<int32_t>(area.height * fb_dimensions.height);
		command_buffer.setScissor(index, vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_pipeline(vuk::PipelineCreateInfo pi) {
		next_pipeline = std::move(pi);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_pipeline(Name p) {
		return bind_pipeline(ptc.ctx.get_named_pipeline(p.data()));
	}

	CommandBuffer& CommandBuffer::bind_vertex_buffer(unsigned binding, const Buffer& buf, unsigned first_attribute, Packed format) {
		attribute_descriptions.resize(std::distance(attribute_descriptions.begin(), std::remove_if(attribute_descriptions.begin(), attribute_descriptions.end(), [&](auto& b) {return b.binding == binding; })));
		binding_descriptions.resize(std::distance(binding_descriptions.begin(), std::remove_if(binding_descriptions.begin(), binding_descriptions.end(), [&](auto& b) {return b.binding == binding; })));

		uint32_t location = first_attribute;
		uint32_t offset = 0;
		for (auto& f : format.list) {
			if (f.ignore) {
				offset += f.size;
			} else {
				vk::VertexInputAttributeDescription viad;
				viad.binding = binding;
				viad.format = f.format;
				viad.location = location;
				viad.offset = offset;
				attribute_descriptions.push_back(viad);
				offset += f.size;
				location++;
			}
		}
	
		vk::VertexInputBindingDescription vibd;
		vibd.binding = binding;
		vibd.inputRate = vk::VertexInputRate::eVertex;
		vibd.stride = offset;
		binding_descriptions.push_back(vibd);

		if(buf.buffer)
			command_buffer.bindVertexBuffers(binding, buf.buffer, buf.offset);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_index_buffer(const Buffer& buf, vk::IndexType type) {
		command_buffer.bindIndexBuffer(buf.buffer, buf.offset, type);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_sampled_image(unsigned set, unsigned binding, vuk::ImageView iv, vk::SamplerCreateInfo sci) {
		sets_used[set] = true;
		set_bindings[set].bindings[binding].type = vk::DescriptorType::eCombinedImageSampler;
		set_bindings[set].bindings[binding].image = { };
		set_bindings[set].bindings[binding].image.image_view = iv;
		set_bindings[set].bindings[binding].image.image_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
		set_bindings[set].bindings[binding].image.sampler = ptc.ctx.wrap(ptc.sampler_cache.acquire(sci));
		set_bindings[set].used.set(binding);

		return *this;
	}

	CommandBuffer& CommandBuffer::bind_sampled_image(unsigned set, unsigned binding, const vuk::Texture& texture, vk::SamplerCreateInfo sampler_create_info) {
		return bind_sampled_image(set, binding, *texture.view, sampler_create_info);
	}

	CommandBuffer& CommandBuffer::bind_sampled_image(unsigned set, unsigned binding, Name name, vk::SamplerCreateInfo sampler_create_info) {
		return bind_sampled_image(set, binding, rg.bound_attachments[name].iv, sampler_create_info);
	}

	CommandBuffer& CommandBuffer::bind_sampled_image(unsigned set, unsigned binding, Name name, vk::ImageViewCreateInfo ivci, vk::SamplerCreateInfo sampler_create_info) {
		ivci.image = rg.bound_attachments[name].image;
		ivci.format = rg.bound_attachments[name].description.format;
		ivci.viewType = vk::ImageViewType::e2D;
		vk::ImageSubresourceRange isr;
		vk::ImageAspectFlagBits aspect;
		if (ivci.format == vk::Format::eD32Sfloat) {
			aspect = vk::ImageAspectFlagBits::eDepth;
		} else {
			aspect = vk::ImageAspectFlagBits::eColor;
		}
		isr.aspectMask = aspect;
		isr.baseArrayLayer = 0;
		isr.layerCount = 1;
		isr.baseMipLevel = 0;
		isr.levelCount = 1;
		ivci.subresourceRange = isr;
	
		vuk::Unique<vuk::ImageView> iv = vuk::Unique<vuk::ImageView>(ptc.ctx, ptc.ctx.wrap(ptc.ctx.device.createImageView(ivci)));

		return bind_sampled_image(set, binding, *iv, sampler_create_info);
	}

	CommandBuffer& CommandBuffer::push_constants(vk::ShaderStageFlags stages, size_t offset, void* data, size_t size) {
		pcrs.push_back(vk::PushConstantRange(stages, (uint32_t)offset, (uint32_t)size));
		void* dst = push_constant_buffer.data() + offset;
		::memcpy(dst, data, size);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_uniform_buffer(unsigned set, unsigned binding, Buffer buffer) {
		sets_used[set] = true;
		set_bindings[set].bindings[binding].type = vk::DescriptorType::eUniformBuffer;
		set_bindings[set].bindings[binding].buffer = vk::DescriptorBufferInfo{ buffer.buffer, buffer.offset, buffer.size };
		set_bindings[set].used.set(binding);
		return *this;
	}

	void* CommandBuffer::_map_scratch_uniform_binding(unsigned set, unsigned binding, size_t size) {
		auto buf = ptc._allocate_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vk::BufferUsageFlagBits::eUniformBuffer, size, true);
		bind_uniform_buffer(set, binding, buf);
		return buf.mapped_ptr;
	}

	CommandBuffer& CommandBuffer::draw(size_t vertex_count, size_t instance_count, size_t first_vertex, size_t first_instance) {
		_bind_graphics_pipeline_state();
		command_buffer.draw((uint32_t)vertex_count, (uint32_t)instance_count, (uint32_t)first_vertex, (uint32_t)first_instance);
		return *this;
	}

	CommandBuffer& CommandBuffer::draw_indexed(size_t index_count, size_t instance_count, size_t first_index, int32_t vertex_offset, size_t first_instance) {
		_bind_graphics_pipeline_state();
		command_buffer.drawIndexed((uint32_t)index_count, (uint32_t)instance_count, (uint32_t)first_index, vertex_offset, (uint32_t)first_instance);
		return *this;
    }

    SecondaryCommandBuffer CommandBuffer::begin_secondary() {
        auto nptc = new vuk::PerThreadContext(ptc.ifc.begin());
        auto scbuf = nptc->commandbuffer_pool.acquire(vk::CommandBufferLevel::eSecondary, 1)[0];
		vk::CommandBufferBeginInfo cbi;
		cbi.flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue;
        vk::CommandBufferInheritanceInfo cbii;
        cbii.renderPass = ongoing_renderpass->renderpass;
        cbii.subpass = ongoing_renderpass->subpass;
        cbii.framebuffer = vk::Framebuffer{};//TODO
		cbi.pInheritanceInfo = &cbii;
		scbuf.begin(cbi);
        SecondaryCommandBuffer scb(rg, *nptc, scbuf);
        scb.ongoing_renderpass = ongoing_renderpass;
        return scb;
    }

    void CommandBuffer::execute(gsl::span<vk::CommandBuffer> scbufs) {
        if(scbufs.size() > 0)
			command_buffer.executeCommands(scbufs.size(), scbufs.data());
	}

	void CommandBuffer::_bind_graphics_pipeline_state() {
		if (next_pipeline) {
			auto& pi = next_pipeline.value();
			// set vertex input
			pi.attribute_descriptions = attribute_descriptions;
			pi.binding_descriptions = binding_descriptions;
			auto& vertex_input_state = pi.vertex_input_state;
			vertex_input_state.pVertexAttributeDescriptions = pi.attribute_descriptions.data();
			vertex_input_state.vertexAttributeDescriptionCount = (uint32_t)pi.attribute_descriptions.size();
			vertex_input_state.pVertexBindingDescriptions = pi.binding_descriptions.data();
			vertex_input_state.vertexBindingDescriptionCount = (uint32_t)pi.binding_descriptions.size();

			pi.render_pass = ongoing_renderpass->renderpass;
			pi.subpass = ongoing_renderpass->subpass;

			pi.dynamic_state.pDynamicStates = pi.dynamic_states.data();
			pi.dynamic_state.dynamicStateCount = gsl::narrow_cast<unsigned>(pi.dynamic_states.size());

			pi.multisample_state.rasterizationSamples = ongoing_renderpass->samples;

			// last blend attachment is replicated to cover all attachments
			if (pi.color_blend_attachments.size() < (size_t)ongoing_renderpass->color_attachments.size()) {
				pi.color_blend_attachments.resize(ongoing_renderpass->color_attachments.size(), pi.color_blend_attachments.back());
			}
			pi.color_blend_state.pAttachments = pi.color_blend_attachments.data();
			pi.color_blend_state.attachmentCount = (uint32_t)pi.color_blend_attachments.size();

			current_pipeline = ptc.pipeline_cache.acquire(pi);
			command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, current_pipeline->pipeline);
			next_pipeline = {};
		}

		for (auto& pcr : pcrs) {
			void* data = push_constant_buffer.data() + pcr.offset;
			command_buffer.pushConstants(current_pipeline->pipeline_layout, pcr.stageFlags, pcr.offset, pcr.size, data);
		}
		pcrs.clear();

		for (size_t i = 0; i < VUK_MAX_SETS; i++) {
			if (!sets_used[i])
				continue;
			set_bindings[i].layout_info = current_pipeline->layout_info[i];
			auto ds = ptc.descriptor_sets.acquire(set_bindings[i]);
			command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, current_pipeline->pipeline_layout, 0, 1, &ds.descriptor_set, 0, nullptr);
			sets_used[i] = false;
			set_bindings[i] = {};
		}
    }

    vk::CommandBuffer SecondaryCommandBuffer::get_buffer() {
        return command_buffer;
    }

    SecondaryCommandBuffer::~SecondaryCommandBuffer() {
        command_buffer.end();
        delete &ptc;
    }

} // namespace vuk