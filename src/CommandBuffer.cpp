#include "CommandBuffer.hpp"
#include "Context.hpp"
#include "RenderGraph.hpp"

namespace vuk {
	CommandBuffer::CommandBuffer(vuk::PerThreadContext& ptc) : ptc(ptc){
		command_buffer = ptc.commandbuffer_pool.acquire(VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1)[0];
	}

	const CommandBuffer::RenderPassInfo& CommandBuffer::get_ongoing_renderpass() const {
		return ongoing_renderpass.value();
	}

	vuk::Buffer CommandBuffer::get_resource_buffer(Name n) const {
		assert(rg);
		return rg->get_resource_buffer(n).buffer;
	}

	vuk::Image CommandBuffer::get_resource_image(Name n) const {
		assert(rg);
		return rg->bound_attachments[n].image;
	}

	vuk::ImageView CommandBuffer::get_resource_image_view(Name n) const {
		assert(rg);
		return rg->bound_attachments[n].iv;
	}

	CommandBuffer& CommandBuffer::set_viewport(unsigned index, vuk::Viewport vp) {
		vkCmdSetViewport(command_buffer, 0, 1, (VkViewport*)&vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_viewport(unsigned index, Area area) {
		vuk::Viewport vp;
		vp.x = (float)area.offset.x;
        vp.y = (float)area.offset.y + (float)area.extent.height;
		vp.width = (float)area.extent.width;
		vp.height = -(float)area.extent.height;
		vp.minDepth = 0.f;
		vp.maxDepth = 1.f;

		vkCmdSetViewport(command_buffer, 0, 1, (VkViewport*)&vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_viewport(unsigned index, Area::Framebuffer area) {
		assert(ongoing_renderpass);
		auto fb_dimensions = ongoing_renderpass->extent;
		vuk::Viewport vp;
		vp.x = area.x * fb_dimensions.width;
		vp.height = -area.height * fb_dimensions.height;
		vp.y = area.y * fb_dimensions.height - vp.height;
		vp.width = area.width * fb_dimensions.width;
		vp.minDepth = 0.f;
		vp.maxDepth = 1.f;
		
		vkCmdSetViewport(command_buffer, 0, 1, (VkViewport*)&vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_scissor(unsigned index, vuk::Rect2D vp) {
		vkCmdSetScissor(command_buffer, 0, 1, (VkRect2D*)&vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_scissor(unsigned index, Area area) {
		vuk::Rect2D rect{ area.offset, area.extent };
		vkCmdSetScissor(command_buffer, 0, 1, (VkRect2D*)&rect);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_scissor(unsigned index, Area::Framebuffer area) {
		assert(ongoing_renderpass);
		auto fb_dimensions = ongoing_renderpass->extent;
		vuk::Rect2D vp;
		vp.offset.x = static_cast<int32_t>(area.x * fb_dimensions.width);
		vp.offset.y = static_cast<int32_t>(area.y * fb_dimensions.height);
		vp.extent.width = static_cast<int32_t>(area.width * fb_dimensions.width);
		vp.extent.height = static_cast<int32_t>(area.height * fb_dimensions.height);

		vkCmdSetScissor(command_buffer, 0, 1, (VkRect2D*)&vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_graphics_pipeline(vuk::PipelineBaseInfo* pi) {
		next_pipeline = pi;
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_graphics_pipeline(Name p) {
		return bind_graphics_pipeline(ptc.ctx.get_named_pipeline(p.data()));
	}

	CommandBuffer& CommandBuffer::bind_compute_pipeline(vuk::ComputePipelineInfo* gpci) {
		next_compute_pipeline = gpci;
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_compute_pipeline(Name p) {
		return bind_compute_pipeline(ptc.ctx.get_named_compute_pipeline(p.data()));
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
				vuk::VertexInputAttributeDescription viad;
				viad.binding = binding;
				viad.format = f.format;
				viad.location = location;
				viad.offset = offset;
				attribute_descriptions.push_back(viad);
				offset += f.size;
				location++;
			}
		}
	
		VkVertexInputBindingDescription vibd;
		vibd.binding = binding;
		vibd.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vibd.stride = offset;
		binding_descriptions.push_back(vibd);

		if (buf.buffer) {
			vkCmdBindVertexBuffers(command_buffer, binding, 1, &buf.buffer, &buf.offset);
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_vertex_buffer(unsigned binding, const Buffer& buf, std::span<vuk::VertexInputAttributeDescription> viads, uint32_t stride) {
		attribute_descriptions.resize(std::distance(attribute_descriptions.begin(), std::remove_if(attribute_descriptions.begin(), attribute_descriptions.end(), [&](auto& b) {return b.binding == binding; })));
		binding_descriptions.resize(std::distance(binding_descriptions.begin(), std::remove_if(binding_descriptions.begin(), binding_descriptions.end(), [&](auto& b) {return b.binding == binding; })));

		attribute_descriptions.insert(attribute_descriptions.end(), viads.begin(), viads.end());

		VkVertexInputBindingDescription vibd;
		vibd.binding = binding;
		vibd.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vibd.stride = stride;
		binding_descriptions.push_back(vibd);

		if (buf.buffer) {
			vkCmdBindVertexBuffers(command_buffer, binding, 1, &buf.buffer, &buf.offset);
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_index_buffer(const Buffer& buf, vuk::IndexType type) {
		vkCmdBindIndexBuffer(command_buffer, buf.buffer, buf.offset, (VkIndexType)type);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_primitive_topology(vuk::PrimitiveTopology topo) {
        topology = topo;
        return *this;
	}

	CommandBuffer& CommandBuffer::bind_sampled_image(unsigned set, unsigned binding, vuk::ImageView iv, vuk::SamplerCreateInfo sci, vuk::ImageLayout il) {
		sets_used[set] = true;
		set_bindings[set].bindings[binding].type = vuk::DescriptorType::eCombinedImageSampler;
		set_bindings[set].bindings[binding].image = vuk::DescriptorImageInfo(ptc.sampler_cache.acquire(sci), iv, il);
		set_bindings[set].used.set(binding);

		return *this;
	}

	CommandBuffer& CommandBuffer::bind_sampled_image(unsigned set, unsigned binding, const vuk::Texture& texture, vuk::SamplerCreateInfo sampler_create_info, vuk::ImageLayout il) {
		return bind_sampled_image(set, binding, *texture.view, sampler_create_info, il);
	}

	CommandBuffer& CommandBuffer::bind_sampled_image(unsigned set, unsigned binding, Name name, vuk::SamplerCreateInfo sampler_create_info) {
		assert(rg);

		auto layout = rg->is_resource_image_in_general_layout(name, current_pass) ? vuk::ImageLayout::eGeneral : vuk::ImageLayout::eShaderReadOnlyOptimal;

		return bind_sampled_image(set, binding, rg->bound_attachments[name].iv, sampler_create_info, layout);
	}

	CommandBuffer& CommandBuffer::bind_sampled_image(unsigned set, unsigned binding, Name name, vuk::ImageViewCreateInfo ivci, vuk::SamplerCreateInfo sampler_create_info) {
		assert(rg);
		ivci.image = rg->bound_attachments[name].image;
        if(ivci.format == vuk::Format{}) {
            ivci.format = vuk::Format(rg->bound_attachments[name].description.format);
        }
		ivci.viewType = vuk::ImageViewType::e2D;
		vuk::ImageSubresourceRange isr;
		vuk::ImageAspectFlagBits aspect;
		if (ivci.format == vuk::Format::eD32Sfloat) {
			aspect = vuk::ImageAspectFlagBits::eDepth;
		} else {
			aspect = vuk::ImageAspectFlagBits::eColor;
		}
		isr.aspectMask = aspect;
		isr.baseArrayLayer = 0;
		isr.layerCount = 1;
		isr.baseMipLevel = 0;
		isr.levelCount = 1;
		ivci.subresourceRange = isr;

		auto layout = rg->is_resource_image_in_general_layout(name, current_pass) ? vuk::ImageLayout::eGeneral : vuk::ImageLayout::eShaderReadOnlyOptimal;
	
		VkImageView image_view;
		vkCreateImageView(ptc.ctx.device, (VkImageViewCreateInfo*)&ivci, nullptr, &image_view);
		vuk::Unique<vuk::ImageView> iv = vuk::Unique<vuk::ImageView>(ptc.ctx, ptc.ctx.wrap(image_view));

		return bind_sampled_image(set, binding, *iv, sampler_create_info, layout);
	}

	CommandBuffer& CommandBuffer::bind_persistent(unsigned set, PersistentDescriptorSet& pda) {
		persistent_sets_used[set] = true;
		persistent_sets[set] = pda.backing_set;
		return *this;
	}

	CommandBuffer& CommandBuffer::push_constants(vuk::ShaderStageFlags stages, size_t offset, void* data, size_t size) {
		pcrs.push_back(VkPushConstantRange{ (VkShaderStageFlags)stages, (uint32_t)offset, (uint32_t)size });
		void* dst = push_constant_buffer.data() + offset;
		::memcpy(dst, data, size);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_uniform_buffer(unsigned set, unsigned binding, Buffer buffer) {
		sets_used[set] = true;
		set_bindings[set].bindings[binding].type = vuk::DescriptorType::eUniformBuffer;
		set_bindings[set].bindings[binding].buffer = VkDescriptorBufferInfo{ buffer.buffer, buffer.offset, buffer.size };
		set_bindings[set].used.set(binding);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_storage_buffer(unsigned set, unsigned binding, Buffer buffer) {
		sets_used[set] = true;
		set_bindings[set].bindings[binding].type = vuk::DescriptorType::eStorageBuffer;
		set_bindings[set].bindings[binding].buffer = VkDescriptorBufferInfo{ buffer.buffer, buffer.offset, buffer.size };
		set_bindings[set].used.set(binding);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_storage_image(unsigned set, unsigned binding, vuk::ImageView image_view) {
		sets_used[set] = true;
		set_bindings[set].bindings[binding].type = vuk::DescriptorType::eStorageImage;
		set_bindings[set].bindings[binding].image = vuk::DescriptorImageInfo({}, image_view, vuk::ImageLayout::eGeneral);
		set_bindings[set].used.set(binding);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_storage_image(unsigned set, unsigned binding, Name name) {
		return bind_storage_image(set, binding, get_resource_image_view(name));
	}

	void* CommandBuffer::_map_scratch_uniform_binding(unsigned set, unsigned binding, size_t size) {
		auto buf = ptc._allocate_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vuk::BufferUsageFlagBits::eUniformBuffer, size, 1, true);
		bind_uniform_buffer(set, binding, buf);
		return buf.mapped_ptr;
	}

	CommandBuffer& CommandBuffer::draw(size_t vertex_count, size_t instance_count, size_t first_vertex, size_t first_instance) {
		_bind_graphics_pipeline_state();
		vkCmdDraw(command_buffer, (uint32_t)vertex_count, (uint32_t)instance_count, (uint32_t)first_vertex, (uint32_t)first_instance);
		return *this;
	}

	CommandBuffer& CommandBuffer::draw_indexed(size_t index_count, size_t instance_count, size_t first_index, int32_t vertex_offset, size_t first_instance) {
		_bind_graphics_pipeline_state();

		vkCmdDrawIndexed(command_buffer, (uint32_t)index_count, (uint32_t)instance_count, (uint32_t)first_index, vertex_offset, (uint32_t)first_instance);
		return *this;
    }

	CommandBuffer& CommandBuffer::draw_indexed_indirect(std::span<vuk::DrawIndexedIndirectCommand> cmds) {
		_bind_graphics_pipeline_state();
		auto buf = ptc._allocate_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vuk::BufferUsageFlagBits::eIndirectBuffer, cmds.size_bytes(), 1, true);
        memcpy(buf.mapped_ptr, cmds.data(), cmds.size_bytes());
		vkCmdDrawIndexedIndirect(command_buffer, buf.buffer, (uint32_t)buf.offset, (uint32_t)cmds.size(), sizeof(vuk::DrawIndexedIndirectCommand));
        return *this;
	}

	CommandBuffer& CommandBuffer::dispatch(size_t size_x, size_t size_y, size_t size_z) {
		_bind_compute_pipeline_state();
		vkCmdDispatch(command_buffer, (uint32_t)size_x, (uint32_t)size_y, (uint32_t)size_z);
		return *this;
	}

	CommandBuffer& CommandBuffer::dispatch_invocations(size_t invocation_count_x, size_t invocation_count_y, size_t invocation_count_z) {
		_bind_compute_pipeline_state();
		auto local_size = current_compute_pipeline->local_size;
		// integer div ceil
		uint32_t x = (uint32_t)(invocation_count_x + local_size[0] - 1) / local_size[0];
		uint32_t y = (uint32_t)(invocation_count_y + local_size[1] - 1) / local_size[1];
		uint32_t z = (uint32_t)(invocation_count_z + local_size[2] - 1) / local_size[2];

		vkCmdDispatch(command_buffer, x, y, z);
		return *this;
	}

    SecondaryCommandBuffer CommandBuffer::begin_secondary() {
        auto nptc = new vuk::PerThreadContext(ptc.ifc.begin());
        auto scbuf = nptc->commandbuffer_pool.acquire(VK_COMMAND_BUFFER_LEVEL_SECONDARY, 1)[0];
		VkCommandBufferBeginInfo cbi;
		cbi.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT | VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VkCommandBufferInheritanceInfo cbii;
        cbii.renderPass = ongoing_renderpass->renderpass;
        cbii.subpass = ongoing_renderpass->subpass;
        cbii.framebuffer = VK_NULL_HANDLE; //TODO
		cbi.pInheritanceInfo = &cbii;
		vkBeginCommandBuffer(scbuf, &cbi);
        return SecondaryCommandBuffer(rg, *nptc, scbuf, ongoing_renderpass);
    }

    void CommandBuffer::execute(std::span<VkCommandBuffer> scbufs) {
		if (scbufs.size() > 0) {
			vkCmdExecuteCommands(command_buffer, (uint32_t)scbufs.size(), scbufs.data());
		}
	}

	void CommandBuffer::resolve_image(Name src, Name dst) {
		assert(rg);
		VkImageResolve ir;
		auto src_image = rg->bound_attachments[src].image;
		auto dst_image = rg->bound_attachments[dst].image;
		vuk::ImageSubresourceLayers isl;
		vuk::ImageAspectFlagBits aspect;
		if (rg->bound_attachments[src].description.format == (VkFormat)vuk::Format::eD32Sfloat) {
			aspect = vuk::ImageAspectFlagBits::eDepth;
		} else {
			aspect = vuk::ImageAspectFlagBits::eColor;
		}
		isl.aspectMask = aspect;
		isl.baseArrayLayer = 0;
		isl.layerCount = 1;
		isl.mipLevel = 0;

		ir.srcOffset = vuk::Offset3D{};
		ir.srcSubresource = isl;
		ir.dstOffset = vuk::Offset3D{};
		ir.dstSubresource = isl;
		ir.extent = static_cast<vuk::Extent3D>(rg->bound_attachments[src].extents);

		auto src_layout = rg->is_resource_image_in_general_layout(src, current_pass) ? vuk::ImageLayout::eGeneral : vuk::ImageLayout::eTransferSrcOptimal;
		auto dst_layout = rg->is_resource_image_in_general_layout(dst, current_pass) ? vuk::ImageLayout::eGeneral : vuk::ImageLayout::eTransferDstOptimal;

		vkCmdResolveImage(command_buffer, src_image, (VkImageLayout)src_layout, dst_image, (VkImageLayout)dst_layout, 1, &ir);
	}

	void CommandBuffer::blit_image(Name src, Name dst, vuk::ImageBlit region, vuk::Filter filter) {
		assert(rg);
		auto src_image = rg->bound_attachments[src].image;
		auto dst_image = rg->bound_attachments[dst].image;

		auto src_layout = rg->is_resource_image_in_general_layout(src, current_pass) ? vuk::ImageLayout::eGeneral : vuk::ImageLayout::eTransferSrcOptimal;
		auto dst_layout = rg->is_resource_image_in_general_layout(dst, current_pass) ? vuk::ImageLayout::eGeneral : vuk::ImageLayout::eTransferDstOptimal;

		vkCmdBlitImage(command_buffer, src_image, (VkImageLayout)src_layout, dst_image, (VkImageLayout)dst_layout, 1, (VkImageBlit*)&region, (VkFilter)filter);
	}

	void CommandBuffer::copy_image_to_buffer(Name src, Name dst, vuk::BufferImageCopy bic) {
		assert(rg);
        auto src_batt = rg->bound_attachments[src];
        auto dst_bbuf = rg->bound_buffers[dst];

		bic.bufferOffset += dst_bbuf.buffer.offset;

		auto src_layout = rg->is_resource_image_in_general_layout(src, current_pass) ? vuk::ImageLayout::eGeneral : vuk::ImageLayout::eTransferSrcOptimal;
		vkCmdCopyImageToBuffer(command_buffer, src_batt.image, (VkImageLayout)src_layout, dst_bbuf.buffer.buffer, 1, (VkBufferImageCopy*)&bic);
	}

	void CommandBuffer::_bind_state(bool graphics) {
		for (auto& pcr : pcrs) {
			void* data = push_constant_buffer.data() + pcr.offset;
			vkCmdPushConstants(command_buffer, current_pipeline->pipeline_layout, pcr.stageFlags, pcr.offset, pcr.size, data);
		}
		pcrs.clear();

		for (unsigned i = 0; i < VUK_MAX_SETS; i++) {
			bool persistent = persistent_sets_used[i];
			if (!sets_used[i] && !persistent_sets_used[i])
				continue;
			set_bindings[i].layout_info = graphics? current_pipeline->layout_info[i] : current_compute_pipeline->layout_info[i];
			if (!persistent) {
				auto ds = ptc.descriptor_sets.acquire(set_bindings[i]);
				vkCmdBindDescriptorSets(command_buffer, graphics ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE, graphics ? current_pipeline->pipeline_layout : current_compute_pipeline->pipeline_layout, i, 1, &ds.descriptor_set, 0, nullptr);
				sets_used[i] = false;
				set_bindings[i] = {};
			} else {
				vkCmdBindDescriptorSets(command_buffer, graphics ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE, graphics ? current_pipeline->pipeline_layout : current_compute_pipeline->pipeline_layout, i, 1, &persistent_sets[i], 0, nullptr);
				persistent_sets_used[i] = false;
				persistent_sets[i] = VK_NULL_HANDLE;
			}
		}
	}

	void CommandBuffer::_bind_compute_pipeline_state() {
		if (next_compute_pipeline) {
			vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, next_compute_pipeline->pipeline);
			current_compute_pipeline = *next_compute_pipeline;
			next_compute_pipeline = nullptr;
		}

		_bind_state(false);
	}

	void CommandBuffer::_bind_graphics_pipeline_state() {
		if (next_pipeline) {
            vuk::PipelineInstanceCreateInfo pi;
            pi.base = next_pipeline;
			// set vertex input
			pi.attribute_descriptions = attribute_descriptions;
			pi.binding_descriptions = binding_descriptions;
			auto& vertex_input_state = pi.vertex_input_state;
			vertex_input_state.pVertexAttributeDescriptions = (VkVertexInputAttributeDescription*)pi.attribute_descriptions.data();
			vertex_input_state.vertexAttributeDescriptionCount = (uint32_t)pi.attribute_descriptions.size();
			vertex_input_state.pVertexBindingDescriptions = pi.binding_descriptions.data();
			vertex_input_state.vertexBindingDescriptionCount = (uint32_t)pi.binding_descriptions.size();

			pi.input_assembly_state.topology = (VkPrimitiveTopology)topology;
            pi.input_assembly_state.primitiveRestartEnable = false;

			pi.render_pass = ongoing_renderpass->renderpass;
			pi.subpass = ongoing_renderpass->subpass;

			pi.dynamic_state.pDynamicStates = next_pipeline->dynamic_states.data();
			pi.dynamic_state.dynamicStateCount = static_cast<unsigned>(next_pipeline->dynamic_states.size());

			pi.multisample_state.rasterizationSamples = (VkSampleCountFlagBits) ongoing_renderpass->samples;

			pi.color_blend_attachments = pi.base->color_blend_attachments;
			// last blend attachment is replicated to cover all attachments
			if (pi.color_blend_attachments.size() < (size_t)ongoing_renderpass->color_attachments.size()) {
				pi.color_blend_attachments.resize(ongoing_renderpass->color_attachments.size(), pi.color_blend_attachments.back());
			}
            pi.color_blend_state = pi.base->color_blend_state;
			pi.color_blend_state.pAttachments = (VkPipelineColorBlendAttachmentState*)pi.color_blend_attachments.data();
			pi.color_blend_state.attachmentCount = (uint32_t)pi.color_blend_attachments.size();

			current_pipeline = ptc.pipeline_cache.acquire(pi);

			vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pipeline->pipeline);
			next_pipeline = nullptr;
		}
		_bind_state(true);
    }

    VkCommandBuffer SecondaryCommandBuffer::get_buffer() {
        return command_buffer;
    }

    SecondaryCommandBuffer::~SecondaryCommandBuffer() {
		vkEndCommandBuffer(command_buffer);
        delete &ptc;
    }

} // namespace vuk