#include "vuk/CommandBuffer.hpp"
#include "RenderGraphUtil.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Context.hpp"
#include "vuk/RenderGraph.hpp"

#define VUK_EARLY_RET()                                                                                                                                        \
	if (!current_error) {                                                                                                                                        \
		return *this;                                                                                                                                              \
	}

namespace vuk {
	uint32_t Ignore::to_size() {
		if (bytes != 0)
			return bytes;
		return format_to_texel_block_size(format);
	}

	FormatOrIgnore::FormatOrIgnore(Format format) : ignore(false), format(format), size(format_to_texel_block_size(format)) {}
	FormatOrIgnore::FormatOrIgnore(Ignore ign) : ignore(true), format(ign.format), size(ign.to_size()) {}

	const CommandBuffer::RenderPassInfo& CommandBuffer::get_ongoing_renderpass() const {
		return ongoing_renderpass.value();
	}

	Result<Buffer> CommandBuffer::get_resource_buffer(Name n) const {
		assert(rg);
		auto res = rg->get_resource_buffer(n, current_pass);
		if (!res) {
			return { expected_error, res.error() };
		}
		return { expected_value, res->buffer };
	}

	Result<Image> CommandBuffer::get_resource_image(Name n) const {
		assert(rg);
		auto res = rg->get_resource_image(n, current_pass);
		if (!res) {
			return { expected_error, res.error() };
		}
		return { expected_value, res->attachment.image };
	}

	Result<ImageView> CommandBuffer::get_resource_image_view(Name n) const {
		assert(rg);
		auto res = rg->get_resource_image(n, current_pass);
		if (!res) {
			return { expected_error, res.error() };
		}
		return { expected_value, res->attachment.image_view };
	}

	Result<ImageAttachment> CommandBuffer::get_resource_image_attachment(Name n) const {
		assert(rg);
		auto res = rg->get_resource_image(n, current_pass);
		if (!res) {
			return { expected_error, res.error() };
		}
		return { expected_value, res->attachment };
	}

	CommandBuffer& CommandBuffer::set_dynamic_state(DynamicStateFlags flags) {
		VUK_EARLY_RET();

		// determine which states change to dynamic now - those states need to be flushed into the command buffer
		DynamicStateFlags not_enabled = DynamicStateFlags{ ~dynamic_state_flags.m_mask }; // has invalid bits, but doesn't matter
		auto to_dynamic = not_enabled & flags;
		if (to_dynamic & DynamicStateFlagBits::eViewport && viewports.size() > 0) {
			vkCmdSetViewport(command_buffer, 0, (uint32_t)viewports.size(), viewports.data());
		}
		if (to_dynamic & DynamicStateFlagBits::eScissor && scissors.size() > 0) {
			vkCmdSetScissor(command_buffer, 0, (uint32_t)scissors.size(), scissors.data());
		}
		if (to_dynamic & DynamicStateFlagBits::eLineWidth) {
			vkCmdSetLineWidth(command_buffer, line_width);
		}
		if (to_dynamic & DynamicStateFlagBits::eDepthBias && rasterization_state) {
			vkCmdSetDepthBias(
			    command_buffer, rasterization_state->depthBiasConstantFactor, rasterization_state->depthBiasClamp, rasterization_state->depthBiasSlopeFactor);
		}
		if (to_dynamic & DynamicStateFlagBits::eBlendConstants && blend_constants) {
			vkCmdSetBlendConstants(command_buffer, blend_constants.value().data());
		}
		if (to_dynamic & DynamicStateFlagBits::eDepthBounds && depth_stencil_state) {
			vkCmdSetDepthBounds(command_buffer, depth_stencil_state->minDepthBounds, depth_stencil_state->maxDepthBounds);
		}
		dynamic_state_flags = flags;
		return *this;
	}

	CommandBuffer& CommandBuffer::set_viewport(unsigned index, Viewport vp) {
		VUK_EARLY_RET();
		if (viewports.size() < (index + 1)) {
			assert(index + 1 <= VUK_MAX_VIEWPORTS);
			viewports.resize(index + 1);
		}
		viewports[index] = vp;

		if (dynamic_state_flags & DynamicStateFlagBits::eViewport) {
			vkCmdSetViewport(command_buffer, index, 1, &viewports[index]);
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::set_viewport(unsigned index, Rect2D area, float min_depth, float max_depth) {
		VUK_EARLY_RET();
		Viewport vp;
		if (area.sizing == Sizing::eAbsolute) {
			vp.x = (float)area.offset.x;
			vp.y = (float)area.offset.y;
			vp.width = (float)area.extent.width;
			vp.height = (float)area.extent.height;
			vp.minDepth = min_depth;
			vp.maxDepth = max_depth;
		} else {
			assert(ongoing_renderpass);
			auto fb_dimensions = ongoing_renderpass->extent;
			vp.x = area._relative.x * fb_dimensions.width;
			vp.height = area._relative.height * fb_dimensions.height;
			vp.y = area._relative.y * fb_dimensions.height;
			vp.width = area._relative.width * fb_dimensions.width;
			vp.minDepth = min_depth;
			vp.maxDepth = max_depth;
		}
		return set_viewport(index, vp);
	}

	CommandBuffer& CommandBuffer::set_scissor(unsigned index, Rect2D area) {
		VUK_EARLY_RET();
		VkRect2D vp;
		if (area.sizing == Sizing::eAbsolute) {
			vp = { area.offset, area.extent };
		} else {
			assert(ongoing_renderpass);
			auto fb_dimensions = ongoing_renderpass->extent;
			vp.offset.x = static_cast<int32_t>(area._relative.x * fb_dimensions.width);
			vp.offset.y = static_cast<int32_t>(area._relative.y * fb_dimensions.height);
			vp.extent.width = static_cast<int32_t>(area._relative.width * fb_dimensions.width);
			vp.extent.height = static_cast<int32_t>(area._relative.height * fb_dimensions.height);
		}
		if (scissors.size() < (index + 1)) {
			assert(index + 1 <= VUK_MAX_SCISSORS);
			scissors.resize(index + 1);
		}
		scissors[index] = vp;
		if (dynamic_state_flags & DynamicStateFlagBits::eScissor) {
			vkCmdSetScissor(command_buffer, index, 1, &scissors[index]);
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::set_rasterization(PipelineRasterizationStateCreateInfo state) {
		VUK_EARLY_RET();
		rasterization_state = state;
		if (state.depthBiasEnable && (dynamic_state_flags & DynamicStateFlagBits::eDepthBias)) {
			vkCmdSetDepthBias(command_buffer, state.depthBiasConstantFactor, state.depthBiasClamp, state.depthBiasSlopeFactor);
		}
		if (state.lineWidth != line_width && (dynamic_state_flags & DynamicStateFlagBits::eLineWidth)) {
			vkCmdSetLineWidth(command_buffer, state.lineWidth);
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::set_depth_stencil(PipelineDepthStencilStateCreateInfo state) {
		VUK_EARLY_RET();
		depth_stencil_state = state;
		if (state.depthBoundsTestEnable && (dynamic_state_flags & DynamicStateFlagBits::eDepthBounds)) {
			vkCmdSetDepthBounds(command_buffer, state.minDepthBounds, state.maxDepthBounds);
		}
		return *this;
	}

	PipelineColorBlendAttachmentState blend_preset_to_pcba(BlendPreset preset) {
		PipelineColorBlendAttachmentState pcba;
		switch (preset) {
		case BlendPreset::eAlphaBlend:
			pcba.blendEnable = true;
			pcba.srcColorBlendFactor = BlendFactor::eSrcAlpha;
			pcba.dstColorBlendFactor = BlendFactor::eOneMinusSrcAlpha;
			pcba.colorBlendOp = BlendOp::eAdd;
			pcba.srcAlphaBlendFactor = BlendFactor::eOne;
			pcba.dstAlphaBlendFactor = BlendFactor::eOneMinusSrcAlpha;
			pcba.alphaBlendOp = BlendOp::eAdd;
			break;
		case BlendPreset::eOff:
			pcba.blendEnable = false;
			break;
		case BlendPreset::ePremultipliedAlphaBlend:
			assert(0 && "NYI");
		}
		return pcba;
	}

	CommandBuffer& CommandBuffer::broadcast_color_blend(PipelineColorBlendAttachmentState state) {
		VUK_EARLY_RET();
		assert(ongoing_renderpass);
		color_blend_attachments[0] = state;
		set_color_blend_attachments.set(0, true);
		broadcast_color_blend_attachment_0 = true;
		return *this;
	}

	CommandBuffer& CommandBuffer::broadcast_color_blend(BlendPreset preset) {
		VUK_EARLY_RET();
		return broadcast_color_blend(blend_preset_to_pcba(preset));
	}

	CommandBuffer& CommandBuffer::set_color_blend(Name att, PipelineColorBlendAttachmentState state) {
		VUK_EARLY_RET();
		assert(ongoing_renderpass);
		auto resolved_name = rg->resolve_name(att, current_pass);
		auto it = std::find(ongoing_renderpass->color_attachment_names.begin(), ongoing_renderpass->color_attachment_names.end(), resolved_name);
		assert(it != ongoing_renderpass->color_attachment_names.end() && "Color attachment name not found.");
		auto idx = std::distance(ongoing_renderpass->color_attachment_names.begin(), it);
		set_color_blend_attachments.set(idx, true);
		color_blend_attachments[idx] = state;
		broadcast_color_blend_attachment_0 = false;
		return *this;
	}

	CommandBuffer& CommandBuffer::set_color_blend(Name att, BlendPreset preset) {
		VUK_EARLY_RET();
		return set_color_blend(att, blend_preset_to_pcba(preset));
	}

	CommandBuffer& CommandBuffer::set_blend_constants(std::array<float, 4> constants) {
		VUK_EARLY_RET();
		blend_constants = constants;
		if (dynamic_state_flags & DynamicStateFlagBits::eBlendConstants) {
			vkCmdSetBlendConstants(command_buffer, constants.data());
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_graphics_pipeline(PipelineBaseInfo* pi) {
		VUK_EARLY_RET();
		assert(ongoing_renderpass);
		next_pipeline = pi;
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_graphics_pipeline(Name p) {
		VUK_EARLY_RET();
		return bind_graphics_pipeline(ctx.get_named_pipeline(p));
	}

	CommandBuffer& CommandBuffer::bind_compute_pipeline(PipelineBaseInfo* gpci) {
		VUK_EARLY_RET();
		assert(!ongoing_renderpass);
		next_compute_pipeline = gpci;
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_compute_pipeline(Name p) {
		VUK_EARLY_RET();
		return bind_compute_pipeline(ctx.get_named_pipeline(p));
	}

	CommandBuffer& CommandBuffer::bind_vertex_buffer(unsigned binding, const Buffer& buf, unsigned first_attribute, Packed format) {
		VUK_EARLY_RET();
		assert(binding < VUK_MAX_ATTRIBUTES && "Vertex buffer binding must be smaller than VUK_MAX_ATTRIBUTES.");
		uint32_t location = first_attribute;
		uint32_t offset = 0;
		for (auto& f : format.list) {
			if (f.ignore) {
				offset += f.size;
			} else {
				VertexInputAttributeDescription viad;
				viad.binding = binding;
				viad.format = f.format;
				viad.location = location;
				viad.offset = offset;
				attribute_descriptions[viad.location] = viad;
				set_attribute_descriptions.set(viad.location, true);
				offset += f.size;
				location++;
			}
		}

		VkVertexInputBindingDescription vibd;
		vibd.binding = binding;
		vibd.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vibd.stride = offset;
		binding_descriptions[binding] = vibd;
		set_binding_descriptions.set(binding, true);

		if (buf.buffer) {
			vkCmdBindVertexBuffers(command_buffer, binding, 1, &buf.buffer, &buf.offset);
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_vertex_buffer(unsigned binding, const Buffer& buf, std::span<VertexInputAttributeDescription> viads, uint32_t stride) {
		VUK_EARLY_RET();
		assert(binding < VUK_MAX_ATTRIBUTES && "Vertex buffer binding must be smaller than VUK_MAX_ATTRIBUTES.");
		for (auto& viad : viads) {
			attribute_descriptions[viad.location] = viad;
			set_attribute_descriptions.set(viad.location, true);
		}

		VkVertexInputBindingDescription vibd;
		vibd.binding = binding;
		vibd.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vibd.stride = stride;
		binding_descriptions[binding] = vibd;
		set_binding_descriptions.set(binding, true);

		if (buf.buffer) {
			vkCmdBindVertexBuffers(command_buffer, binding, 1, &buf.buffer, &buf.offset);
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_index_buffer(const Buffer& buf, IndexType type) {
		VUK_EARLY_RET();
		vkCmdBindIndexBuffer(command_buffer, buf.buffer, buf.offset, (VkIndexType)type);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_primitive_topology(PrimitiveTopology topo) {
		VUK_EARLY_RET();
		topology = topo;
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_persistent(unsigned set, PersistentDescriptorSet& pda) {
		VUK_EARLY_RET();
		persistent_sets_to_bind[set] = true;
		persistent_sets[set] = { pda.backing_set, pda.set_layout };
		return *this;
	}

	CommandBuffer& CommandBuffer::push_constants(ShaderStageFlags stages, size_t offset, void* data, size_t size) {
		VUK_EARLY_RET();
		pcrs.push_back(VkPushConstantRange{ (VkShaderStageFlags)stages, (uint32_t)offset, (uint32_t)size });
		void* dst = push_constant_buffer.data() + offset;
		::memcpy(dst, data, size);
		return *this;
	}

	CommandBuffer& CommandBuffer::specialize_constants(uint32_t constant_id, void* data, size_t size) {
		VUK_EARLY_RET();
		auto v = spec_map_entries.emplace(constant_id, SpecEntry{ size == sizeof(double) });
		memcpy(&v.first->second.data, data, size);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_buffer(unsigned set, unsigned binding, const Buffer& buffer) {
		VUK_EARLY_RET();
		sets_to_bind[set] = true;
		set_bindings[set].bindings[binding].type = DescriptorType::eUniformBuffer; // just means buffer
		set_bindings[set].bindings[binding].buffer = VkDescriptorBufferInfo{ buffer.buffer, buffer.offset, buffer.size };
		set_bindings[set].used.set(binding);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_buffer(unsigned set, unsigned binding, Name name) {
		VUK_EARLY_RET();
		auto res = rg->get_resource_buffer(name, current_pass);
		if (!res) {
			current_error = std::move(res);
			return *this;
		}
		return bind_buffer(set, binding, res->buffer);
	}

	CommandBuffer& CommandBuffer::bind_image(unsigned set, unsigned binding, Name resource_name) {
		VUK_EARLY_RET();
		auto res = rg->get_resource_image(resource_name, current_pass);
		if (!res) {
			current_error = std::move(res);
			return *this;
		}
		auto res_gl = rg->is_resource_image_in_general_layout(resource_name, current_pass);
		if (!res_gl) {
			current_error = std::move(res);
			return *this;
		}

		auto layout = *res_gl ? ImageLayout::eGeneral : ImageLayout::eShaderReadOnlyOptimal;

		return bind_image(set, binding, res->attachment.image_view, layout);
	}

	CommandBuffer& CommandBuffer::bind_image(unsigned set, unsigned binding, ImageView image_view, ImageLayout layout) {
		VUK_EARLY_RET();
		sets_to_bind[set] = true;
		auto& db = set_bindings[set].bindings[binding];
		// if previous descriptor was not an image, we reset the DescriptorImageInfo
		if (db.type != DescriptorType::eStorageImage && db.type != DescriptorType::eSampledImage && db.type != DescriptorType::eSampler &&
		    db.type != DescriptorType::eCombinedImageSampler) {
			db.image = { {}, {}, {} };
		}
		db.image.set_image_view(image_view);
		db.image.dii.imageLayout = (VkImageLayout)layout;
		// if it was just a sampler, we upgrade to combined (has both image and sampler) - otherwise just image
		db.type = db.type == DescriptorType::eSampler ? DescriptorType::eCombinedImageSampler : DescriptorType::eSampledImage;
		set_bindings[set].used.set(binding);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_sampler(unsigned set, unsigned binding, SamplerCreateInfo sci) {
		VUK_EARLY_RET();
		sets_to_bind[set] = true;
		auto& db = set_bindings[set].bindings[binding];
		// if previous descriptor was not an image, we reset the DescriptorImageInfo
		if (db.type != DescriptorType::eStorageImage && db.type != DescriptorType::eSampledImage && db.type != DescriptorType::eSampler &&
		    db.type != DescriptorType::eCombinedImageSampler) {
			db.image = { {}, {}, {} };
		}
		db.image.set_sampler(ctx.acquire_sampler(sci, ctx.get_frame_count()));
		// if it was just an image, we upgrade to combined (has both image and sampler) - otherwise just sampler
		db.type = db.type == DescriptorType::eSampledImage ? DescriptorType::eCombinedImageSampler : DescriptorType::eSampler;
		set_bindings[set].used.set(binding);
		return *this;
	}

	void* CommandBuffer::_map_scratch_uniform_binding(unsigned set, unsigned binding, size_t size) {
		if (!current_error) {
			return nullptr;
		}

		auto res = allocate_buffer_cross_device(*allocator, { MemoryUsage::eCPUtoGPU, size, 1 });
		if (!res) {
			current_error = std::move(res);
			return nullptr;
		} else {
			auto& buf = res->get();
			bind_buffer(set, binding, buf);
			return buf.mapped_ptr;
		}
	}

	CommandBuffer& CommandBuffer::draw(size_t vertex_count, size_t instance_count, size_t first_vertex, size_t first_instance) {
		VUK_EARLY_RET();
		if (!_bind_graphics_pipeline_state()) {
			return *this;
		}
		vkCmdDraw(command_buffer, (uint32_t)vertex_count, (uint32_t)instance_count, (uint32_t)first_vertex, (uint32_t)first_instance);
		return *this;
	}

	CommandBuffer& CommandBuffer::draw_indexed(size_t index_count, size_t instance_count, size_t first_index, int32_t vertex_offset, size_t first_instance) {
		VUK_EARLY_RET();
		if (!_bind_graphics_pipeline_state()) {
			return *this;
		}

		vkCmdDrawIndexed(command_buffer, (uint32_t)index_count, (uint32_t)instance_count, (uint32_t)first_index, vertex_offset, (uint32_t)first_instance);
		return *this;
	}

	CommandBuffer& CommandBuffer::draw_indexed_indirect(size_t command_count, const Buffer& indirect_buffer) {
		VUK_EARLY_RET();
		if (!_bind_graphics_pipeline_state()) {
			return *this;
		}
		vkCmdDrawIndexedIndirect(
		    command_buffer, indirect_buffer.buffer, (uint32_t)indirect_buffer.offset, (uint32_t)command_count, sizeof(DrawIndexedIndirectCommand));
		return *this;
	}

	CommandBuffer& CommandBuffer::draw_indexed_indirect(std::span<DrawIndexedIndirectCommand> cmds) {
		VUK_EARLY_RET();
		if (!_bind_graphics_pipeline_state()) {
			return *this;
		}

		auto res = allocate_buffer_cross_device(*allocator, { MemoryUsage::eCPUtoGPU, cmds.size_bytes(), 1 });
		if (!res) {
			current_error = std::move(res);
			return *this;
		}

		auto& buf = *res;
		memcpy(buf->mapped_ptr, cmds.data(), cmds.size_bytes());
		vkCmdDrawIndexedIndirect(command_buffer, buf->buffer, (uint32_t)buf->offset, (uint32_t)cmds.size(), sizeof(DrawIndexedIndirectCommand));
		return *this;
	}

	CommandBuffer& CommandBuffer::draw_indexed_indirect_count(size_t max_draw_count, const Buffer& indirect_buffer, const Buffer& count_buffer) {
		VUK_EARLY_RET();
		if (!_bind_graphics_pipeline_state()) {
			return *this;
		}
		vkCmdDrawIndexedIndirectCount(command_buffer,
		                              indirect_buffer.buffer,
		                              indirect_buffer.offset,
		                              count_buffer.buffer,
		                              count_buffer.offset,
		                              (uint32_t)max_draw_count,
		                              sizeof(DrawIndexedIndirectCommand));
		return *this;
	}

	CommandBuffer& CommandBuffer::dispatch(size_t size_x, size_t size_y, size_t size_z) {
		VUK_EARLY_RET();
		if (!_bind_compute_pipeline_state()) {
			return *this;
		}
		vkCmdDispatch(command_buffer, (uint32_t)size_x, (uint32_t)size_y, (uint32_t)size_z);
		return *this;
	}

	CommandBuffer& CommandBuffer::dispatch_invocations(size_t invocation_count_x, size_t invocation_count_y, size_t invocation_count_z) {
		VUK_EARLY_RET();
		if (!_bind_compute_pipeline_state()) {
			return *this;
		}
		auto local_size = current_compute_pipeline->local_size;
		// integer div ceil
		uint32_t x = (uint32_t)(invocation_count_x + local_size[0] - 1) / local_size[0];
		uint32_t y = (uint32_t)(invocation_count_y + local_size[1] - 1) / local_size[1];
		uint32_t z = (uint32_t)(invocation_count_z + local_size[2] - 1) / local_size[2];

		vkCmdDispatch(command_buffer, x, y, z);
		return *this;
	}

	CommandBuffer& CommandBuffer::dispatch_indirect(const Buffer& indirect_buffer) {
		VUK_EARLY_RET();
		if (!_bind_compute_pipeline_state()) {
			return *this;
		}
		vkCmdDispatchIndirect(command_buffer, indirect_buffer.buffer, indirect_buffer.offset);
		return *this;
	}

	CommandBuffer& CommandBuffer::clear_image(Name src, Clear c) {
		VUK_EARLY_RET();

		assert(rg);
		auto res = rg->get_resource_image(src, current_pass);
		if (!res) {
			current_error = std::move(res);
			return *this;
		}

		auto res_gl = rg->is_resource_image_in_general_layout(src, current_pass);
		if (!res_gl) {
			current_error = std::move(res_gl);
			return *this;
		}
		auto layout = *res_gl ? ImageLayout::eGeneral : ImageLayout::eTransferDstOptimal;

		VkImageSubresourceRange isr = {};
		auto aspect = format_to_aspect((Format)res->description.format);
		isr.aspectMask = (VkImageAspectFlags)aspect;
		isr.baseArrayLayer = 0;
		isr.layerCount = VK_REMAINING_ARRAY_LAYERS;
		isr.baseMipLevel = 0;
		isr.levelCount = VK_REMAINING_MIP_LEVELS;

		if (aspect == ImageAspectFlagBits::eColor) {
			vkCmdClearColorImage(command_buffer, res->attachment.image, (VkImageLayout)layout, &c.c.color, 1, &isr);
		} else if (aspect & (ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil)) {
			vkCmdClearDepthStencilImage(command_buffer, res->attachment.image, (VkImageLayout)layout, &c.c.depthStencil, 1, &isr);
		}

		return *this;
	}

	CommandBuffer& CommandBuffer::resolve_image(Name src, Name dst) {
		VUK_EARLY_RET();
		assert(rg);
		VkImageResolve ir;
		auto src_res = rg->get_resource_image(src, current_pass);
		if (!src_res) {
			current_error = std::move(src_res);
			return *this;
		}
		auto src_image = src_res->attachment.image;
		auto dst_res = rg->get_resource_image(dst, current_pass);
		if (!dst_res) {
			current_error = std::move(dst_res);
			return *this;
		}
		auto dst_image = dst_res->attachment.image;
		ImageSubresourceLayers isl;
		ImageAspectFlagBits aspect;
		if (dst_res->description.format == (VkFormat)Format::eD32Sfloat) {
			aspect = ImageAspectFlagBits::eDepth;
		} else {
			aspect = ImageAspectFlagBits::eColor;
		}
		isl.aspectMask = aspect;
		isl.baseArrayLayer = 0;
		isl.layerCount = 1;
		isl.mipLevel = 0;

		ir.srcOffset = Offset3D{};
		ir.srcSubresource = isl;
		ir.dstOffset = Offset3D{};
		ir.dstSubresource = isl;
		ir.extent = static_cast<Extent3D>(src_res->attachment.extent.extent);

		auto res_gl_src = rg->is_resource_image_in_general_layout(src, current_pass);
		if (!res_gl_src) {
			current_error = std::move(res_gl_src);
			return *this;
		}
		auto res_gl_dst = rg->is_resource_image_in_general_layout(dst, current_pass);
		if (!res_gl_dst) {
			current_error = std::move(res_gl_dst);
			return *this;
		}

		auto src_layout = *res_gl_src ? ImageLayout::eGeneral : ImageLayout::eTransferSrcOptimal;
		auto dst_layout = *res_gl_dst ? ImageLayout::eGeneral : ImageLayout::eTransferDstOptimal;

		vkCmdResolveImage(command_buffer, src_image, (VkImageLayout)src_layout, dst_image, (VkImageLayout)dst_layout, 1, &ir);

		return *this;
	}

	CommandBuffer& CommandBuffer::blit_image(Name src, Name dst, ImageBlit region, Filter filter) {
		VUK_EARLY_RET();
		assert(rg);
		auto src_res = rg->get_resource_image(src, current_pass);
		if (!src_res) {
			current_error = std::move(src_res);
			return *this;
		}
		auto src_image = src_res->attachment.image;
		auto dst_res = rg->get_resource_image(dst, current_pass);
		if (!dst_res) {
			current_error = std::move(dst_res);
			return *this;
		}
		auto dst_image = dst_res->attachment.image;

		auto res_gl_src = rg->is_resource_image_in_general_layout(src, current_pass);
		if (!res_gl_src) {
			current_error = std::move(res_gl_src);
			return *this;
		}
		auto res_gl_dst = rg->is_resource_image_in_general_layout(dst, current_pass);
		if (!res_gl_dst) {
			current_error = std::move(res_gl_dst);
			return *this;
		}

		auto src_layout = *res_gl_src ? ImageLayout::eGeneral : ImageLayout::eTransferSrcOptimal;
		auto dst_layout = *res_gl_dst ? ImageLayout::eGeneral : ImageLayout::eTransferDstOptimal;

		vkCmdBlitImage(command_buffer, src_image, (VkImageLayout)src_layout, dst_image, (VkImageLayout)dst_layout, 1, (VkImageBlit*)&region, (VkFilter)filter);

		return *this;
	}

	CommandBuffer& CommandBuffer::copy_buffer_to_image(Name src, Name dst, BufferImageCopy bic) {
		VUK_EARLY_RET();
		assert(rg);
		auto src_res = rg->get_resource_buffer(src, current_pass);
		if (!src_res) {
			current_error = std::move(src_res);
			return *this;
		}
		auto src_bbuf = src_res->buffer;
		bic.bufferOffset += src_bbuf.offset;

		auto dst_res = rg->get_resource_image(dst, current_pass);
		if (!dst_res) {
			current_error = std::move(dst_res);
			return *this;
		}
		auto dst_image = dst_res->attachment.image;

		auto res_gl = rg->is_resource_image_in_general_layout(dst, current_pass);
		if (!res_gl) {
			current_error = std::move(res_gl);
			return *this;
		}
		auto dst_layout = *res_gl ? ImageLayout::eGeneral : ImageLayout::eTransferDstOptimal;
		vkCmdCopyBufferToImage(command_buffer, src_bbuf.buffer, dst_image, (VkImageLayout)dst_layout, 1, (VkBufferImageCopy*)&bic);

		return *this;
	}

	CommandBuffer& CommandBuffer::copy_image_to_buffer(Name src, Name dst, BufferImageCopy bic) {
		VUK_EARLY_RET();
		assert(rg);
		auto src_res = rg->get_resource_image(src, current_pass);
		if (!src_res) {
			current_error = std::move(src_res);
			return *this;
		}
		auto src_image = src_res->attachment.image;
		auto dst_res = rg->get_resource_buffer(dst, current_pass);
		if (!dst_res) {
			current_error = std::move(dst_res);
			return *this;
		}
		auto dst_bbuf = dst_res->buffer;

		bic.bufferOffset += dst_bbuf.offset;

		auto res_gl = rg->is_resource_image_in_general_layout(src, current_pass);
		if (!res_gl) {
			current_error = std::move(res_gl);
			return *this;
		}
		auto src_layout = *res_gl ? ImageLayout::eGeneral : ImageLayout::eTransferSrcOptimal;
		vkCmdCopyImageToBuffer(command_buffer, src_image, (VkImageLayout)src_layout, dst_bbuf.buffer, 1, (VkBufferImageCopy*)&bic);

		return *this;
	}

	CommandBuffer& CommandBuffer::copy_buffer(Name src, Name dst, size_t size) {
		VUK_EARLY_RET();
		assert(rg);
		auto src_res = rg->get_resource_buffer(src, current_pass);
		if (!src_res) {
			current_error = std::move(src_res);
			return *this;
		}
		auto src_bbuf = src_res->buffer;
		auto dst_res = rg->get_resource_buffer(dst, current_pass);
		if (!dst_res) {
			current_error = std::move(dst_res);
			return *this;
		}
		auto dst_bbuf = dst_res->buffer;
		if (src_bbuf.buffer == dst_bbuf.buffer) {
			bool overlap_a = src_bbuf.offset > dst_bbuf.offset && src_bbuf.offset < (dst_bbuf.offset + dst_bbuf.size);
			bool overlap_b = dst_bbuf.offset > src_bbuf.offset && dst_bbuf.offset < (src_bbuf.offset + src_bbuf.size);
			assert(!overlap_a && !overlap_b);
		}
		VkBufferCopy bc{};
		bc.srcOffset += src_bbuf.offset;
		bc.dstOffset += dst_bbuf.offset;
		bc.size = size;

		vkCmdCopyBuffer(command_buffer, src_bbuf.buffer, dst_bbuf.buffer, 1, &bc);
		return *this;
	}

	CommandBuffer& CommandBuffer::copy_buffer(const Buffer& src, const Buffer& dst) {
		VUK_EARLY_RET();

		assert(src.size == dst.size);
		if (src.buffer == dst.buffer) {
			bool overlap_a = src.offset > dst.offset && src.offset < (dst.offset + dst.size);
			bool overlap_b = dst.offset > src.offset && dst.offset < (src.offset + src.size);
			assert(!overlap_a && !overlap_b);
		}

		VkBufferCopy bc{};
		bc.srcOffset += src.offset;
		bc.dstOffset += dst.offset;
		bc.size = src.size;

		vkCmdCopyBuffer(command_buffer, src.buffer, dst.buffer, 1, &bc);
		return *this;
	}

	CommandBuffer& CommandBuffer::memory_barrier(Access src_access, Access dst_access) {
		VkMemoryBarrier mb{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		auto src_use = to_use(src_access);
		auto dst_use = to_use(dst_access);
		mb.srcAccessMask = is_read_access(src_use) ? 0 : (VkAccessFlags)src_use.access;
		mb.dstAccessMask = (VkAccessFlags)dst_use.access;
		vkCmdPipelineBarrier(command_buffer, (VkPipelineStageFlags)src_use.stages, (VkPipelineStageFlags)dst_use.stages, {}, 1, &mb, 0, nullptr, 0, nullptr);
		return *this;
	}

	CommandBuffer& CommandBuffer::image_barrier(Name src, vuk::Access src_acc, vuk::Access dst_acc, uint32_t mip_level, uint32_t level_count) {
		VUK_EARLY_RET();
		assert(rg);
		auto src_res = rg->get_resource_image(src, current_pass);
		if (!src_res) {
			current_error = std::move(src_res);
			return *this;
		}
		auto src_image = src_res->attachment.image;

		VkImageSubresourceRange isr = {};
		isr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		isr.baseArrayLayer = 0;
		isr.layerCount = VK_REMAINING_ARRAY_LAYERS;
		isr.baseMipLevel = mip_level;
		isr.levelCount = level_count;
		VkImageMemoryBarrier imb{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		imb.image = src_image;
		auto src_use = to_use(src_acc);
		auto dst_use = to_use(dst_acc);
		imb.srcAccessMask = (VkAccessFlags)src_use.access;
		imb.dstAccessMask = (VkAccessFlags)dst_use.access;

		auto res_gl = rg->is_resource_image_in_general_layout(src, current_pass);
		if (!res_gl) {
			current_error = std::move(res_gl);
			return *this;
		}

		if (*res_gl) {
			imb.oldLayout = imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		} else {
			imb.oldLayout = (VkImageLayout)src_use.layout;
			imb.newLayout = (VkImageLayout)dst_use.layout;
		}
		imb.subresourceRange = isr;
		vkCmdPipelineBarrier(command_buffer, (VkPipelineStageFlags)src_use.stages, (VkPipelineStageFlags)dst_use.stages, {}, 0, nullptr, 0, nullptr, 1, &imb);

		return *this;
	}

	CommandBuffer& CommandBuffer::write_timestamp(Query q, PipelineStageFlagBits stage) {
		VUK_EARLY_RET();

		vuk::TimestampQuery tsq;
		vuk::TimestampQueryCreateInfo ci{ .query = q };

		auto res = allocator->allocate_timestamp_queries(std::span{ &tsq, 1 }, std::span{ &ci, 1 });
		if (!res) {
			current_error = std::move(res);
			return *this;
		}

		vkCmdWriteTimestamp(command_buffer, (VkPipelineStageFlagBits)stage, tsq.pool, tsq.id);
		return *this;
	}

	Result<void> CommandBuffer::result() {
		return std::move(current_error);
	}

	bool CommandBuffer::_bind_state(bool graphics) {
		for (auto& pcr : pcrs) {
			void* data = push_constant_buffer.data() + pcr.offset;
			vkCmdPushConstants(
			    command_buffer, graphics ? current_pipeline->pipeline_layout : current_compute_pipeline->pipeline_layout, pcr.stageFlags, pcr.offset, pcr.size, data);
		}
		pcrs.clear();

		auto sets_mask = sets_to_bind.to_ulong();
		auto persistent_sets_mask = persistent_sets_to_bind.to_ulong();
		uint32_t highest_undisturbed_binding_required = 0;
		uint32_t lowest_disturbed_binding = VUK_MAX_SETS;
		for (unsigned i = 0; i < VUK_MAX_SETS; i++) {
			bool set_to_bind = sets_mask & (1 << i);
			bool persistent_set_to_bind = persistent_sets_mask & (1 << i);

			// binding validation
			auto& pipeline_set_layout = graphics ? current_pipeline->layout_info[i].layout : current_compute_pipeline->layout_info[i].layout;
			if (pipeline_set_layout != VK_NULL_HANDLE) {                      // set in the layout
				if (!sets_used[i] && !set_to_bind && !persistent_set_to_bind) { // never set in the cbuf & not requested to bind now
					assert(false && "Pipeline layout contains set, but never set in CommandBuffer or disturbed by a previous set composition or binding.");
					return false;
				} else if (!set_to_bind && !persistent_set_to_bind) { // but not requested to bind now
					// validate that current set is compatible (== same set layout)
					assert(set_layouts_used[i] == pipeline_set_layout && "Previously bound set is incompatible with currently bound pipeline.");
					// this set is compatible, but we require it to be undisturbed
					highest_undisturbed_binding_required = std::max(highest_undisturbed_binding_required, i);
					// detect if during this binding we disturb a set that we depend on
					assert(highest_undisturbed_binding_required < lowest_disturbed_binding &&
					       "Set composition disturbs previously bound set that is not recomposed or bound for this drawcall.");
					continue;
				}
			} else {                                         // not set in the layout
				if (!set_to_bind && !persistent_set_to_bind) { // not requested to bind now, noop
					continue;
				} else { // requested to bind now
					assert(false && "Set layout doesn't contain set, but set in CommandBuffer.");
					return false;
				}
			}
			// if the newly bound DS has a different set layout than the previously bound set, then it disturbs all the sets at higher indices
			bool is_disturbing = set_layouts_used[i] != pipeline_set_layout;
			if (is_disturbing) {
				lowest_disturbed_binding = std::min(lowest_disturbed_binding, i + 1);
			}
			set_bindings[i].layout_info = graphics ? &current_pipeline->layout_info[i] : &current_compute_pipeline->layout_info[i];
			if (!persistent_set_to_bind) {
				auto sb = set_bindings[i].finalize();
				auto& pipeline_set_bindings = graphics ? current_pipeline->base->dslcis[i].bindings : current_compute_pipeline->base->dslcis[i].bindings;
				for (uint64_t j = 0; j < pipeline_set_bindings.size(); j++) {
					auto& pipe_binding = pipeline_set_bindings[j];
					auto& cbuf_binding = sb.bindings[j];

					auto pipe_dtype = (DescriptorType)pipe_binding.descriptorType;
					auto cbuf_dtype = cbuf_binding.type;

					// untyped buffer descriptor inference
					if (cbuf_dtype == DescriptorType::eUniformBuffer && pipe_dtype == DescriptorType::eStorageBuffer) {
						cbuf_binding.type = DescriptorType::eStorageBuffer;
						continue;
					}
					// storage image from any image
					if ((cbuf_dtype == DescriptorType::eSampledImage || cbuf_dtype == DescriptorType::eCombinedImageSampler) &&
					    pipe_dtype == DescriptorType::eStorageImage) {
						cbuf_binding.type = DescriptorType::eStorageImage;
						continue;
					}
					// just sampler -> fine to have image and sampler
					if (cbuf_dtype == DescriptorType::eCombinedImageSampler && pipe_dtype == DescriptorType::eSampler) {
						cbuf_binding.type = DescriptorType::eSampler;
						continue;
					}
					// just image -> fine to have image and sampler
					if (cbuf_dtype == DescriptorType::eCombinedImageSampler && pipe_dtype == DescriptorType::eSampledImage) {
						cbuf_binding.type = DescriptorType::eSampledImage;
						continue;
					}
					// diagnose missing sampler or image
					if (cbuf_dtype == DescriptorType::eSampler && pipe_dtype == DescriptorType::eCombinedImageSampler) {
						assert(false && "Descriptor is combined image-sampler, but only sampler was bound.");
						return false;
					}
					if (cbuf_dtype == DescriptorType::eSampledImage && pipe_dtype == DescriptorType::eCombinedImageSampler) {
						assert(false && "Descriptor is combined image-sampler, but only image was bound.");
						return false;
					}
					if (pipe_dtype != cbuf_dtype) {
						assert(false && "Attempting to bind the wrong descriptor type.");
						return false;
					}
				}

				Unique<DescriptorSet> ds;
				if (auto ret = allocator->allocate_descriptor_sets(std::span{ &*ds, 1 }, std::span{ &sb, 1 }); !ret) {
					current_error = std::move(ret);
					return false;
				}
				vkCmdBindDescriptorSets(command_buffer,
				                        graphics ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE,
				                        graphics ? current_pipeline->pipeline_layout : current_compute_pipeline->pipeline_layout,
				                        i,
				                        1,
				                        &ds->descriptor_set,
				                        0,
				                        nullptr);
				set_layouts_used[i] = ds->layout_info.layout;
			} else {
				vkCmdBindDescriptorSets(command_buffer,
				                        graphics ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE,
				                        graphics ? current_pipeline->pipeline_layout : current_compute_pipeline->pipeline_layout,
				                        i,
				                        1,
				                        &persistent_sets[i].first,
				                        0,
				                        nullptr);
				set_layouts_used[i] = persistent_sets[i].second;
			}
			set_bindings[i].used.reset();
		}
		auto sets_bound = sets_to_bind | persistent_sets_to_bind;            // these sets we bound freshly, valid
		for (unsigned i = lowest_disturbed_binding; i < VUK_MAX_SETS; i++) { // clear the slots where the binding was disturbed
			sets_used.set(i, false);
		}
		sets_used |= sets_bound;
		sets_to_bind.reset();
		persistent_sets_to_bind.reset();
		return true;
	}

	bool CommandBuffer::_bind_compute_pipeline_state() {
		if (next_compute_pipeline) {
			ComputePipelineInstanceCreateInfo pi;
			pi.base = next_compute_pipeline;

			bool empty = true;
			unsigned offset = 0;
			unsigned idx = 0;
			for (auto& sc : pi.base->reflection_info.spec_constants) {
				auto it = spec_map_entries.find(sc.binding);
				if (it != spec_map_entries.end()) {
					auto& map_e = it->second;
					unsigned size = map_e.is_double ? (unsigned)sizeof(double) : 4;
					pi.specialization_map_entries.push_back(VkSpecializationMapEntry{ sc.binding, offset, size });
					memcpy(pi.specialization_constant_data.data() + offset, map_e.data, size);
					offset += size;
					empty = false;
				}
			}

			if (!empty) {
				VkSpecializationInfo& si = pi.specialization_info;
				si.pMapEntries = pi.specialization_map_entries.data();
				si.mapEntryCount = (uint32_t)pi.specialization_map_entries.size();
				si.pData = pi.specialization_constant_data.data();
				si.dataSize = pi.specialization_constant_data.size();

				pi.base->psscis[0].pSpecializationInfo = &pi.specialization_info;
			}

			current_compute_pipeline = ctx.acquire_pipeline(pi, ctx.get_frame_count());

			vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_compute_pipeline->pipeline);
			next_compute_pipeline = nullptr;
		}

		return _bind_state(false);
	}

	template<class T>
	void write(std::byte*& data_ptr, const T& data) {
		memcpy(data_ptr, &data, sizeof(T));
		data_ptr += sizeof(T);
	};

	bool CommandBuffer::_bind_graphics_pipeline_state() {
		if (next_pipeline) {
			PipelineInstanceCreateInfo pi;
			pi.base = next_pipeline;
			pi.render_pass = ongoing_renderpass->renderpass;
			pi.dynamic_state_flags = dynamic_state_flags;
			auto& records = pi.records;
			if (ongoing_renderpass->subpass > 0) {
				records.nonzero_subpass = true;
				pi.extended_size += sizeof(uint8_t);
			}
			pi.topology = (VkPrimitiveTopology)topology;
			pi.primitive_restart_enable = false;

			// VERTEX INPUT
			Bitset<VUK_MAX_ATTRIBUTES> used_bindings = {};
			if (attribute_descriptions.size() > 0 && binding_descriptions.size() > 0 && pi.base->reflection_info.attributes.size() > 0) {
				records.vertex_input = true;
				for (unsigned i = 0; i < pi.base->reflection_info.attributes.size(); i++) {
					auto& attr = pi.base->reflection_info.attributes[i];
					assert(set_attribute_descriptions.test(i) && "Pipeline expects attribute, but was never set in command buffer.");
					used_bindings.set(attribute_descriptions[i].binding, true);
				}

				pi.extended_size += (uint16_t)pi.base->reflection_info.attributes.size() * sizeof(PipelineInstanceCreateInfo::VertexInputAttributeDescription);
				pi.extended_size += sizeof(uint8_t);
				pi.extended_size += (uint16_t)used_bindings.count() * sizeof(PipelineInstanceCreateInfo::VertexInputBindingDescription);
			}

			// BLEND STATE
			// attachmentCount says how many attachments
			pi.attachmentCount = (uint8_t)ongoing_renderpass->color_attachments.size();
			bool rasterization = ongoing_renderpass->depth_stencil_attachment || pi.attachmentCount > 0;

			if (pi.attachmentCount > 0) {
				assert(set_color_blend_attachments.count() > 0 && "If a pass has a color attachment, you must set at least one color blend state.");
				records.broadcast_color_blend_attachment_0 = broadcast_color_blend_attachment_0;

				if (broadcast_color_blend_attachment_0) {
					assert(set_color_blend_attachments.test(0) && "Broadcast turned on, but no blend state set.");
					if (color_blend_attachments[0] != PipelineColorBlendAttachmentState{}) {
						records.color_blend_attachments = true;
						pi.extended_size += sizeof(PipelineInstanceCreateInfo::PipelineColorBlendAttachmentState);
					}
				} else {
					assert(set_color_blend_attachments.count() >= pi.attachmentCount &&
					       "If color blend state is not broadcast, you must set it for each color attachment.");
					records.color_blend_attachments = true;
					pi.extended_size += pi.attachmentCount * sizeof(PipelineInstanceCreateInfo::PipelineColorBlendAttachmentState);
				}
			}

			records.logic_op = false; // TODO: logic op unsupported
			if (blend_constants && !(dynamic_state_flags & DynamicStateFlagBits::eBlendConstants)) {
				records.blend_constants = true;
				pi.extended_size += sizeof(float) * 4;
			}

			unsigned spec_const_size = 0;
			Bitset<VUK_MAX_SPECIALIZATIONCONSTANT_RANGES> set_constants = {};
			if (spec_map_entries.size() > 0 && pi.base->reflection_info.spec_constants.size() > 0) {
				for (unsigned i = 0; i < pi.base->reflection_info.spec_constants.size(); i++) {
					auto& sc = pi.base->reflection_info.spec_constants[i];
					auto size = sc.type == Program::Type::edouble ? sizeof(double) : 4;
					auto it = spec_map_entries.find(sc.binding);
					if (it != spec_map_entries.end()) {
						spec_const_size += (uint32_t)size;
						set_constants.set(i, true);
					}
				}
				records.specialization_constants = true;
				pi.extended_size += (uint16_t)sizeof(set_constants);
				pi.extended_size += (uint16_t)spec_const_size;
			}

			if (rasterization) {
				assert(rasterization_state && "If a pass has a depth/stencil or color attachment, you must set the rasterization state.");

				pi.cullMode = (VkCullModeFlags)rasterization_state->cullMode;
				PipelineRasterizationStateCreateInfo def{ .cullMode = rasterization_state->cullMode };
				if (dynamic_state_flags & DynamicStateFlagBits::eDepthBias) {
					def.depthBiasConstantFactor = rasterization_state->depthBiasConstantFactor;
					def.depthBiasClamp = rasterization_state->depthBiasClamp;
					def.depthBiasSlopeFactor = rasterization_state->depthBiasSlopeFactor;
				} else {
					// TODO: static depth bias unsupported
					assert(rasterization_state->depthBiasConstantFactor == def.depthBiasConstantFactor);
					assert(rasterization_state->depthBiasClamp == def.depthBiasClamp);
					assert(rasterization_state->depthBiasSlopeFactor == def.depthBiasSlopeFactor);
				}
				records.depth_bias_enable = rasterization_state->depthBiasEnable; // the enable itself is not dynamic state in core
				if (*rasterization_state != def) {
					records.non_trivial_raster_state = true;
					pi.extended_size += sizeof(PipelineInstanceCreateInfo::RasterizationState);
				}
			}

			if (ongoing_renderpass->depth_stencil_attachment) {
				assert(depth_stencil_state && "If a pass has a depth/stencil attachment, you must set the depth/stencil state.");

				records.depth_stencil = true;
				pi.extended_size += sizeof(PipelineInstanceCreateInfo::Depth);

				assert(depth_stencil_state->stencilTestEnable == false);     // TODO: stencil unsupported
				assert(depth_stencil_state->depthBoundsTestEnable == false); // TODO: depth bounds unsupported
			}

			if (ongoing_renderpass->samples != SampleCountFlagBits::e1) {
				records.more_than_one_sample = true;
				pi.extended_size += sizeof(PipelineInstanceCreateInfo::Multisample);
			}

			if (rasterization) {
				if (viewports.size() > 0) {
					records.viewports = true;
					pi.extended_size += sizeof(uint8_t);
					if (!(dynamic_state_flags & DynamicStateFlagBits::eViewport)) {
						pi.extended_size += (uint16_t)viewports.size() * sizeof(VkViewport);
					}
				} else if (!(dynamic_state_flags & DynamicStateFlagBits::eViewport)) {
					assert("If a pass has a depth/stencil or color attachment, you must set at least one viewport.");
				}
			}

			if (rasterization) {
				if (scissors.size() > 0) {
					records.scissors = true;
					pi.extended_size += sizeof(uint8_t);
					if (!(dynamic_state_flags & DynamicStateFlagBits::eScissor)) {
						pi.extended_size += (uint16_t)scissors.size() * sizeof(VkRect2D);
					}
				} else if (!(dynamic_state_flags & DynamicStateFlagBits::eScissor)) {
					assert("If a pass has a depth/stencil or color attachment, you must set at least one scissor.");
				}
			}
			// small buffer optimization:
			// if the extended data fits, then we put it inline in the key
			std::byte* data_ptr;
			std::byte* data_start_ptr;
			if (pi.is_inline()) {
				data_start_ptr = data_ptr = pi.inline_data;
			} else { // otherwise we allocate
				pi.extended_data = new std::byte[pi.extended_size];
				data_start_ptr = data_ptr = pi.extended_data;
			}
			// start writing packed stream
			if (ongoing_renderpass->subpass > 0) {
				write<uint8_t>(data_ptr, ongoing_renderpass->subpass);
			}

			if (records.vertex_input) {
				for (unsigned i = 0; i < pi.base->reflection_info.attributes.size(); i++) {
					auto& attr = pi.base->reflection_info.attributes[i];
					auto& att = attribute_descriptions[i];
					PipelineInstanceCreateInfo::VertexInputAttributeDescription viad{
						.format = att.format, .offset = att.offset, .location = (uint8_t)att.location, .binding = (uint8_t)att.binding
					};
					write(data_ptr, viad);
				}
				write<uint8_t>(data_ptr, (uint8_t)used_bindings.count());
				for (unsigned i = 0; i < VUK_MAX_ATTRIBUTES; i++) {
					if (used_bindings.test(i)) {
						auto& bin = binding_descriptions[i];
						PipelineInstanceCreateInfo::VertexInputBindingDescription vibd{ .stride = bin.stride,
							                                                              .inputRate = (uint32_t)bin.inputRate,
							                                                              .binding = (uint8_t)bin.binding };
						write(data_ptr, vibd);
					}
				}
			}

			if (records.color_blend_attachments) {
				uint32_t num_pcba_to_write = records.broadcast_color_blend_attachment_0 ? 1 : (uint32_t)color_blend_attachments.size();
				for (uint32_t i = 0; i < num_pcba_to_write; i++) {
					auto& cba = color_blend_attachments[i];
					PipelineInstanceCreateInfo::PipelineColorBlendAttachmentState pcba{ .blendEnable = cba.blendEnable,
						                                                                  .srcColorBlendFactor = cba.srcColorBlendFactor,
						                                                                  .dstColorBlendFactor = cba.dstColorBlendFactor,
						                                                                  .colorBlendOp = cba.colorBlendOp,
						                                                                  .srcAlphaBlendFactor = cba.srcAlphaBlendFactor,
						                                                                  .dstAlphaBlendFactor = cba.dstAlphaBlendFactor,
						                                                                  .alphaBlendOp = cba.alphaBlendOp,
						                                                                  .colorWriteMask = (uint32_t)cba.colorWriteMask };
					write(data_ptr, pcba);
				}
			}

			if (blend_constants && !(dynamic_state_flags & DynamicStateFlagBits::eBlendConstants)) {
				memcpy(data_ptr, &*blend_constants, sizeof(float) * 4);
				data_ptr += sizeof(float) * 4;
			}

			if (records.specialization_constants) {
				write(data_ptr, set_constants);
				for (unsigned i = 0; i < VUK_MAX_SPECIALIZATIONCONSTANT_RANGES; i++) {
					if (set_constants.test(i)) {
						auto& sc = pi.base->reflection_info.spec_constants[i];
						auto size = sc.type == Program::Type::edouble ? sizeof(double) : 4;
						auto& map_e = spec_map_entries.find(sc.binding)->second;
						memcpy(data_ptr, map_e.data, size);
						data_ptr += size;
					}
				}
			}

			if (records.non_trivial_raster_state) {
				PipelineInstanceCreateInfo::RasterizationState rs{ .depthClampEnable = (bool)rasterization_state->depthClampEnable,
					                                                 .rasterizerDiscardEnable = (bool)rasterization_state->rasterizerDiscardEnable,
					                                                 .polygonMode = (uint8_t)rasterization_state->polygonMode,
					                                                 .frontFace = (uint8_t)rasterization_state->frontFace };
				write(data_ptr, rs);
				// TODO: support depth bias
			}

			if (ongoing_renderpass->depth_stencil_attachment) {
				PipelineInstanceCreateInfo::Depth ds = { .depthTestEnable = (bool)depth_stencil_state->depthTestEnable,
					                                       .depthWriteEnable = (bool)depth_stencil_state->depthWriteEnable,
					                                       .depthCompareOp = (uint8_t)depth_stencil_state->depthCompareOp };
				write(data_ptr, ds);
				// TODO: support stencil
				// TODO: support depth bounds
			}

			if (ongoing_renderpass->samples != SampleCountFlagBits::e1) {
				PipelineInstanceCreateInfo::Multisample ms{ .rasterization_samples = (VkSampleCountFlagBits)ongoing_renderpass->samples };
				write(data_ptr, ms);
			}

			if (viewports.size() > 0) {
				write<uint8_t>(data_ptr, (uint8_t)viewports.size());
				if (!(dynamic_state_flags & DynamicStateFlagBits::eViewport)) {
					for (const auto& vp : viewports) {
						write(data_ptr, vp);
					}
				}
			}

			if (scissors.size() > 0) {
				write<uint8_t>(data_ptr, (uint8_t)scissors.size());
				if (!(dynamic_state_flags & DynamicStateFlagBits::eScissor)) {
					for (const auto& sc : scissors) {
						write(data_ptr, sc);
					}
				}
			}

			assert(data_ptr - data_start_ptr == pi.extended_size); // sanity check: we wrote all the data we wanted to
			// acquire_pipeline makes copy of extended_data if it needs to
			current_pipeline = ctx.acquire_pipeline(pi, ctx.get_frame_count());
			if (!pi.is_inline()) {
				delete pi.extended_data;
			}

			vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pipeline->pipeline);
			next_pipeline = nullptr;
		}
		return _bind_state(true);
	}

} // namespace vuk
