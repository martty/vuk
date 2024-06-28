#include "vuk/runtime/CommandBuffer.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/SyncLowering.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"

#include <cmath>
#include <fmt/printf.h>

#define VUK_EARLY_RET()                                                                                                                                        \
	if (!current_error) {                                                                                                                                        \
		return *this;                                                                                                                                              \
	}

// hand-inlined versions of the most common bitset ops, because if these are not inlined, the overhead is terrible
#define VUK_SB_SET(bitset, pos, value)                                                                                                                         \
	if constexpr (decltype(bitset)::n_words == 1) {                                                                                                              \
		if (value) {                                                                                                                                               \
			bitset.words[0] |= 1ULL << pos;                                                                                                                          \
		} else {                                                                                                                                                   \
			bitset.words[0] &= ~(1ULL << pos);                                                                                                                       \
		}                                                                                                                                                          \
	} else {                                                                                                                                                     \
		bitset.set(pos, value);                                                                                                                                    \
	}

#define VUK_SB_COUNT(bitset, dst)                                                                                                                              \
	if constexpr (decltype(bitset)::n_words == 1) {                                                                                                              \
		using T = uint64_t;                                                                                                                                        \
		uint64_t v = bitset.words[0];                                                                                                                              \
		v = v - ((v >> 1) & (T) ~(T)0 / 3);                                                                                                                        \
		v = (v & (T) ~(T)0 / 15 * 3) + ((v >> 2) & (T) ~(T)0 / 15 * 3);                                                                                            \
		v = (v + (v >> 4)) & (T) ~(T)0 / 255 * 15;                                                                                                                 \
		dst = (T)(v * ((T) ~(T)0 / 255)) >> (sizeof(T) - 1) * CHAR_BIT;                                                                                            \
	} else {                                                                                                                                                     \
		dst = bitset.count();                                                                                                                                      \
	}

#define VUK_SB_TEST(bitset, pos, dst)                                                                                                                          \
	if constexpr (decltype(bitset)::n_words == 1) {                                                                                                              \
		dst = bitset.words[0] & 1ULL << pos;                                                                                                                       \
	} else {                                                                                                                                                     \
		dst = bitset.test(pos);                                                                                                                                    \
	}

namespace vuk {
	uint32_t Ignore::to_size() {
		if (bytes != 0)
			return bytes;
		return format_to_texel_block_size(format);
	}

	FormatOrIgnore::FormatOrIgnore(Format format) : ignore(false), format(format), size(format_to_texel_block_size(format)) {}
	FormatOrIgnore::FormatOrIgnore(Ignore ign) : ignore(true), format(ign.format), size(ign.to_size()) {}

	// for rendergraph

	CommandBuffer::CommandBuffer(ExecutableRenderGraph& rg, Runtime& ctx, Allocator& allocator, VkCommandBuffer cb) :
	    rg(&rg),
	    ctx(ctx),
	    allocator(&allocator),
	    command_buffer(cb),
	    ds_strategy_flags(ctx.default_descriptor_set_strategy) {}

	CommandBuffer::CommandBuffer(ExecutableRenderGraph& rg, Runtime& ctx, Allocator& allocator, VkCommandBuffer cb, std::optional<RenderPassInfo> ongoing) :
	    rg(&rg),
	    ctx(ctx),
	    allocator(&allocator),
	    command_buffer(cb),
	    ongoing_render_pass(ongoing),
	    ds_strategy_flags(ctx.default_descriptor_set_strategy) {}

	const CommandBuffer::RenderPassInfo& CommandBuffer::get_ongoing_render_pass() const {
		return ongoing_render_pass.value();
	}

	CommandBuffer& CommandBuffer::set_descriptor_set_strategy(DescriptorSetStrategyFlags ds_strategy_flags) {
		this->ds_strategy_flags = ds_strategy_flags;
		return *this;
	}

	CommandBuffer& CommandBuffer::set_dynamic_state(DynamicStateFlags flags) {
		VUK_EARLY_RET();

		// determine which states change to dynamic now - those states need to be flushed into the command buffer
		DynamicStateFlags not_enabled = DynamicStateFlags{ ~dynamic_state_flags.m_mask }; // has invalid bits, but doesn't matter
		auto to_dynamic = not_enabled & flags;
		if (to_dynamic & DynamicStateFlagBits::eViewport && viewports.size() > 0) {
			ctx.vkCmdSetViewport(command_buffer, 0, (uint32_t)viewports.size(), viewports.data());
		}
		if (to_dynamic & DynamicStateFlagBits::eScissor && scissors.size() > 0) {
			ctx.vkCmdSetScissor(command_buffer, 0, (uint32_t)scissors.size(), scissors.data());
		}
		if (to_dynamic & DynamicStateFlagBits::eLineWidth) {
			ctx.vkCmdSetLineWidth(command_buffer, line_width);
		}
		if (to_dynamic & DynamicStateFlagBits::eDepthBias && rasterization_state) {
			ctx.vkCmdSetDepthBias(
			    command_buffer, rasterization_state->depthBiasConstantFactor, rasterization_state->depthBiasClamp, rasterization_state->depthBiasSlopeFactor);
		}
		if (to_dynamic & DynamicStateFlagBits::eBlendConstants && blend_constants) {
			ctx.vkCmdSetBlendConstants(command_buffer, blend_constants.value().data());
		}
		if (to_dynamic & DynamicStateFlagBits::eDepthBounds && depth_stencil_state) {
			ctx.vkCmdSetDepthBounds(command_buffer, depth_stencil_state->minDepthBounds, depth_stencil_state->maxDepthBounds);
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
			ctx.vkCmdSetViewport(command_buffer, index, 1, &viewports[index]);
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
			assert(ongoing_render_pass);
			auto fb_dimensions = ongoing_render_pass->extent;
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
			assert(ongoing_render_pass);
			auto fb_dimensions = ongoing_render_pass->extent;
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
			ctx.vkCmdSetScissor(command_buffer, index, 1, &scissors[index]);
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::set_rasterization(PipelineRasterizationStateCreateInfo state) {
		VUK_EARLY_RET();
		rasterization_state = state;
		if (state.depthBiasEnable && (dynamic_state_flags & DynamicStateFlagBits::eDepthBias)) {
			ctx.vkCmdSetDepthBias(command_buffer, state.depthBiasConstantFactor, state.depthBiasClamp, state.depthBiasSlopeFactor);
		}
		if (state.lineWidth != line_width && (dynamic_state_flags & DynamicStateFlagBits::eLineWidth)) {
			ctx.vkCmdSetLineWidth(command_buffer, state.lineWidth);
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::set_depth_stencil(PipelineDepthStencilStateCreateInfo state) {
		VUK_EARLY_RET();
		depth_stencil_state = state;
		if (state.depthBoundsTestEnable && (dynamic_state_flags & DynamicStateFlagBits::eDepthBounds)) {
			ctx.vkCmdSetDepthBounds(command_buffer, state.minDepthBounds, state.maxDepthBounds);
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::set_conservative(PipelineRasterizationConservativeStateCreateInfo state) {
		VUK_EARLY_RET();
		conservative_state = state;
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
		assert(ongoing_render_pass);
		color_blend_attachments[0] = state;
		set_color_blend_attachments.set(0, true);
		broadcast_color_blend_attachment_0 = true;
		return *this;
	}

	CommandBuffer& CommandBuffer::broadcast_color_blend(BlendPreset preset) {
		VUK_EARLY_RET();
		return broadcast_color_blend(blend_preset_to_pcba(preset));
	}

	CommandBuffer& CommandBuffer::set_color_blend(const ImageAttachment& att, PipelineColorBlendAttachmentState state) {
		VUK_EARLY_RET();
		assert(ongoing_render_pass);

		auto it = std::find(ongoing_render_pass->color_attachment_ivs.begin(), ongoing_render_pass->color_attachment_ivs.end(), att.image_view);
		assert(it != ongoing_render_pass->color_attachment_ivs.end() && "Color attachment name not found.");
		auto idx = std::distance(ongoing_render_pass->color_attachment_ivs.begin(), it);
		set_color_blend_attachments.set(idx, true);
		color_blend_attachments[idx] = state;
		broadcast_color_blend_attachment_0 = false;
		return *this;
	}

	CommandBuffer& CommandBuffer::set_color_blend(const ImageAttachment& att, BlendPreset preset) {
		VUK_EARLY_RET();
		return set_color_blend(att, blend_preset_to_pcba(preset));
	}

	CommandBuffer& CommandBuffer::set_blend_constants(std::array<float, 4> constants) {
		VUK_EARLY_RET();
		blend_constants = constants;
		if (dynamic_state_flags & DynamicStateFlagBits::eBlendConstants) {
			ctx.vkCmdSetBlendConstants(command_buffer, constants.data());
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_graphics_pipeline(PipelineBaseInfo* pi) {
		VUK_EARLY_RET();
		assert(ongoing_render_pass);
		next_pipeline = pi;
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_graphics_pipeline(Name p) {
		VUK_EARLY_RET();
		return bind_graphics_pipeline(ctx.get_named_pipeline(p));
	}

	CommandBuffer& CommandBuffer::bind_compute_pipeline(PipelineBaseInfo* gpci) {
		VUK_EARLY_RET();
		assert(!ongoing_render_pass);
		next_compute_pipeline = gpci;
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_compute_pipeline(Name p) {
		VUK_EARLY_RET();
		return bind_compute_pipeline(ctx.get_named_pipeline(p));
	}

	CommandBuffer& CommandBuffer::bind_ray_tracing_pipeline(PipelineBaseInfo* gpci) {
		VUK_EARLY_RET();
		assert(!ongoing_render_pass);
		next_ray_tracing_pipeline = gpci;
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_ray_tracing_pipeline(Name p) {
		VUK_EARLY_RET();
		return bind_ray_tracing_pipeline(ctx.get_named_pipeline(p));
	}

	CommandBuffer& CommandBuffer::bind_vertex_buffer(unsigned binding, const Buffer& buf, unsigned first_attribute, Packed format, VertexInputRate input_rate) {
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
				VUK_SB_SET(set_attribute_descriptions, viad.location, true);
				offset += f.size;
				location++;
			}
		}

		VkVertexInputBindingDescription vibd;
		vibd.binding = binding;
		vibd.inputRate = (VkVertexInputRate)input_rate;
		vibd.stride = offset;
		binding_descriptions[binding] = vibd;
		VUK_SB_SET(set_binding_descriptions, binding, true);

		if (buf.buffer) {
			ctx.vkCmdBindVertexBuffers(command_buffer, binding, 1, &buf.buffer, &buf.offset);
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_vertex_buffer(unsigned binding, const Buffer& buf, std::span<VertexInputAttributeDescription> viads, uint32_t stride, VertexInputRate input_rate) {
		VUK_EARLY_RET();
		assert(binding < VUK_MAX_ATTRIBUTES && "Vertex buffer binding must be smaller than VUK_MAX_ATTRIBUTES.");
		for (auto& viad : viads) {
			attribute_descriptions[viad.location] = viad;
			VUK_SB_SET(set_attribute_descriptions, viad.location, true);
		}

		VkVertexInputBindingDescription vibd;
		vibd.binding = binding;
		vibd.inputRate = (VkVertexInputRate)input_rate;
		vibd.stride = stride;
		binding_descriptions[binding] = vibd;
		VUK_SB_SET(set_binding_descriptions, binding, true);

		if (buf.buffer) {
			ctx.vkCmdBindVertexBuffers(command_buffer, binding, 1, &buf.buffer, &buf.offset);
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_index_buffer(const Buffer& buf, IndexType type) {
		VUK_EARLY_RET();
		ctx.vkCmdBindIndexBuffer(command_buffer, buf.buffer, buf.offset, (VkIndexType)type);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_primitive_topology(PrimitiveTopology topo) {
		VUK_EARLY_RET();
		topology = topo;
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_persistent(unsigned set, PersistentDescriptorSet& pda) {
		VUK_EARLY_RET();
		assert(set < VUK_MAX_SETS);
		persistent_sets_to_bind.set(set, true);
		persistent_sets[set] = { pda.backing_set, pda.set_layout };
		return *this;
	}

	CommandBuffer& CommandBuffer::push_constants(ShaderStageFlags stages, size_t offset, void* data, size_t size) {
		VUK_EARLY_RET();
		assert(offset + size <= VUK_MAX_PUSHCONSTANT_SIZE);
		pcrs.push_back(VkPushConstantRange{ (VkShaderStageFlags)stages, (uint32_t)offset, (uint32_t)size });
		void* dst = push_constant_buffer + offset;
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
		assert(set < VUK_MAX_SETS);
		assert(binding < VUK_MAX_BINDINGS);
		sets_to_bind.set(set, true);
		set_bindings[set].bindings[binding].type = DescriptorType::eUniformBuffer; // just means buffer
		set_bindings[set].bindings[binding].buffer = VkDescriptorBufferInfo{ buffer.buffer, buffer.offset, buffer.size };
		set_bindings[set].used.set(binding);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_image(unsigned set, unsigned binding, const ImageAttachment& ia) {
		VUK_EARLY_RET();
		if (ia.image_view != ImageView{}) {
			bind_image(set, binding, ia.image_view, ia.layout);
		} else {
			assert(ia.image);
			auto res = allocate_image_view(*allocator, ia);
			if (!res) {
				current_error = std::move(res);
				return *this;
			} else {
				bind_image(set, binding, **res, ia.layout);
			}
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_image(unsigned set, unsigned binding, const ImageAttachment& ia, Ref def) {
		VUK_EARLY_RET();
		if (ia.image_view != ImageView{}) {
			bind_image(set, binding, ia.image_view, ia.layout);
		} else {
			assert(ia.image);
			auto res = allocate_image_view(*allocator, ia);
			if (!res) {
				current_error = std::move(res);
				return *this;
			} else {
				auto node = def.node;
				if (node->debug_info && node->debug_info->result_names.size() > 0 && !node->debug_info->result_names[0].empty()) {
					ctx.set_name((**res).payload, node->debug_info->result_names[0]);
				} else {
					printf("");
				}
				bind_image(set, binding, **res, ia.layout);
			}
		}
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_image(unsigned set, unsigned binding, ImageView image_view, ImageLayout layout) {
		VUK_EARLY_RET();
		assert(set < VUK_MAX_SETS);
		assert(binding < VUK_MAX_BINDINGS);
		assert(image_view.payload != VK_NULL_HANDLE);
		sets_to_bind.set(set, true);
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
		assert(set < VUK_MAX_SETS);
		assert(binding < VUK_MAX_BINDINGS);
		sets_to_bind.set(set, true);
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

	void* CommandBuffer::_scratch_buffer(unsigned set, unsigned binding, size_t size) {
		if (!current_error) {
			return nullptr;
		}

		auto res = allocate_buffer(*allocator, { MemoryUsage::eCPUtoGPU, size, 1 });
		if (!res) {
			current_error = std::move(res);
			return nullptr;
		} else {
			auto& buf = res->get();
			bind_buffer(set, binding, buf);
			return buf.mapped_ptr;
		}
	}

	CommandBuffer& CommandBuffer::bind_acceleration_structure(unsigned set, unsigned binding, VkAccelerationStructureKHR tlas) {
		VUK_EARLY_RET();
		assert(set < VUK_MAX_SETS);
		assert(binding < VUK_MAX_BINDINGS);
		sets_to_bind.set(set, true);
		auto& db = set_bindings[set].bindings[binding];
		db.as.as = tlas;
		db.as.wds.accelerationStructureCount = 1;
		db.type = DescriptorType::eAccelerationStructureKHR;
		set_bindings[set].used.set(binding);
		return *this;
	}

	CommandBuffer& CommandBuffer::draw(size_t vertex_count, size_t instance_count, size_t first_vertex, size_t first_instance) {
		VUK_EARLY_RET();
		if (!_bind_graphics_pipeline_state()) {
			return *this;
		}
		ctx.vkCmdDraw(command_buffer, (uint32_t)vertex_count, (uint32_t)instance_count, (uint32_t)first_vertex, (uint32_t)first_instance);
		return *this;
	}

	CommandBuffer& CommandBuffer::draw_indirect(size_t command_count, const Buffer& indirect_buffer) {
		VUK_EARLY_RET();
		if (!_bind_graphics_pipeline_state()) {
			return *this;
		}
		ctx.vkCmdDrawIndirect(command_buffer, indirect_buffer.buffer, (uint32_t)indirect_buffer.offset, (uint32_t)command_count, sizeof(DrawIndirectCommand));
		return *this;
	}

	CommandBuffer& CommandBuffer::draw_indirect(std::span<DrawIndirectCommand> commands) {
		VUK_EARLY_RET();
		if (!_bind_graphics_pipeline_state()) {
			return *this;
		}

		auto res = allocate_buffer(*allocator, { MemoryUsage::eCPUtoGPU, commands.size_bytes(), 1 });
		if (!res) {
			current_error = std::move(res);
			return *this;
		}

		auto& buf = *res;
		memcpy(buf->mapped_ptr, commands.data(), commands.size_bytes());
		ctx.vkCmdDrawIndirect(command_buffer, buf->buffer, (uint32_t)buf->offset, (uint32_t)commands.size(), sizeof(DrawIndirectCommand));
		return *this;
	}

	CommandBuffer& CommandBuffer::draw_indexed(size_t index_count, size_t instance_count, size_t first_index, int32_t vertex_offset, size_t first_instance) {
		VUK_EARLY_RET();
		if (!_bind_graphics_pipeline_state()) {
			return *this;
		}

		ctx.vkCmdDrawIndexed(command_buffer, (uint32_t)index_count, (uint32_t)instance_count, (uint32_t)first_index, vertex_offset, (uint32_t)first_instance);
		return *this;
	}

	CommandBuffer& CommandBuffer::draw_indexed_indirect(size_t command_count, const Buffer& indirect_buffer) {
		VUK_EARLY_RET();
		if (!_bind_graphics_pipeline_state()) {
			return *this;
		}
		ctx.vkCmdDrawIndexedIndirect(
		    command_buffer, indirect_buffer.buffer, (uint32_t)indirect_buffer.offset, (uint32_t)command_count, sizeof(DrawIndexedIndirectCommand));
		return *this;
	}

	CommandBuffer& CommandBuffer::draw_indexed_indirect(std::span<DrawIndexedIndirectCommand> cmds) {
		VUK_EARLY_RET();
		if (!_bind_graphics_pipeline_state()) {
			return *this;
		}

		auto res = allocate_buffer(*allocator, { MemoryUsage::eCPUtoGPU, cmds.size_bytes(), 1 });
		if (!res) {
			current_error = std::move(res);
			return *this;
		}

		auto& buf = *res;
		memcpy(buf->mapped_ptr, cmds.data(), cmds.size_bytes());
		ctx.vkCmdDrawIndexedIndirect(command_buffer, buf->buffer, (uint32_t)buf->offset, (uint32_t)cmds.size(), sizeof(DrawIndexedIndirectCommand));
		return *this;
	}

	CommandBuffer& CommandBuffer::draw_indexed_indirect_count(size_t max_draw_count, const Buffer& indirect_buffer, const Buffer& count_buffer) {
		VUK_EARLY_RET();
		if (!_bind_graphics_pipeline_state()) {
			return *this;
		}
		ctx.vkCmdDrawIndexedIndirectCount(command_buffer,
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
		ctx.vkCmdDispatch(command_buffer, (uint32_t)size_x, (uint32_t)size_y, (uint32_t)size_z);
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

		ctx.vkCmdDispatch(command_buffer, x, y, z);
		return *this;
	}

	CommandBuffer& CommandBuffer::dispatch_invocations_per_pixel(const ImageAttachment& ia,
	                                                             float invocations_per_pixel_scale_x,
	                                                             float invocations_per_pixel_scale_y,
	                                                             float invocations_per_pixel_scale_z) {
		auto extent = ia.extent;

		return dispatch_invocations((uint32_t)std::ceil(invocations_per_pixel_scale_x * extent.width),
		                            (uint32_t)std::ceil(invocations_per_pixel_scale_y * extent.height),
		                            (uint32_t)std::ceil(invocations_per_pixel_scale_z * extent.depth));
	}

	CommandBuffer& CommandBuffer::dispatch_invocations_per_element(const Buffer& buffer, size_t element_size, float invocations_per_element_scale) {
		auto count = (uint32_t)std::ceil(invocations_per_element_scale * idivceil(buffer.size, element_size));

		return dispatch_invocations(count, 1, 1);
	}

	CommandBuffer& CommandBuffer::dispatch_indirect(const Buffer& indirect_buffer) {
		VUK_EARLY_RET();
		if (!_bind_compute_pipeline_state()) {
			return *this;
		}
		ctx.vkCmdDispatchIndirect(command_buffer, indirect_buffer.buffer, indirect_buffer.offset);
		return *this;
	}

	CommandBuffer& CommandBuffer::trace_rays(size_t size_x, size_t size_y, size_t size_z) {
		VUK_EARLY_RET();
		if (!_bind_ray_tracing_pipeline_state()) {
			return *this;
		}

		auto& pipe = *current_ray_tracing_pipeline;

		ctx.vkCmdTraceRaysKHR(
		    command_buffer, &pipe.rgen_region, &pipe.miss_region, &pipe.hit_region, &pipe.call_region, (uint32_t)size_x, (uint32_t)size_y, (uint32_t)size_z);
		return *this;
	}

	CommandBuffer& CommandBuffer::clear_image(const ImageAttachment& src, Clear c) {
		VUK_EARLY_RET();

		assert(rg);

		auto aspect = format_to_aspect(src.format);

		if (!ongoing_render_pass) {
			VkImageSubresourceRange isr = {};
			isr.aspectMask = (VkImageAspectFlags)aspect;
			isr.baseArrayLayer = src.base_layer;
			isr.layerCount = src.layer_count;
			isr.baseMipLevel = src.base_level;
			isr.levelCount = src.level_count;
			if (aspect == ImageAspectFlagBits::eColor) {
				ctx.vkCmdClearColorImage(command_buffer, src.image.image, (VkImageLayout)src.layout, &c.c.color, 1, &isr);
			} else if (aspect & (ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil)) {
				ctx.vkCmdClearDepthStencilImage(command_buffer, src.image.image, (VkImageLayout)src.layout, &c.c.depthStencil, 1, &isr);
			}
		} else {
			VkClearAttachment clr = {};
			clr.aspectMask = (VkImageAspectFlags)aspect;
			clr.clearValue = c.c;
			if (aspect == ImageAspectFlagBits::eColor) {
				auto it = std::find(ongoing_render_pass->color_attachment_ivs.begin(), ongoing_render_pass->color_attachment_ivs.end(), src.image_view);
				assert(it != ongoing_render_pass->color_attachment_ivs.end() && "Color attachment name not found.");
				auto idx = std::distance(ongoing_render_pass->color_attachment_ivs.begin(), it);
				clr.colorAttachment = (uint32_t)idx;
			}
			VkClearRect rect = {};
			rect.baseArrayLayer = src.base_layer;
			rect.layerCount = src.layer_count;
			rect.rect = {
				(int32_t)0,
				(int32_t)0,
				src.extent.width,
				src.extent.height,
			};
			ctx.vkCmdClearAttachments(command_buffer, 1, &clr, 1, &rect);
		}

		return *this;
	}

	CommandBuffer& CommandBuffer::resolve_image(const ImageAttachment& src, const ImageAttachment& dst) {
		VUK_EARLY_RET();
		assert(rg);
		VkImageResolve ir;

		ImageSubresourceLayers isl;
		ImageAspectFlagBits aspect;
		if (dst.format == Format::eD32Sfloat) {
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
		ir.extent = static_cast<Extent3D>(src.extent);

		ctx.vkCmdResolveImage(command_buffer, src.image.image, (VkImageLayout)src.layout, dst.image.image, (VkImageLayout)dst.layout, 1, &ir);

		return *this;
	}

	CommandBuffer& CommandBuffer::blit_image(const ImageAttachment& src, const ImageAttachment& dst, ImageBlit region, Filter filter) {
		VUK_EARLY_RET();
		assert(rg);

		ctx.vkCmdBlitImage(
		    command_buffer, src.image.image, (VkImageLayout)src.layout, dst.image.image, (VkImageLayout)dst.layout, 1, (VkImageBlit*)&region, (VkFilter)filter);

		return *this;
	}

	CommandBuffer& CommandBuffer::copy_image(const ImageAttachment& src, const ImageAttachment& dst, ImageCopy region) {
		VUK_EARLY_RET();
		assert(rg);

		ctx.vkCmdCopyImage(command_buffer, src.image.image, (VkImageLayout)src.layout, dst.image.image, (VkImageLayout)dst.layout, 1, (VkImageCopy*)&region);

		return *this;
	}

	CommandBuffer& CommandBuffer::copy_buffer_to_image(const Buffer& src, const ImageAttachment& dst, BufferImageCopy bic) {
		VUK_EARLY_RET();
		assert(rg);

		ctx.vkCmdCopyBufferToImage(command_buffer, src.buffer, dst.image.image, (VkImageLayout)dst.layout, 1, (VkBufferImageCopy*)&bic);

		return *this;
	}

	CommandBuffer& CommandBuffer::copy_image_to_buffer(const ImageAttachment& src, const Buffer& dst, BufferImageCopy bic) {
		VUK_EARLY_RET();
		assert(rg);

		ctx.vkCmdCopyImageToBuffer(command_buffer, src.image.image, (VkImageLayout)src.layout, dst.buffer, 1, (VkBufferImageCopy*)&bic);

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

		ctx.vkCmdCopyBuffer(command_buffer, src.buffer, dst.buffer, 1, &bc);
		return *this;
	}

	CommandBuffer& CommandBuffer::fill_buffer(const Buffer& dst, uint32_t data) {
		ctx.vkCmdFillBuffer(command_buffer, dst.buffer, dst.offset, dst.size, data);
		return *this;
	}

	CommandBuffer& CommandBuffer::update_buffer(const Buffer& dst, const void* data) {
		ctx.vkCmdUpdateBuffer(command_buffer, dst.buffer, dst.offset, dst.size, data);
		return *this;
	}

	CommandBuffer& CommandBuffer::memory_barrier(Access src_access, Access dst_access) {
		VkMemoryBarrier mb{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		auto src_use = to_use(src_access);
		auto dst_use = to_use(dst_access);
		mb.srcAccessMask = is_readonly_access(src_use) ? 0 : (VkAccessFlags)src_use.access;
		mb.dstAccessMask = (VkAccessFlags)dst_use.access;
		ctx.vkCmdPipelineBarrier(command_buffer, (VkPipelineStageFlags)src_use.stages, (VkPipelineStageFlags)dst_use.stages, {}, 1, &mb, 0, nullptr, 0, nullptr);
		return *this;
	}

	CommandBuffer& CommandBuffer::image_barrier(const ImageAttachment& src, vuk::Access src_acc, vuk::Access dst_acc, uint32_t mip_level, uint32_t level_count) {
		VUK_EARLY_RET();
		assert(rg);

		// TODO: fill these out from attachment
		VkImageSubresourceRange isr = {};
		isr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		isr.baseArrayLayer = 0;
		isr.layerCount = VK_REMAINING_ARRAY_LAYERS;
		isr.baseMipLevel = mip_level;
		isr.levelCount = level_count;
		VkImageMemoryBarrier imb{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		imb.image = src.image.image;
		auto src_use = to_use(src_acc);
		auto dst_use = to_use(dst_acc);
		imb.srcAccessMask = (VkAccessFlags)src_use.access;
		imb.dstAccessMask = (VkAccessFlags)dst_use.access;

		// TODO: questionable
		if (src.layout == ImageLayout::eGeneral) {
			imb.oldLayout = imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		} else {
			imb.oldLayout = (VkImageLayout)src_use.layout;
			imb.newLayout = (VkImageLayout)dst_use.layout;
		}
		imb.subresourceRange = isr;
		ctx.vkCmdPipelineBarrier(command_buffer, (VkPipelineStageFlags)src_use.stages, (VkPipelineStageFlags)dst_use.stages, {}, 0, nullptr, 0, nullptr, 1, &imb);

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

		ctx.vkCmdWriteTimestamp(command_buffer, (VkPipelineStageFlagBits)stage, tsq.pool, tsq.id);
		return *this;
	}

	CommandBuffer& CommandBuffer::build_acceleration_structures(uint32_t info_count,
	                                                            const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
	                                                            const VkAccelerationStructureBuildRangeInfoKHR* const* ppBuildRangeInfos) {
		VUK_EARLY_RET();

		ctx.vkCmdBuildAccelerationStructuresKHR(command_buffer, info_count, pInfos, ppBuildRangeInfos);
		return *this;
	}

	Result<void> CommandBuffer::result() {
		return std::move(current_error);
	}

	VkCommandBuffer CommandBuffer::bind_compute_state() {
		auto result = _bind_compute_pipeline_state();
		assert(result);
		return command_buffer;
	}
	VkCommandBuffer CommandBuffer::bind_graphics_state() {
		auto result = _bind_graphics_pipeline_state();
		assert(result);
		return command_buffer;
	}
	VkCommandBuffer CommandBuffer::bind_ray_tracing_state() {
		auto result = _bind_ray_tracing_pipeline_state();
		assert(result);
		return command_buffer;
	}

	std::string to_string(vuk::DescriptorType dt) {
		switch (dt) {
		case DescriptorType::eUniformBuffer:
			return "Uniform Buffer";
		case DescriptorType::eStorageBuffer:
			return "Storage Buffer";
		case DescriptorType::eSampledImage:
			return "Sampled Image";
		case DescriptorType::eSampler:
			return "Sampler";
		case DescriptorType::eCombinedImageSampler:
			return "Combined Image-Sampler";
		case DescriptorType::eStorageImage:
			return "Storage Image";
		case DescriptorType::eAccelerationStructureKHR:
			return "Acceleration Structure";
		}
		assert(0);
		return "";
	}

	bool CommandBuffer::_bind_state(PipeType pipe_type) {
		VkPipelineLayout current_layout;
		switch (pipe_type) {
		case PipeType::eGraphics:
			current_layout = current_graphics_pipeline->pipeline_layout;
			break;
		case PipeType::eCompute:
			current_layout = current_compute_pipeline->pipeline_layout;
			break;
		case PipeType::eRayTracing:
			current_layout = current_ray_tracing_pipeline->pipeline_layout;
			break;
		}
		VkPipelineBindPoint bind_point;
		switch (pipe_type) {
		case PipeType::eGraphics:
			bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
			break;
		case PipeType::eCompute:
			bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
			break;
		case PipeType::eRayTracing:
			bind_point = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
			break;
		}

		for (auto& pcr : pcrs) {
			void* data = push_constant_buffer + pcr.offset;
			ctx.vkCmdPushConstants(command_buffer, current_layout, pcr.stageFlags, pcr.offset, pcr.size, data);
		}
		pcrs.clear();

		auto sets_mask = sets_to_bind.to_ulong();
		auto persistent_sets_mask = persistent_sets_to_bind.to_ulong();
		uint64_t highest_undisturbed_binding_required = 0;
		uint64_t lowest_disturbed_binding = VUK_MAX_SETS;
		for (size_t set_index = 0; set_index < VUK_MAX_SETS; set_index++) {
			bool set_to_bind = sets_mask & (1ULL << set_index);
			bool persistent_set_to_bind = persistent_sets_mask & (1ULL << set_index);

			VkDescriptorSetLayout pipeline_set_layout;
			DescriptorSetLayoutAllocInfo* ds_layout_alloc_info;
			switch (pipe_type) {
			case PipeType::eGraphics:
				ds_layout_alloc_info = &current_graphics_pipeline->layout_info[set_index];
				break;
			case PipeType::eCompute:
				ds_layout_alloc_info = &current_compute_pipeline->layout_info[set_index];
				break;
			case PipeType::eRayTracing:
				ds_layout_alloc_info = &current_ray_tracing_pipeline->layout_info[set_index];
				break;
			}
			pipeline_set_layout = ds_layout_alloc_info->layout;

			// binding validation
			if (pipeline_set_layout != VK_NULL_HANDLE) { // set in the layout
				bool is_used;
				VUK_SB_TEST(sets_used, set_index, is_used);
				if (!is_used && !set_to_bind && !persistent_set_to_bind) { // never set in the cbuf & not requested to bind now
					fmt::print(stderr, "Shader declares (set: {}), but never set in CommandBuffer or disturbed by a previous set composition or binding.", set_index);
					assert(false && "Shader declares set, but never set in CommandBuffer or disturbed by a previous set composition or binding (see stderr).");
					return false;
				} else if (!set_to_bind && !persistent_set_to_bind) { // but not requested to bind now
					// validate that current set is compatible (== same set layout)
					assert(set_layouts_used[set_index] == pipeline_set_layout && "Previously bound set is incompatible with currently bound pipeline.");
					// this set is compatible, but we require it to be undisturbed
					highest_undisturbed_binding_required = std::max(highest_undisturbed_binding_required, set_index);
					// detect if during this binding we disturb a set that we depend on
					assert(highest_undisturbed_binding_required < lowest_disturbed_binding &&
					       "Set composition disturbs previously bound set that is not recomposed or bound for this drawcall.");
					continue;
				}
			} else {                                         // not set in the layout
				if (!set_to_bind && !persistent_set_to_bind) { // not requested to bind now, noop
					continue;
				} else { // requested to bind now
					fmt::fprintf(stderr, "Attempting to bind descriptor(s)/set to (set: {}) not declared in shader.", set_index);
					fmt::print(stderr, "Attempting to bind descriptor(s)/set to (set: {}) not declared in shader.", set_index);
					assert(false && "Attempting to bind descriptor(s)/set to set not declared in shader (see stderr).");
					return false;
				}
			}
			// if the newly bound DS has a different set layout than the previously bound set, then it disturbs all the sets at higher indices
			bool is_disturbing = set_layouts_used[set_index] != pipeline_set_layout;
			if (is_disturbing) {
				lowest_disturbed_binding = std::min(lowest_disturbed_binding, set_index + 1);
			}

			switch (pipe_type) {
			case PipeType::eGraphics:
				set_bindings[set_index].layout_info = &current_graphics_pipeline->layout_info[set_index];
				break;
			case PipeType::eCompute:
				set_bindings[set_index].layout_info = &current_compute_pipeline->layout_info[set_index];
				break;
			case PipeType::eRayTracing:
				set_bindings[set_index].layout_info = &current_ray_tracing_pipeline->layout_info[set_index];
				break;
			}

			if (!persistent_set_to_bind) {
				std::vector<VkDescriptorSetLayoutBinding>* ppipeline_set_bindings;
				DescriptorSetLayoutCreateInfo* dslci;
				switch (pipe_type) {
				case PipeType::eGraphics:
					dslci = &current_graphics_pipeline->base->dslcis[set_index];
					break;
				case PipeType::eCompute:
					dslci = &current_compute_pipeline->base->dslcis[set_index];
					break;
				case PipeType::eRayTracing:
					dslci = &current_ray_tracing_pipeline->base->dslcis[set_index];
					break;
				}
				ppipeline_set_bindings = &dslci->bindings;
				auto& pipeline_set_bindings = *ppipeline_set_bindings;
				auto sb = set_bindings[set_index].finalize(dslci->used_bindings);

				for (uint64_t j = 0; j < pipeline_set_bindings.size(); j++) {
					auto& pipe_binding = pipeline_set_bindings[j];
					auto& cbuf_binding = sb.bindings[pipeline_set_bindings[j].binding];

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
						fmt::print(stderr,
						           "Shader has declared (set: {}, binding: {}) combined image-sampler, but only sampler was bound.",
						           set_index,
						           pipeline_set_bindings[j].binding);
						assert(false && "Descriptor is combined image-sampler, but only sampler was bound.");
						return false;
					}
					if (cbuf_dtype == DescriptorType::eSampledImage && pipe_dtype == DescriptorType::eCombinedImageSampler) {
						fmt::print(stderr,
						           "Shader has declared (set: {}, binding: {}) combined image-sampler, but only image was bound.",
						           set_index,
						           pipeline_set_bindings[j].binding);
						assert(false && "Descriptor is combined image-sampler, but only image was bound.");
						return false;
					}
					if (pipe_dtype != cbuf_dtype) {
						bool optional;
						VUK_SB_TEST(dslci->optional, j, optional);
						if (optional) { // this was an optional binding with a mismatched or missing bound resource -> forgo writing
							VUK_SB_SET(sb.used, j, false);
						} else {
							if (cbuf_dtype == vuk::DescriptorType(127)) {
								fmt::print(stderr, "Shader has declared (set: {}, binding: {}) that was not bound.", set_index, pipeline_set_bindings[j].binding);
								assert(false && "Descriptor layout contains binding that was not bound (see stderr).");
							} else {
								fmt::print(stderr,
								           "Shader has declared (set: {}, binding: {}) with type <{}> - tried to bind <{}>.",
								           set_index,
								           pipeline_set_bindings[j].binding,
								           to_string(pipe_dtype),
								           to_string(cbuf_dtype));
								assert(false && "Attempting to bind the wrong descriptor type (see stderr).");
							}
							return false;
						}
					}
				}

				auto strategy = ds_strategy_flags.m_mask == 0 ? DescriptorSetStrategyFlagBits::eCommon : ds_strategy_flags;
				Unique<DescriptorSet> ds;
				if (strategy & DescriptorSetStrategyFlagBits::ePerLayout) {
					if (auto ret = allocator->allocate_descriptor_sets_with_value(std::span{ &*ds, 1 }, std::span{ &sb, 1 }); !ret) {
						current_error = std::move(ret);
						return false;
					}
				} else if (strategy & DescriptorSetStrategyFlagBits::eCommon) {
					if (auto ret = allocator->allocate_descriptor_sets(std::span{ &*ds, 1 }, std::span{ ds_layout_alloc_info, 1 }); !ret) {
						current_error = std::move(ret);
						return false;
					}

					auto& cinfo = sb;
					auto mask = cinfo.used.to_ulong();
					uint32_t leading_ones = num_leading_ones((uint32_t)mask);
					VkWriteDescriptorSet writes[VUK_MAX_BINDINGS];
					int j = 0;
					for (uint32_t i = 0; i < leading_ones; i++, j++) {
						bool used;
						VUK_SB_TEST(cinfo.used, i, used);
						if (!used) {
							j--;
							continue;
						}
						auto& write = writes[j];
						write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
						auto& binding = cinfo.bindings[i];
						write.descriptorType = DescriptorBinding::vk_descriptor_type(binding.type);
						write.dstArrayElement = 0;
						write.descriptorCount = 1;
						write.dstBinding = i;
						write.dstSet = ds->descriptor_set;
						switch (binding.type) {
						case DescriptorType::eUniformBuffer:
						case DescriptorType::eStorageBuffer:
							write.pBufferInfo = &binding.buffer;
							break;
						case DescriptorType::eSampledImage:
						case DescriptorType::eSampler:
						case DescriptorType::eCombinedImageSampler:
						case DescriptorType::eStorageImage:
							write.pImageInfo = &binding.image.dii;
							break;
						case DescriptorType::eAccelerationStructureKHR:
							binding.as.wds = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
							binding.as.wds.accelerationStructureCount = 1;
							binding.as.wds.pAccelerationStructures = &binding.as.as;
							write.pNext = &binding.as.wds;
							break;
						default:
							assert(0);
						}
					}
					ctx.vkUpdateDescriptorSets(allocator->get_context().device, j, writes, 0, nullptr);
				} else {
					assert(0 && "Unimplemented DS strategy");
				}

				ctx.vkCmdBindDescriptorSets(command_buffer, bind_point, current_layout, (uint32_t)set_index, 1, &ds->descriptor_set, 0, nullptr);
				set_layouts_used[set_index] = ds->layout_info.layout;
			} else {
				ctx.vkCmdBindDescriptorSets(command_buffer, bind_point, current_layout, (uint32_t)set_index, 1, &persistent_sets[set_index].first, 0, nullptr);
				set_layouts_used[set_index] = persistent_sets[set_index].second;
			}
		}
		auto sets_bound = sets_to_bind | persistent_sets_to_bind;            // these sets we bound freshly, valid
		for (uint64_t i = lowest_disturbed_binding; i < VUK_MAX_SETS; i++) { // clear the slots where the binding was disturbed
			VUK_SB_SET(sets_used, i, false);
		}
		sets_used = sets_used | sets_bound;
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
			for (auto& sc : pi.base->reflection_info.spec_constants) {
				auto it = spec_map_entries.find(sc.binding);
				if (it != spec_map_entries.end()) {
					auto& map_e = it->second;
					unsigned size = map_e.is_double ? (unsigned)sizeof(double) : 4;
					assert(pi.specialization_map_entries.size() < VUK_MAX_SPECIALIZATIONCONSTANT_RANGES);
					pi.specialization_map_entries.push_back(VkSpecializationMapEntry{ sc.binding, offset, size });
					assert(offset + size < VUK_MAX_SPECIALIZATIONCONSTANT_SIZE);
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

			current_compute_pipeline = ComputePipelineInfo{};
			allocator->allocate_compute_pipelines(std::span{ &current_compute_pipeline.value(), 1 }, std::span{ &pi, 1 });
			// drop pipeline immediately
			allocator->deallocate(std::span{ &current_compute_pipeline.value(), 1 });

			ctx.vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, current_compute_pipeline->pipeline);
			next_compute_pipeline = nullptr;
		}

		return _bind_state(PipeType::eCompute);
	}

	template<class T>
	void write(std::byte*& data_ptr, const T& data) {
		memcpy(data_ptr, &data, sizeof(T));
		data_ptr += sizeof(T);
	};

	bool CommandBuffer::_bind_graphics_pipeline_state() {
		if (next_pipeline) {
			GraphicsPipelineInstanceCreateInfo pi;
			pi.base = next_pipeline;
			pi.render_pass = ongoing_render_pass->render_pass;
			pi.dynamic_state_flags = dynamic_state_flags.m_mask;
			auto& records = pi.records;
			if (ongoing_render_pass->subpass > 0) {
				records.nonzero_subpass = true;
				pi.extended_size += sizeof(uint8_t);
			}
			pi.topology = (VkPrimitiveTopology)topology;
			pi.primitive_restart_enable = false;

			// VERTEX INPUT
			Bitset<VUK_MAX_ATTRIBUTES> used_bindings = {};
			if (pi.base->reflection_info.attributes.size() > 0) {
				records.vertex_input = true;
				for (unsigned i = 0; i < pi.base->reflection_info.attributes.size(); i++) {
					auto& reflected_att = pi.base->reflection_info.attributes[i];
					assert(set_attribute_descriptions.test(reflected_att.location) && "Pipeline expects attribute, but was never set in command buffer.");
					VUK_SB_SET(used_bindings, attribute_descriptions[reflected_att.location].binding, true);
				}

				pi.extended_size += (uint16_t)pi.base->reflection_info.attributes.size() * sizeof(GraphicsPipelineInstanceCreateInfo::VertexInputAttributeDescription);
				pi.extended_size += sizeof(uint8_t);
				uint64_t count;
				VUK_SB_COUNT(used_bindings, count);
				pi.extended_size += (uint16_t)count * sizeof(GraphicsPipelineInstanceCreateInfo::VertexInputBindingDescription);
			}

			// BLEND STATE
			// attachmentCount says how many attachments
			pi.attachmentCount = (uint8_t)ongoing_render_pass->color_attachments.size();
			bool rasterization = ongoing_render_pass->depth_stencil_attachment || pi.attachmentCount > 0;

			if (pi.attachmentCount > 0) {
				uint64_t count;
				VUK_SB_COUNT(set_color_blend_attachments, count);
				assert(count > 0 && "If a pass has a color attachment, you must set at least one color blend state.");
				records.broadcast_color_blend_attachment_0 = broadcast_color_blend_attachment_0;

				if (broadcast_color_blend_attachment_0) {
					bool set;
					VUK_SB_TEST(set_color_blend_attachments, 0, set);
					assert(set && "Broadcast turned on, but no blend state set.");
					if (color_blend_attachments[0] != PipelineColorBlendAttachmentState{}) {
						records.color_blend_attachments = true;
						pi.extended_size += sizeof(GraphicsPipelineInstanceCreateInfo::PipelineColorBlendAttachmentState);
					}
				} else {
					assert(count >= pi.attachmentCount && "If color blend state is not broadcast, you must set it for each color attachment.");
					records.color_blend_attachments = true;
					pi.extended_size += (uint16_t)(pi.attachmentCount * sizeof(GraphicsPipelineInstanceCreateInfo::PipelineColorBlendAttachmentState));
				}
			}

			records.logic_op = false; // TODO: logic op unsupported
			if (blend_constants && !(dynamic_state_flags & DynamicStateFlagBits::eBlendConstants)) {
				records.blend_constants = true;
				pi.extended_size += sizeof(float) * 4;
			}

			unsigned spec_const_size = 0;
			Bitset<VUK_MAX_SPECIALIZATIONCONSTANT_RANGES> set_constants = {};
			assert(pi.base->reflection_info.spec_constants.size() < VUK_MAX_SPECIALIZATIONCONSTANT_RANGES);
			if (spec_map_entries.size() > 0 && pi.base->reflection_info.spec_constants.size() > 0) {
				for (unsigned i = 0; i < pi.base->reflection_info.spec_constants.size(); i++) {
					auto& sc = pi.base->reflection_info.spec_constants[i];
					auto size = sc.type == Program::Type::edouble ? sizeof(double) : 4;
					auto it = spec_map_entries.find(sc.binding);
					if (it != spec_map_entries.end()) {
						spec_const_size += (uint32_t)size;
						VUK_SB_SET(set_constants, i, true);
					}
				}
				records.specialization_constants = true;
				assert(spec_const_size < VUK_MAX_SPECIALIZATIONCONSTANT_SIZE);
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
					pi.extended_size += sizeof(GraphicsPipelineInstanceCreateInfo::RasterizationState);
				}
			}

			if (conservative_state) {
				records.conservative_rasterization_enabled = true;
				pi.extended_size += sizeof(GraphicsPipelineInstanceCreateInfo::ConservativeState);
			}

			if (ongoing_render_pass->depth_stencil_attachment) {
				assert(depth_stencil_state && "If a pass has a depth/stencil attachment, you must set the depth/stencil state.");

				records.depth_stencil = true;
				pi.extended_size += sizeof(GraphicsPipelineInstanceCreateInfo::Depth);

				if (depth_stencil_state->stencilTestEnable) {
					records.stencil_state = true;
					pi.extended_size += sizeof(GraphicsPipelineInstanceCreateInfo::Stencil);
				}

				if (depth_stencil_state->depthBoundsTestEnable) {
					records.depth_bounds = true;
					pi.extended_size += sizeof(GraphicsPipelineInstanceCreateInfo::DepthBounds);
				}
			}

			if (ongoing_render_pass->samples != SampleCountFlagBits::e1) {
				records.more_than_one_sample = true;
				pi.extended_size += sizeof(GraphicsPipelineInstanceCreateInfo::Multisample);
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
			if (ongoing_render_pass->subpass > 0) {
				write<uint8_t>(data_ptr, ongoing_render_pass->subpass);
			}

			if (records.vertex_input) {
				for (unsigned i = 0; i < pi.base->reflection_info.attributes.size(); i++) {
					auto& reflected_att = pi.base->reflection_info.attributes[i];
					auto& att = attribute_descriptions[reflected_att.location];
					GraphicsPipelineInstanceCreateInfo::VertexInputAttributeDescription viad{
						.format = att.format, .offset = att.offset, .location = (uint8_t)att.location, .binding = (uint8_t)att.binding
					};
					write(data_ptr, viad);
				}
				uint64_t count;
				VUK_SB_COUNT(used_bindings, count);
				write<uint8_t>(data_ptr, (uint8_t)count);
				for (unsigned i = 0; i < VUK_MAX_ATTRIBUTES; i++) {
					bool used;
					VUK_SB_TEST(used_bindings, i, used);
					if (used) {
						auto& bin = binding_descriptions[i];
						GraphicsPipelineInstanceCreateInfo::VertexInputBindingDescription vibd{ .stride = bin.stride,
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
					GraphicsPipelineInstanceCreateInfo::PipelineColorBlendAttachmentState pcba{ .blendEnable = cba.blendEnable,
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
					bool set;
					VUK_SB_TEST(set_constants, i, set);
					if (set) {
						auto& sc = pi.base->reflection_info.spec_constants[i];
						auto size = sc.type == Program::Type::edouble ? sizeof(double) : 4;
						auto& map_e = spec_map_entries.find(sc.binding)->second;
						memcpy(data_ptr, map_e.data, size);
						data_ptr += size;
					}
				}
			}

			if (records.non_trivial_raster_state) {
				GraphicsPipelineInstanceCreateInfo::RasterizationState rs{ .depthClampEnable = (bool)rasterization_state->depthClampEnable,
					                                                         .rasterizerDiscardEnable = (bool)rasterization_state->rasterizerDiscardEnable,
					                                                         .polygonMode = (uint8_t)rasterization_state->polygonMode,
					                                                         .frontFace = (uint8_t)rasterization_state->frontFace };
				write(data_ptr, rs);
				// TODO: support depth bias
			}

			if (records.conservative_rasterization_enabled) {
				GraphicsPipelineInstanceCreateInfo::ConservativeState cs{ .conservativeMode = (uint8_t)conservative_state->mode,
					                                                        .overestimationAmount = conservative_state->overestimationAmount };
				write(data_ptr, cs);
			}

			if (ongoing_render_pass->depth_stencil_attachment) {
				GraphicsPipelineInstanceCreateInfo::Depth ds = { .depthTestEnable = (bool)depth_stencil_state->depthTestEnable,
					                                               .depthWriteEnable = (bool)depth_stencil_state->depthWriteEnable,
					                                               .depthCompareOp = (uint8_t)depth_stencil_state->depthCompareOp };
				write(data_ptr, ds);

				if (depth_stencil_state->stencilTestEnable) {
					GraphicsPipelineInstanceCreateInfo::Stencil ss = { .front = depth_stencil_state->front, .back = depth_stencil_state->back };
					write(data_ptr, ss);
				}

				if (depth_stencil_state->depthBoundsTestEnable) {
					GraphicsPipelineInstanceCreateInfo::DepthBounds dps = { .minDepthBounds = depth_stencil_state->minDepthBounds,
						                                                      .maxDepthBounds = depth_stencil_state->maxDepthBounds };
					write(data_ptr, dps);
				}
			}

			if (ongoing_render_pass->samples != SampleCountFlagBits::e1) {
				GraphicsPipelineInstanceCreateInfo::Multisample ms{ .rasterization_samples = (uint32_t)ongoing_render_pass->samples };
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
			current_graphics_pipeline = GraphicsPipelineInfo{};
			allocator->allocate_graphics_pipelines(std::span{ &current_graphics_pipeline.value(), 1 }, std::span{ &pi, 1 });
			if (!pi.is_inline()) {
				delete pi.extended_data;
			}
			// drop pipeline immediately
			allocator->deallocate(std::span{ &current_graphics_pipeline.value(), 1 });

			ctx.vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, current_graphics_pipeline->pipeline);
			next_pipeline = nullptr;
		}
		return _bind_state(PipeType::eGraphics);
	}
	bool CommandBuffer::_bind_ray_tracing_pipeline_state() {
		if (next_ray_tracing_pipeline) {
			RayTracingPipelineInstanceCreateInfo pi;
			pi.base = next_ray_tracing_pipeline;

			bool empty = true;
			unsigned offset = 0;
			for (auto& sc : pi.base->reflection_info.spec_constants) {
				auto it = spec_map_entries.find(sc.binding);
				if (it != spec_map_entries.end()) {
					auto& map_e = it->second;
					unsigned size = map_e.is_double ? (unsigned)sizeof(double) : 4;
					assert(pi.specialization_map_entries.size() < VUK_MAX_SPECIALIZATIONCONSTANT_RANGES);
					pi.specialization_map_entries.push_back(VkSpecializationMapEntry{ sc.binding, offset, size });
					assert(offset + size < VUK_MAX_SPECIALIZATIONCONSTANT_SIZE);
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

			current_ray_tracing_pipeline = RayTracingPipelineInfo{};
			allocator->allocate_ray_tracing_pipelines(std::span{ &current_ray_tracing_pipeline.value(), 1 }, std::span{ &pi, 1 });
			// drop pipeline immediately
			allocator->deallocate(std::span{ &current_ray_tracing_pipeline.value(), 1 });

			ctx.vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, current_ray_tracing_pipeline->pipeline);
			next_ray_tracing_pipeline = nullptr;
		}

		return _bind_state(PipeType::eRayTracing);
	}
} // namespace vuk
