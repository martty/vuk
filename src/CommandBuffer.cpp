#include "vuk/CommandBuffer.hpp"
#include "vuk/Context.hpp"
#include "vuk/RenderGraph.hpp"
#include "RenderGraphUtil.hpp"
#include <cstdio>

namespace vuk {
	uint32_t Ignore::to_size() {
		if (bytes != 0) return bytes;
		return format_to_texel_block_size(format);
	}

	FormatOrIgnore::FormatOrIgnore(vuk::Format format) : ignore(false), format(format), size(format_to_texel_block_size(format)) {
	}
	FormatOrIgnore::FormatOrIgnore(Ignore ign) : ignore(true), format(ign.format), size(ign.to_size()) {
	}

	CommandBuffer::CommandBuffer(vuk::PerThreadContext& ptc) : ptc(ptc) {
		command_buffer = ptc.acquire_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
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
		return rg->get_resource_image(n).image;
	}

	vuk::ImageView CommandBuffer::get_resource_image_view(Name n) const {
		assert(rg);
		return rg->get_resource_image(n).iv;
	}

	CommandBuffer& CommandBuffer::set_viewport(unsigned index, vuk::Viewport vp) {
		vkCmdSetViewport(command_buffer, 0, 1, (VkViewport*)&vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_viewport(unsigned index, Rect2D area, float min_depth, float max_depth) {
		vuk::Viewport vp;
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

		vkCmdSetViewport(command_buffer, 0, 1, (VkViewport*)&vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_scissor(unsigned index, Rect2D area) {
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
		vkCmdSetScissor(command_buffer, 0, 1, &vp);
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
		set_bindings[set].bindings[binding].image = vuk::DescriptorImageInfo(ptc.acquire_sampler(sci), iv, il);
		set_bindings[set].used.set(binding);

		return *this;
	}

	CommandBuffer& CommandBuffer::bind_sampled_image(unsigned set, unsigned binding, const vuk::Texture& texture, vuk::SamplerCreateInfo sampler_create_info, vuk::ImageLayout il) {
		return bind_sampled_image(set, binding, *texture.view, sampler_create_info, il);
	}

	CommandBuffer& CommandBuffer::bind_sampled_image(unsigned set, unsigned binding, Name name, vuk::SamplerCreateInfo sampler_create_info) {
		assert(rg);

		auto layout = rg->is_resource_image_in_general_layout(name, current_pass) ? vuk::ImageLayout::eGeneral : vuk::ImageLayout::eShaderReadOnlyOptimal;

		return bind_sampled_image(set, binding, rg->get_resource_image(name).iv, sampler_create_info, layout);
	}

	CommandBuffer& CommandBuffer::bind_sampled_image(unsigned set, unsigned binding, Name name, vuk::ImageViewCreateInfo ivci, vuk::SamplerCreateInfo sampler_create_info) {
		assert(rg);
		ivci.image = rg->get_resource_image(name).image;
		if (ivci.format == vuk::Format{}) {
			ivci.format = vuk::Format(rg->get_resource_image(name).description.format);
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

		vuk::Unique<vuk::ImageView> iv = ptc.ctx.create_image_view(ivci);
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

	void CommandBuffer::_inline_compute_helper(const char* source, std::vector<SymType> arg_types, const char * file) {
		// parse and strip C++ header
		std::string_view sv(source);
		auto end_of_header = sv.find("{");
		auto body = sv.substr(end_of_header);
		auto header = sv.substr(0, end_of_header);
		auto att_start = header.find_last_of("[[");
		auto att_end = header.find_last_of("]]");
		unsigned sizes[3] = {1,1,1};
		if (att_start != std::string::npos) {
			std::string_view atts = header.substr(att_start + 1);
			atts = atts.substr(atts.find_first_not_of(' '));
			sscanf(atts.data(), "local_size(%d,%d,%d)", &sizes[0], &sizes[1], &sizes[2]);
			printf("");
		}
		std::string glsl = "#version 460\n#pragma shader_stage(compute)\n\nlayout(local_size_x = ";
		glsl += std::to_string(sizes[0]) + ", local_size_y = " + std::to_string(sizes[1]) + ", local_size_z = " + std::to_string(sizes[2]) + ") in;\n";
		// extract arg names
		std::vector<std::string> arg_names;
		header = header.substr(0, att_start - 1);
		header = header.substr(header.find_first_of('(') + 1);
		header = header.substr(0, header.find_last_of(')'));
		size_t last_comma = std::string::npos;
		do {
			auto prev_comma = last_comma + 1;
			last_comma = header.find_first_of(',', prev_comma);
			std::string_view decl = header.substr(prev_comma, last_comma);
			// parse argument declaration
			auto last_space = decl.find_last_of(' ');
			std::string_view var_name = decl.substr(last_space + 1);
			arg_names.push_back(std::string(var_name));
		} while (last_comma != std::string::npos);
		
		// generate header
		unsigned binding_counter = 0;
		// descriptors
		bool emit_pc = false;
		for (unsigned i = 0; i < arg_types.size(); i++) {
			auto& arg = arg_types[i];
			if (arg.type != SymType::Type::eDescriptor) {
				if (arg.type == SymType::Type::ePushConstant) {
					emit_pc = true;
				}
				continue;
			}
			std::string& arg_name = arg_names[i];
			std::string block_name = "_" + arg_name + "_";
			std::string dst_buf;
			unsigned chars = snprintf(nullptr, 0, arg.declaration.c_str(), binding_counter, block_name.c_str(), arg.GLSL_type.c_str(), arg_name.c_str());
			dst_buf.resize(chars);
			sprintf(dst_buf.data(), arg.declaration.c_str(), binding_counter, block_name.c_str(), arg.GLSL_type.c_str(), arg_name.c_str());
			
			glsl += dst_buf;
			binding_counter++;
		}
		// push consts
		if (emit_pc) {
			glsl += "layout(push_constant) uniform _pc_ {\n";
			for (unsigned i = 0; i < arg_types.size(); i++) {
				auto& arg = arg_types[i];
				if (arg.type != SymType::Type::ePushConstant) {
					continue;
				}
				std::string& arg_name = arg_names[i];
				std::string dst_buf;
				unsigned chars = snprintf(nullptr, 0, arg.declaration.c_str(), arg.GLSL_type.c_str(), arg_name.c_str());
				dst_buf.resize(chars);
				sprintf(dst_buf.data(), arg.declaration.c_str(), arg.GLSL_type.c_str(), arg_name.c_str());

				glsl += dst_buf;
				binding_counter++;
			}
			glsl += "};\n";
		}
		// finish text
		glsl += "void main()";
		glsl += body;
		// create compute pipeline
		vuk::ComputePipelineCreateInfo pci;
		pci.add_shader(glsl, file);
		auto pipeinfo = ptc.ctx.get_pipeline(pci);
		bind_compute_pipeline(pipeinfo);
	}

	SecondaryCommandBuffer CommandBuffer::begin_secondary() {
		auto nptc = new vuk::PerThreadContext(ptc.ifc.begin());
		auto scbuf = nptc->acquire_command_buffer(VK_COMMAND_BUFFER_LEVEL_SECONDARY);
		VkCommandBufferBeginInfo cbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
									 .flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT | VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
		VkCommandBufferInheritanceInfo cbii{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO };
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

	void CommandBuffer::clear_image(Name src, Clear c) {
		// TODO: depth images
		assert(rg);
		auto att = rg->get_resource_image(src);
		auto layout = rg->is_resource_image_in_general_layout(src, current_pass) ? vuk::ImageLayout::eGeneral : vuk::ImageLayout::eTransferDstOptimal;
		VkImageSubresourceRange isr = {};
		isr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		isr.baseArrayLayer = 0;
		isr.layerCount = VK_REMAINING_ARRAY_LAYERS;
		isr.baseMipLevel = 0;
		isr.levelCount = VK_REMAINING_MIP_LEVELS;
		vkCmdClearColorImage(command_buffer, att.image, (VkImageLayout)layout, &c.c.color, 1, &isr);
	}

	void CommandBuffer::resolve_image(Name src, Name dst) {
		assert(rg);
		VkImageResolve ir;
		auto src_image = rg->get_resource_image(src).image;
		auto dst_image = rg->get_resource_image(dst).image;
		vuk::ImageSubresourceLayers isl;
		vuk::ImageAspectFlagBits aspect;
		if (rg->get_resource_image(dst).description.format == (VkFormat)vuk::Format::eD32Sfloat) {
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
		ir.extent = static_cast<vuk::Extent3D>(rg->get_resource_image(src).extents.extent);

		auto src_layout = rg->is_resource_image_in_general_layout(src, current_pass) ? vuk::ImageLayout::eGeneral : vuk::ImageLayout::eTransferSrcOptimal;
		auto dst_layout = rg->is_resource_image_in_general_layout(dst, current_pass) ? vuk::ImageLayout::eGeneral : vuk::ImageLayout::eTransferDstOptimal;

		vkCmdResolveImage(command_buffer, src_image, (VkImageLayout)src_layout, dst_image, (VkImageLayout)dst_layout, 1, &ir);
	}

	void CommandBuffer::blit_image(Name src, Name dst, vuk::ImageBlit region, vuk::Filter filter) {
		assert(rg);
		auto src_image = rg->get_resource_image(src).image;
		auto dst_image = rg->get_resource_image(dst).image;

		auto src_layout = rg->is_resource_image_in_general_layout(src, current_pass) ? vuk::ImageLayout::eGeneral : vuk::ImageLayout::eTransferSrcOptimal;
		auto dst_layout = rg->is_resource_image_in_general_layout(dst, current_pass) ? vuk::ImageLayout::eGeneral : vuk::ImageLayout::eTransferDstOptimal;

		vkCmdBlitImage(command_buffer, src_image, (VkImageLayout)src_layout, dst_image, (VkImageLayout)dst_layout, 1, (VkImageBlit*)&region, (VkFilter)filter);
	}

	void CommandBuffer::copy_image_to_buffer(Name src, Name dst, vuk::BufferImageCopy bic) {
		assert(rg);
		auto src_batt = rg->get_resource_image(src);
		auto dst_bbuf = rg->get_resource_buffer(dst);

		bic.bufferOffset += dst_bbuf.buffer.offset;

		auto src_layout = rg->is_resource_image_in_general_layout(src, current_pass) ? vuk::ImageLayout::eGeneral : vuk::ImageLayout::eTransferSrcOptimal;
		vkCmdCopyImageToBuffer(command_buffer, src_batt.image, (VkImageLayout)src_layout, dst_bbuf.buffer.buffer, 1, (VkBufferImageCopy*)&bic);
	}

	void CommandBuffer::image_barrier(Name src, vuk::Access src_acc, vuk::Access dst_acc) {
		assert(rg);
		auto att = rg->get_resource_image(src);

		VkImageSubresourceRange isr = {};
		isr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		isr.baseArrayLayer = 0;
		isr.layerCount = VK_REMAINING_ARRAY_LAYERS;
		isr.baseMipLevel = 0;
		isr.levelCount = VK_REMAINING_MIP_LEVELS;
		VkImageMemoryBarrier imb{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		imb.image = att.image;
		auto src_use = to_use(src_acc);
		auto dst_use = to_use(dst_acc);
		imb.srcAccessMask = (VkAccessFlags)src_use.access;
		imb.dstAccessMask = (VkAccessFlags)dst_use.access;
		if (rg->is_resource_image_in_general_layout(src, current_pass)) {
			imb.oldLayout = imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		} else {
			imb.oldLayout = (VkImageLayout)src_use.layout;
			imb.newLayout = (VkImageLayout)dst_use.layout;
		}
		imb.subresourceRange = isr;
		vkCmdPipelineBarrier(command_buffer, (VkPipelineStageFlags)src_use.stages, (VkPipelineStageFlags)dst_use.stages, {}, 0, nullptr, 0, nullptr, 1, &imb);
	}

	void CommandBuffer::_bind_state(bool graphics) {
		for (auto& pcr : pcrs) {
			void* data = push_constant_buffer.data() + pcr.offset;
			vkCmdPushConstants(command_buffer, graphics ? current_pipeline->pipeline_layout : current_compute_pipeline->pipeline_layout, pcr.stageFlags, pcr.offset, pcr.size, data);
		}
		pcrs.clear();

		for (unsigned i = 0; i < VUK_MAX_SETS; i++) {
			bool persistent = persistent_sets_used[i];
			if (!sets_used[i] && !persistent_sets_used[i])
				continue;
			set_bindings[i].layout_info = graphics ? current_pipeline->layout_info[i] : current_compute_pipeline->layout_info[i];
			if (!persistent) {
				auto ds = ptc.acquire_descriptorset(set_bindings[i]);
				vkCmdBindDescriptorSets(command_buffer, graphics ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE, graphics ? current_pipeline->pipeline_layout : current_compute_pipeline->pipeline_layout, i, 1, &ds.descriptor_set, 0, nullptr);
			} else {
				vkCmdBindDescriptorSets(command_buffer, graphics ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE, graphics ? current_pipeline->pipeline_layout : current_compute_pipeline->pipeline_layout, i, 1, &persistent_sets[i], 0, nullptr);
			}
			set_bindings[i].used.reset();
		}
		sets_used.reset();
		persistent_sets_used.reset();
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
			pi.attribute_descriptions = std::move(attribute_descriptions);
			pi.binding_descriptions = std::move(binding_descriptions);
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

			pi.multisample_state.rasterizationSamples = (VkSampleCountFlagBits)ongoing_renderpass->samples;

			pi.color_blend_attachments = pi.base->color_blend_attachments;
			// last blend attachment is replicated to cover all attachments
			if (pi.color_blend_attachments.size() < (size_t)ongoing_renderpass->color_attachments.size()) {
				pi.color_blend_attachments.resize(ongoing_renderpass->color_attachments.size(), pi.color_blend_attachments.back());
			}
			pi.color_blend_state = pi.base->color_blend_state;
			pi.color_blend_state.pAttachments = (VkPipelineColorBlendAttachmentState*)pi.color_blend_attachments.data();
			pi.color_blend_state.attachmentCount = (uint32_t)pi.color_blend_attachments.size();

			current_pipeline = ptc.acquire_pipeline(pi);

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
		delete& ptc;
	}

} // namespace vuk