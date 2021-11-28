#include "vuk/Context.hpp"
#include "ContextImpl.hpp"

vuk::PerThreadContext::PerThreadContext(InflightContext& ifc, unsigned tid) : ctx(ifc.ctx), ifc(ifc), tid(tid), impl(new PTCImpl(ifc, *this)) {
}

vuk::PerThreadContext::~PerThreadContext() {
	ifc.destroy(std::move(impl->image_recycle));
	ifc.destroy(std::move(impl->image_view_recycle));
	delete impl;
}

void vuk::PerThreadContext::destroy(vuk::Image image) {
	impl->image_recycle.push_back(image);
}

void vuk::PerThreadContext::destroy(vuk::ImageView image) {
	impl->image_view_recycle.push_back(image.payload);
}

void vuk::PerThreadContext::destroy(vuk::DescriptorSet ds) {
	// note that since we collect at integer times FC, we are releasing the DS back to the right pool
	ctx.impl->pool_cache.acquire(ds.layout_info, ifc.absolute_frame).free_sets.enqueue(ds.descriptor_set);
}

vuk::Unique<vuk::PersistentDescriptorSet> vuk::PerThreadContext::create_persistent_descriptorset(DescriptorSetLayoutCreateInfo dslci,
	unsigned num_descriptors) {
	dslci.dslci.bindingCount = (uint32_t)dslci.bindings.size();
	dslci.dslci.pBindings = dslci.bindings.data();
	VkDescriptorSetLayoutBindingFlagsCreateInfo dslbfci{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
	if (dslci.flags.size() > 0) {
		dslbfci.bindingCount = (uint32_t)dslci.bindings.size();
		dslbfci.pBindingFlags = dslci.flags.data();
		dslci.dslci.pNext = &dslbfci;
	}
	auto& dslai = ctx.impl->descriptor_set_layouts.acquire(dslci, ifc.absolute_frame);
	return create_persistent_descriptorset(dslai, num_descriptors);
}

vuk::Unique<vuk::PersistentDescriptorSet> vuk::PerThreadContext::create_persistent_descriptorset(const DescriptorSetLayoutAllocInfo& dslai, unsigned num_descriptors) {
	vuk::PersistentDescriptorSet tda;
	auto dsl = dslai.layout;
	VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
	dpci.maxSets = 1;
	std::array<VkDescriptorPoolSize, 12> descriptor_counts = {};
	uint32_t used_idx = 0;
	for (auto i = 0; i < descriptor_counts.size(); i++) {
		bool used = false;
		// create non-variable count descriptors
		if (dslai.descriptor_counts[i] > 0) {
			auto& d = descriptor_counts[used_idx];
			d.type = VkDescriptorType(i);
			d.descriptorCount = dslai.descriptor_counts[i];
			used = true;
		}
		// create variable count descriptors
		if (dslai.variable_count_binding != (unsigned)-1 &&
			dslai.variable_count_binding_type == vuk::DescriptorType(i)) {
			auto& d = descriptor_counts[used_idx];
			d.type = VkDescriptorType(i);
			d.descriptorCount += num_descriptors;
			used = true;
		}
		if (used) {
			used_idx++;
		}
	}

	dpci.pPoolSizes = descriptor_counts.data();
	dpci.poolSizeCount = used_idx;
	vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &tda.backing_pool);
	VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	dsai.descriptorPool = tda.backing_pool;
	dsai.descriptorSetCount = 1;
	dsai.pSetLayouts = &dsl;
	VkDescriptorSetVariableDescriptorCountAllocateInfo dsvdcai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO };
	dsvdcai.descriptorSetCount = 1;
	dsvdcai.pDescriptorCounts = &num_descriptors;
	dsai.pNext = &dsvdcai;

	vkAllocateDescriptorSets(ctx.device, &dsai, &tda.backing_set);
	// TODO: we need more information here to handled arrayed bindings properly
	// for now we assume no arrayed bindings outside of the variable count one
	for (auto& bindings : tda.descriptor_bindings) {
		bindings.resize(1);
	}
	if (dslai.variable_count_binding != (unsigned)-1) {
		tda.descriptor_bindings[dslai.variable_count_binding].resize(num_descriptors);
	}
	return Unique<PersistentDescriptorSet>(ctx, std::move(tda));
}

vuk::Unique<vuk::PersistentDescriptorSet> vuk::PerThreadContext::create_persistent_descriptorset(const PipelineBaseInfo& base, unsigned set, unsigned num_descriptors) {
	return create_persistent_descriptorset(base.layout_info[set], num_descriptors);
}

vuk::Unique<vuk::PersistentDescriptorSet> vuk::PerThreadContext::create_persistent_descriptorset(const ComputePipelineBaseInfo& base, unsigned set, unsigned num_descriptors) {
	return create_persistent_descriptorset(base.layout_info[set], num_descriptors);
}

void vuk::PerThreadContext::commit_persistent_descriptorset(vuk::PersistentDescriptorSet& array) {
	vkUpdateDescriptorSets(ctx.device, (uint32_t)array.pending_writes.size(), array.pending_writes.data(), 0, nullptr);
	array.pending_writes.clear();
}

size_t vuk::Context::get_allocation_size(Buffer buf) {
	return impl->allocator.get_allocation_size(buf);
}

vuk::NUnique<vuk::Buffer> vuk::Context::allocate_buffer(NAllocator& allocator, MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment) {
	bool create_mapped = mem_usage == MemoryUsage::eCPUonly || mem_usage == MemoryUsage::eCPUtoGPU || mem_usage == MemoryUsage::eGPUtoCPU;
	NUnique<Buffer> buf(allocator);
	BufferCreateInfo bci{ mem_usage, vuk::BufferUsageFlagBits::eTransferDst | buffer_usage, size, alignment};
	auto ret = allocator.allocate_buffers(std::span{ &*buf, 1 }, std::span{ &bci, 1 }, VUK_HERE_AND_NOW()); // TODO: dropping error
	return buf;
}


std::pair<vuk::Texture, vuk::TransferStub> vuk::Context::create_texture(NAllocator& allocator, vuk::Format format, vuk::Extent3D extent, void* data, bool generate_mips) {
	vuk::ImageCreateInfo ici;
	ici.format = format;
	ici.extent = extent;
	ici.samples = vuk::Samples::e1;
	ici.initialLayout = vuk::ImageLayout::eUndefined;
	ici.tiling = vuk::ImageTiling::eOptimal;
	ici.usage = vuk::ImageUsageFlagBits::eTransferSrc | vuk::ImageUsageFlagBits::eTransferDst | vuk::ImageUsageFlagBits::eSampled;
	ici.mipLevels = generate_mips ? (uint32_t)log2f((float)std::max(extent.width, extent.height)) + 1 : 1;
	ici.arrayLayers = 1;
	auto tex = allocate_texture(ici);
	auto stub = upload(allocator, *tex.image, format, extent, 0, std::span<std::byte>((std::byte*)data, compute_image_size(format, extent)), generate_mips);
	return { std::move(tex), stub };
}

void vuk::Context::dma_task() {
	std::lock_guard _(impl->transfer_mutex);
	while (!impl->pending_transfers.empty() && vkGetFenceStatus(device, impl->pending_transfers.front().fence) == VK_SUCCESS) {
		auto last = impl->pending_transfers.front();
		last_transfer_complete = last.last_transfer_id;
		impl->pending_transfers.pop();
	}

	if (impl->buffer_transfer_commands.empty() && impl->bufferimage_transfer_commands.empty()) return;
	auto cbuf = *allocate_hl_commandbuffer(get_direct_allocator(), { .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .queue_family_index = transfer_queue_family_index }, VUK_HERE_AND_NOW());
	VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	vkBeginCommandBuffer(*cbuf, &cbi);
	size_t last = 0;
	while (!impl->buffer_transfer_commands.empty()) {
		auto task = impl->buffer_transfer_commands.front();
		impl->buffer_transfer_commands.pop();
		VkBufferCopy bc;
		bc.dstOffset = task.dst.offset;
		bc.srcOffset = task.src.offset;
		bc.size = task.src.size;
		vkCmdCopyBuffer(*cbuf, task.src.buffer, task.dst.buffer, 1, &bc);
		last = std::max(last, task.stub.id);
	}
	while (!impl->bufferimage_transfer_commands.empty()) {
		auto task = impl->bufferimage_transfer_commands.front();
		impl->bufferimage_transfer_commands.pop();
		record_buffer_image_copy(cbuf->command_buffer, task);
		last = std::max(last, task.stub.id);
	}
	vkEndCommandBuffer(*cbuf);
	auto fence = *allocate_fence(get_direct_allocator(), VUK_HERE_AND_NOW());
	VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cbuf->command_buffer;
	submit_graphics(si, *fence);
	impl->pending_transfers.emplace(PendingTransfer{ last, *fence });

	// TODO:
	// wait all transfers, immediately

	while (!impl->pending_transfers.empty()) {
		vkWaitForFences(device, 1, &impl->pending_transfers.front().fence, true, UINT64_MAX);
		auto last = impl->pending_transfers.front();
		last_transfer_complete = last.last_transfer_id;
		impl->pending_transfers.pop();
	}
}

vuk::SampledImage& vuk::PerThreadContext::make_sampled_image(vuk::ImageView iv, vuk::SamplerCreateInfo sci) {
	vuk::SampledImage si(vuk::SampledImage::Global{ iv, sci, vuk::ImageLayout::eShaderReadOnlyOptimal });
	return impl->sampled_images.acquire(si);
}

vuk::SampledImage& vuk::PerThreadContext::make_sampled_image(Name n, vuk::SamplerCreateInfo sci) {
	vuk::SampledImage si(vuk::SampledImage::RenderGraphAttachment{ n, sci, {}, vuk::ImageLayout::eShaderReadOnlyOptimal });
	return impl->sampled_images.acquire(si);
}

vuk::SampledImage& vuk::PerThreadContext::make_sampled_image(Name n, vuk::ImageViewCreateInfo ivci, vuk::SamplerCreateInfo sci) {
	vuk::SampledImage si(vuk::SampledImage::RenderGraphAttachment{ n, sci, ivci, vuk::ImageLayout::eShaderReadOnlyOptimal });
	return impl->sampled_images.acquire(si);
}

vuk::DescriptorSet vuk::Context::create(const create_info_t<vuk::DescriptorSet>& cinfo) {
	auto& pool = impl->pool_cache.acquire(cinfo.layout_info, frame_counter);
	auto ds = pool.acquire(*this, cinfo.layout_info);
	auto mask = cinfo.used.to_ulong();
	uint32_t leading_ones = num_leading_ones(mask);
	std::array<VkWriteDescriptorSet, VUK_MAX_BINDINGS> writes = {};
	int j = 0;
	for (uint32_t i = 0; i < leading_ones; i++, j++) {
		if (!cinfo.used.test(i)) {
			j--;
			continue;
		}
		auto& write = writes[j];
		write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
		auto& binding = cinfo.bindings[i];
		write.descriptorType = (VkDescriptorType)binding.type;
		write.dstArrayElement = 0;
		write.descriptorCount = 1;
		write.dstBinding = i;
		write.dstSet = ds;
		switch (binding.type) {
		case vuk::DescriptorType::eUniformBuffer:
		case vuk::DescriptorType::eStorageBuffer:
			write.pBufferInfo = &binding.buffer;
			break;
		case vuk::DescriptorType::eSampledImage:
		case vuk::DescriptorType::eSampler:
		case vuk::DescriptorType::eCombinedImageSampler:
		case vuk::DescriptorType::eStorageImage:
			write.pImageInfo = &binding.image.dii;
			break;
		default:
			assert(0);
		}
	}
	vkUpdateDescriptorSets(device, j, writes.data(), 0, nullptr);
	return { ds, cinfo.layout_info };
}

vuk::LinearAllocator vuk::Context::create(const create_info_t<vuk::LinearAllocator>& cinfo) {
	return impl->allocator.allocate_linear(cinfo.mem_usage, cinfo.buffer_usage);
}

vuk::RGImage vuk::Context::create(const create_info_t<vuk::RGImage>& cinfo) {
	RGImage res{};
	res.image = impl->allocator.create_image_for_rendertarget(cinfo.ici);
	auto ivci = cinfo.ivci;
	ivci.image = res.image;
	std::string name = std::string("Image: RenderTarget ") + std::string(cinfo.name.to_sv());
	debug.set_name(res.image, Name(name));
	name = std::string("ImageView: RenderTarget ") + std::string(cinfo.name.to_sv());
	// skip creating image views for images that can't be viewed
	if (cinfo.ici.usage & (vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eDepthStencilAttachment | vuk::ImageUsageFlagBits::eInputAttachment | vuk::ImageUsageFlagBits::eSampled | vuk::ImageUsageFlagBits::eStorage)) {
		VkImageView iv;
		vkCreateImageView(device, (VkImageViewCreateInfo*)&ivci, nullptr, &iv);
		res.image_view = wrap(iv, ivci);
		debug.set_name(res.image_view.payload, Name(name));
	}
	return res;
}

VkRenderPass vuk::Context::create(const create_info_t<VkRenderPass>& cinfo) {
	VkRenderPass rp;
	vkCreateRenderPass(device, &cinfo, nullptr, &rp);
	return rp;
}

template<class T>
T read(const std::byte*& data_ptr) {
	T t;
	memcpy(&t, data_ptr, sizeof(T));
	data_ptr += sizeof(T);
	return t;
};

vuk::PipelineInfo vuk::Context::create(const create_info_t<PipelineInfo>& cinfo) {
	// create gfx pipeline
	VkGraphicsPipelineCreateInfo gpci{ .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
	gpci.renderPass = cinfo.render_pass;
	gpci.layout = cinfo.base->pipeline_layout;
	auto psscis = cinfo.base->psscis;
	gpci.pStages = psscis.data();
	gpci.stageCount = (uint32_t)psscis.size();

	// read variable sized data
	const std::byte* data_ptr = cinfo.is_inline() ? cinfo.inline_data : cinfo.extended_data;

	// subpass
	if (cinfo.records.nonzero_subpass) {
		gpci.subpass = read<uint8_t>(data_ptr);
	}

	// INPUT ASSEMBLY
	VkPipelineInputAssemblyStateCreateInfo input_assembly_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = cinfo.topology, .primitiveRestartEnable = cinfo.primitive_restart_enable };
	gpci.pInputAssemblyState = &input_assembly_state;
	// VERTEX INPUT
	vuk::fixed_vector<VkVertexInputBindingDescription, VUK_MAX_ATTRIBUTES> vibds;
	vuk::fixed_vector<VkVertexInputAttributeDescription, VUK_MAX_ATTRIBUTES> viads;
	VkPipelineVertexInputStateCreateInfo vertex_input_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
	if (cinfo.records.vertex_input) {
		viads.resize(cinfo.base->reflection_info.attributes.size());
		for (auto& viad : viads) {
			auto compressed = read<PipelineInstanceCreateInfo::VertexInputAttributeDescription>(data_ptr);
			viad.binding = compressed.binding;
			viad.location = compressed.location;
			viad.format = (VkFormat)compressed.format;
			viad.offset = compressed.offset;
		}
		vertex_input_state.pVertexAttributeDescriptions = viads.data();
		vertex_input_state.vertexAttributeDescriptionCount = (uint32_t)viads.size();

		vibds.resize(read<uint8_t>(data_ptr));
		for (auto& vibd : vibds) {
			auto compressed = read<PipelineInstanceCreateInfo::VertexInputBindingDescription>(data_ptr);
			vibd.binding = compressed.binding;
			vibd.inputRate = (VkVertexInputRate)compressed.inputRate;
			vibd.stride = compressed.stride;
		}
		vertex_input_state.pVertexBindingDescriptions = vibds.data();
		vertex_input_state.vertexBindingDescriptionCount = (uint32_t)vibds.size();
	}
	gpci.pVertexInputState = &vertex_input_state;
	// PIPELINE COLOR BLEND ATTACHMENTS
	VkPipelineColorBlendStateCreateInfo color_blend_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = cinfo.attachmentCount
	};
	auto default_writemask = vuk::ColorComponentFlagBits::eR | vuk::ColorComponentFlagBits::eG | vuk::ColorComponentFlagBits::eB | vuk::ColorComponentFlagBits::eA;
	std::vector<VkPipelineColorBlendAttachmentState> pcbas(cinfo.attachmentCount,
		VkPipelineColorBlendAttachmentState{
			.blendEnable = false,
			.colorWriteMask = (VkColorComponentFlags)default_writemask
		}
	);
	if (cinfo.records.color_blend_attachments) {
		if (!cinfo.records.broadcast_color_blend_attachment_0) {
			for (auto& pcba : pcbas) {
				auto compressed = read<PipelineInstanceCreateInfo::PipelineColorBlendAttachmentState>(data_ptr);
				pcba = {
					compressed.blendEnable,
					(VkBlendFactor)compressed.srcColorBlendFactor,
					(VkBlendFactor)compressed.dstColorBlendFactor,
					(VkBlendOp)compressed.colorBlendOp,
					(VkBlendFactor)compressed.srcAlphaBlendFactor,
					(VkBlendFactor)compressed.dstAlphaBlendFactor,
					(VkBlendOp)compressed.alphaBlendOp,
					compressed.colorWriteMask
				};
			}
		} else { // handle broadcast
			auto compressed = read<PipelineInstanceCreateInfo::PipelineColorBlendAttachmentState>(data_ptr);
			for (auto& pcba : pcbas) {
				pcba = {
					compressed.blendEnable,
					(VkBlendFactor)compressed.srcColorBlendFactor,
					(VkBlendFactor)compressed.dstColorBlendFactor,
					(VkBlendOp)compressed.colorBlendOp,
					(VkBlendFactor)compressed.srcAlphaBlendFactor,
					(VkBlendFactor)compressed.dstAlphaBlendFactor,
					(VkBlendOp)compressed.alphaBlendOp,
					compressed.colorWriteMask
				};
			}
		}
	}
	if (cinfo.records.logic_op) {
		auto compressed = read<PipelineInstanceCreateInfo::BlendStateLogicOp>(data_ptr);
		color_blend_state.logicOpEnable = true;
		color_blend_state.logicOp = compressed.logic_op;
	}
	if (cinfo.records.blend_constants) {
		memcpy(&color_blend_state.blendConstants, data_ptr, sizeof(float) * 4);
		data_ptr += sizeof(float) * 4;
	}

	color_blend_state.pAttachments = pcbas.data();
	color_blend_state.attachmentCount = (uint32_t)pcbas.size();
	gpci.pColorBlendState = &color_blend_state;

	// SPECIALIZATION CONSTANTS
	vuk::fixed_vector<VkSpecializationInfo, vuk::graphics_stage_count> specialization_infos;
	vuk::fixed_vector<VkSpecializationMapEntry, VUK_MAX_SPECIALIZATIONCONSTANT_RANGES> specialization_map_entries;
	uint16_t specialization_constant_data_size = 0;
	const std::byte* specialization_constant_data = nullptr;
	if (cinfo.records.specialization_constants) {
		Bitset<VUK_MAX_SPECIALIZATIONCONSTANT_RANGES> set_constants = {};
		set_constants = read<Bitset<VUK_MAX_SPECIALIZATIONCONSTANT_RANGES>>(data_ptr);
		specialization_constant_data = data_ptr;

		for (unsigned i = 0; i < cinfo.base->reflection_info.spec_constants.size(); i++) {
			auto& sc = cinfo.base->reflection_info.spec_constants[i];
			uint16_t size = sc.type == vuk::Program::Type::edouble ? (uint16_t)sizeof(double) : 4;
			if (set_constants.test(i)) {
				specialization_constant_data_size += size;
			}
		}
		data_ptr += specialization_constant_data_size;

		uint16_t entry_offset = 0;
		for (uint32_t i = 0; i < psscis.size(); i++) {
			auto& pssci = psscis[i];
			uint16_t data_offset = 0;
			uint16_t current_entry_offset = entry_offset;
			for (unsigned i = 0; i < cinfo.base->reflection_info.spec_constants.size(); i++) {
				auto& sc = cinfo.base->reflection_info.spec_constants[i];
				auto size = sc.type == vuk::Program::Type::edouble ? sizeof(double) : 4;
				if (sc.stage & pssci.stage) {
					specialization_map_entries.emplace_back(VkSpecializationMapEntry{
						sc.binding,
						data_offset,
						size
						});
					data_offset += (uint16_t)size;
					entry_offset++;
				}
			}

			VkSpecializationInfo si;
			si.pMapEntries = specialization_map_entries.data() + current_entry_offset;
			si.mapEntryCount = (uint32_t)specialization_map_entries.size() - current_entry_offset;
			si.pData = specialization_constant_data;
			si.dataSize = specialization_constant_data_size;
			if (si.mapEntryCount > 0) {
				specialization_infos.push_back(si);
				pssci.pSpecializationInfo = &specialization_infos.back();
			}
		}
	}

	// RASTER STATE
	VkPipelineRasterizationStateCreateInfo rasterization_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = cinfo.cullMode,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.f
	};
	if (cinfo.records.non_trivial_raster_state) {
		auto rs = read<PipelineInstanceCreateInfo::RasterizationState>(data_ptr);
		rasterization_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = rs.depthClampEnable,
		.rasterizerDiscardEnable = rs.rasterizerDiscardEnable,
		.polygonMode = (VkPolygonMode)rs.polygonMode,
		.cullMode = cinfo.cullMode,
		.frontFace = (VkFrontFace)rs.frontFace,
		.lineWidth = 1.f
		};
	}
	if (cinfo.records.depth_bias) {
		auto db = read<PipelineInstanceCreateInfo::DepthBias>(data_ptr);
		rasterization_state.depthBiasEnable = true;
		rasterization_state.depthBiasClamp = db.depthBiasClamp;
		rasterization_state.depthBiasConstantFactor = db.depthBiasConstantFactor;
		rasterization_state.depthBiasSlopeFactor = db.depthBiasSlopeFactor;
	}
	if (cinfo.records.line_width_not_1) {
		rasterization_state.lineWidth = read<float>(data_ptr);
	}
	gpci.pRasterizationState = &rasterization_state;

	// DEPTH - STENCIL STATE
	VkPipelineDepthStencilStateCreateInfo depth_stencil_state{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
	if (cinfo.records.depth_stencil) {
		auto d = read<PipelineInstanceCreateInfo::Depth>(data_ptr);
		depth_stencil_state.depthTestEnable = d.depthTestEnable;
		depth_stencil_state.depthWriteEnable = d.depthWriteEnable;
		depth_stencil_state.depthCompareOp = (VkCompareOp)d.depthCompareOp;
		if (cinfo.records.depth_bounds) {
			auto db = read<PipelineInstanceCreateInfo::DepthBounds>(data_ptr);
			depth_stencil_state.depthBoundsTestEnable = true;
			depth_stencil_state.minDepthBounds = db.minDepthBounds;
			depth_stencil_state.maxDepthBounds = db.maxDepthBounds;
		}
		if (cinfo.records.stencil_state) {
			auto s = read<PipelineInstanceCreateInfo::Stencil>(data_ptr);
			depth_stencil_state.stencilTestEnable = true;
			depth_stencil_state.front = s.front;
			depth_stencil_state.back = s.back;
		}
		gpci.pDepthStencilState = &depth_stencil_state;
	}

	// MULTISAMPLE STATE
	VkPipelineMultisampleStateCreateInfo multisample_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
	if (cinfo.records.more_than_one_sample) {
		auto ms = read<PipelineInstanceCreateInfo::Multisample>(data_ptr);
		multisample_state.rasterizationSamples = ms.rasterization_samples;
		multisample_state.alphaToCoverageEnable = ms.alpha_to_coverage_enable;
		multisample_state.alphaToOneEnable = ms.alpha_to_one_enable;
		multisample_state.minSampleShading = ms.min_sample_shading;
		multisample_state.sampleShadingEnable = ms.sample_shading_enable;
		multisample_state.pSampleMask = nullptr; // not yet supported
	}
	gpci.pMultisampleState = &multisample_state;

	// VIEWPORTS
	const VkViewport* viewports = nullptr;
	uint8_t num_viewports = 1;
	if (cinfo.records.viewports) {
		num_viewports = read<uint8_t>(data_ptr);
		viewports = reinterpret_cast<const VkViewport*>(data_ptr);
		data_ptr += num_viewports * sizeof(VkViewport);
	}

	// SCISSORS
	const VkRect2D* scissors = nullptr;
	uint8_t num_scissors = 1;
	if (cinfo.records.scissors) {
		num_scissors = read<uint8_t>(data_ptr);
		scissors = reinterpret_cast<const VkRect2D*>(data_ptr);
		data_ptr += num_scissors * sizeof(VkRect2D);
	}

	VkPipelineViewportStateCreateInfo viewport_state{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
	viewport_state.pViewports = viewports;
	viewport_state.viewportCount = num_viewports;
	viewport_state.pScissors = scissors;
	viewport_state.scissorCount = num_scissors;
	gpci.pViewportState = &viewport_state;

	VkPipelineDynamicStateCreateInfo dynamic_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
	dynamic_state.dynamicStateCount = std::popcount(cinfo.dynamic_state_flags.m_mask);
	vuk::fixed_vector<VkDynamicState, VkDynamicState::VK_DYNAMIC_STATE_DEPTH_BOUNDS> dyn_states;
	uint64_t dyn_state_cnt = 0;
	uint64_t mask = cinfo.dynamic_state_flags.m_mask;
	while (mask > 0) {
		bool set = mask & 0x1;
		if (set) {
			dyn_states.push_back((VkDynamicState)dyn_state_cnt); // TODO: we will need a switch here instead of a cast when handling EXT
		}
		mask >>= 1;
		dyn_state_cnt++;
	}
	dynamic_state.pDynamicStates = dyn_states.data();
	gpci.pDynamicState = &dynamic_state;

	VkPipeline pipeline;
	VkResult res = vkCreateGraphicsPipelines(device, impl->vk_pipeline_cache, 1, &gpci, nullptr, &pipeline);
	assert(res == VK_SUCCESS);
	debug.set_name(pipeline, cinfo.base->pipeline_name);
	return { pipeline, gpci.layout, cinfo.base->layout_info };
}

vuk::ComputePipelineInfo vuk::Context::create(const create_info_t<ComputePipelineInfo>& cinfo) {
	// create compute pipeline
	VkComputePipelineCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	cpci.layout = cinfo.base->pipeline_layout;
	cpci.stage = cinfo.base->pssci;

	VkPipeline pipeline;
	VkResult res = vkCreateComputePipelines(device, impl->vk_pipeline_cache, 1, &cpci, nullptr, &pipeline);
	assert(res == VK_SUCCESS);
	debug.set_name(pipeline, cinfo.base->pipeline_name);
	return { pipeline, cpci.layout, cinfo.base->layout_info, cinfo.base->reflection_info.local_size };
}


vuk::Unique<VkFramebuffer> vuk::Context::create(const create_info_t<VkFramebuffer>& cinfo) {
	VkFramebuffer fb;
	vkCreateFramebuffer(device, &cinfo, nullptr, &fb);
	return vuk::Unique(*this, fb);
}

vuk::Sampler vuk::Context::create(const create_info_t<vuk::Sampler>& cinfo) {
	VkSampler s;
	vkCreateSampler(device, (VkSamplerCreateInfo*)&cinfo, nullptr, &s);
	return wrap(s);
}

vuk::DescriptorPool vuk::Context::create(const create_info_t<vuk::DescriptorPool>& cinfo) {
	return vuk::DescriptorPool{};
}

vuk::Program vuk::PerThreadContext::get_pipeline_reflection_info(const vuk::PipelineBaseCreateInfo& pci) {
	auto& res = ctx.impl->pipelinebase_cache.acquire(pci, ifc.absolute_frame);
	return res.reflection_info;
}

vuk::Program vuk::PerThreadContext::get_pipeline_reflection_info(const vuk::ComputePipelineBaseCreateInfo& pci) {
	auto& res = ctx.impl->compute_pipelinebase_cache.acquire(pci, ifc.absolute_frame);
	return res.reflection_info;
}

vuk::TimestampQuery vuk::PerThreadContext::register_timestamp_query(vuk::Query handle) {
	auto query_slot = impl->tsquery_pool.acquire(1)[0];
	auto& mapping = impl->tsquery_pool.pool.id_to_value_mapping;
	mapping.emplace_back(handle.id, query_slot.id);
	return query_slot;
}

VkRenderPass vuk::Context::acquire_renderpass(const vuk::RenderPassCreateInfo& rpci, uint64_t absolute_frame) {
	return impl->renderpass_cache.acquire(rpci, absolute_frame);
}

vuk::RGImage vuk::Context::acquire_rendertarget(const vuk::RGCI& rgci, uint64_t absolute_frame) {
	return impl->transient_images.acquire(rgci, absolute_frame);
}

vuk::Sampler vuk::Context::acquire_sampler(const vuk::SamplerCreateInfo& sci, uint64_t absolute_frame) {
	return impl->sampler_cache.acquire(sci, absolute_frame);
}

vuk::DescriptorSet vuk::Context::acquire_descriptorset(const vuk::SetBinding& sb, uint64_t absolute_frame) {
	return impl->descriptor_sets.acquire(sb, absolute_frame);
}

vuk::PipelineInfo vuk::Context::acquire_pipeline(const vuk::PipelineInstanceCreateInfo& pici, uint64_t absolute_frame) {
	return impl->pipeline_cache.acquire(pici, absolute_frame);
}

vuk::ComputePipelineInfo vuk::Context::acquire_pipeline(const vuk::ComputePipelineInstanceCreateInfo& pici, uint64_t absolute_frame) {
	return impl->compute_pipeline_cache.acquire(pici, absolute_frame);
}

const plf::colony<vuk::SampledImage>& vuk::PerThreadContext::get_sampled_images() {
	return impl->sampled_images.pool.values;
}


