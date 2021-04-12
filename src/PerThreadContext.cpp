#include "vuk/Context.hpp"
#include "ContextImpl.hpp"

/*
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
	tda.descriptor_bindings.resize(num_descriptors);
	return Unique<PersistentDescriptorSet>(ctx, std::move(tda));
}

vuk::Unique<vuk::PersistentDescriptorSet> vuk::PerThreadContext::create_persistent_descriptorset(const PipelineBaseInfo& base, unsigned set, unsigned num_descriptors) {
	return create_persistent_descriptorset(base.layout_info[set], num_descriptors);
}

vuk::Unique<vuk::PersistentDescriptorSet> vuk::PerThreadContext::create_persistent_descriptorset(const ComputePipelineInfo& base, unsigned set, unsigned num_descriptors) {
	return create_persistent_descriptorset(base.layout_info[set], num_descriptors);
}

void vuk::PerThreadContext::commit_persistent_descriptorset(vuk::PersistentDescriptorSet& array) {
	vkUpdateDescriptorSets(ctx.device, (uint32_t)array.pending_writes.size(), array.pending_writes.data(), 0, nullptr);
	array.pending_writes.clear();
}

size_t vuk::PerThreadContext::get_allocation_size(Buffer buf) {
	return ctx.impl->allocator.get_allocation_size(buf);
}

vuk::Buffer vuk::PerThreadContext::allocate_scratch_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment) {
	bool create_mapped = mem_usage == MemoryUsage::eCPUonly || mem_usage == MemoryUsage::eCPUtoGPU || mem_usage == MemoryUsage::eGPUtoCPU;
	PoolSelect ps{ mem_usage, buffer_usage };
	auto& pool = impl->scratch_buffers.acquire(ps);
	return ctx.impl->allocator.allocate_buffer(pool, size, alignment, create_mapped);
}

vuk::Unique<vuk::Buffer> vuk::PerThreadContext::allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment) {
	bool create_mapped = mem_usage == MemoryUsage::eCPUonly || mem_usage == MemoryUsage::eCPUtoGPU || mem_usage == MemoryUsage::eGPUtoCPU;
	return vuk::Unique<Buffer>(ctx, ctx.impl->allocator.allocate_buffer(mem_usage, buffer_usage, size, alignment, create_mapped));
}

bool vuk::PerThreadContext::is_ready(const TransferStub& stub) {
	return ifc->last_transfer_complete >= stub.id;
}

void vuk::PerThreadContext::wait_all_transfers() {
	// TODO: remove when we go MT
	dma_task(); // run one transfer so it is more easy to follow
	return ifc->wait_all_transfers();
}

vuk::Texture vuk::PerThreadContext::allocate_texture(vuk::ImageCreateInfo ici) {
	auto tex = ctx.allocate_texture(ici);
	return tex;
}


std::pair<vuk::Texture, vuk::TransferStub> vuk::PerThreadContext::create_texture(vuk::Format format, vuk::Extent3D extent, void* data, bool generate_mips) {
	vuk::ImageCreateInfo ici;
	ici.format = format;
	ici.extent = extent;
	ici.samples = vuk::Samples::e1;
	ici.imageType = vuk::ImageType::e2D;
	ici.initialLayout = vuk::ImageLayout::eUndefined;
	ici.tiling = vuk::ImageTiling::eOptimal;
	ici.usage = vuk::ImageUsageFlagBits::eTransferSrc | vuk::ImageUsageFlagBits::eTransferDst | vuk::ImageUsageFlagBits::eSampled;
	ici.mipLevels = generate_mips ? (uint32_t)log2f((float)std::max(extent.width, extent.height)) + 1 : 1;
	ici.arrayLayers = 1;
	auto tex = ctx.allocate_texture(ici);
	auto stub = upload(*tex.image, format, extent, 0, std::span<std::byte>((std::byte*)data, compute_image_size(format, extent)), generate_mips);
	return { std::move(tex), stub };
}

void vuk::PerThreadContext::dma_task() {
	std::lock_guard _(ifc->impl->transfer_mutex);
	while (!ifc->impl->pending_transfers.empty() && vkGetFenceStatus(ctx.device, ifc->impl->pending_transfers.front().fence) == VK_SUCCESS) {
		auto last = ifc->impl->pending_transfers.front();
		ifc->last_transfer_complete = last.last_transfer_id;
		ifc->impl->pending_transfers.pop();
	}

	if (ifc->impl->buffer_transfer_commands.empty() && ifc->impl->bufferimage_transfer_commands.empty()) return;
	auto cbuf = impl->commandbuffer_pool.acquire(VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1)[0];
	VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	vkBeginCommandBuffer(cbuf, &cbi);
	size_t last = 0;
	while (!ifc->impl->buffer_transfer_commands.empty()) {
		auto task = ifc->impl->buffer_transfer_commands.front();
		ifc->impl->buffer_transfer_commands.pop();
		VkBufferCopy bc;
		bc.dstOffset = task.dst.offset;
		bc.srcOffset = task.src.offset;
		bc.size = task.src.size;
		vkCmdCopyBuffer(cbuf, task.src.buffer, task.dst.buffer, 1, &bc);
		last = std::max(last, task.stub.id);
	}
	while (!ifc->impl->bufferimage_transfer_commands.empty()) {
		auto task = ifc->impl->bufferimage_transfer_commands.front();
		ifc->impl->bufferimage_transfer_commands.pop();
		record_buffer_image_copy(cbuf, task);
		last = std::max(last, task.stub.id);
	}
	vkEndCommandBuffer(cbuf);
	auto fence = impl->fence_pool.acquire(1)[0];
	VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cbuf;
	ctx.submit_graphics(si, fence);
	ifc->impl->pending_transfers.emplace(PendingTransfer{ last, fence });
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

vuk::LinearAllocator vuk::PerThreadContext::create(const create_info_t<vuk::LinearAllocator>& cinfo) {
	return ctx.impl->allocator.allocate_linear(cinfo.mem_usage, cinfo.buffer_usage);
}

vuk::Program vuk::PerThreadContext::get_pipeline_reflection_info(vuk::PipelineBaseCreateInfo pci) {
	return ctx.impl->pipelinebase_cache.acquire(pci, ifc->absolute_frame).reflection_info;
}

vuk::TimestampQuery vuk::PerThreadContext::register_timestamp_query(vuk::Query handle) {
	auto query_slot = impl->tsquery_pool.acquire(1)[0];
	auto& mapping = impl->tsquery_pool.pool.id_to_value_mapping;
	mapping.emplace_back(handle.id, query_slot.id);
	return query_slot;
}

void vuk::PerThreadContext::free(Token token) {
	TokenData* data = &ctx.impl->get_token_data(token);
	//vkDestroySemaphore(ctx.device, data.resources->sema, nullptr);
	while (data != nullptr) {
		ctx.impl->cleanup_transient_bundle_recursively(data->resources);
		data = data->next;
	}
}

const plf::colony<vuk::SampledImage>& vuk::PerThreadContext::get_sampled_images() {
	return impl->sampled_images.pool.values;
}*/