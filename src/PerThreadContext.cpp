#include "vuk/Context.hpp"
#include "ContextImpl.hpp"

vuk::PerThreadContext::PerThreadContext(InflightContext& ifc, unsigned tid) : ctx(ifc.ctx), ifc(&ifc), tid(tid), impl(new PTCImpl(ifc, *this)) {
}

vuk::PerThreadContext::~PerThreadContext() {
	if (impl) {
		ifc->destroy(std::move(impl->image_recycle));
		ifc->destroy(std::move(impl->image_view_recycle));
		ifc->destroy(std::move(impl->linear_allocators));
		delete impl;
	}
}

vuk::PerThreadContext vuk::PerThreadContext::clone() {
	return ifc->begin();
}

void vuk::PerThreadContext::destroy(vuk::Image image) {
	impl->image_recycle.push_back(image);
}

void vuk::PerThreadContext::destroy(vuk::ImageView image) {
	impl->image_view_recycle.push_back(image.payload);
}

void vuk::PerThreadContext::destroy(vuk::DescriptorSet ds) {
	// note that since we collect at integer times FC, we are releasing the DS back to the right pool
	ctx.impl->pool_cache.acquire(ds.layout_info, ifc->absolute_frame).free_sets.enqueue(ds.descriptor_set);
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

vuk::DescriptorSet vuk::PerThreadContext::create(const create_info_t<vuk::DescriptorSet>& cinfo) {
	auto& pool = ctx.impl->pool_cache.acquire(cinfo.layout_info, ifc->absolute_frame);
	auto ds = pool.acquire(*this, cinfo.layout_info);
	auto mask = cinfo.used.to_ulong();
	unsigned long leading_ones = num_leading_ones(mask);
	std::array<VkWriteDescriptorSet, VUK_MAX_BINDINGS> writes = {};
	int j = 0;
	for (int i = 0; i < leading_ones; i++, j++) {
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
	vkUpdateDescriptorSets(ctx.device, j, writes.data(), 0, nullptr);
	return { ds, cinfo.layout_info };
}

vuk::LinearAllocator vuk::PerThreadContext::create(const create_info_t<vuk::LinearAllocator>& cinfo) {
	return ctx.impl->allocator.allocate_linear(cinfo.mem_usage, cinfo.buffer_usage);
}

vuk::RGImage vuk::PerThreadContext::create(const create_info_t<vuk::RGImage>& cinfo) {
	return ctx.allocate(cinfo);
}

VkRenderPass vuk::PerThreadContext::create(const create_info_t<VkRenderPass>& cinfo) {
	return ctx.allocate(cinfo);
}

vuk::ShaderModule vuk::PerThreadContext::create(const create_info_t<vuk::ShaderModule>& cinfo) {
	return ctx.allocate(cinfo);
}

vuk::PipelineBaseInfo vuk::PerThreadContext::create(const create_info_t<PipelineBaseInfo>& cinfo) {
	return ctx.allocate(cinfo);
}

vuk::PipelineInfo vuk::PerThreadContext::create(const create_info_t<PipelineInfo>& cinfo) {
	return ctx.allocate(cinfo);
}

vuk::ComputePipelineInfo vuk::PerThreadContext::create(const create_info_t<ComputePipelineInfo>& cinfo) {
	return ctx.allocate(cinfo);
}

VkFramebuffer vuk::PerThreadContext::create(const create_info_t<VkFramebuffer>& cinfo) {
	return ctx.allocate(cinfo);
}

vuk::Sampler vuk::PerThreadContext::create(const create_info_t<vuk::Sampler>& cinfo) {
	return ctx.allocate(cinfo);
}

vuk::DescriptorSetLayoutAllocInfo vuk::PerThreadContext::create(const create_info_t<vuk::DescriptorSetLayoutAllocInfo>& cinfo) {
	return ctx.allocate(cinfo);
}

VkPipelineLayout vuk::PerThreadContext::create(const create_info_t<VkPipelineLayout>& cinfo) {
	return ctx.allocate(cinfo);
}

vuk::DescriptorPool vuk::PerThreadContext::create(const create_info_t<vuk::DescriptorPool>& cinfo) {
	return vuk::DescriptorPool{};
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

vuk::Token vuk::PerThreadContext::submit(vuk::Token token, vuk::Domain domain) {
	TokenData::TokenType token_type;
	/*if (domain & vuk::Domain::eHost) {
		token_type = TokenData::TokenType::eTimeline;
	}*/
	TokenData* data = &ctx.impl->get_token_data(token);
	data->token_type = TokenData::TokenType::eTimeline;
	assert(data->state == TokenData::State::eArmed);

	std::vector<VkCommandBuffer> cbufs;

	vuk::LinearResourceAllocator* allocator;
	if (!data->resources) {
		// TODO: map domain to queue family
		data->resources = ctx.impl->get_linear_allocator(ctx.graphics_queue_family_index);
	}

	allocator = data->resources;

	while (data != nullptr) {
		if (data->rg) {
			ExecutableRenderGraph erg = std::move(*data->rg).link(ctx);
			cbufs.push_back(erg.execute(*allocator, {}).command_buffers[0]); // TODO: the waits and signals
		}
		data->state = TokenData::State::ePending;
		data = data->next;
	}

	if (cbufs.size() == 0) {
		//TODO: free token
		ctx.impl->get_token_data(token).state = TokenData::State::eComplete;
		return token;
	}

	allocator->sema = allocator->acquire_timeline_semaphore();
	
	// enqueue for destruction on frame end
	impl->linear_allocators.push_back(allocator);

	VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = cbufs.size();
	si.pCommandBuffers = cbufs.data();
	VkTimelineSemaphoreSubmitInfo tssi{ .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
	uint64_t signal = 1;
	si.pSignalSemaphores = &allocator->sema;
	si.signalSemaphoreCount = 1;
	tssi.pSignalSemaphoreValues = &signal;
	tssi.signalSemaphoreValueCount = 1;
	si.pNext = &tssi;

	allocator->fence = allocator->acquire_fence();

	ctx.submit_graphics(si, allocator->fence);

	return token;
}

void vuk::PerThreadContext::wait(Token token) {
	TokenData& data = ctx.impl->get_token_data(token);
	if (data.state == TokenData::State::eComplete) {
		return;
	}
	assert(data.state == TokenData::State::ePending && "Token must have been submitted to be waited on.");
	assert(data.token_type == TokenData::TokenType::eTimeline && "Can only wait on Timeline tokens on host");
	VkSemaphoreWaitInfo swi{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
	swi.pSemaphores = &data.resources->sema;
	uint64_t value = 1;
	swi.pValues = &value;
	swi.semaphoreCount = 1;
	vkWaitSemaphores(ctx.device, &swi, UINT64_MAX);
	data.state = TokenData::State::eComplete;
	free(token);
}

void vuk::PerThreadContext::free(Token token) {
	TokenData* data = &ctx.impl->get_token_data(token);
	//vkDestroySemaphore(ctx.device, data.resources->sema, nullptr);
	while (data != nullptr) {
		ctx.impl->cleanup_transient_bundle_recursively(data->resources);
		data = data->next;
	}
}

VkFence vuk::PerThreadContext::acquire_fence() {
	return impl->fence_pool.acquire(1)[0];
}

VkCommandBuffer vuk::PerThreadContext::acquire_command_buffer(VkCommandBufferLevel level) {
	return impl->commandbuffer_pool.acquire(level, 1)[0];
}

VkSemaphore vuk::PerThreadContext::acquire_semaphore() {
	return impl->semaphore_pool.acquire(1)[0];
}

VkFramebuffer vuk::PerThreadContext::acquire_framebuffer(const vuk::FramebufferCreateInfo& fbci) {
	return ctx.impl->framebuffer_cache.acquire(fbci, ifc->absolute_frame);
}

VkRenderPass vuk::PerThreadContext::acquire_renderpass(const vuk::RenderPassCreateInfo& rpci) {
	return ctx.impl->renderpass_cache.acquire(rpci, ifc->absolute_frame);
}

vuk::RGImage vuk::PerThreadContext::acquire_rendertarget(const vuk::RGCI& rgci) {
	return ctx.impl->transient_images.acquire(rgci, ifc->absolute_frame);
}

vuk::Sampler vuk::PerThreadContext::acquire_sampler(const vuk::SamplerCreateInfo& sci) {
	return ctx.impl->sampler_cache.acquire(sci);
}

vuk::DescriptorSet vuk::PerThreadContext::acquire_descriptorset(const vuk::SetBinding& sb) {
	return impl->descriptor_sets.acquire(sb);
}

vuk::PipelineInfo vuk::PerThreadContext::acquire_pipeline(const vuk::PipelineInstanceCreateInfo& pici) {
	return ctx.impl->pipeline_cache.acquire(pici, ifc->absolute_frame);
}

const plf::colony<vuk::SampledImage>& vuk::PerThreadContext::get_sampled_images() {
	return impl->sampled_images.pool.values;
}


