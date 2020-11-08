#include "vuk/Context.hpp"
#include "ContextImpl.hpp"

vuk::PerThreadContext::PerThreadContext(InflightContext& ifc, unsigned tid) : ctx(ifc.ctx), ifc(ifc), tid(tid),
commandbuffer_pool(ifc.commandbuffer_pools.get_view(*this)),
semaphore_pool(ifc.semaphore_pools.get_view(*this)),
fence_pool(ifc.fence_pools.get_view(*this)),
pipeline_cache(*this, ifc.pipeline_cache),
compute_pipeline_cache(*this, ifc.compute_pipeline_cache),
pipelinebase_cache(*this, ifc.pipelinebase_cache),
renderpass_cache(*this, ifc.renderpass_cache),
framebuffer_cache(*this, ifc.framebuffer_cache),
transient_images(*this, ifc.transient_images),
scratch_buffers(*this, ifc.scratch_buffers),
descriptor_sets(*this, ifc.descriptor_sets),
sampler_cache(*this, ifc.sampler_cache),
sampled_images(ifc.sampled_images.get_view(*this)),
pool_cache(*this, ifc.pool_cache),
shader_modules(*this, ifc.shader_modules),
descriptor_set_layouts(*this, ifc.descriptor_set_layouts),
pipeline_layouts(*this, ifc.pipeline_layouts) {
}

vuk::PerThreadContext::~PerThreadContext() {
	ifc.destroy(std::move(image_recycle));
	ifc.destroy(std::move(image_view_recycle));
}

void vuk::PerThreadContext::destroy(vuk::Image image) {
	image_recycle.push_back(image);
}

void vuk::PerThreadContext::destroy(vuk::ImageView image) {
	image_view_recycle.push_back(image.payload);
}

void vuk::PerThreadContext::destroy(vuk::DescriptorSet ds) {
	// note that since we collect at integer times FC, we are releasing the DS back to the right pool
	pool_cache.acquire(ds.layout_info).free_sets.enqueue(ds.descriptor_set);
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

vuk::Buffer vuk::PerThreadContext::_allocate_scratch_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment,
	bool create_mapped) {
	PoolSelect ps{ mem_usage, buffer_usage };
	auto& pool = scratch_buffers.acquire(ps);
	return ifc.ctx.impl->allocator.allocate_buffer(pool, size, alignment, create_mapped);
}

vuk::Unique<vuk::Buffer> vuk::PerThreadContext::_allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment, bool create_mapped) {
	return vuk::Unique<Buffer>(ifc.ctx, ifc.ctx.impl->allocator.allocate_buffer(mem_usage, buffer_usage, size, alignment, create_mapped));
}


bool vuk::PerThreadContext::is_ready(const TransferStub& stub) {
	return ifc.last_transfer_complete >= stub.id;
}

void vuk::PerThreadContext::wait_all_transfers() {
	// TODO: remove when we go MT
	dma_task(); // run one transfer so it is more easy to follow
	return ifc.wait_all_transfers();
}

vuk::Texture vuk::PerThreadContext::allocate_texture(vuk::ImageCreateInfo ici) {
	auto tex = ctx.allocate_texture(ici);
	return tex;
}


std::pair<vuk::Texture, vuk::TransferStub> vuk::PerThreadContext::create_texture(vuk::Format format, vuk::Extent3D extents, void* data) {
	vuk::ImageCreateInfo ici;
	ici.format = format;
	ici.extent = extents;
	ici.samples = vuk::Samples::e1;
	ici.imageType = vuk::ImageType::e2D;
	ici.initialLayout = vuk::ImageLayout::eUndefined;
	ici.tiling = vuk::ImageTiling::eOptimal;
	ici.usage = vuk::ImageUsageFlagBits::eTransferSrc | vuk::ImageUsageFlagBits::eTransferDst | vuk::ImageUsageFlagBits::eSampled;
	ici.mipLevels = ici.arrayLayers = 1;
	auto tex = ctx.allocate_texture(ici);
	auto stub = upload(*tex.image, extents, std::span<std::byte>((std::byte*)data, extents.width * extents.height * extents.depth * 4), false);
	return { std::move(tex), stub };
}

void vuk::PerThreadContext::dma_task() {
	std::lock_guard _(ifc.transfer_mutex);
	while (!ifc.pending_transfers.empty() && vkGetFenceStatus(ctx.device, ifc.pending_transfers.front().fence) == VK_SUCCESS) {
		auto last = ifc.pending_transfers.front();
		ifc.last_transfer_complete = last.last_transfer_id;
		ifc.pending_transfers.pop();
	}

	if (ifc.buffer_transfer_commands.empty() && ifc.bufferimage_transfer_commands.empty()) return;
	auto cbuf = commandbuffer_pool.acquire(VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1)[0];
	VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	vkBeginCommandBuffer(cbuf, &cbi);
	size_t last = 0;
	while (!ifc.buffer_transfer_commands.empty()) {
		auto task = ifc.buffer_transfer_commands.front();
		ifc.buffer_transfer_commands.pop();
		VkBufferCopy bc;
		bc.dstOffset = task.dst.offset;
		bc.srcOffset = task.src.offset;
		bc.size = task.src.size;
		vkCmdCopyBuffer(cbuf, task.src.buffer, task.dst.buffer, 1, &bc);
		last = std::max(last, task.stub.id);
	}
	while (!ifc.bufferimage_transfer_commands.empty()) {
		auto task = ifc.bufferimage_transfer_commands.front();
		ifc.bufferimage_transfer_commands.pop();
		record_buffer_image_copy(cbuf, task);
		last = std::max(last, task.stub.id);
	}
	vkEndCommandBuffer(cbuf);
	auto fence = fence_pool.acquire(1)[0];
	VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cbuf;
	vkQueueSubmit(ifc.ctx.graphics_queue, 1, &si, fence);
	ifc.pending_transfers.emplace(InflightContext::PendingTransfer{ last, fence });
}

vuk::SampledImage& vuk::PerThreadContext::make_sampled_image(vuk::ImageView iv, vuk::SamplerCreateInfo sci) {
	vuk::SampledImage si(vuk::SampledImage::Global{ iv, sci, vuk::ImageLayout::eShaderReadOnlyOptimal });
	return sampled_images.acquire(si);
}

vuk::SampledImage& vuk::PerThreadContext::make_sampled_image(Name n, vuk::SamplerCreateInfo sci) {
	vuk::SampledImage si(vuk::SampledImage::RenderGraphAttachment{ n, sci, {}, vuk::ImageLayout::eShaderReadOnlyOptimal });
	return sampled_images.acquire(si);
}

vuk::SampledImage& vuk::PerThreadContext::make_sampled_image(Name n, vuk::ImageViewCreateInfo ivci, vuk::SamplerCreateInfo sci) {
	vuk::SampledImage si(vuk::SampledImage::RenderGraphAttachment{ n, sci, ivci, vuk::ImageLayout::eShaderReadOnlyOptimal });
	return sampled_images.acquire(si);
}

vuk::DescriptorSet vuk::PerThreadContext::create(const create_info_t<vuk::DescriptorSet>& cinfo) {
	auto& pool = pool_cache.acquire(cinfo.layout_info);
	auto ds = pool.acquire(*this, cinfo.layout_info);
	auto mask = cinfo.used.to_ulong();
	unsigned long leading_ones = num_leading_ones(mask);
	std::array<VkWriteDescriptorSet, VUK_MAX_BINDINGS> writes;
	for (unsigned i = 0; i < leading_ones; i++) {
		if (!cinfo.used.test(i)) continue;
		auto& write = writes[i];
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
	vkUpdateDescriptorSets(ctx.device, leading_ones, writes.data(), 0, nullptr);
	return { ds, cinfo.layout_info };
}

vuk::Allocator::Linear vuk::PerThreadContext::create(const create_info_t<vuk::Allocator::Linear>& cinfo) {
	return ctx.impl->allocator.allocate_linear(cinfo.mem_usage, cinfo.buffer_usage);
}


vuk::RGImage vuk::PerThreadContext::create(const create_info_t<vuk::RGImage>& cinfo) {
	RGImage res{};
	res.image = ctx.impl->allocator.create_image_for_rendertarget(cinfo.ici);
	auto ivci = cinfo.ivci;
	ivci.image = res.image;
	std::string name = std::string("Image: RenderTarget ") + std::string(cinfo.name);
	ctx.debug.set_name(res.image, name);
	name = std::string("ImageView: RenderTarget ") + std::string(cinfo.name);
	// skip creating image views for images that can't be viewed
	if (cinfo.ici.usage & (vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eDepthStencilAttachment | vuk::ImageUsageFlagBits::eInputAttachment | vuk::ImageUsageFlagBits::eSampled | vuk::ImageUsageFlagBits::eStorage)) {
		VkImageView iv;
		vkCreateImageView(ctx.device, (VkImageViewCreateInfo*)&ivci, nullptr, &iv);
		res.image_view = ctx.wrap(iv);
		ctx.debug.set_name(res.image_view.payload, name);
	}
	return res;
}

VkRenderPass vuk::PerThreadContext::create(const create_info_t<VkRenderPass>& cinfo) {
	VkRenderPass rp;
	vkCreateRenderPass(ctx.device, &cinfo, nullptr, &rp);
	return rp;
}

vuk::ShaderModule vuk::PerThreadContext::create(const create_info_t<vuk::ShaderModule>& cinfo) {
	return ctx.create(cinfo);
}

vuk::PipelineBaseInfo vuk::PerThreadContext::create(const create_info_t<PipelineBaseInfo>& cinfo) {
	return ctx.create(cinfo);
}

vuk::PipelineInfo vuk::PerThreadContext::create(const create_info_t<PipelineInfo>& cinfo) {
	// create gfx pipeline
	VkGraphicsPipelineCreateInfo gpci = cinfo.to_vk();
	gpci.layout = cinfo.base->pipeline_layout;
	gpci.pStages = cinfo.base->psscis.data();
	gpci.stageCount = (uint32_t)cinfo.base->psscis.size();

	VkPipeline pipeline;
	vkCreateGraphicsPipelines(ctx.device, ctx.impl->vk_pipeline_cache, 1, &gpci, nullptr, &pipeline);
	ctx.debug.set_name(pipeline, cinfo.base->pipeline_name);
	return { pipeline, gpci.layout, cinfo.base->layout_info };
}

vuk::ComputePipelineInfo vuk::PerThreadContext::create(const create_info_t<ComputePipelineInfo>& cinfo) {
	return ctx.create(cinfo);
}

VkFramebuffer vuk::PerThreadContext::create(const create_info_t<VkFramebuffer>& cinfo) {
	VkFramebuffer fb;
	vkCreateFramebuffer(ctx.device, &cinfo, nullptr, &fb);
	return fb;
}

vuk::Sampler vuk::PerThreadContext::create(const create_info_t<vuk::Sampler>& cinfo) {
	VkSampler s;
	vkCreateSampler(ctx.device, (VkSamplerCreateInfo*)&cinfo, nullptr, &s);
	return ctx.wrap(s);
}

vuk::DescriptorSetLayoutAllocInfo vuk::PerThreadContext::create(const create_info_t<vuk::DescriptorSetLayoutAllocInfo>& cinfo) {
	return ctx.create(cinfo);
}

VkPipelineLayout vuk::PerThreadContext::create(const create_info_t<VkPipelineLayout>& cinfo) {
	return ctx.create(cinfo);
}

vuk::DescriptorPool vuk::PerThreadContext::create(const create_info_t<vuk::DescriptorPool>& cinfo) {
	return vuk::DescriptorPool{};
}

vuk::Program vuk::PerThreadContext::get_pipeline_reflection_info(vuk::PipelineBaseCreateInfo pci) {
	auto& res = pipelinebase_cache.acquire(pci);
	return res.reflection_info;
}


