#include "Context.hpp"
#include "RenderGraph.hpp"
#include <shaderc/shaderc.hpp>
#include <algorithm>
#include "Program.hpp"
#include <fstream>
#include <sstream>
#include <spirv_cross.hpp>

void burp(const std::string& in, const std::string& path) {
	std::ofstream output(path.c_str(), std::ios::trunc);
	if (!output.is_open()) {
	}
	output << in;
	output.close();
}

vuk::InflightContext::InflightContext(Context& ctx, size_t absolute_frame, std::lock_guard<std::mutex>&& recycle_guard) :
	ctx(ctx),
	absolute_frame(absolute_frame),
	frame(absolute_frame% Context::FC),
	fence_pools(ctx.fence_pools.get_view(*this)), // must be first, so we wait for the fences
	commandbuffer_pools(ctx.cbuf_pools.get_view(*this)),
	semaphore_pools(ctx.semaphore_pools.get_view(*this)),
	pipeline_cache(*this, ctx.pipeline_cache),
	compute_pipeline_cache(*this, ctx.compute_pipeline_cache),
	pipelinebase_cache(*this, ctx.pipelinebase_cache),
	renderpass_cache(*this, ctx.renderpass_cache),
	framebuffer_cache(*this, ctx.framebuffer_cache),
	transient_images(*this, ctx.transient_images),
	scratch_buffers(*this, ctx.scratch_buffers),
	descriptor_sets(*this, ctx.descriptor_sets),
	sampler_cache(*this, ctx.sampler_cache),
	sampled_images(ctx.sampled_images.get_view(*this)),
	pool_cache(*this, ctx.pool_cache),
	shader_modules(*this, ctx.shader_modules),
	descriptor_set_layouts(*this, ctx.descriptor_set_layouts),
	pipeline_layouts(*this, ctx.pipeline_layouts) {

	// image recycling
	for (auto& img : ctx.image_recycle[frame]) {
		ctx.allocator.destroy_image(img);
	}
	ctx.image_recycle[frame].clear();

	for (auto& iv : ctx.image_view_recycle[frame]) {
		vkDestroyImageView(ctx.device, iv, nullptr);
	}
	ctx.image_view_recycle[frame].clear();

	for (auto& p : ctx.pipeline_recycle[frame]) {
		vkDestroyPipeline(ctx.device, p, nullptr);
	}
	ctx.pipeline_recycle[frame].clear();

	for (auto& b : ctx.buffer_recycle[frame]) {
		ctx.allocator.free_buffer(b);
	}
	ctx.buffer_recycle[frame].clear();

	for (auto& pds : ctx.pds_recycle[frame]) {
		vkDestroyDescriptorPool(ctx.device, pds.backing_pool, nullptr);
	}
	ctx.pds_recycle[frame].clear();

	for (auto& [k, v] : scratch_buffers.cache.data[frame].lru_map) {
		ctx.allocator.reset_pool(v.value);
	}

	auto ptc = begin();
	ptc.descriptor_sets.collect(Context::FC * 2);
	ptc.transient_images.collect(Context::FC * 2);
	ptc.scratch_buffers.collect(Context::FC * 2);
}

vuk::TransferStub vuk::InflightContext::enqueue_transfer(Buffer src, Buffer dst) {
	std::lock_guard _(transfer_mutex);
	TransferStub stub{ transfer_id++ };
	buffer_transfer_commands.push({ src, dst, stub });
	return stub;
}

vuk::TransferStub vuk::InflightContext::enqueue_transfer(Buffer src, vuk::Image dst, vuk::Extent3D extent, bool generate_mips) {
	std::lock_guard _(transfer_mutex);
	TransferStub stub{ transfer_id++ };
	bufferimage_transfer_commands.push({ src, dst, extent, generate_mips, stub });
	return stub;
}

void vuk::InflightContext::wait_all_transfers() {
	std::lock_guard _(transfer_mutex);

	while (!pending_transfers.empty()) {
		vkWaitForFences(ctx.device, 1, &pending_transfers.front().fence, true, UINT64_MAX);
		auto last = pending_transfers.front();
		last_transfer_complete = last.last_transfer_id;
		pending_transfers.pop();
	}
}

void vuk::InflightContext::destroy(std::vector<vuk::Image>&& images) {
	std::lock_guard _(recycle_lock);
	ctx.image_recycle[frame].insert(ctx.image_recycle[frame].end(), images.begin(), images.end());
}


bool vuk::execute_submit_and_present_to_one(PerThreadContext& ptc, RenderGraph& rg, SwapchainRef swapchain, bool use_secondary_command_buffers) {
	auto present_rdy = ptc.semaphore_pool.acquire(1)[0];
	uint32_t image_index = (uint32_t)-1;
	VkResult acq_result = vkAcquireNextImageKHR(ptc.ctx.device, swapchain->swapchain, UINT64_MAX, present_rdy, VK_NULL_HANDLE, &image_index);
	if (acq_result != VK_SUCCESS) {
		VkSubmitInfo si { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
		si.commandBufferCount = 0;
		si.pCommandBuffers = nullptr;
		si.waitSemaphoreCount = 1;
		si.pWaitSemaphores = &present_rdy;
		VkPipelineStageFlags flags = (VkPipelineStageFlags)vuk::PipelineStageFlagBits::eTopOfPipe;
		si.pWaitDstStageMask = &flags;

		return false;
	}

	auto render_complete = ptc.semaphore_pool.acquire(1)[0];
	std::vector<std::pair<SwapChainRef, size_t>> swapchains_with_indexes = { { swapchain, image_index } };

	auto cb = rg.execute(ptc, swapchains_with_indexes, use_secondary_command_buffers);

	VkSubmitInfo si { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cb;
	si.pSignalSemaphores = &render_complete;
	si.signalSemaphoreCount = 1;
	si.waitSemaphoreCount = 1;
	si.pWaitSemaphores = &present_rdy;
	VkPipelineStageFlags flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	si.pWaitDstStageMask = &flags;
	auto fence = ptc.fence_pool.acquire(1)[0];
	{
		std::lock_guard _(ptc.ctx.gfx_queue_lock);
		vkQueueSubmit(ptc.ctx.graphics_queue, 1, &si, fence);

		VkPresentInfoKHR pi{ .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
		pi.swapchainCount = 1;
		pi.pSwapchains = &swapchain->swapchain;
		pi.pImageIndices = &image_index;
		pi.waitSemaphoreCount = 1;
		pi.pWaitSemaphores = &render_complete;
		auto present_result = vkQueuePresentKHR(ptc.ctx.graphics_queue, &pi);
		return present_result == VK_SUCCESS;
	}
	return true;
}

void vuk::execute_submit_and_wait(PerThreadContext& ptc, RenderGraph& rg, bool use_secondary_command_buffers) {
	auto cbuf = rg.execute(ptc, {}, use_secondary_command_buffers);
	// get an unpooled fence
	VkFenceCreateInfo fci{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	VkFence fence;
	vkCreateFence(ptc.ctx.device, &fci, nullptr, &fence);
	VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cbuf;
	{
		std::lock_guard _(ptc.ctx.gfx_queue_lock);
		vkQueueSubmit(ptc.ctx.graphics_queue, 1, &si, fence);
	}
	vkWaitForFences(ptc.ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
	vkDestroyFence(ptc.ctx.device, fence, nullptr);
}

bool vuk::Context::DebugUtils::enabled() {
	return setDebugUtilsObjectNameEXT != nullptr;
}

vuk::Context::DebugUtils::DebugUtils(Context& ctx) : ctx(ctx) {
	setDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(ctx.device, "vkSetDebugUtilsObjectNameEXT");
	cmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(ctx.device, "vkCmdBeginDebugUtilsLabelEXT");
	cmdEndDebugUtilsLabelEXT = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(ctx.device, "vkCmdEndDebugUtilsLabelEXT");
}

void vuk::Context::DebugUtils::set_name(const vuk::Texture& tex, Name name) {
	if (!enabled()) return;
	set_name(tex.image.get(), name);
	set_name(tex.view.get().payload, name);
}

void vuk::Context::DebugUtils::begin_region(const VkCommandBuffer& cb, Name name, std::array<float, 4> color) {
	if (!enabled()) return;
	VkDebugUtilsLabelEXT label = { .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
	label.pLabelName = name.data();
	::memcpy(label.color, color.data(), sizeof(float) * 4);
	cmdBeginDebugUtilsLabelEXT(cb, &label);
}

void vuk::Context::DebugUtils::end_region(const VkCommandBuffer& cb) {
	if (!enabled()) return;
	cmdEndDebugUtilsLabelEXT(cb);
}

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

void vuk::PersistentDescriptorSet::update_combined_image_sampler(PerThreadContext& ptc, unsigned binding, unsigned array_index, vuk::ImageView iv, vuk::SamplerCreateInfo sci, vuk::ImageLayout layout) {
	descriptor_bindings[array_index].image = vuk::DescriptorImageInfo(ptc.sampler_cache.acquire(sci), iv, layout);
	descriptor_bindings[array_index].type = vuk::DescriptorType::eCombinedImageSampler;
	VkWriteDescriptorSet wds = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	wds.descriptorCount = 1;
	wds.descriptorType = (VkDescriptorType)vuk::DescriptorType::eCombinedImageSampler;
	wds.dstArrayElement = array_index;
	wds.dstBinding = binding;
	wds.pImageInfo = &descriptor_bindings[array_index].image.dii;
	wds.dstSet = backing_set;
	pending_writes.push_back(wds);
}

void vuk::PerThreadContext::commit_persistent_descriptorset(vuk::PersistentDescriptorSet& array) {
	vkUpdateDescriptorSets(ctx.device, (uint32_t)array.pending_writes.size(), array.pending_writes.data(), 0, nullptr);
	array.pending_writes.clear();
}

size_t vuk::PerThreadContext::get_allocation_size(Buffer buf) {
	return ctx.allocator.get_allocation_size(buf);
}

vuk::Buffer vuk::PerThreadContext::_allocate_scratch_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment,
	bool create_mapped) {
	PoolSelect ps{ mem_usage, buffer_usage };
	auto& pool = scratch_buffers.acquire(ps);
	return ifc.ctx.allocator.allocate_buffer(pool, size, alignment, create_mapped);
}

vuk::Unique<vuk::Buffer> vuk::PerThreadContext::_allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment, bool create_mapped) {
	return vuk::Unique<Buffer>(ifc.ctx, ifc.ctx.allocator.allocate_buffer(mem_usage, buffer_usage, size, alignment, create_mapped));
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

void record_buffer_image_copy(VkCommandBuffer& cbuf, vuk::InflightContext::BufferImageCopyCommand& task) {
	VkBufferImageCopy bc;
	bc.bufferOffset = task.src.offset;
	bc.imageOffset = VkOffset3D{ 0, 0, 0 };
	bc.bufferRowLength = 0;
	bc.bufferImageHeight = 0;
	bc.imageExtent = task.extent;
	bc.imageSubresource.aspectMask = (VkImageAspectFlagBits)vuk::ImageAspectFlagBits::eColor;
	bc.imageSubresource.baseArrayLayer = 0;
	bc.imageSubresource.mipLevel = 0;
	bc.imageSubresource.layerCount = 1;

	VkImageMemoryBarrier copy_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	copy_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	copy_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eUndefined;
	copy_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
	copy_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	copy_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	copy_barrier.image = task.dst;
	copy_barrier.subresourceRange.aspectMask = bc.imageSubresource.aspectMask;
	copy_barrier.subresourceRange.layerCount = bc.imageSubresource.layerCount;
	copy_barrier.subresourceRange.baseArrayLayer = bc.imageSubresource.baseArrayLayer;
	copy_barrier.subresourceRange.baseMipLevel = bc.imageSubresource.mipLevel;
	copy_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;

	// transition top mip to transfersrc
	VkImageMemoryBarrier top_mip_to_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	top_mip_to_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	top_mip_to_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	top_mip_to_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
	top_mip_to_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eTransferSrcOptimal;
	top_mip_to_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	top_mip_to_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	top_mip_to_barrier.image = task.dst;
	top_mip_to_barrier.subresourceRange = copy_barrier.subresourceRange;
	top_mip_to_barrier.subresourceRange.levelCount = 1;

	// transition top mip to SROO
	VkImageMemoryBarrier top_mip_use_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	top_mip_use_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	top_mip_use_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	top_mip_use_barrier.oldLayout = task.generate_mips ? (VkImageLayout)vuk::ImageLayout::eTransferSrcOptimal : (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
	top_mip_use_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eShaderReadOnlyOptimal;
	top_mip_use_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	top_mip_use_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	top_mip_use_barrier.image = task.dst;
	top_mip_use_barrier.subresourceRange = copy_barrier.subresourceRange;
	top_mip_use_barrier.subresourceRange.levelCount = 1;

	// transition rest of the mips to SROO
	VkImageMemoryBarrier use_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };;
	use_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	use_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	use_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
	use_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eShaderReadOnlyOptimal;
	use_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier.image = task.dst;
	use_barrier.subresourceRange = copy_barrier.subresourceRange;
	use_barrier.subresourceRange.baseMipLevel = 1;

	vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &copy_barrier);
	vkCmdCopyBufferToImage(cbuf, task.src.buffer, task.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);
	if (task.generate_mips) {
		vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &top_mip_to_barrier);
		auto mips = (uint32_t)std::min(std::log2f((float)task.extent.width), std::log2f((float)task.extent.height));

		for (uint32_t miplevel = 1; miplevel < mips; miplevel++) {
			VkImageBlit blit;
			blit.srcSubresource.aspectMask = copy_barrier.subresourceRange.aspectMask;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.srcSubresource.mipLevel = 0;
			blit.srcOffsets[0] = VkOffset3D{ 0 };
			blit.srcOffsets[1] = VkOffset3D{ (int32_t)task.extent.width, (int32_t)task.extent.height, (int32_t)task.extent.depth };
			blit.dstSubresource = blit.srcSubresource;
			blit.dstSubresource.mipLevel = miplevel;
			blit.dstOffsets[0] = VkOffset3D{ 0 };
			blit.dstOffsets[1] = VkOffset3D{ (int32_t)task.extent.width >> miplevel, (int32_t)task.extent.height >> miplevel, (int32_t)task.extent.depth };
			vkCmdBlitImage(cbuf, task.dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, task.dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
		}

		vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &use_barrier);
	}

	vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &top_mip_use_barrier);
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
	return ctx.allocator.allocate_linear(cinfo.mem_usage, cinfo.buffer_usage);
}


vuk::RGImage vuk::PerThreadContext::create(const create_info_t<vuk::RGImage>& cinfo) {
	RGImage res{};
	res.image = ctx.allocator.create_image_for_rendertarget(cinfo.ici);
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

vuk::ShaderModule vuk::Context::create(const create_info_t<vuk::ShaderModule>& cinfo) {
	shaderc::Compiler compiler;
	shaderc::CompileOptions options;
	options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);

	shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(cinfo.source, shaderc_glsl_infer_from_source, cinfo.filename.c_str(), options);

	if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
		std::string message = result.GetErrorMessage().c_str();
		throw ShaderCompilationException{ message };
	} else {
		std::vector<uint32_t> spirv(result.cbegin(), result.cend());

		spirv_cross::Compiler refl(spirv.data(), spirv.size());
		vuk::Program p;
		auto stage = p.introspect(refl);

		VkShaderModuleCreateInfo moduleCreateInfo{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
		moduleCreateInfo.codeSize = spirv.size() * sizeof(uint32_t);
		moduleCreateInfo.pCode = spirv.data();
		VkShaderModule sm;
		vkCreateShaderModule(device, &moduleCreateInfo, nullptr, &sm);
		std::string name = "ShaderModule: " + cinfo.filename;
		debug.set_name(sm, name);
		return { sm, p, stage };
	}
}

vuk::ShaderModule vuk::PerThreadContext::create(const create_info_t<vuk::ShaderModule>& cinfo) {
	return ctx.create(cinfo);
}

vuk::PipelineBaseInfo vuk::Context::create(const create_info_t<PipelineBaseInfo>& cinfo) {
	std::vector<VkPipelineShaderStageCreateInfo> psscis;

	// accumulate descriptors from all stages
	vuk::Program accumulated_reflection;
	std::string pipe_name = "Pipeline:";
	for (auto i = 0; i < cinfo.shaders.size(); i++) {
		auto contents = cinfo.shaders[i];
		if (contents.empty())
			continue;
		auto& sm = shader_modules.acquire({ contents, cinfo.shader_paths[i] });
		VkPipelineShaderStageCreateInfo shader_stage{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
		shader_stage.pSpecializationInfo = nullptr;
		shader_stage.stage = sm.stage;
		shader_stage.module = sm.shader_module;
		shader_stage.pName = "main"; //TODO: make param
		psscis.push_back(shader_stage);
		accumulated_reflection.append(sm.reflection_info);
		pipe_name += cinfo.shader_paths[i] + "+";
	}
	pipe_name = pipe_name.substr(0, pipe_name.size() - 1); //trim off last "+"

	// acquire descriptor set layouts (1 per set)
	// acquire pipeline layout
	vuk::PipelineLayoutCreateInfo plci;
	plci.dslcis = vuk::PipelineBaseCreateInfo::build_descriptor_layouts(accumulated_reflection, cinfo);
	plci.pcrs.insert(plci.pcrs.begin(), accumulated_reflection.push_constant_ranges.begin(), accumulated_reflection.push_constant_ranges.end());
	plci.plci.pushConstantRangeCount = (uint32_t)accumulated_reflection.push_constant_ranges.size();
	plci.plci.pPushConstantRanges = accumulated_reflection.push_constant_ranges.data();
	std::array<vuk::DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> dslai;
	std::vector<VkDescriptorSetLayout> dsls;
	for (auto& dsl : plci.dslcis) {
		dsl.dslci.bindingCount = (uint32_t)dsl.bindings.size();
		dsl.dslci.pBindings = dsl.bindings.data();
		VkDescriptorSetLayoutBindingFlagsCreateInfo dslbfci{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
		if (dsl.flags.size() > 0) {
			dslbfci.bindingCount = (uint32_t)dsl.bindings.size();
			dslbfci.pBindingFlags = dsl.flags.data();
			dsl.dslci.pNext = &dslbfci;
		}
		auto descset_layout_alloc_info = descriptor_set_layouts.acquire(dsl);
		dslai[dsl.index] = descset_layout_alloc_info;
		dsls.push_back(dslai[dsl.index].layout);
	}
	plci.plci.pSetLayouts = dsls.data();
	plci.plci.setLayoutCount = (uint32_t)dsls.size();

	PipelineBaseInfo pbi;
	pbi.psscis = std::move(psscis);
	pbi.color_blend_attachments = cinfo.color_blend_attachments;
	pbi.color_blend_state = cinfo.color_blend_state;
	pbi.depth_stencil_state = cinfo.depth_stencil_state;
	pbi.layout_info = dslai;
	pbi.pipeline_layout = pipeline_layouts.acquire(plci);
	pbi.rasterization_state = cinfo.rasterization_state;
	pbi.pipeline_name = std::move(pipe_name);
	pbi.reflection_info = accumulated_reflection;
	pbi.binding_flags = cinfo.binding_flags;
	pbi.variable_count_max = cinfo.variable_count_max;
	return pbi;
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
	vkCreateGraphicsPipelines(ctx.device, ctx.vk_pipeline_cache, 1, &gpci, nullptr, &pipeline);
	ctx.debug.set_name(pipeline, cinfo.base->pipeline_name);
	return { pipeline, gpci.layout, cinfo.base->layout_info };
}

vuk::ComputePipelineInfo vuk::Context::create(const create_info_t<vuk::ComputePipelineInfo>& cinfo) {
	VkPipelineShaderStageCreateInfo shader_stage{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	std::string pipe_name = "Compute:";
	auto& sm = shader_modules.acquire({ cinfo.shader, cinfo.shader_path });
	shader_stage.pSpecializationInfo = nullptr;
	shader_stage.stage = sm.stage;
	shader_stage.module = sm.shader_module;
	shader_stage.pName = "main"; //TODO: make param
	pipe_name += cinfo.shader_path;

	vuk::PipelineLayoutCreateInfo plci;
	plci.dslcis = vuk::PipelineBaseCreateInfo::build_descriptor_layouts(sm.reflection_info, cinfo);
	plci.pcrs.insert(plci.pcrs.begin(), sm.reflection_info.push_constant_ranges.begin(), sm.reflection_info.push_constant_ranges.end());
	plci.plci.pushConstantRangeCount = (uint32_t)sm.reflection_info.push_constant_ranges.size();
	plci.plci.pPushConstantRanges = sm.reflection_info.push_constant_ranges.data();
	std::array<vuk::DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> dslai;
	std::vector<VkDescriptorSetLayout> dsls;
	for (auto& dsl : plci.dslcis) {
		dsl.dslci.bindingCount = (uint32_t)dsl.bindings.size();
		dsl.dslci.pBindings = dsl.bindings.data();
		VkDescriptorSetLayoutBindingFlagsCreateInfo dslbfci{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
		if (dsl.flags.size() > 0) {
			dslbfci.bindingCount = (uint32_t)dsl.bindings.size();
			dslbfci.pBindingFlags = dsl.flags.data();
			dsl.dslci.pNext = &dslbfci;
		}
		auto descset_layout_alloc_info = descriptor_set_layouts.acquire(dsl);
		dslai[dsl.index] = descset_layout_alloc_info;
		dsls.push_back(dslai[dsl.index].layout);
	}
	plci.plci.pSetLayouts = dsls.data();
	plci.plci.setLayoutCount = (uint32_t)dsls.size();

	VkComputePipelineCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	cpci.stage = shader_stage;
	cpci.layout = pipeline_layouts.acquire(plci);
	VkPipeline pipeline;
	vkCreateComputePipelines(device, vk_pipeline_cache, 1, &cpci, nullptr, &pipeline);
	debug.set_name(pipeline, pipe_name);
	return { { pipeline, cpci.layout, dslai }, sm.reflection_info.local_size };
}

bool vuk::Context::load_pipeline_cache(std::span<uint8_t> data) {
	VkPipelineCacheCreateInfo pcci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = data.size_bytes(), .pInitialData = data.data() };
	vkDestroyPipelineCache(device, vk_pipeline_cache, nullptr);
	vkCreatePipelineCache(device, &pcci, nullptr, &vk_pipeline_cache);
	return true;
}

std::vector<uint8_t> vuk::Context::save_pipeline_cache() {
	size_t size;
	std::vector<uint8_t> data;
	vkGetPipelineCacheData(device, vk_pipeline_cache, &size, nullptr);
	data.resize(size);
	vkGetPipelineCacheData(device, vk_pipeline_cache, &size, data.data());
	return data;
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

vuk::DescriptorSetLayoutAllocInfo vuk::Context::create(const create_info_t<vuk::DescriptorSetLayoutAllocInfo>& cinfo) {
	vuk::DescriptorSetLayoutAllocInfo ret;
	vkCreateDescriptorSetLayout(device, &cinfo.dslci, nullptr, &ret.layout);
	for (size_t i = 0; i < cinfo.bindings.size(); i++) {
		auto& b = cinfo.bindings[i];
		// if this is not a variable count binding, add it to the descriptor count
		if (cinfo.flags.size() <= i || !(cinfo.flags[i] & to_integral(vuk::DescriptorBindingFlagBits::eVariableDescriptorCount))) {
			ret.descriptor_counts[to_integral(b.descriptorType)] += b.descriptorCount;
		} else { // a variable count binding
			ret.variable_count_binding = (uint32_t)i;
			ret.variable_count_binding_type = vuk::DescriptorType(b.descriptorType);
			ret.variable_count_binding_max_size = b.descriptorCount;
		}
	}
	return ret;
}

vuk::DescriptorSetLayoutAllocInfo vuk::PerThreadContext::create(const create_info_t<vuk::DescriptorSetLayoutAllocInfo>& cinfo) {
	return ctx.create(cinfo);
}

VkPipelineLayout vuk::Context::create(const create_info_t<VkPipelineLayout>& cinfo) {
	VkPipelineLayout pl;
	vkCreatePipelineLayout(device, &cinfo.plci, nullptr, &pl);
	return pl;
}

VkPipelineLayout vuk::PerThreadContext::create(const create_info_t<VkPipelineLayout>& cinfo) {
	return ctx.create(cinfo);
}

vuk::DescriptorPool vuk::PerThreadContext::create(const create_info_t<vuk::DescriptorPool>& cinfo) {
	return vuk::DescriptorPool{};
}

vuk::SwapchainRef vuk::Context::add_swapchain(Swapchain sw) {
	std::lock_guard _(swapchains_lock);
	sw.image_views.reserve(sw._ivs.size());
	for (auto& v : sw._ivs) {
		sw.image_views.push_back(wrap(v));
	}

	return &*swapchains.emplace(sw);
}

vuk::Context::Context(VkInstance instance, VkDevice device, VkPhysicalDevice physical_device, VkQueue graphics) :
	instance(instance),
	device(device),
	physical_device(physical_device),
	graphics_queue(graphics),
	allocator(instance, device, physical_device),
	cbuf_pools(*this),
	semaphore_pools(*this),
	fence_pools(*this),
	pipelinebase_cache(*this),
	pipeline_cache(*this),
	compute_pipeline_cache(*this),
	renderpass_cache(*this),
	framebuffer_cache(*this),
	transient_images(*this),
	scratch_buffers(*this),
	pool_cache(*this),
	descriptor_sets(*this),
	sampler_cache(*this),
	sampled_images(*this),
	shader_modules(*this),
	descriptor_set_layouts(*this),
	pipeline_layouts(*this),
	debug(*this) {

	VkPipelineCacheCreateInfo pcci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
	vkCreatePipelineCache(device, &pcci, nullptr, &vk_pipeline_cache);
}

void vuk::Context::create_named_pipeline(const char* name, vuk::PipelineBaseCreateInfo ci) {
	std::lock_guard _(named_pipelines_lock);
	named_pipelines.insert_or_assign(name, &pipelinebase_cache.acquire(std::move(ci)));
}

void vuk::Context::create_named_pipeline(const char* name, vuk::ComputePipelineCreateInfo ci) {
	std::lock_guard _(named_pipelines_lock);
	named_compute_pipelines.insert_or_assign(name, &compute_pipeline_cache.acquire(std::move(ci)));
}

vuk::PipelineBaseInfo* vuk::Context::get_named_pipeline(const char* name) {
	std::lock_guard _(named_pipelines_lock);
	return named_pipelines.at(name);
}

vuk::ComputePipelineInfo* vuk::Context::get_named_compute_pipeline(const char* name) {
	std::lock_guard _(named_pipelines_lock);
	return named_compute_pipelines.at(name);
}

vuk::PipelineBaseInfo* vuk::Context::get_pipeline(const vuk::PipelineBaseCreateInfo& pbci) {
	return &pipelinebase_cache.acquire(pbci);
}

vuk::ComputePipelineInfo* vuk::Context::get_pipeline(const vuk::ComputePipelineCreateInfo& pbci) {
	return &compute_pipeline_cache.acquire(pbci);
}

vuk::Program vuk::PerThreadContext::get_pipeline_reflection_info(vuk::PipelineBaseCreateInfo pci) {
	auto& res = pipelinebase_cache.acquire(pci);
	return res.reflection_info;
}

vuk::Program vuk::Context::get_pipeline_reflection_info(vuk::PipelineBaseCreateInfo pci) {
	auto& res = pipelinebase_cache.acquire(pci);
	return res.reflection_info;
}

vuk::ShaderModule vuk::Context::compile_shader(std::string source, Name path) {
	vuk::ShaderModuleCreateInfo sci;
	sci.filename = path;
	sci.source = std::move(source);
	auto sm = shader_modules.remove(sci);
	if (sm) {
		vkDestroyShaderModule(device, sm->shader_module, nullptr);
	}
	return shader_modules.acquire(sci);
}

vuk::Context::UploadResult vuk::Context::fenced_upload(std::span<BufferUpload> uploads) {
	// get a one time command buffer
	auto tid = get_thread_index ? get_thread_index() : 0;

	VkCommandBuffer cbuf;
	{
		std::lock_guard _(one_time_pool_lock);
		if (xfer_one_time_pools.size() < (tid + 1)) {
			xfer_one_time_pools.resize(tid + 1, VK_NULL_HANDLE);
		}

		auto& pool = xfer_one_time_pools[tid];
		if (pool == VK_NULL_HANDLE) {
			VkCommandPoolCreateInfo cpci;
			cpci.queueFamilyIndex = transfer_queue_family_index;
			cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
			vkCreateCommandPool(device, &cpci, nullptr, &pool);
		}

		VkCommandBufferAllocateInfo cbai;
		cbai.commandPool = pool;
		cbai.commandBufferCount = 1;

		vkAllocateCommandBuffers(device, &cbai, &cbuf);
	}
	VkCommandBufferBeginInfo cbi;
	cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cbuf, &cbi);

	size_t size = 0;
	for (auto& upload : uploads) {
		size += upload.data.size();
	}

	// create 1 big staging buffer
	auto staging_alloc = allocator.allocate_buffer(vuk::MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, size, 1, true);
	auto staging = staging_alloc;
	for (auto& upload : uploads) {
		// copy to staging
		::memcpy(staging.mapped_ptr, upload.data.data(), upload.data.size());

		VkBufferCopy bc;
		bc.dstOffset = upload.dst.offset;
		bc.srcOffset = staging.offset;
		bc.size = upload.data.size();
		vkCmdCopyBuffer(cbuf, staging.buffer, upload.dst.buffer, 1, &bc);

		staging.offset += upload.data.size();
		staging.mapped_ptr = static_cast<unsigned char*>(staging.mapped_ptr) + upload.data.size();
	}
	vkEndCommandBuffer(cbuf);
	// get an unpooled fence
	VkFenceCreateInfo fci{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	VkFence fence;
	vkCreateFence(device, &fci, nullptr, &fence);
	VkSubmitInfo si;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cbuf;
	{
		std::lock_guard _(xfer_queue_lock);
		vkQueueSubmit(transfer_queue, 1, &si, fence);
	}
	return { fence, cbuf, staging_alloc, true, tid };
}

vuk::Context::UploadResult vuk::Context::fenced_upload(std::span<ImageUpload> uploads) {
	// get a one time command buffer
	auto tid = get_thread_index ? get_thread_index() : 0;
	VkCommandBuffer cbuf;
	{
		std::lock_guard _(one_time_pool_lock);
		if (one_time_pools.size() < (tid + 1)) {
			one_time_pools.resize(tid + 1, VK_NULL_HANDLE);
		}
		auto& pool = one_time_pools[tid];
		if (pool == VK_NULL_HANDLE) {
			VkCommandPoolCreateInfo cpci;
			cpci.queueFamilyIndex = transfer_queue_family_index;
			cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
			vkCreateCommandPool(device, &cpci, nullptr, &pool);
		}
		VkCommandBufferAllocateInfo cbai;
		cbai.commandPool = pool;
		cbai.commandBufferCount = 1;

		vkAllocateCommandBuffers(device, &cbai, &cbuf);
	}
	VkCommandBufferBeginInfo cbi;
	cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cbuf, &cbi);

	size_t size = 0;
	for (auto& upload : uploads) {
		size += upload.data.size();
	}

	// create 1 big staging buffer
	auto staging_alloc = allocator.allocate_buffer(vuk::MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, size, 1, true);
	auto staging = staging_alloc;
	for (auto& upload : uploads) {
		// copy to staging
		::memcpy(staging.mapped_ptr, upload.data.data(), upload.data.size());

		InflightContext::BufferImageCopyCommand task;
		task.src = staging;
		task.dst = upload.dst;
		task.extent = upload.extent;
		task.generate_mips = true;
		record_buffer_image_copy(cbuf, task);
		staging.offset += upload.data.size();
		staging.mapped_ptr = static_cast<unsigned char*>(staging.mapped_ptr) + upload.data.size();
	}
	vkEndCommandBuffer(cbuf);
	// get an unpooled fence
	VkFenceCreateInfo fci{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	VkFence fence;
	vkCreateFence(device, &fci, nullptr, &fence);
	VkSubmitInfo si;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cbuf;
	{
		std::lock_guard _(gfx_queue_lock);
		vkQueueSubmit(graphics_queue, 1, &si, fence);
	}
	return { fence, cbuf, staging_alloc, false, tid };
}

void vuk::Context::free_upload_resources(const UploadResult& ur) {
	auto& pools = ur.is_buffer ? xfer_one_time_pools : one_time_pools;
	std::lock_guard _(one_time_pool_lock);
	vkFreeCommandBuffers(device, pools[ur.thread_index], 1, &ur.command_buffer);
	allocator.free_buffer(ur.staging);
	vkDestroyFence(device, ur.fence, nullptr);
}

vuk::Buffer vuk::Context::allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment) {
	return allocator.allocate_buffer(mem_usage, buffer_usage, size, alignment, false);
}

vuk::Texture vuk::Context::allocate_texture(vuk::ImageCreateInfo ici) {
	auto dst = allocator.create_image(ici);
	vuk::ImageViewCreateInfo ivci;
	ivci.format = ici.format;
	ivci.image = dst;
	ivci.subresourceRange.aspectMask = vuk::ImageAspectFlagBits::eColor;
	ivci.subresourceRange.baseArrayLayer = 0;
	ivci.subresourceRange.baseMipLevel = 0;
	ivci.subresourceRange.layerCount = 1;
	ivci.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
	ivci.viewType = vuk::ImageViewType::e2D;
	VkImageView iv;
	vkCreateImageView(device, (VkImageViewCreateInfo*)&ivci, nullptr, &iv);
	vuk::Texture tex{ vuk::Unique<vuk::Image>(*this, dst), vuk::Unique<vuk::ImageView>(*this, wrap(iv)) };
	tex.extent = ici.extent;
	tex.format = ici.format;
	return tex;
}


void vuk::Context::enqueue_destroy(vuk::Image i) {
	std::lock_guard _(recycle_locks[frame_counter % FC]);
	image_recycle[frame_counter % FC].push_back(i);
}

void vuk::Context::enqueue_destroy(vuk::ImageView iv) {
	std::lock_guard _(recycle_locks[frame_counter % FC]);
	image_view_recycle[frame_counter % FC].push_back(iv.payload);
}

void vuk::Context::enqueue_destroy(VkPipeline p) {
	std::lock_guard _(recycle_locks[frame_counter % FC]);
	pipeline_recycle[frame_counter % FC].push_back(p);
}

void vuk::Context::enqueue_destroy(vuk::Buffer b) {
	std::lock_guard _(recycle_locks[frame_counter % FC]);
	buffer_recycle[frame_counter % FC].push_back(b);
}

void vuk::Context::enqueue_destroy(vuk::PersistentDescriptorSet b) {
	std::lock_guard _(recycle_locks[frame_counter % FC]);
	pds_recycle[frame_counter % FC].push_back(std::move(b));
}


void vuk::Context::destroy(const RGImage& image) {
	vkDestroyImageView(device, image.image_view.payload, nullptr);
	allocator.destroy_image(image.image);
}

void vuk::Context::destroy(const Allocator::Pool& v) {
	allocator.destroy(v);
}

void vuk::Context::destroy(const Allocator::Linear& v) {
	allocator.destroy(v);
}


void vuk::Context::destroy(const vuk::DescriptorPool& dp) {
	for (auto& p : dp.pools) {
		vkDestroyDescriptorPool(device, p, nullptr);
	}
}

void vuk::Context::destroy(const vuk::PipelineInfo& pi) {
	vkDestroyPipeline(device, pi.pipeline, nullptr);
}

void vuk::Context::destroy(const vuk::ShaderModule& sm) {
	vkDestroyShaderModule(device, sm.shader_module, nullptr);
}

void vuk::Context::destroy(const vuk::DescriptorSetLayoutAllocInfo& ds) {
	vkDestroyDescriptorSetLayout(device, ds.layout, nullptr);
}

void vuk::Context::destroy(const VkPipelineLayout& pl) {
	vkDestroyPipelineLayout(device, pl, nullptr);
}

void vuk::Context::destroy(const VkRenderPass& rp) {
	vkDestroyRenderPass(device, rp, nullptr);
}

void vuk::Context::destroy(const vuk::DescriptorSet&) {
	// no-op, we destroy the pools
}

void vuk::Context::destroy(const VkFramebuffer& fb) {
	vkDestroyFramebuffer(device, fb, nullptr);
}

void vuk::Context::destroy(const vuk::Sampler& sa) {
	vkDestroySampler(device, sa.payload, nullptr);
}

void vuk::Context::destroy(const vuk::PipelineBaseInfo& pbi) {
	// no-op, we don't own device objects
}

vuk::Context::~Context() {
	vkDeviceWaitIdle(device);
	for (auto& s : swapchains) {
		for (auto& swiv : s.image_views) {
			vkDestroyImageView(device, swiv.payload, nullptr);
		}
		vkDestroySwapchainKHR(device, s.swapchain, nullptr);
	}
	for (auto& cp : one_time_pools) {
		if (cp != VK_NULL_HANDLE) {
			vkDestroyCommandPool(device, cp, nullptr);
		}
	}
	for (auto& cp : xfer_one_time_pools) {
		if (cp != VK_NULL_HANDLE) {
			vkDestroyCommandPool(device, cp, nullptr);
		}
	}
	vkDestroyPipelineCache(device, vk_pipeline_cache, nullptr);
}

vuk::InflightContext vuk::Context::begin() {
	std::lock_guard _(begin_frame_lock);
	std::lock_guard recycle(recycle_locks[_next(frame_counter.load(), FC)]);
	return InflightContext(*this, ++frame_counter, std::move(recycle));
}

void vuk::Context::wait_idle() {
	vkDeviceWaitIdle(device);
}

void vuk::InflightContext::destroy(std::vector<VkImageView>&& images) {
	std::lock_guard _(recycle_lock);
	ctx.image_view_recycle[frame].insert(ctx.image_view_recycle[frame].end(), images.begin(), images.end());
}

vuk::PerThreadContext vuk::InflightContext::begin() {
	return PerThreadContext{ *this, ctx.get_thread_index ? ctx.get_thread_index() : 0 };
}

