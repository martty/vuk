#include <algorithm>

#include <vuk/Context.hpp>
#include <ContextImpl.hpp>
#include <vuk/RenderGraph.hpp>
#include <vuk/CommandBuffer.hpp>

vuk::Context::Context(ContextCreateParameters params) :
	instance(params.instance),
	device(params.device),
	physical_device(params.physical_device),
	graphics_queue(params.graphics_queue),
	graphics_queue_family_index(params.graphics_queue_family_index),
	transfer_queue(params.transfer_queue),
	transfer_queue_family_index(params.transfer_queue_family_index),
	impl(new ContextImpl(*this)) {
}

bool vuk::DebugUtils::enabled() {
	return setDebugUtilsObjectNameEXT != nullptr;
}

vuk::DebugUtils::DebugUtils(VkDevice device) : device(device) {
	setDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT");
	cmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(device, "vkCmdBeginDebugUtilsLabelEXT");
	cmdEndDebugUtilsLabelEXT = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(device, "vkCmdEndDebugUtilsLabelEXT");
}

void vuk::DebugUtils::set_name(const vuk::Texture& tex, Name name) {
	if (!enabled()) return;
	set_name(tex.image.get(), name);
	set_name(tex.view.get().payload, name);
}

void vuk::DebugUtils::begin_region(const VkCommandBuffer& cb, Name name, std::array<float, 4> color) {
	if (!enabled()) return;
	VkDebugUtilsLabelEXT label = { .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
	label.pLabelName = name.c_str();
	::memcpy(label.color, color.data(), sizeof(float) * 4);
	cmdBeginDebugUtilsLabelEXT(cb, &label);
}

void vuk::DebugUtils::end_region(const VkCommandBuffer& cb) {
	if (!enabled()) return;
	cmdEndDebugUtilsLabelEXT(cb);
}

void vuk::Context::submit_graphics(VkSubmitInfo si, VkFence fence) {
	std::lock_guard _(impl->gfx_queue_lock);
	VkResult result = vkQueueSubmit(graphics_queue, 1, &si, fence);
	assert(result == VK_SUCCESS);
}

void vuk::Context::submit_transfer(VkSubmitInfo si, VkFence fence) {
	std::lock_guard _(impl->xfer_queue_lock);
	assert(vkQueueSubmit(transfer_queue, 1, &si, fence) == VK_SUCCESS);
}

void vuk::PersistentDescriptorSet::update_combined_image_sampler(PerThreadContext& ptc, unsigned binding, unsigned array_index, vuk::ImageView iv, vuk::SamplerCreateInfo sci, vuk::ImageLayout layout) {
	descriptor_bindings[array_index].image = vuk::DescriptorImageInfo(ptc.acquire_sampler(sci), iv, layout);
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

void vuk::PersistentDescriptorSet::update_storage_image(PerThreadContext& ptc, unsigned binding, unsigned array_index, vuk::ImageView iv) {
	descriptor_bindings[array_index].image = vuk::DescriptorImageInfo({}, iv, vuk::ImageLayout::eGeneral);
	descriptor_bindings[array_index].type = vuk::DescriptorType::eStorageImage;
	VkWriteDescriptorSet wds = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	wds.descriptorCount = 1;
	wds.descriptorType = (VkDescriptorType)vuk::DescriptorType::eStorageImage;
	wds.dstArrayElement = array_index;
	wds.dstBinding = binding;
	wds.pImageInfo = &descriptor_bindings[array_index].image.dii;
	wds.dstSet = backing_set;
	pending_writes.push_back(wds);
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

vuk::Query vuk::Context::create_timestamp_query() {
	return { impl->query_id_counter++ };
}

vuk::SwapchainRef vuk::Context::add_swapchain(Swapchain sw) {
	std::lock_guard _(impl->swapchains_lock);
	for (auto& v : sw.image_views) {
		v = wrap(v.payload);
	}

	return &*impl->swapchains.emplace(sw);
}

void vuk::Context::remove_swapchain(SwapchainRef sw) {
	std::lock_guard _(impl->swapchains_lock);
	for (auto it = impl->swapchains.begin(); it != impl->swapchains.end(); it++) {
		if (&*it == sw) {
			impl->swapchains.erase(it);
			return;
		}
	}
}

void vuk::Context::create_named_pipeline(vuk::Name name, vuk::PipelineBaseCreateInfo ci) {
	std::lock_guard _(impl->named_pipelines_lock);
	impl->named_pipelines.insert_or_assign(name, &impl->pipelinebase_cache.acquire(std::move(ci)));
}

void vuk::Context::create_named_pipeline(vuk::Name name, vuk::ComputePipelineCreateInfo ci) {
	std::lock_guard _(impl->named_pipelines_lock);
	impl->named_compute_pipelines.insert_or_assign(name, &impl->compute_pipeline_cache.acquire(std::move(ci)));
}

vuk::PipelineBaseInfo* vuk::Context::get_named_pipeline(vuk::Name name) {
	std::lock_guard _(impl->named_pipelines_lock);
	return impl->named_pipelines.at(name);
}

vuk::ComputePipelineInfo* vuk::Context::get_named_compute_pipeline(vuk::Name name) {
	std::lock_guard _(impl->named_pipelines_lock);
	return impl->named_compute_pipelines.at(name);
}

vuk::PipelineBaseInfo* vuk::Context::get_pipeline(const vuk::PipelineBaseCreateInfo& pbci) {
	return &impl->pipelinebase_cache.acquire(pbci);
}

vuk::ComputePipelineInfo* vuk::Context::get_pipeline(const vuk::ComputePipelineCreateInfo& pbci) {
	return &impl->compute_pipeline_cache.acquire(pbci);
}

vuk::Program vuk::Context::get_pipeline_reflection_info(vuk::PipelineBaseCreateInfo pci) {
	auto& res = impl->pipelinebase_cache.acquire(pci);
	return res.reflection_info;
}

vuk::ShaderModule vuk::Context::compile_shader(ShaderSource source, std::string path) {
	vuk::ShaderModuleCreateInfo sci;
	sci.filename = std::move(path);
	sci.source = std::move(source);
	auto sm = impl->shader_modules.remove(sci);
	if (sm) {
		vkDestroyShaderModule(device, sm->shader_module, nullptr);
	}
	return impl->shader_modules.acquire(sci);
}

vuk::Context::TransientSubmitStub vuk::Context::fenced_upload(std::span<UploadItem> uploads, uint32_t dst_queue_family) {
	LinearResourceAllocator* bundle = impl->get_linear_allocator(transfer_queue_family_index);
	LinearResourceAllocator* head_bundle = bundle;
	VkCommandBuffer xfercbuf = bundle->acquire_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	VkCommandBufferBeginInfo cbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	vkBeginCommandBuffer(xfercbuf, &cbi);

	bool any_image_transfers = false;

	// size a staging buffer to fit all the uploads, with proper alignment
	size_t size = 0;
	size_t biggest_align = 1;
	for (auto& upload : uploads) {
		if (!upload.is_buffer) {
			size = align_up(size, (size_t)format_to_texel_block_size(upload.image.format));
			size += upload.image.data.size();
			biggest_align = std::max(biggest_align, (size_t)format_to_texel_block_size(upload.image.format));
			any_image_transfers = true;
		} else {
			size += upload.buffer.data.size();
		}
	}

	VkCommandBuffer dstcbuf;
	LinearResourceAllocator* dst_bundle = nullptr;
	// image transfers will finish on the dst queue, get a bundle for them and hook it up to our transfer bundle
	if (any_image_transfers) {
		dst_bundle = impl->get_linear_allocator(dst_queue_family);
		dst_bundle->next = bundle;
		dstcbuf = bundle->acquire_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		VkCommandBufferBeginInfo cbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
		vkBeginCommandBuffer(dstcbuf, &cbi);
		head_bundle = dst_bundle;
	}

	// create 1 big staging buffer
	// we need to have this aligned for the first upload
	// as that corresponds to offset = 0
	bundle->buffer = impl->allocator.allocate_buffer(vuk::MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, size, biggest_align, true);
	auto staging = bundle->buffer;

	for (auto& upload : uploads) {
		// realign offset
		auto aligned = upload.is_buffer ? staging.offset : align_up(staging.offset, (size_t)format_to_texel_block_size(upload.image.format));
		auto delta = aligned - staging.offset;
		staging.offset = aligned;
		staging.mapped_ptr = staging.mapped_ptr + delta;

		// copy to staging
		auto& data = upload.is_buffer ? upload.buffer.data : upload.image.data;
		::memcpy(staging.mapped_ptr, data.data(), data.size());

		if (upload.is_buffer) {
			VkBufferCopy bc;
			bc.dstOffset = upload.buffer.dst.offset;
			bc.srcOffset = staging.offset;
			bc.size = upload.buffer.data.size();
			vkCmdCopyBuffer(xfercbuf, staging.buffer, upload.buffer.dst.buffer, 1, &bc);
		} else {
			// perform buffer->image copy on the xfer queue
			VkBufferImageCopy bc;
			bc.bufferOffset = staging.offset;
			bc.imageOffset = VkOffset3D{ 0, 0, 0 };
			bc.bufferRowLength = 0;
			bc.bufferImageHeight = 0;
			bc.imageExtent = upload.image.extent;
			bc.imageSubresource.aspectMask = (VkImageAspectFlags)format_to_aspect(upload.image.format);
			bc.imageSubresource.baseArrayLayer = upload.image.base_array_layer;
			bc.imageSubresource.mipLevel = upload.image.mip_level;
			bc.imageSubresource.layerCount = 1;

			VkImageMemoryBarrier copy_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
			copy_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			copy_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eUndefined;
			copy_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
			copy_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			copy_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			copy_barrier.image = upload.image.dst;
			copy_barrier.subresourceRange.aspectMask = bc.imageSubresource.aspectMask;
			copy_barrier.subresourceRange.layerCount = bc.imageSubresource.layerCount;
			copy_barrier.subresourceRange.baseArrayLayer = bc.imageSubresource.baseArrayLayer;
			copy_barrier.subresourceRange.baseMipLevel = bc.imageSubresource.mipLevel;
			copy_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;

			vkCmdPipelineBarrier(xfercbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &copy_barrier);
			vkCmdCopyBufferToImage(xfercbuf, staging.buffer, upload.image.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);

			if (!upload.image.generate_mips) {
				// transition the mips to SROO on xfer & release to dst_queue_family
				VkImageMemoryBarrier release_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };;
				release_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				release_barrier.dstAccessMask = 0; // ignored
				release_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
				release_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eShaderReadOnlyOptimal;
				release_barrier.dstQueueFamilyIndex = transfer_queue_family_index;
				release_barrier.srcQueueFamilyIndex = dst_queue_family;
				release_barrier.image = upload.image.dst;
				release_barrier.subresourceRange = copy_barrier.subresourceRange;
				vkCmdPipelineBarrier(xfercbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &release_barrier);

				// acquire on dst_queue_family
				VkImageMemoryBarrier acq_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };;
				acq_barrier.srcAccessMask = 0; // ignored
				acq_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; // TODO: maybe memory read?
				acq_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
				acq_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eShaderReadOnlyOptimal;
				acq_barrier.dstQueueFamilyIndex = transfer_queue_family_index;
				acq_barrier.srcQueueFamilyIndex = dst_queue_family;
				acq_barrier.image = upload.image.dst;
				acq_barrier.subresourceRange = copy_barrier.subresourceRange;
				// no wait, no delay, we wait on host
				vkCmdPipelineBarrier(dstcbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &acq_barrier);
			} else {
				// for now, we blit, which requires the gfx queue
				assert(dst_queue_family == graphics_queue_family_index);
				// release to dst_queue_family
				VkImageMemoryBarrier release_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };;
				release_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				release_barrier.dstAccessMask = 0; // ignored
				release_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
				release_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
				release_barrier.srcQueueFamilyIndex = transfer_queue_family_index;
				release_barrier.dstQueueFamilyIndex = dst_queue_family;
				release_barrier.image = upload.image.dst;
				release_barrier.subresourceRange = copy_barrier.subresourceRange;
				vkCmdPipelineBarrier(xfercbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &release_barrier);

				// acquire on dst_queue_family
				VkImageMemoryBarrier acq_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };;
				acq_barrier.srcAccessMask = 0; // ignored
				acq_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				acq_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
				acq_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
				acq_barrier.srcQueueFamilyIndex = transfer_queue_family_index;
				acq_barrier.dstQueueFamilyIndex = dst_queue_family;
				acq_barrier.image = upload.image.dst;
				acq_barrier.subresourceRange = copy_barrier.subresourceRange;

				// no wait, no delay, sync'd by the sema
				vkCmdPipelineBarrier(dstcbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acq_barrier);
				MipGenerateCommand task;
				task.dst = upload.image.dst;
				task.extent = upload.image.extent;
				task.base_array_layer = upload.image.base_array_layer;
				task.base_mip_level = upload.image.mip_level;
				task.layer_count = 1;
				task.format = upload.image.format;
				record_mip_gen(dstcbuf, task, vuk::ImageLayout::eTransferDstOptimal);
			}
		}

		staging.offset += data.size();
		staging.mapped_ptr = staging.mapped_ptr + data.size();
	}
	vkEndCommandBuffer(xfercbuf);

	// get a fence, submit command buffer
	VkFence fence = impl->get_unpooled_fence();
	VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = 1;
	si.pCommandBuffers = &xfercbuf;
	if (!any_image_transfers) {
		// only buffers, single submit to transfer, fence waits on single cbuf
		submit_transfer(si, fence);
	} else {
		vkEndCommandBuffer(dstcbuf);
		// buffers and images, submit to transfer, signal sema
		si.signalSemaphoreCount = 1;
		auto sema = impl->get_unpooled_sema();
		si.pSignalSemaphores = &sema;
		submit_transfer(si, VkFence{ VK_NULL_HANDLE });
		// second submit, to dst queue ideally, but for now to graphics
		si.signalSemaphoreCount = 0;
		si.waitSemaphoreCount = 1;
		si.pWaitSemaphores = &sema;
		// mipping happens in STAGE_TRANSFER for now
		VkPipelineStageFlags wait = VK_PIPELINE_STAGE_TRANSFER_BIT;
		si.pWaitDstStageMask = &wait;
		si.pCommandBuffers = &dstcbuf;
		// stash semaphore
		head_bundle->sema = sema;
		// submit with fence
		submit_graphics(si, fence);
	}

	head_bundle->fence = fence;
	return head_bundle;
}

bool vuk::Context::poll_upload(LinearResourceAllocator* ur) {
	if (vkGetFenceStatus(device, ur->fence) == VK_SUCCESS) {
		std::lock_guard _(impl->transient_submit_lock);
		impl->cleanup_transient_bundle_recursively(ur);
		return true;
	} else {
		return false;
	}
}

vuk::Unique<vuk::Buffer> vuk::Context::allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment) {
	bool create_mapped = mem_usage == MemoryUsage::eCPUonly || mem_usage == MemoryUsage::eCPUtoGPU || mem_usage == MemoryUsage::eGPUtoCPU;
	return Unique{ *this, impl->allocator.allocate_buffer(mem_usage, buffer_usage, size, alignment, create_mapped) };
}

vuk::Texture vuk::Context::allocate_texture(const vuk::ImageCreateInfo& ici) {
	auto dst = impl->allocator.create_image(ici);
	vuk::ImageViewCreateInfo ivci;
	ivci.format = ici.format;
	ivci.image = dst;
	ivci.subresourceRange.aspectMask = format_to_aspect(ici.format);
	ivci.subresourceRange.baseArrayLayer = 0;
	ivci.subresourceRange.baseMipLevel = 0;
	ivci.subresourceRange.layerCount = 1;
	ivci.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
	ivci.viewType = vuk::ImageViewType::e2D;
	vuk::Texture tex{ Unique<Image>(*this, dst), create_image_view(ivci) };
	tex.extent = ici.extent;
	tex.format = ici.format;
	return tex;
}

vuk::Unique<vuk::ImageView> vuk::Context::create_image_view(vuk::ImageViewCreateInfo ivci) {
	VkImageView iv;
	vkCreateImageView(device, (VkImageViewCreateInfo*)&ivci, nullptr, &iv);
	return vuk::Unique<vuk::ImageView>(*this, wrap(iv));
}

VkRenderPass vuk::Context::acquire_renderpass(const vuk::RenderPassCreateInfo& rpci) {
	return impl->renderpass_cache.acquire(rpci, frame_counter);
}

void vuk::Context::enqueue_destroy(vuk::Image i) {
	std::lock_guard _(impl->recycle_locks[frame_counter % FC]);
	impl->image_recycle[frame_counter % FC].push_back(i);
}

void vuk::Context::enqueue_destroy(vuk::ImageView iv) {
	std::lock_guard _(impl->recycle_locks[frame_counter % FC]);
	impl->image_view_recycle[frame_counter % FC].push_back(iv.payload);
}

void vuk::Context::enqueue_destroy(VkPipeline p) {
	std::lock_guard _(impl->recycle_locks[frame_counter % FC]);
	impl->pipeline_recycle[frame_counter % FC].push_back(p);
}

void vuk::Context::enqueue_destroy(vuk::Buffer b) {
	std::lock_guard _(impl->recycle_locks[frame_counter % FC]);
	impl->buffer_recycle[frame_counter % FC].push_back(b);
}

void vuk::Context::enqueue_destroy(vuk::PersistentDescriptorSet b) {
	std::lock_guard _(impl->recycle_locks[frame_counter % FC]);
	impl->pds_recycle[frame_counter % FC].push_back(std::move(b));
}

vuk::Token vuk::Context::create_token() {
	return impl->create_token();
}

vuk::TokenData& vuk::Context::get_token_data(Token t) {
	return impl->get_token_data(t);
}

void vuk::TokenWithContext::operator+=(Token other) {
	auto& data = ctx.impl->get_token_data(token);
	assert(data.next == nullptr);
	data.next = &ctx.impl->get_token_data(other);
}

vuk::TokenWithContext vuk::Context::transition_image(vuk::Texture& t, vuk::Access src_access, vuk::Access dst_access) {
	Token tok = impl->create_token();
	auto& data = impl->get_token_data(tok);
	data.rg = std::make_unique<vuk::RenderGraph>();
	data.rg->attach_image("_transition", ImageAttachment::from_texture(t), src_access, dst_access);
	data.state = TokenData::State::eArmed;
	return { *this, tok };
}

vuk::TokenWithContext vuk::Context::copy_to_image(vuk::Image dst, vuk::Format format, vuk::Extent3D extent, uint32_t base_layer, void* src_data, size_t size) {
	Token tok = impl->create_token();
	auto& data = impl->get_token_data(tok);

	data.resources = impl->get_linear_allocator(transfer_queue_family_index);

	// compute staging buffer alignment as texel block size
	size_t alignment = format_to_texel_block_size(format);
	auto src = impl->allocator.allocate_buffer(vuk::MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, size, alignment, true);
	memcpy(src.mapped_ptr, src_data, size);
	data.resources->buffer = src;

	data.rg = std::make_unique<vuk::RenderGraph>();

	data.rg->add_pass({
		.executes_on = vuk::Domain::eTransfer,
		.resources = {"_dst"_image(vuk::Access::eTransferDst), "_src"_buffer(vuk::Access::eTransferSrc)},
		.execute = [base_layer](vuk::CommandBuffer& command_buffer) {
			command_buffer.copy_image_to_buffer("_src", "_dst", vuk::BufferImageCopy{.imageSubresource = {.baseArrayLayer = base_layer}});
		}
		});

	Access src_access = vuk::eNone;
	vuk::ImageAttachment ia;
	ia.image = dst;
	ia.image_view = {};
	ia.format = format;
	ia.extent = vuk::Extent2D{ extent.width, extent.height };
	ia.sample_count = vuk::Samples::e1;

	data.rg->attach_buffer("_src", src, vuk::Access::eNone, vuk::Access::eNone);
	data.rg->attach_image("_dst", ia, src_access, vuk::eConsume);
	data.state = TokenData::State::eArmed;
	return { *this, tok };
}

vuk::TokenWithContext vuk::Context::copy_to_buffer(vuk::Domain copy_domain, vuk::Buffer buffer, void* src_data, size_t size) {
	Token tok = impl->create_token();
	auto& data = impl->get_token_data(tok);

	// host-mapped buffers just get memcpys
	if (buffer.mapped_ptr) {
		memcpy(buffer.mapped_ptr, src_data, size);
		data.state = TokenData::State::eComplete;
		return { *this, tok };
	}

	data.resources = impl->get_linear_allocator(graphics_queue_family_index);

	auto src = impl->allocator.allocate_buffer(vuk::MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, size, 1, true);
	memcpy(src.mapped_ptr, src_data, size);
	data.resources->buffer = src;

	data.rg = std::make_unique<vuk::RenderGraph>();
	data.rg->add_pass({
		.executes_on = copy_domain,
		.resources = {"_dst"_buffer(vuk::Access::eTransferDst), "_src"_buffer(vuk::Access::eTransferSrc)},
		.execute = [size](vuk::CommandBuffer& command_buffer) {
			command_buffer.copy_buffer("_src", "_dst", VkBufferCopy{.size = size});
		} });
	data.rg->attach_buffer("_src", src, vuk::Access::eNone, vuk::Access::eNone);
	data.rg->attach_buffer("_dst", buffer, vuk::Access::eNone, vuk::Access::eNone);
	data.state = TokenData::State::eArmed;
	return { *this, tok };
}

void vuk::GlobalAllocator::deallocate(const RGImage& image) {
	vkDestroyImageView(device, image.image_view.payload, nullptr);
	device_memory_allocator->destroy_image(image.image);
}

void vuk::GlobalAllocator::deallocate(const PoolAllocator& v) {
	device_memory_allocator->destroy(v);
}

void vuk::GlobalAllocator::deallocate(const LinearAllocator& v) {
	device_memory_allocator->destroy(v);
}

void vuk::GlobalAllocator::deallocate(const vuk::DescriptorPool& dp) {
	for (auto& p : dp.pools) {
		vkDestroyDescriptorPool(device, p, nullptr);
	}
}

void vuk::GlobalAllocator::deallocate(const vuk::PipelineInfo& pi) {
	vkDestroyPipeline(device, pi.pipeline, nullptr);
}

void vuk::GlobalAllocator::deallocate(const vuk::ShaderModule& sm) {
	vkDestroyShaderModule(device, sm.shader_module, nullptr);
}

void vuk::GlobalAllocator::deallocate(const vuk::DescriptorSetLayoutAllocInfo& ds) {
	vkDestroyDescriptorSetLayout(device, ds.layout, nullptr);
}

void vuk::GlobalAllocator::deallocate(const VkPipelineLayout& pl) {
	vkDestroyPipelineLayout(device, pl, nullptr);
}

void vuk::GlobalAllocator::deallocate(const VkRenderPass& rp) {
	vkDestroyRenderPass(device, rp, nullptr);
}

void vuk::GlobalAllocator::deallocate(const vuk::DescriptorSet&) {
	// no-op, we destroy the pools
}

void vuk::GlobalAllocator::deallocate(const VkFramebuffer& fb) {
	vkDestroyFramebuffer(device, fb, nullptr);
}

void vuk::GlobalAllocator::deallocate(const vuk::Sampler& sa) {
	vkDestroySampler(device, sa.payload, nullptr);
}

void vuk::GlobalAllocator::deallocate(const vuk::PipelineBaseInfo& pbi) {
	// no-op, we don't own device objects
}

vuk::Context::~Context() {
	vkDeviceWaitIdle(device);
	for (auto& s : impl->swapchains) {
		for (auto& swiv : s.image_views) {
			vkDestroyImageView(device, swiv.payload, nullptr);
		}
		vkDestroySwapchainKHR(device, s.swapchain, nullptr);
	}
	for (auto& cp : impl->transient_submit_bundles) {
		if (cp.cpool != VK_NULL_HANDLE) {
			vkDestroyCommandPool(device, cp.cpool, nullptr);
		}
	}
	vkDestroyPipelineCache(device, impl->vk_pipeline_cache, nullptr);
	delete impl;
}

vuk::InflightContext vuk::Context::begin() {
	std::lock_guard _(impl->begin_frame_lock);
	std::lock_guard recycle(impl->recycle_locks[_next(frame_counter.load(), FC)]);
	return InflightContext(*this, ++frame_counter, std::move(recycle));
}

void vuk::Context::wait_idle() {
	vkDeviceWaitIdle(device);
}