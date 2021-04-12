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
	/*descriptor_bindings[array_index].image = vuk::DescriptorImageInfo(ptc.acquire_sampler(sci), iv, layout);
	descriptor_bindings[array_index].type = vuk::DescriptorType::eCombinedImageSampler;
	VkWriteDescriptorSet wds = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	wds.descriptorCount = 1;
	wds.descriptorType = (VkDescriptorType)vuk::DescriptorType::eCombinedImageSampler;
	wds.dstArrayElement = array_index;
	wds.dstBinding = binding;
	wds.pImageInfo = &descriptor_bindings[array_index].image.dii;
	wds.dstSet = backing_set;
	pending_writes.push_back(wds);*/
}

void vuk::PersistentDescriptorSet::update_storage_image(PerThreadContext& ptc, unsigned binding, unsigned array_index, vuk::ImageView iv) {
	/*descriptor_bindings[array_index].image = vuk::DescriptorImageInfo({}, iv, vuk::ImageLayout::eGeneral);
	descriptor_bindings[array_index].type = vuk::DescriptorType::eStorageImage;
	VkWriteDescriptorSet wds = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	wds.descriptorCount = 1;
	wds.descriptorType = (VkDescriptorType)vuk::DescriptorType::eStorageImage;
	wds.dstArrayElement = array_index;
	wds.dstBinding = binding;
	wds.pImageInfo = &descriptor_bindings[array_index].image.dii;
	wds.dstSet = backing_set;
	pending_writes.push_back(wds);*/
}

/*vuk::Query vuk::Context::create_timestamp_query() {
	return { impl->query_id_counter++ };
}*/

void vuk::Context::create_named_pipeline(vuk::Name name, vuk::PipelineBaseInfo& ci) {
	std::lock_guard _(impl->named_pipelines_lock);
	impl->named_pipelines.insert_or_assign(name, &ci);
}

void vuk::Context::create_named_pipeline(vuk::Name name, vuk::ComputePipelineCreateInfo ci) {
	std::lock_guard _(impl->named_pipelines_lock);
	//impl->named_compute_pipelines.insert_or_assign(name, &impl->compute_pipeline_cache.acquire(std::move(ci)));
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
	return nullptr;// &impl->pipelinebase_cache.acquire(pbci);
}

vuk::ComputePipelineInfo* vuk::Context::get_pipeline(const vuk::ComputePipelineCreateInfo& pbci) {
	return nullptr;// &impl->compute_pipeline_cache.acquire(pbci);
}

vuk::Program vuk::Context::get_pipeline_reflection_info(vuk::PipelineBaseCreateInfo pci) {
	return {};
	/*auto& res = impl->pipelinebase_cache.acquire(pci);
	return res.reflection_info;*/
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

void vuk::Context::destroy_token(Token token) {
	TokenData* data = &get_token_data(token);
	//vkDestroySemaphore(ctx.device, data.resources->sema, nullptr);
	while (data != nullptr) {
		impl->cleanup_transient_bundle_recursively(data->resources);
		data = data->next;
	}
}

vuk::Token vuk::Context::submit(Allocator& allocator, vuk::Token token, vuk::Domain domain) {
	TokenData::TokenType token_type;
	/*if (domain & vuk::Domain::eHost) {
		token_type = TokenData::TokenType::eTimeline;
	}*/
	TokenData* data = &allocator.get_token_data(token);
	TokenData* odata = data;
	data->token_type = TokenData::TokenType::eTimeline;
	assert(data->state == TokenData::State::eArmed);

	std::vector<VkCommandBuffer> cbufs;

	LinearResourceAllocator<Allocator>* lallocator;
	if (!data->resources) {
		// TODO: map domain to queue family
		data->resources = new LinearResourceAllocator<Allocator>(allocator);
	}

	lallocator = data->resources;

	while (data != nullptr) {
		if (data->rg) {
			ExecutableRenderGraph erg = std::move(*data->rg).link();
			cbufs.push_back(erg.execute(*lallocator, {}).command_buffers[0]); // TODO: the waits and signals
		}
		data->state = TokenData::State::ePending;
		data = data->next;
	}

	if (cbufs.size() == 0) {
		//TODO: free token
		odata->state = TokenData::State::eComplete;
		return token;
	}

	lallocator->sema = lallocator->allocate_timeline_semaphore(0, 0, VUK_HERE());

	// enqueue for destruction on frame end
	//impl->linear_allocators.push_back(allocator);

	VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = cbufs.size();
	si.pCommandBuffers = cbufs.data();
	VkTimelineSemaphoreSubmitInfo tssi{ .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
	uint64_t signal = 1;
	si.pSignalSemaphores = &lallocator->sema;
	si.signalSemaphoreCount = 1;
	tssi.pSignalSemaphoreValues = &signal;
	tssi.signalSemaphoreValueCount = 1;
	si.pNext = &tssi;

	submit_graphics(si, {});

	return token;
}

void vuk::Context::wait(vuk::Allocator& allocator, Token token) {
	TokenData& data = allocator.get_token_data(token);
	if (data.state == TokenData::State::eComplete) {
		return;
	}
	assert(data.state == TokenData::State::ePending && "Token must have been submitted to be waited on.");
	assert(data.token_type == TokenData::TokenType::eTimeline && "Can only wait on Timeline tokens on host");
	VkSemaphoreWaitInfo swi{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };

	auto* resources = static_cast<LinearResourceAllocator<Allocator>*>(data.resources);
	swi.pSemaphores = &resources->sema;
	uint64_t value = 1;
	swi.pValues = &value;
	swi.semaphoreCount = 1;
	vkWaitSemaphores(device, &swi, UINT64_MAX);
	data.state = TokenData::State::eComplete;
	allocator.destroy(token);
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
	//vkDestroyPipelineCache(device, impl->vk_pipeline_cache, nullptr);
	delete impl;
}

void vuk::Context::wait_idle() {
	vkDeviceWaitIdle(device);
}