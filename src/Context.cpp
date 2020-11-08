#include <shaderc/shaderc.hpp>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <spirv_cross.hpp>

#include "vuk/Context.hpp"
#include "ContextImpl.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/Program.hpp"

void burp(const std::string& in, const std::string& path) {
	std::ofstream output(path.c_str(), std::ios::trunc);
	if (!output.is_open()) {
	}
	output << in;
	output.close();
}

vuk::Context::Context(VkInstance instance, VkDevice device, VkPhysicalDevice physical_device, VkQueue graphics) :
	instance(instance),
	device(device),
	physical_device(physical_device),
	graphics_queue(graphics),
	debug(*this),
	impl(new ContextImpl(*this)){}


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

void vuk::Context::submit_graphics(VkSubmitInfo si, VkFence fence) {
	std::lock_guard _(impl->gfx_queue_lock);
	vkQueueSubmit(graphics_queue, 1, &si, fence);
}

void vuk::Context::submit_transfer(VkSubmitInfo si, VkFence fence) {
	std::lock_guard _(impl->xfer_queue_lock);
	vkQueueSubmit(transfer_queue, 1, &si, fence);
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

vuk::PipelineBaseInfo vuk::Context::create(const create_info_t<PipelineBaseInfo>& cinfo) {
	std::vector<VkPipelineShaderStageCreateInfo> psscis;

	// accumulate descriptors from all stages
	vuk::Program accumulated_reflection;
	std::string pipe_name = "Pipeline:";
	for (auto i = 0; i < cinfo.shaders.size(); i++) {
		auto contents = cinfo.shaders[i];
		if (contents.empty())
			continue;
		auto& sm = impl->shader_modules.acquire({ contents, cinfo.shader_paths[i] });
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
		auto descset_layout_alloc_info = impl->descriptor_set_layouts.acquire(dsl);
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
	pbi.pipeline_layout = impl->pipeline_layouts.acquire(plci);
	pbi.rasterization_state = cinfo.rasterization_state;
	pbi.pipeline_name = std::move(pipe_name);
	pbi.reflection_info = accumulated_reflection;
	pbi.binding_flags = cinfo.binding_flags;
	pbi.variable_count_max = cinfo.variable_count_max;
	return pbi;
}

vuk::ComputePipelineInfo vuk::Context::create(const create_info_t<vuk::ComputePipelineInfo>& cinfo) {
	VkPipelineShaderStageCreateInfo shader_stage{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	std::string pipe_name = "Compute:";
	auto& sm = impl->shader_modules.acquire({ cinfo.shader, cinfo.shader_path });
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
		auto descset_layout_alloc_info = impl->descriptor_set_layouts.acquire(dsl);
		dslai[dsl.index] = descset_layout_alloc_info;
		dsls.push_back(dslai[dsl.index].layout);
	}
	plci.plci.pSetLayouts = dsls.data();
	plci.plci.setLayoutCount = (uint32_t)dsls.size();

	VkComputePipelineCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	cpci.stage = shader_stage;
	cpci.layout = impl->pipeline_layouts.acquire(plci);
	VkPipeline pipeline;
	vkCreateComputePipelines(device, impl->vk_pipeline_cache, 1, &cpci, nullptr, &pipeline);
	debug.set_name(pipeline, pipe_name);
	return { { pipeline, cpci.layout, dslai }, sm.reflection_info.local_size };
}

bool vuk::Context::load_pipeline_cache(std::span<uint8_t> data) {
	VkPipelineCacheCreateInfo pcci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = data.size_bytes(), .pInitialData = data.data() };
	vkDestroyPipelineCache(device, impl->vk_pipeline_cache, nullptr);
	vkCreatePipelineCache(device, &pcci, nullptr, &impl->vk_pipeline_cache);
	return true;
}

std::vector<uint8_t> vuk::Context::save_pipeline_cache() {
	size_t size;
	std::vector<uint8_t> data;
	vkGetPipelineCacheData(device, impl->vk_pipeline_cache, &size, nullptr);
	data.resize(size);
	vkGetPipelineCacheData(device, impl->vk_pipeline_cache, &size, data.data());
	return data;
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

VkPipelineLayout vuk::Context::create(const create_info_t<VkPipelineLayout>& cinfo) {
	VkPipelineLayout pl;
	vkCreatePipelineLayout(device, &cinfo.plci, nullptr, &pl);
	return pl;
}

vuk::SwapchainRef vuk::Context::add_swapchain(Swapchain sw) {
	std::lock_guard _(impl->swapchains_lock);
	sw.image_views.reserve(sw._ivs.size());
	for (auto& v : sw._ivs) {
		sw.image_views.push_back(wrap(v));
	}

	return &*impl->swapchains.emplace(sw);
}

void vuk::Context::create_named_pipeline(const char* name, vuk::PipelineBaseCreateInfo ci) {
	std::lock_guard _(impl->named_pipelines_lock);
	impl->named_pipelines.insert_or_assign(name, &impl->pipelinebase_cache.acquire(std::move(ci)));
}

void vuk::Context::create_named_pipeline(const char* name, vuk::ComputePipelineCreateInfo ci) {
	std::lock_guard _(impl->named_pipelines_lock);
	impl->named_compute_pipelines.insert_or_assign(name, &impl->compute_pipeline_cache.acquire(std::move(ci)));
}

vuk::PipelineBaseInfo* vuk::Context::get_named_pipeline(const char* name) {
	std::lock_guard _(impl->named_pipelines_lock);
	return impl->named_pipelines.at(name);
}

vuk::ComputePipelineInfo* vuk::Context::get_named_compute_pipeline(const char* name) {
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

vuk::ShaderModule vuk::Context::compile_shader(std::string source, Name path) {
	vuk::ShaderModuleCreateInfo sci;
	sci.filename = path;
	sci.source = std::move(source);
	auto sm = impl->shader_modules.remove(sci);
	if (sm) {
		vkDestroyShaderModule(device, sm->shader_module, nullptr);
	}
	return impl->shader_modules.acquire(sci);
}

vuk::Context::UploadResult vuk::Context::fenced_upload(std::span<BufferUpload> uploads) {
	// get a one time command buffer
	auto tid = get_thread_index ? get_thread_index() : 0;

	VkCommandBuffer cbuf;
	{
		std::lock_guard _(impl->one_time_pool_lock);
		if (impl->xfer_one_time_pools.size() < (tid + 1)) {
			impl->xfer_one_time_pools.resize(tid + 1, VK_NULL_HANDLE);
		}

		auto& pool = impl->xfer_one_time_pools[tid];
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
	auto staging_alloc = impl->allocator.allocate_buffer(vuk::MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, size, 1, true);
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
	submit_transfer(si, fence);
	return { fence, cbuf, staging_alloc, true, tid };
}

vuk::Context::UploadResult vuk::Context::fenced_upload(std::span<ImageUpload> uploads) {
	// get a one time command buffer
	auto tid = get_thread_index ? get_thread_index() : 0;
	VkCommandBuffer cbuf;
	{
		std::lock_guard _(impl->one_time_pool_lock);
		if (impl->one_time_pools.size() < (tid + 1)) {
			impl->one_time_pools.resize(tid + 1, VK_NULL_HANDLE);
		}
		auto& pool = impl->one_time_pools[tid];
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
	auto staging_alloc = impl->allocator.allocate_buffer(vuk::MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, size, 1, true);
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
	submit_graphics(si, fence);
	return { fence, cbuf, staging_alloc, false, tid };
}

void vuk::Context::free_upload_resources(const UploadResult& ur) {
	auto& pools = ur.is_buffer ? impl->xfer_one_time_pools : impl->one_time_pools;
	std::lock_guard _(impl->one_time_pool_lock);
	vkFreeCommandBuffers(device, pools[ur.thread_index], 1, &ur.command_buffer);
	impl->allocator.free_buffer(ur.staging);
	vkDestroyFence(device, ur.fence, nullptr);
}

vuk::Buffer vuk::Context::allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment) {
	return impl->allocator.allocate_buffer(mem_usage, buffer_usage, size, alignment, false);
}

vuk::Texture vuk::Context::allocate_texture(vuk::ImageCreateInfo ici) {
	auto dst = impl->allocator.create_image(ici);
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


void vuk::Context::destroy(const RGImage& image) {
	vkDestroyImageView(device, image.image_view.payload, nullptr);
	impl->allocator.destroy_image(image.image);
}

void vuk::Context::destroy(const Allocator::Pool& v) {
	impl->allocator.destroy(v);
}

void vuk::Context::destroy(const Allocator::Linear& v) {
	impl->allocator.destroy(v);
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
	for (auto& s : impl->swapchains) {
		for (auto& swiv : s.image_views) {
			vkDestroyImageView(device, swiv.payload, nullptr);
		}
		vkDestroySwapchainKHR(device, s.swapchain, nullptr);
	}
	for (auto& cp : impl->one_time_pools) {
		if (cp != VK_NULL_HANDLE) {
			vkDestroyCommandPool(device, cp, nullptr);
		}
	}
	for (auto& cp : impl->xfer_one_time_pools) {
		if (cp != VK_NULL_HANDLE) {
			vkDestroyCommandPool(device, cp, nullptr);
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