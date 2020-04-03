#include "Context.hpp"
#include "Context.hpp"
#include "Context.hpp"
#include "Context.hpp"
#include "Context.hpp"
#include "Context.hpp"
#include "Context.hpp"
#include "Context.hpp"
#include "Context.hpp"
#include "RenderGraph.hpp"
#include <shaderc/shaderc.hpp>
#include <algorithm>
#include "Program.hpp"
#include <fstream>
#include <sstream>
#include <spirv_cross.hpp>

std::string slurp(const std::string& path) {
	std::ostringstream buf;
	std::ifstream input(path.c_str());
	buf << input.rdbuf();
	return buf.str();
}

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
		ctx.device.destroy(iv);
	}
	ctx.image_view_recycle[frame].clear();

	for (auto& sb : scratch_buffers.cache.data[frame].pool) {
		ctx.allocator.reset_pool(sb);
	}

	auto ptc = begin();
	ptc.descriptor_sets.collect(Context::FC * 2);
}

vuk::TransferStub vuk::InflightContext::enqueue_transfer(Buffer src, Buffer dst) {
	std::lock_guard _(transfer_mutex);
	TransferStub stub{ transfer_id++ };
	buffer_transfer_commands.push({ src, dst, stub });
	return stub;
}

vuk::TransferStub vuk::InflightContext::enqueue_transfer(Buffer src, vk::Image dst, vk::Extent3D extent) {
	std::lock_guard _(transfer_mutex);
	TransferStub stub{ transfer_id++ };
	bufferimage_transfer_commands.push({ src, dst, extent, stub });
	return stub;
}

void vuk::InflightContext::wait_all_transfers() {
	std::lock_guard _(transfer_mutex);

	while (!pending_transfers.empty()) {
		ctx.device.waitForFences(pending_transfers.front().fence, true, UINT64_MAX);
		auto last = pending_transfers.front();
		last_transfer_complete = last.last_transfer_id;
		pending_transfers.pop();
	}
}

void vuk::InflightContext::destroy(std::vector<vk::Image>&& images) {
	std::lock_guard _(recycle_lock);
	ctx.image_recycle[frame].insert(ctx.image_recycle[frame].end(), images.begin(), images.end());
}


void vuk::execute_submit_and_present_to_one(PerThreadContext& ptc, RenderGraph& rg, SwapchainRef swapchain) {
	auto render_complete = ptc.semaphore_pool.acquire(1)[0];
	auto present_rdy = ptc.semaphore_pool.acquire(1)[0];
	auto acq_result = ptc.ctx.device.acquireNextImageKHR(swapchain->swapchain, UINT64_MAX, present_rdy, vk::Fence{});
	auto index = acq_result.value;

	std::vector<std::pair<SwapChainRef, size_t>> swapchains_with_indexes = { { swapchain, index } };

	auto cb = rg.execute(ptc, swapchains_with_indexes);

	vk::SubmitInfo si;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cb;
	si.pSignalSemaphores = &render_complete;
	si.signalSemaphoreCount = 1;
	si.waitSemaphoreCount = 1;
	si.pWaitSemaphores = &present_rdy;
	vk::PipelineStageFlags flags = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	si.pWaitDstStageMask = &flags;
	auto fence = ptc.fence_pool.acquire(1)[0];
	ptc.ctx.graphics_queue.submit(si, fence);
	vk::PresentInfoKHR pi;
	pi.swapchainCount = 1;
	pi.pSwapchains = &swapchain->swapchain;
	pi.pImageIndices = &acq_result.value;
	pi.waitSemaphoreCount = 1;
	pi.pWaitSemaphores = &render_complete;
	ptc.ctx.graphics_queue.presentKHR(pi);
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

void vuk::Context::DebugUtils::begin_region(const vk::CommandBuffer& cb, Name name, std::array<float, 4> color) {
	if (!enabled()) return;
	VkDebugUtilsLabelEXT label;
	label.pNext = nullptr;
	label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
	label.pLabelName = name.data();
	::memcpy(label.color, color.data(), sizeof(float) * 4);
	cmdBeginDebugUtilsLabelEXT(cb, &label);
}

void vuk::Context::DebugUtils::end_region(const vk::CommandBuffer& cb) {
	if (!enabled()) return;
	cmdEndDebugUtilsLabelEXT(cb);
}

vuk::PerThreadContext::PerThreadContext(InflightContext& ifc, unsigned tid) : ctx(ifc.ctx), ifc(ifc), tid(tid),
commandbuffer_pool(ifc.commandbuffer_pools.get_view(*this)),
semaphore_pool(ifc.semaphore_pools.get_view(*this)),
fence_pool(ifc.fence_pools.get_view(*this)),
pipeline_cache(*this, ifc.pipeline_cache),
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
pipeline_layouts(*this, ifc.pipeline_layouts) {}

vuk::PerThreadContext::~PerThreadContext() {
	ifc.destroy(std::move(image_recycle));
	ifc.destroy(std::move(image_view_recycle));
}

void vuk::PerThreadContext::destroy(vk::Image image) {
	image_recycle.push_back(image);
}

void vuk::PerThreadContext::destroy(vuk::ImageView image) {
	image_view_recycle.push_back(image.payload);
}

void vuk::PerThreadContext::destroy(vuk::DescriptorSet ds) {
	// note that since we collect at integer times FC, we are releasing the DS back to the right pool
	pool_cache.acquire(ds.layout_info).free_sets.push_back(ds.descriptor_set);
}

vuk::Buffer vuk::PerThreadContext::_allocate_scratch_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, size_t size, bool create_mapped) {
	auto& pool = scratch_buffers.acquire({ mem_usage, buffer_usage });
	return ifc.ctx.allocator.allocate_buffer(pool, size, create_mapped);
}

bool vuk::PerThreadContext::is_ready(const TransferStub& stub) {
	return ifc.last_transfer_complete >= stub.id;
}

void vuk::PerThreadContext::wait_all_transfers() {
	// TODO: remove when we go MT
	dma_task(); // run one transfer so it is more easy to follow
	return ifc.wait_all_transfers();
}

std::pair<vuk::Texture, vuk::TransferStub> vuk::PerThreadContext::create_texture(vk::Format format, vk::Extent3D extents, void* data) {
	vk::ImageCreateInfo ici;
	ici.format = format;
	ici.extent = extents;
	ici.arrayLayers = 1;
	ici.initialLayout = vk::ImageLayout::eUndefined;
	ici.mipLevels = 1;
	ici.imageType = vk::ImageType::e2D;
	ici.samples = vk::SampleCountFlagBits::e1;
	ici.tiling = vk::ImageTiling::eOptimal;
	ici.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	auto dst = ifc.ctx.allocator.create_image(ici);
	auto stub = upload(dst, extents, gsl::span<std::byte>((std::byte*)data, extents.width * extents.height * extents.depth * 4));
	vk::ImageViewCreateInfo ivci;
	ivci.format = format;
	ivci.image = dst;
	ivci.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	ivci.subresourceRange.baseArrayLayer = 0;
	ivci.subresourceRange.baseMipLevel = 0;
	ivci.subresourceRange.layerCount = 1;
	ivci.subresourceRange.levelCount = 1;
	ivci.viewType = vk::ImageViewType::e2D;
	auto iv = ifc.ctx.device.createImageView(ivci);
	vuk::Texture tex{ vuk::Unique<vk::Image>(ifc.ctx, dst), vuk::Unique<vuk::ImageView>(ifc.ctx, ifc.ctx.wrap(iv)) };
	tex.extent = extents;
	tex.format = format;
	return { std::move(tex), stub };
}

void vuk::PerThreadContext::dma_task() {
	std::lock_guard _(ifc.transfer_mutex);
	while (!ifc.pending_transfers.empty() && ctx.device.getFenceStatus(ifc.pending_transfers.front().fence) == vk::Result::eSuccess) {
		auto last = ifc.pending_transfers.front();
		ifc.last_transfer_complete = last.last_transfer_id;
		ifc.pending_transfers.pop();
	}

	if (ifc.buffer_transfer_commands.empty() && ifc.bufferimage_transfer_commands.empty()) return;
	auto cbuf = commandbuffer_pool.acquire(1)[0];
	cbuf.begin(vk::CommandBufferBeginInfo{});
	size_t last = 0;
	while (!ifc.buffer_transfer_commands.empty()) {
		auto task = ifc.buffer_transfer_commands.front();
		ifc.buffer_transfer_commands.pop();
		vk::BufferCopy bc;
		bc.dstOffset = task.dst.offset;
		bc.srcOffset = task.src.offset;
		bc.size = task.src.size;
		cbuf.copyBuffer(task.src.buffer, task.dst.buffer, bc);
		last = std::max(last, task.stub.id);
	}
	while (!ifc.bufferimage_transfer_commands.empty()) {
		auto task = ifc.bufferimage_transfer_commands.front();
		ifc.bufferimage_transfer_commands.pop();
		vk::BufferImageCopy bc;
		bc.bufferOffset = task.src.offset;
		bc.imageOffset = 0;
		bc.bufferRowLength = 0;
		bc.bufferImageHeight = 0;
		bc.imageExtent = task.extent;
		bc.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		bc.imageSubresource.baseArrayLayer = 0;
		bc.imageSubresource.mipLevel = 0;
		bc.imageSubresource.layerCount = 1;

		vk::ImageMemoryBarrier copy_barrier;
		copy_barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		copy_barrier.oldLayout = vk::ImageLayout::eUndefined;
		copy_barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
		copy_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		copy_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		copy_barrier.image = task.dst;
		copy_barrier.subresourceRange.aspectMask = bc.imageSubresource.aspectMask;
		copy_barrier.subresourceRange.layerCount = bc.imageSubresource.layerCount;
		copy_barrier.subresourceRange.baseArrayLayer = bc.imageSubresource.baseArrayLayer;
		copy_barrier.subresourceRange.baseMipLevel = bc.imageSubresource.mipLevel;
		copy_barrier.subresourceRange.levelCount = 1;

		vk::ImageMemoryBarrier use_barrier;
		use_barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		use_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
		use_barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
		use_barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		use_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		use_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		use_barrier.image = task.dst;
		use_barrier.subresourceRange = copy_barrier.subresourceRange;

		cbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits(0), {}, {}, copy_barrier);
		cbuf.copyBufferToImage(task.src.buffer, task.dst, vk::ImageLayout::eTransferDstOptimal, bc);
		cbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, vk::DependencyFlagBits(0), {}, {}, use_barrier);
		last = std::max(last, task.stub.id);
	}
	cbuf.end();
	auto fence = fence_pool.acquire(1)[0];
	vk::SubmitInfo si;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cbuf;
	ifc.ctx.graphics_queue.submit(si, fence);
	ifc.pending_transfers.emplace(InflightContext::PendingTransfer{ last, fence });
}

vuk::SampledImage& vuk::PerThreadContext::make_sampled_image(vuk::ImageView iv, vk::SamplerCreateInfo sci) {
	vuk::SampledImage si(vuk::SampledImage::Global{ iv, sci, vk::ImageLayout::eShaderReadOnlyOptimal });
	return sampled_images.acquire(si);
}

vuk::SampledImage& vuk::PerThreadContext::make_sampled_image(Name n, vk::SamplerCreateInfo sci) {
	vuk::SampledImage si(vuk::SampledImage::RenderGraphAttachment{ n, sci, vk::ImageLayout::eShaderReadOnlyOptimal });
	return sampled_images.acquire(si);
}



vuk::DescriptorSet vuk::PerThreadContext::create(const create_info_t<vuk::DescriptorSet>& cinfo) {
	auto& pool = pool_cache.acquire(cinfo.layout_info);
	auto ds = pool.acquire(*this, cinfo.layout_info);
	auto mask = cinfo.used.to_ulong();
	unsigned long leading_ones = num_leading_ones(mask);
	std::array<vk::WriteDescriptorSet, VUK_MAX_BINDINGS> writes;
	std::array<vk::DescriptorImageInfo, VUK_MAX_BINDINGS> diis;
	for (unsigned i = 0; i < leading_ones; i++) {
		if (!cinfo.used.test(i)) continue;
		auto& write = writes[i];
		auto& binding = cinfo.bindings[i];
		write.descriptorType = binding.type;
		write.dstArrayElement = 0;
		write.descriptorCount = 1;
		write.dstBinding = i;
		write.dstSet = ds;
		switch (binding.type) {
		case vk::DescriptorType::eUniformBuffer:
		case vk::DescriptorType::eStorageBuffer:
			write.pBufferInfo = &binding.buffer;
			break;
		case vk::DescriptorType::eSampledImage:
		case vk::DescriptorType::eSampler:
		case vk::DescriptorType::eCombinedImageSampler:
			diis[i] = binding.image;
			write.pImageInfo = &diis[i];
			break;
		default:
			assert(0);
		}
	}
	ctx.device.updateDescriptorSets(leading_ones, writes.data(), 0, nullptr);
	return { ds, cinfo.layout_info };
}

vuk::Allocator::Pool vuk::PerThreadContext::create(const create_info_t<vuk::Allocator::Pool>& cinfo) {
	return ctx.allocator.allocate_pool(cinfo.mem_usage, cinfo.buffer_usage);
}

vuk::RGImage vuk::PerThreadContext::create(const create_info_t<vuk::RGImage>& cinfo) {
	RGImage res;
	res.image = ctx.allocator.create_image_for_rendertarget(cinfo.ici);
	auto ivci = cinfo.ivci;
	ivci.image = res.image;
	std::string name = std::string("Image: RenderTarget ") + std::string(cinfo.name);
	ctx.debug.set_name(res.image, name);
	name = std::string("ImageView: RenderTarget ") + std::string(cinfo.name);
	res.image_view = ctx.wrap(ctx.device.createImageView(ivci));
	ctx.debug.set_name(res.image_view.payload, name);
	return res;
}

vk::RenderPass vuk::PerThreadContext::create(const create_info_t<vk::RenderPass>& cinfo) {
	return ctx.device.createRenderPass(cinfo);
}

vuk::ShaderModule vuk::PerThreadContext::create(const create_info_t<vuk::ShaderModule>& cinfo) {
	shaderc::Compiler compiler;
	shaderc::CompileOptions options;

	shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(cinfo.source, shaderc_glsl_infer_from_source, cinfo.filename.c_str(), options);

	if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
		printf("%s", module.GetErrorMessage().c_str());
		//Platform::log->error("%s", module.GetErrorMessage().c_str());
		return {};
	} else {
		std::vector<uint32_t> spirv(module.cbegin(), module.cend());

		spirv_cross::Compiler refl(spirv.data(), spirv.size());
		vuk::Program p;
		auto stage = p.introspect(refl);

		vk::ShaderModuleCreateInfo moduleCreateInfo;
		moduleCreateInfo.codeSize = spirv.size() * sizeof(uint32_t);
		moduleCreateInfo.pCode = (uint32_t*)spirv.data();
		auto module = ctx.device.createShaderModule(moduleCreateInfo);
		std::string name = "ShaderModule: " + cinfo.filename;
		ctx.debug.set_name(module, name);
		return { module, p, stage };
	}
}

vuk::PipelineInfo vuk::PerThreadContext::create(const create_info_t<PipelineInfo>& cinfo) {
	std::vector<vk::PipelineShaderStageCreateInfo> psscis;

	// accumulate descriptors from all stages
	vuk::Program accumulated_reflection;
	std::string pipe_name = "Pipeline:";
	for (auto& path : cinfo.shaders) {
		auto contents = slurp(path);
		auto& sm = shader_modules.acquire({ contents, path });
		vk::PipelineShaderStageCreateInfo shaderStage;
		shaderStage.pSpecializationInfo = nullptr;
		shaderStage.stage = sm.stage;
		shaderStage.module = sm.shader_module;
		shaderStage.pName = "main"; //TODO: make param
		psscis.push_back(shaderStage);
		accumulated_reflection.append(sm.reflection_info);
		pipe_name += path + "+";
	}
	pipe_name = pipe_name.substr(0, pipe_name.size() - 1); //trim off last "+"
														   // acquire descriptor set layouts (1 per set)
														   // acquire pipeline layout
	vuk::PipelineLayoutCreateInfo plci;
	plci.dslcis = vuk::PipelineCreateInfo::build_descriptor_layouts(accumulated_reflection);
	plci.pcrs = accumulated_reflection.push_constant_ranges;
	plci.plci.pushConstantRangeCount = (uint32_t)accumulated_reflection.push_constant_ranges.size();
	plci.plci.pPushConstantRanges = accumulated_reflection.push_constant_ranges.data();
	std::array<vuk::DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> dslai;
	std::vector<vk::DescriptorSetLayout> dsls;
	for (auto& dsl : plci.dslcis) {
		dsl.dslci.bindingCount = (uint32_t)dsl.bindings.size();
		dsl.dslci.pBindings = dsl.bindings.data();
		auto l = descriptor_set_layouts.acquire(dsl);
		dslai[dsl.index] = l;
		dsls.push_back(dslai[dsl.index].layout);
	}
	plci.plci.pSetLayouts = dsls.data();
	plci.plci.setLayoutCount = (uint32_t)dsls.size();
	// create gfx pipeline
	vk::GraphicsPipelineCreateInfo gpci = cinfo.to_vk();
	gpci.layout = pipeline_layouts.acquire(plci);
	gpci.pStages = psscis.data();
	gpci.stageCount = (uint32_t)psscis.size();

	auto pipeline = ctx.device.createGraphicsPipeline(*ctx.vk_pipeline_cache, gpci);
	ctx.debug.set_name(pipeline, pipe_name);
	return { pipeline, gpci.layout, dslai };
}

vk::Framebuffer vuk::PerThreadContext::create(const create_info_t<vk::Framebuffer>& cinfo) {
	return ctx.device.createFramebuffer(cinfo);
}

vk::Sampler vuk::PerThreadContext::create(const create_info_t<vk::Sampler>& cinfo) {
	return ctx.device.createSampler(cinfo);
}

vuk::DescriptorSetLayoutAllocInfo vuk::PerThreadContext::create(const create_info_t<vuk::DescriptorSetLayoutAllocInfo>& cinfo) {
	vuk::DescriptorSetLayoutAllocInfo ret;
	ret.layout = ctx.device.createDescriptorSetLayout(cinfo.dslci);
	for (auto& b : cinfo.bindings) {
		ret.descriptor_counts[to_integral(b.descriptorType)] += b.descriptorCount;
	}
	return ret;
}

vk::PipelineLayout vuk::PerThreadContext::create(const create_info_t<vk::PipelineLayout>& cinfo) {
	return ctx.device.createPipelineLayout(cinfo.plci);
}

vuk::DescriptorPool vuk::PerThreadContext::create(const create_info_t<vuk::DescriptorPool>& cinfo) {
	return vuk::DescriptorPool{};
}

vuk::SwapchainRef vuk::Context::add_swapchain(Swapchain sw) {
	std::lock_guard _(swapchains_lock);
	sw.image_views.reserve(sw._ivs.size());
	for (auto& v : sw._ivs) {
		sw.image_views.push_back(wrap(vk::ImageView{ v }));
	}

	return &*swapchains.emplace(sw);
}

vuk::Context::Context(vk::Instance instance, vk::Device device, vk::PhysicalDevice physical_device, vk::Queue graphics) :
	instance(instance),
	device(device),
	physical_device(physical_device),
	graphics_queue(graphics),
	allocator(instance, device, physical_device),
	cbuf_pools(*this),
	semaphore_pools(*this),
	fence_pools(*this),
	pipeline_cache(*this),
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
	vk_pipeline_cache = device.createPipelineCacheUnique({});
}

void vuk::Context::create_named_pipeline(const char* name, vuk::PipelineCreateInfo ci) {
	std::lock_guard _(named_pipelines_lock);
	named_pipelines.emplace(name, std::move(ci));
}

vuk::PipelineCreateInfo vuk::Context::get_named_pipeline(const char* name) {
	return named_pipelines.at(name);
}

void vuk::Context::invalidate_shadermodule_and_pipelines(Name filename) {
	vuk::ShaderModuleCreateInfo sci;
	sci.filename = filename;
	auto sm = shader_modules.invalidate(sci);
	auto pipe = pipeline_cache.invalidate([&](auto& ci, auto& p) {
		if (std::find(ci.shaders.begin(), ci.shaders.end(), std::string(filename)) != ci.shaders.end()) return true;
		return false;
	});
	while (pipe != std::nullopt) {
		pipe = pipeline_cache.invalidate([&](auto& ci, auto& p) {
			if (std::find(ci.shaders.begin(), ci.shaders.end(), std::string(filename)) != ci.shaders.end()) return true;
			return false;
		});
	}
}

void vuk::Context::enqueue_destroy(vk::Image i) {
	std::lock_guard _(recycle_locks[frame_counter % FC]);
	image_recycle[frame_counter % FC].push_back(i);
}

void vuk::Context::enqueue_destroy(vuk::ImageView iv) {
	std::lock_guard _(recycle_locks[frame_counter % FC]);
	image_view_recycle[frame_counter % FC].push_back(iv.payload);
}

void vuk::Context::destroy(const RGImage& image) {
	device.destroy(image.image_view.payload);
	allocator.destroy_image(image.image);
}

void vuk::Context::destroy(const Allocator::Pool& v) {
	allocator.destroy_scratch_pool(v);
}

void vuk::Context::destroy(const vuk::DescriptorPool& dp) {
	for (auto& p : dp.pools) {
		device.destroy(p);
	}
}

void vuk::Context::destroy(vuk::PipelineInfo pi) {
	device.destroy(pi.pipeline);
}

void vuk::Context::destroy(vuk::ShaderModule sm) {
	device.destroy(sm.shader_module);
}

void vuk::Context::destroy(vuk::DescriptorSetLayoutAllocInfo ds) {
	device.destroy(ds.layout);
}

void vuk::Context::destroy(vk::PipelineLayout pl) {
	device.destroy(pl);
}

void vuk::Context::destroy(vk::RenderPass rp) {
	device.destroy(rp);
}

void vuk::Context::destroy(vuk::DescriptorSet) {
	// no-op, we destroy the pools
}

void vuk::Context::destroy(vk::Framebuffer fb) {
	device.destroy(fb);
}

void vuk::Context::destroy(vk::Sampler sa) {
	device.destroy(sa);
}

vuk::Context::~Context() {
	device.waitIdle();
	for (auto& s : swapchains) {
		for (auto& swiv : s.image_views) {
			device.destroy(swiv.payload);
		}
		device.destroy(s.swapchain);
	}
}

vuk::InflightContext vuk::Context::begin() {
	std::lock_guard _(begin_frame_lock);
	std::lock_guard recycle(recycle_locks[_next(frame_counter.load(), FC)]);
	return InflightContext(*this, frame_counter++, std::move(recycle));
}

void vuk::Context::wait_idle() {
	device.waitIdle();
}

void vuk::InflightContext::destroy(std::vector<vk::ImageView>&& images) {
	std::lock_guard _(recycle_lock);
	ctx.image_view_recycle[frame].insert(ctx.image_view_recycle[frame].end(), images.begin(), images.end());
}

vuk::PerThreadContext vuk::InflightContext::begin() {
	return PerThreadContext{ *this, 0 };
}

