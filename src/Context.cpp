#include "Context.hpp"
#include "RenderGraph.hpp"
#include <shaderc/shaderc.hpp>
#include "Program.hpp"
#include <fstream>
#include <sstream>
#include <spirv_cross.hpp>

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

void vuk::Context::DebugUtils::set_name(const vuk::ImageView& iv, Name name) {
	if (!enabled()) return;
	vk::DebugUtilsObjectNameInfoEXT info;
	info.pObjectName = name.data();
	info.objectType = iv.payload.objectType;
	info.objectHandle = reinterpret_cast<uint64_t>((VkImageView)iv.payload);
	setDebugUtilsObjectNameEXT(ctx.device, &(VkDebugUtilsObjectNameInfoEXT)info);
}

void vuk::Context::DebugUtils::begin_region(const vk::CommandBuffer& cb, Name name, std::array<float, 4> color) {
	if (!enabled()) return;
	vk::DebugUtilsLabelEXT label;
	label.pLabelName = name.data();
	::memcpy(label.color, color.data(), sizeof(float) * 4);
	cmdBeginDebugUtilsLabelEXT(cb, &(VkDebugUtilsLabelEXT)label);
}

void vuk::Context::DebugUtils::end_region(const vk::CommandBuffer& cb) {
	if (!enabled()) return;
	cmdEndDebugUtilsLabelEXT(cb);
}
