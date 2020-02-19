#include <stdio.h>
#include <vector>
#include <unordered_map>
#include "VkBootstrap.h"
#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
#include "Pipeline.hpp"
#include "Program.hpp"

GLFWwindow* create_window_glfw(bool resize = true)
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    if (!resize) glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    return glfwCreateWindow(640, 480, "Vulkan Triangle", NULL, NULL);
}
void destroy_window_glfw(GLFWwindow* window)
{
    glfwDestroyWindow(window);
    glfwTerminate();
}
VkSurfaceKHR create_surface_glfw(VkInstance instance, GLFWwindow* window){
    VkSurfaceKHR surface = nullptr;
    VkResult err = glfwCreateWindowSurface(instance, window, NULL, &surface);
    if (err)
    {
        const char* error_msg;
        int ret = glfwGetError(&error_msg);
        if (ret != 0)
        {
            std::cout << ret << " ";
            if (error_msg != nullptr) std::cout << error_msg;
            std::cout << "\n";
        }
        surface = nullptr;
    }
    return surface;
}

#include "Context.hpp"
#include "Cache.hpp"

void device_init() {
	vkb::InstanceBuilder builder;
	builder.setup_validation_layers()
		.set_debug_callback([](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			VkDebugUtilsMessageTypeFlagsEXT messageType,
			const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
			void* pUserData) -> VkBool32 {
				auto ms = vkb::to_string_message_severity(messageSeverity);
				auto mt = vkb::to_string_message_type(messageType);
				printf("[%s: %s](user defined)\n%s\n", ms, mt, pCallbackData->pMessage);
				return VK_FALSE;
			})
		.set_app_name("vuk_example")
				.set_engine_name("vuk")
				.set_api_version(1, 2, 0)
				.set_app_version(0, 1, 0);
			auto inst_ret = builder.build();
			if (!inst_ret.has_value()) {
				// error
			}
			vkb::Instance inst = inst_ret.value();

			vkb::PhysicalDeviceSelector selector{ inst };
			auto window = create_window_glfw();
			auto surface = create_surface_glfw(inst.instance, window);
			selector.set_surface(surface)
				.set_minimum_version(1, 0);
			auto phys_ret = selector.select();
			if (!phys_ret.has_value()) {
				// error
			}
			vkb::PhysicalDevice physical_device = phys_ret.value();

			vkb::DeviceBuilder device_builder{ physical_device };
			auto dev_ret = device_builder.build();
			if (!dev_ret.has_value()) {
				// error
			}
			vkb::Device vkbdevice = dev_ret.value();
			vk::Queue graphics_queue = vkb::get_graphics_queue(vkbdevice).value();
			vk::Device device = vkbdevice.device;

			vkb::SwapchainBuilder swb(vkbdevice);
			auto vkswapchain = swb.build();
			vk::SwapchainKHR swapchain = vkswapchain->swapchain;
			vk::AttachmentReference attachmentReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };
			{
				vuk::Context context(device);
				// Subpass containing first draw
				vk::SubpassDescription subpass;
				subpass.colorAttachmentCount = 1;
				subpass.pColorAttachments = &attachmentReference;

				vk::SubpassDependency dependency;
				dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
				dependency.dstSubpass = 0;
				// .srcStageMask needs to be a part of pWaitDstStageMask in the WSI semaphore.
				dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
				dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
				dependency.srcAccessMask = vk::AccessFlags{};
				dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

				vk::AttachmentDescription attachmentDescription;
				attachmentDescription.format = vk::Format(vkswapchain->image_format);
				attachmentDescription.loadOp = vk::AttachmentLoadOp::eClear;
				attachmentDescription.storeOp = vk::AttachmentStoreOp::eStore;
				// The image will automatically be transitioned from UNDEFINED to COLOR_ATTACHMENT_OPTIMAL for rendering, then out to PRESENT_SRC_KHR at the end.
				attachmentDescription.initialLayout = vk::ImageLayout::eUndefined;
				// Presenting images in Vulkan requires a special layout.
				attachmentDescription.finalLayout = vk::ImageLayout::ePresentSrcKHR;

				vk::RenderPassCreateInfo rpci;
				rpci.attachmentCount = 1;
				rpci.pAttachments = &attachmentDescription;
				rpci.subpassCount = 1;
				rpci.pSubpasses = &subpass;
				rpci.dependencyCount = 1;
				rpci.pDependencies = &dependency;
				
				{
					vk::GraphicsPipelineCreateInfo gpci;
					gpci.stageCount = 2;
					Program prog;
					prog.shaders.push_back("../../triangle.vert");
					prog.shaders.push_back("../../triangle.frag");
					prog.compile("");
					prog.link(device);
					Pipeline pipe(&prog);
					pipe.descriptorSetLayout = device.createDescriptorSetLayout(pipe.descriptorLayout);
					pipe.pipelineLayoutCreateInfo.pSetLayouts = &pipe.descriptorSetLayout;
					pipe.pipelineLayoutCreateInfo.setLayoutCount = 1;
					pipe.pipelineLayout = device.createPipelineLayout(pipe.pipelineLayoutCreateInfo);
					gpci.layout = pipe.pipelineLayout;
					gpci.stageCount = prog.pipeline_shader_stage_CIs.size();
					gpci.pStages = prog.pipeline_shader_stage_CIs.data();
					gpci.pVertexInputState = &pipe.inputState;
					pipe.inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;
					gpci.pInputAssemblyState = &pipe.inputAssemblyState;
					pipe.rasterizationState.lineWidth = 1.f;
					gpci.pRasterizationState = &pipe.rasterizationState;
					pipe.colorBlendState.attachmentCount = 1;
					vk::PipelineColorBlendAttachmentState pcba;
					pcba.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
					pipe.colorBlendState.pAttachments = &pcba;
					gpci.pColorBlendState = &pipe.colorBlendState;
					gpci.pMultisampleState = &pipe.multisampleState;
					gpci.pViewportState = &pipe.viewportState;
					gpci.pDepthStencilState = &pipe.depthStencilState;
					gpci.pDynamicState = &pipe.dynamicState;
					//auto gp = device.createGraphicsPipelineUnique(*context.pipeline_cache, gpci);

					auto swapimages = vkb::get_swapchain_images(*vkswapchain);
					auto swapimageviews = *vkb::get_swapchain_image_views(*vkswapchain, *swapimages);
					auto swapChainImages = device.getSwapchainImagesKHR(swapchain);
					vk::ImageViewCreateInfo colorAttachmentView;
					colorAttachmentView.format = vk::Format(vkswapchain->image_format);
					colorAttachmentView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
					colorAttachmentView.subresourceRange.levelCount = 1;
					colorAttachmentView.subresourceRange.layerCount = 1;
					colorAttachmentView.viewType = vk::ImageViewType::e2D;

					while (!glfwWindowShouldClose(window)) {
						glfwPollEvents();
						auto ictx = context.begin();
						auto pfc = ictx.begin();
						auto cbufs = pfc.commandbuffer_pool.acquire(1);
						auto& cbuf = cbufs[0];

						gpci.renderPass = ictx.renderpass_cache.acquire(rpci);
						auto pipe = ictx.pipeline_cache.acquire(gpci);
						auto render_complete = pfc.semaphore_pool.acquire(1)[0];
						auto present_rdy = pfc.semaphore_pool.acquire(1)[0];
						auto acq_result = device.acquireNextImageKHR(swapchain, UINT64_MAX, present_rdy, vk::Fence{});
						auto index = acq_result.value;
						vk::CommandBufferBeginInfo cbi;
						cbi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
						cbuf.begin(cbi);

						vk::FramebufferCreateInfo fci;
						fci.attachmentCount = 1;
						fci.layers = 1;
						fci.renderPass = gpci.renderPass;
						fci.width = vkswapchain->extent.width;
						fci.height = vkswapchain->extent.height;
						vk::ImageView view = swapimageviews[index];
						fci.pAttachments = &view;
						auto fb = device.createFramebufferUnique(fci);

						vk::RenderPassBeginInfo rbi;
						rbi.renderPass = gpci.renderPass;
						rbi.framebuffer = *fb;
						rbi.clearValueCount = 1;
						vk::ClearColorValue ccv;
						ccv.setFloat32({ 0.3f, 0.3f, 0.3f, 1.f });
						vk::ClearValue cv;
						cv.setColor(ccv);
						rbi.pClearValues = &cv;
						cbuf.beginRenderPass(rbi, vk::SubpassContents::eInline);
						cbuf.setViewport(0, vk::Viewport(0, vkswapchain->extent.height, vkswapchain->extent.width, -1.f * (float)vkswapchain->extent.height, 0.f, 1.f));
						cbuf.setScissor(0, vk::Rect2D({ 0,0 }, { vkswapchain->extent.width, vkswapchain->extent.height }));
						cbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe);
						cbuf.draw(3, 1, 0, 0);
						cbuf.endRenderPass();
						cbuf.end();
						vk::SubmitInfo si;
						si.commandBufferCount = 1;
						si.pCommandBuffers = &cbuf;
						si.pSignalSemaphores = &render_complete;
						si.signalSemaphoreCount = 1;
						si.waitSemaphoreCount = 1;
						si.pWaitSemaphores = &present_rdy;
						vk::PipelineStageFlags flags = vk::PipelineStageFlagBits::eAllGraphics;
						si.pWaitDstStageMask = &flags;
						graphics_queue.submit(si, {});

						vk::PresentInfoKHR pi;
						pi.swapchainCount = 1;
						pi.pSwapchains = &swapchain;
						pi.pImageIndices = &acq_result.value;
						pi.waitSemaphoreCount = 1;
						pi.pWaitSemaphores = &render_complete;
						graphics_queue.presentKHR(pi);
						graphics_queue.waitIdle();
					}
				}
			}
			vkDestroySurfaceKHR(inst.instance, surface, nullptr);
			destroy_window_glfw(window);
			vkb::destroy_device(dev_ret.value());
			vkb::destroy_instance(inst);
}

int main() {
	device_init();
}