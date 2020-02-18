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
#include <atomic>
#include <gsl/span>
namespace vuk {
	template<class T, size_t FC>
	struct Pool {
		std::array<std::vector<T>, FC> values;
		std::array<size_t, FC> needles;
	};

	class Context;
	class InflightContext;

	template<class T>
	struct PoolView;

	template<class T>
	struct PooledType;

	template<>
	struct PooledType<vk::CommandBuffer> {
		vk::UniqueCommandPool pool;
		std::vector<vk::CommandBuffer> command_buffers;
		size_t needle = 0;
	};

	template<size_t FC>
	struct Pool<vk::CommandBuffer, FC> {
		Context& ctx;
		Pool(Context& ctx);

		std::array<PooledType<vk::CommandBuffer>, FC> values;

		PoolView<vk::CommandBuffer> get_view(InflightContext& ctx);
		void reset(unsigned frame);
	};

	template<class T>
	struct PoolView {
		InflightContext& ifc;
		PooledType<T>& pool;

		PoolView(InflightContext& ifc, PooledType<T>& pool) : ifc(ifc), pool(pool) {}

		gsl::span<T> acquire(size_t count);
	};

	class Context {
	public:
		constexpr static size_t FC = 3;

		vk::Device device;
		Pool<vk::CommandBuffer, FC> cbuf_pools;

		Context(vk::Device device) : device(device), cbuf_pools(*this) {}

		std::atomic<size_t> frame_counter = 0;
		InflightContext begin();
	};

	unsigned prev_(unsigned frame, unsigned amt, unsigned FC) {
		return ((frame - amt) % FC) + ((frame >= amt) ? 0 : FC - 1);
	}

	class InflightContext {
	public:
		Context& ctx;
		unsigned frame;
		PoolView<vk::CommandBuffer> commandbuffer_pool;
		InflightContext(Context& ctx, unsigned frame) : ctx(ctx), frame(frame),
			commandbuffer_pool(ctx.cbuf_pools.get_view(*this)){
			ctx.cbuf_pools.reset(prev_(frame, 1, ctx.FC));
		}
	};

	template<size_t FC>
	Pool<vk::CommandBuffer, FC>::Pool(Context& ctx) : ctx(ctx){
		for (auto& v : values) {
			v.pool = ctx.device.createCommandPoolUnique({});
		}
	}

	template<size_t FC>
	PoolView<vk::CommandBuffer> Pool<vk::CommandBuffer, FC>::get_view(InflightContext& ctx) {
		return PoolView<vk::CommandBuffer>(ctx, values[ctx.frame]);
	}

	gsl::span<vk::CommandBuffer> PoolView<vk::CommandBuffer>::acquire(size_t count) {
		if (pool.command_buffers.size() >= (pool.needle + count)) {
			gsl::span<vk::CommandBuffer> ret{ &*pool.command_buffers.begin() + pool.needle, (ptrdiff_t)count };
			pool.needle += count;
			return ret;
		}
		auto remaining = pool.command_buffers.size() - pool.needle;
		vk::CommandBufferAllocateInfo cbai;
		cbai.commandBufferCount = count - remaining;
		cbai.commandPool = *pool.pool;
		cbai.level = vk::CommandBufferLevel::ePrimary;
		auto nalloc = ifc.ctx.device.allocateCommandBuffers(cbai);
		pool.command_buffers.insert(pool.command_buffers.end(), nalloc.begin(), nalloc.end());
		gsl::span<vk::CommandBuffer> ret{ &*pool.command_buffers.begin() + pool.needle, (ptrdiff_t)count };
		pool.needle += count;
		return ret;
	}

	template<size_t FC>
	void Pool<vk::CommandBuffer, FC>::reset(unsigned frame) {
		vk::CommandPoolResetFlags flags = {};
		ctx.device.resetCommandPool(*values[frame].pool, flags);
		values[frame].needle = 0;
	}

	InflightContext Context::begin() {
		return InflightContext(*this, frame_counter++ % FC);
	}
};

void device_init()
{
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

			vk::RenderPassCreateInfo renderPassCreateInfo;
			renderPassCreateInfo.attachmentCount = 1;
			renderPassCreateInfo.pAttachments = &attachmentDescription;
			renderPassCreateInfo.subpassCount = 1;
			renderPassCreateInfo.pSubpasses = &subpass;
			renderPassCreateInfo.dependencyCount = 1;
			renderPassCreateInfo.pDependencies = &dependency;
			auto rp = device.createRenderPass(renderPassCreateInfo);
			{
				vk::PipelineCacheCreateInfo pci;
				auto pc = device.createPipelineCacheUnique(pci);
				vk::GraphicsPipelineCreateInfo gpci;
				gpci.renderPass = rp;
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
				auto gp = device.createGraphicsPipelineUnique(*pc, gpci);

				auto cp = device.createCommandPoolUnique({});
				vk::CommandBufferAllocateInfo cba;
				cba.commandBufferCount = 1;
				cba.commandPool = *cp;

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
					auto cbufs = ictx.commandbuffer_pool.acquire(1);
					auto& cbuf = cbufs[0];

					auto render_complete = device.createSemaphoreUnique({});
					auto present_rdy = device.createSemaphoreUnique({});
					auto acq_result = device.acquireNextImageKHR(swapchain, UINT64_MAX, *present_rdy, vk::Fence{});
					auto index = acq_result.value;
					vk::CommandBufferBeginInfo cbi;
					cbi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
					cbuf.begin(cbi);

					vk::FramebufferCreateInfo fci;
					fci.attachmentCount = 1;
					fci.layers = 1;
					fci.renderPass = rp;
					fci.width = vkswapchain->extent.width;
					fci.height = vkswapchain->extent.height;
					vk::ImageView view = swapimageviews[index];
					fci.pAttachments = &view;
					auto fb = device.createFramebufferUnique(fci);

					vk::RenderPassBeginInfo rbi;
					rbi.renderPass = rp;
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
					cbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, *gp);
					cbuf.draw(3, 1, 0, 0);
					cbuf.endRenderPass();
					cbuf.end();
					vk::SubmitInfo si;
					si.commandBufferCount = 1;
					si.pCommandBuffers = &cbuf;
					si.pSignalSemaphores = &*render_complete;
					si.signalSemaphoreCount = 1;
					si.waitSemaphoreCount = 1;
					si.pWaitSemaphores = &*present_rdy;
					vk::PipelineStageFlags flags = vk::PipelineStageFlagBits::eAllGraphics;
					si.pWaitDstStageMask = &flags;
					graphics_queue.submit(si, {});

					vk::PresentInfoKHR pi;
					pi.swapchainCount = 1;
					pi.pSwapchains = &swapchain;
					pi.pImageIndices = &acq_result.value;
					pi.waitSemaphoreCount = 1;
					pi.pWaitSemaphores = &*render_complete;
					graphics_queue.presentKHR(pi);
					graphics_queue.waitIdle();
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