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
#include <plf_colony.h>
#include <mutex>
namespace vuk {
	class Context;
	class InflightContext;
	class PerThreadContext;

	template<class T>
	struct PooledType {
		std::vector<T> values;
		size_t needle = 0;

		PooledType(Context&) {}
		gsl::span<T> acquire(PerThreadContext& ptc, size_t count);
		void reset(Context& ctx) { needle = 0; }
		void free(Context& ctx);
	};

	template<>
	struct PooledType<vk::CommandBuffer> {
		vk::UniqueCommandPool pool;
		std::vector<vk::CommandBuffer> values;
		size_t needle = 0;

		PooledType(Context&);
		gsl::span<vk::CommandBuffer> acquire(PerThreadContext& ptc, size_t count);
		void reset(Context&);
		void free(Context&);
	};

	template<class T, size_t FC>
	struct PFView;

	template<class T, size_t FC>
	struct Pool {
		std::mutex lock;
		plf::colony<PooledType<T>> store;
		std::array<plf::colony<PooledType<T>>, FC> per_frame_storage;
		Context& ctx;

		Pool(Context& ctx) : ctx(ctx) {	}

		PooledType<T>* acquire_one_into(plf::colony<PooledType<T>>& dst) {
			std::lock_guard _(lock);
			if (!store.empty()) {
				auto& last_elem = *(--store.end());
				auto new_it = dst.emplace(std::move(last_elem));
				store.erase(--store.end());
				return &*new_it;
			} else {
				return &*dst.emplace(PooledType<T>(ctx));
			}
		}

		void reset(unsigned frame) {
			std::lock_guard _(lock);
			for (auto& t : per_frame_storage[frame]) {
				t.reset(ctx);
			}
			store.splice(per_frame_storage[frame]);
		}

		~Pool() {
			// return all to pool
			for (auto& pf : per_frame_storage) {
				for (auto& s : pf) {
					s.free(ctx);
				}
			}
			for (auto& s : store) {
				s.free(ctx);
			}
		}

		PFView<T, FC> get_view(InflightContext& ctx);
	};

	template<class T>
	struct PFPTView {
		PerThreadContext& ptc;
		PooledType<T>& pool;

		PFPTView(PerThreadContext& ptc, PooledType<T>& pool) : ptc(ptc), pool(pool) {}

		gsl::span<T> acquire(size_t count) {
			return pool.acquire(ptc, count);
		}
	};

	template<class T, size_t FC>
	struct PFView {
		std::mutex lock;
		Pool<T, FC>& storage;
		InflightContext& ifc;
		plf::colony<PooledType<T>>& frame_values;

		PFView(InflightContext& ifc, Pool<T, FC>& storage, plf::colony<PooledType<T>>& fv) : ifc(ifc), storage(storage), frame_values(fv) {}

		PFPTView<T> get_view(PerThreadContext& ptc) {
			std::lock_guard _(lock);
			return { ptc, *storage.acquire_one_into(frame_values) };
		}
	};

	class Context {
	public:
		constexpr static size_t FC = 3;

		vk::Device device;
		Pool<vk::CommandBuffer, FC> cbuf_pools;
		Pool<vk::Semaphore, FC> semaphore_pools;

		Context(vk::Device device) : device(device), 
			cbuf_pools(*this),
			semaphore_pools(*this)
		{}

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
		PFView<vk::CommandBuffer, Context::FC> commandbuffer_pools;
		PFView<vk::Semaphore, Context::FC> semaphore_pools;

		InflightContext(Context& ctx, unsigned frame) : ctx(ctx), frame(frame),
			commandbuffer_pools(ctx.cbuf_pools.get_view(*this)),
			semaphore_pools(ctx.semaphore_pools.get_view(*this))
		{
			auto prev_frame = prev_(frame, 1, Context::FC);
			ctx.cbuf_pools.reset(prev_frame);
			ctx.semaphore_pools.reset(prev_frame);
		}

		PerThreadContext begin();
	};

	InflightContext Context::begin() {
		return InflightContext(*this, frame_counter++ % FC);
	}

	class PerThreadContext {
	public:
		Context& ctx;
		InflightContext& ifc;
		unsigned tid;
		PFPTView<vk::CommandBuffer> commandbuffer_pool;
		PFPTView<vk::Semaphore> semaphore_pool;

		PerThreadContext(InflightContext& ifc, unsigned tid) : ctx(ifc.ctx), ifc(ifc), tid(tid),
			commandbuffer_pool(ifc.commandbuffer_pools.get_view(*this)),
			semaphore_pool(ifc.semaphore_pools.get_view(*this))
		{}

	};
	
	PerThreadContext InflightContext::begin() {
		return PerThreadContext{*this, 0};
	}

	template<class T, size_t FC>
	PFView<T, FC> Pool<T, FC>::get_view(InflightContext& ctx) {
		return PFView<T, FC>(ctx, *this, per_frame_storage[ctx.frame]);
	}

	// pools
	
	gsl::span<vk::Semaphore> PooledType<vk::Semaphore>::acquire(PerThreadContext& ptc, size_t count) {
		if (values.size() < (needle + count)) {
			auto remaining = values.size() - needle;
			for (auto i = 0; i < remaining; i++) {
				auto nalloc = ptc.ctx.device.createSemaphore({});
				values.push_back(nalloc);
			}
		}
		gsl::span<vk::Semaphore> ret{ &*values.begin() + needle, (ptrdiff_t)count };
		needle += count;
		return ret;
	}

	template<class T>
	void PooledType<T>::free(Context& ctx){
		for (auto& v : values) {
			ctx.device.destroy(v);
		}
	}

	// vk::CommandBuffer pool
	PooledType<vk::CommandBuffer>::PooledType(Context& ctx){
		pool = ctx.device.createCommandPoolUnique({});
	}

	gsl::span<vk::CommandBuffer> PooledType<vk::CommandBuffer>::acquire(PerThreadContext& ptc, size_t count) {
		if (values.size() < (needle + count)) {
			auto remaining = values.size() - needle;
			vk::CommandBufferAllocateInfo cbai;
			cbai.commandBufferCount = count - remaining;
			cbai.commandPool = *pool;
			cbai.level = vk::CommandBufferLevel::ePrimary;
			auto nalloc = ptc.ctx.device.allocateCommandBuffers(cbai);
			values.insert(values.end(), nalloc.begin(), nalloc.end());
		}
		gsl::span<vk::CommandBuffer> ret{ &*values.begin() + needle, (ptrdiff_t)count };
		needle += count;
		return ret;
	}
	void PooledType<vk::CommandBuffer>::reset(Context& ctx) {
		vk::CommandPoolResetFlags flags = {};
		ctx.device.resetCommandPool(*pool, flags);
		needle = 0;
	}
	
	void PooledType<vk::CommandBuffer>::free(Context& ctx){
		ctx.device.freeCommandBuffers(*pool, values);
		pool.reset();
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
						auto pfc = ictx.begin();
						auto cbufs = pfc.commandbuffer_pool.acquire(1);
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
			}
			vkDestroySurfaceKHR(inst.instance, surface, nullptr);
			destroy_window_glfw(window);
			vkb::destroy_device(dev_ret.value());
			vkb::destroy_instance(inst);
}

int main() {
	device_init();
}