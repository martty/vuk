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
#include "RenderGraph.hpp"
#include "Allocator.hpp"
#include "CommandBuffer.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

using glm::vec3;
using glm::vec2;

struct Vertex {
	vec3 position;
	vec3 normal;
	vec3 tangent;
	vec3 bitangent;
	vec2 uv_coordinates;
};

using Mesh = std::pair<std::vector<Vertex>, std::vector<unsigned>>;

Mesh generate_cube() {
	// clang-format off
	return Mesh(std::vector<Vertex> {
		// back
		Vertex{{-1, -1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {1, 0}},  Vertex{{1, 1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {0, 1}},
		Vertex{{1, -1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {0, 0}},   Vertex{{1, 1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {0, 1}},
		Vertex{{-1, -1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {1, 0}},  Vertex{{-1, 1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {1, 1}},
		// front 
		Vertex{{-1, -1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {0, 0}},   Vertex{{1, -1, 1}, {0, 0, 1},{1, 0.0, 0}, {0, 1, 0}, {1, 0}},
		Vertex{{1, 1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {1, 1}},     Vertex{{1, 1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {1, 1}},
		Vertex{{-1, 1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {0, 1}},    Vertex{{-1, -1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {0, 0}},
		// left 
		Vertex{{-1, 1, -1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1}},    Vertex{{-1, -1, -1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 0}},
		Vertex{{-1, 1, 1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 1}},     Vertex{{-1, -1, -1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 0}},
		Vertex{{-1, -1, 1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0}},    Vertex{{-1, 1, 1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 1}},
		// right 
		Vertex{{1, 1, 1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {0, 1}},      Vertex{{1, -1, -1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {1, 0}},
		Vertex{{1, 1, -1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {1, 1}},     Vertex{{1, -1, -1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {1, 0}},
		Vertex{{1, 1, 1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {0, 1}},      Vertex{{1, -1, 1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {0, 0}},
		// bottom 
		Vertex{{-1, -1, -1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0}},   Vertex{{1, -1, -1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {1, 0}},
		Vertex{{1, -1, 1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {1, 1}},     Vertex{{1, -1, 1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {1, 1}},
		Vertex{{-1, -1, 1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 1}},    Vertex{{-1, -1, -1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0}},
		// top 
		Vertex{{-1, 1, -1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {0, 1}},    Vertex{{1, 1, 1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {1, 0}},
		Vertex{{1, 1, -1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {1, 1}},     Vertex{{1, 1, 1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {1, 0}},
		Vertex{{-1, 1, -1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {0, 1}},    Vertex{{-1, 1, 1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {0, 0}} },
		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35 });
	// clang-format on
}

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
			swb.set_desired_format(vk::SurfaceFormatKHR(vk::Format::eR8G8B8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear));
			auto vkswapchain = swb.build();
			vk::SwapchainKHR swapchain = vkswapchain->swapchain;

			int x, y, chans;
			auto doge_image = stbi_load("../../doge.png", &x, &y, &chans, 4);
			

			{
				vuk::Context context(device, physical_device.phys_device);
				context.graphics_queue = graphics_queue;
				{
					{
						vk::GraphicsPipelineCreateInfo gpci;
						Program* prog = new Program;
						prog->shaders.push_back("../../triangle.vert");
						prog->shaders.push_back("../../triangle.frag");
						prog->compile("");
						prog->link(device);
						Pipeline* pipe = new Pipeline(prog);
						pipe->descriptorSetLayout = device.createDescriptorSetLayout(pipe->descriptorLayout);
						pipe->pipelineLayoutCreateInfo.pSetLayouts = &pipe->descriptorSetLayout;
						pipe->pipelineLayoutCreateInfo.setLayoutCount = 1;
						pipe->pipelineLayout = device.createPipelineLayout(pipe->pipelineLayoutCreateInfo);
						gpci.layout = pipe->pipelineLayout;
						gpci.stageCount = prog->pipeline_shader_stage_CIs.size();
						gpci.pStages = prog->pipeline_shader_stage_CIs.data();
						gpci.pVertexInputState = &pipe->inputState;
						pipe->inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;
						gpci.pInputAssemblyState = &pipe->inputAssemblyState;
						pipe->rasterizationState.lineWidth = 1.f;
						gpci.pRasterizationState = &pipe->rasterizationState;
						pipe->colorBlendState.attachmentCount = 1;
						vk::PipelineColorBlendAttachmentState pcba;
						pcba.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
						pipe->colorBlendState.pAttachments = &pcba;
						gpci.pColorBlendState = &pipe->colorBlendState;
						gpci.pMultisampleState = &pipe->multisampleState;
						gpci.pViewportState = &pipe->viewportState;
						gpci.pDepthStencilState = &pipe->depthStencilState;
						gpci.pDynamicState = &pipe->dynamicState;
						vuk::PipelineCreateInfo pci;
						pci.gpci = gpci;
						pci.layout_info.layout = pipe->descriptorSetLayout;
						pci.pipeline_layout = pipe->pipelineLayout;
						context.named_pipelines.emplace("triangle", pci);
					}
					{
						vk::GraphicsPipelineCreateInfo gpci;
						Program* prog = new Program();
						prog->shaders.push_back("../../cube.vert");
						prog->shaders.push_back("../../triangle.frag");
						prog->compile("");
						prog->link(device);
						Pipeline* pipe = new Pipeline(prog);
						pipe->descriptorSetLayout = device.createDescriptorSetLayout(pipe->descriptorLayout);
						pipe->pipelineLayoutCreateInfo.pSetLayouts = &pipe->descriptorSetLayout;
						pipe->pipelineLayoutCreateInfo.setLayoutCount = 1;
						pipe->pipelineLayout = device.createPipelineLayout(pipe->pipelineLayoutCreateInfo);
						gpci.layout = pipe->pipelineLayout;
						gpci.stageCount = prog->pipeline_shader_stage_CIs.size();
						gpci.pStages = prog->pipeline_shader_stage_CIs.data();
						gpci.pVertexInputState = &pipe->inputState;
						pipe->inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;
						gpci.pInputAssemblyState = &pipe->inputAssemblyState;
						pipe->rasterizationState.lineWidth = 1.f;
						gpci.pRasterizationState = &pipe->rasterizationState;
						pipe->colorBlendState.attachmentCount = 1;
						vk::PipelineColorBlendAttachmentState pcba;
						pcba.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
						pipe->colorBlendState.pAttachments = &pcba;
						gpci.pColorBlendState = &pipe->colorBlendState;
						gpci.pMultisampleState = &pipe->multisampleState;
						gpci.pViewportState = &pipe->viewportState;
						pipe->depthStencilState.depthWriteEnable = true;
						pipe->depthStencilState.depthCompareOp = vk::CompareOp::eAlways;
						pipe->depthStencilState.depthTestEnable = true;
						gpci.pDepthStencilState = &pipe->depthStencilState;
						gpci.pDynamicState = &pipe->dynamicState;

						vuk::PipelineCreateInfo pci;
						pci.gpci = gpci;
						pci.layout_info.layout = pipe->descriptorSetLayout;
						pci.pipeline_layout = pipe->pipelineLayout;

						context.named_pipelines.emplace("cube", pci);
					}

					{
						vk::GraphicsPipelineCreateInfo gpci;
						Program* prog = new Program();
						prog->shaders.push_back("../../ubo_test.vert");
						prog->shaders.push_back("../../triangle_depthshaded.frag");
						prog->compile("");
						prog->link(device);
						Pipeline* pipe = new Pipeline(prog);
						pipe->descriptorSetLayout = device.createDescriptorSetLayout(pipe->descriptorLayout);
						pipe->pipelineLayoutCreateInfo.pSetLayouts = &pipe->descriptorSetLayout;
						pipe->pipelineLayoutCreateInfo.setLayoutCount = 1;
						pipe->pipelineLayout = device.createPipelineLayout(pipe->pipelineLayoutCreateInfo);
						gpci.layout = pipe->pipelineLayout;
						gpci.stageCount = prog->pipeline_shader_stage_CIs.size();
						gpci.pStages = prog->pipeline_shader_stage_CIs.data();
						vk::VertexInputAttributeDescription viad;
						viad.binding = 0;
						viad.format = vk::Format::eR32G32B32Sfloat;
						viad.location = 0;
						viad.offset = 0;
						pipe->attributeDescriptions.push_back(viad);
						pipe->inputState.vertexAttributeDescriptionCount = pipe->attributeDescriptions.size();
						pipe->inputState.pVertexAttributeDescriptions = pipe->attributeDescriptions.data();
						vk::VertexInputBindingDescription vibd;
						vibd.binding = 0;
						vibd.inputRate = vk::VertexInputRate::eVertex;
						vibd.stride = sizeof(Vertex);
						pipe->bindingDescriptions.push_back(vibd);
						pipe->inputState.vertexBindingDescriptionCount = pipe->bindingDescriptions.size();
						pipe->inputState.pVertexBindingDescriptions = pipe->bindingDescriptions.data();
						gpci.pVertexInputState = &pipe->inputState;
						pipe->inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;
						gpci.pInputAssemblyState = &pipe->inputAssemblyState;
						pipe->rasterizationState.lineWidth = 1.f;
						gpci.pRasterizationState = &pipe->rasterizationState;
						pipe->colorBlendState.attachmentCount = 1;
						vk::PipelineColorBlendAttachmentState pcba;
						pcba.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
						pipe->colorBlendState.pAttachments = &pcba;
						gpci.pColorBlendState = &pipe->colorBlendState;
						gpci.pMultisampleState = &pipe->multisampleState;
						gpci.pViewportState = &pipe->viewportState;
						pipe->depthStencilState.depthWriteEnable = true;
						pipe->depthStencilState.depthCompareOp = vk::CompareOp::eLessOrEqual;
						pipe->depthStencilState.depthTestEnable = true;
						gpci.pDepthStencilState = &pipe->depthStencilState;
						gpci.pDynamicState = &pipe->dynamicState;

						vuk::PipelineCreateInfo pci{};
						pci.gpci = gpci;
						pci.layout_info.layout = pipe->descriptorSetLayout;
						pci.layout_info.descriptor_counts[to_integral(vk::DescriptorType::eUniformBuffer)] = 2;
						pci.pipeline_layout = pipe->pipelineLayout;


						context.named_pipelines.emplace("vatt", pci);
					}

					{
						vk::GraphicsPipelineCreateInfo gpci;
						Program* prog = new Program();
						prog->shaders.push_back("../../ubo_test_tex.vert");
						prog->shaders.push_back("../../triangle_depthshaded_tex.frag");
						prog->compile("");
						prog->link(device);
						Pipeline* pipe = new Pipeline(prog);
						pipe->descriptorSetLayout = device.createDescriptorSetLayout(pipe->descriptorLayout);
						pipe->pipelineLayoutCreateInfo.pSetLayouts = &pipe->descriptorSetLayout;
						pipe->pipelineLayoutCreateInfo.setLayoutCount = 1;
						pipe->pipelineLayout = device.createPipelineLayout(pipe->pipelineLayoutCreateInfo);
						gpci.layout = pipe->pipelineLayout;
						gpci.stageCount = prog->pipeline_shader_stage_CIs.size();
						gpci.pStages = prog->pipeline_shader_stage_CIs.data();
						{
							vk::VertexInputAttributeDescription viad;
							viad.binding = 0;
							viad.format = vk::Format::eR32G32B32Sfloat;
							viad.location = 0;
							viad.offset = 0;
							pipe->attributeDescriptions.push_back(viad);
						}
						{
							vk::VertexInputAttributeDescription viad;
							viad.binding = 0;
							viad.format = vk::Format::eR32G32Sfloat;
							viad.location = 1;
							viad.offset = offsetof(Vertex, uv_coordinates);
							pipe->attributeDescriptions.push_back(viad);
						}
						pipe->inputState.vertexAttributeDescriptionCount = pipe->attributeDescriptions.size();
						pipe->inputState.pVertexAttributeDescriptions = pipe->attributeDescriptions.data();
						vk::VertexInputBindingDescription vibd;
						vibd.binding = 0;
						vibd.inputRate = vk::VertexInputRate::eVertex;
						vibd.stride = sizeof(Vertex);
						pipe->bindingDescriptions.push_back(vibd);
						pipe->inputState.vertexBindingDescriptionCount = pipe->bindingDescriptions.size();
						pipe->inputState.pVertexBindingDescriptions = pipe->bindingDescriptions.data();
						gpci.pVertexInputState = &pipe->inputState;
						pipe->inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;
						gpci.pInputAssemblyState = &pipe->inputAssemblyState;
						pipe->rasterizationState.lineWidth = 1.f;
						gpci.pRasterizationState = &pipe->rasterizationState;
						pipe->colorBlendState.attachmentCount = 1;
						vk::PipelineColorBlendAttachmentState pcba;
						pcba.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
						pipe->colorBlendState.pAttachments = &pcba;
						gpci.pColorBlendState = &pipe->colorBlendState;
						gpci.pMultisampleState = &pipe->multisampleState;
						gpci.pViewportState = &pipe->viewportState;
						pipe->depthStencilState.depthWriteEnable = true;
						pipe->depthStencilState.depthCompareOp = vk::CompareOp::eLessOrEqual;
						pipe->depthStencilState.depthTestEnable = true;
						gpci.pDepthStencilState = &pipe->depthStencilState;
						gpci.pDynamicState = &pipe->dynamicState;

						vuk::PipelineCreateInfo pci{};
						pci.gpci = gpci;
						pci.layout_info.layout = pipe->descriptorSetLayout;
						pci.layout_info.descriptor_counts[to_integral(vk::DescriptorType::eUniformBuffer)] = 2;
						pci.layout_info.descriptor_counts[to_integral(vk::DescriptorType::eCombinedImageSampler)] = 1;
						pci.pipeline_layout = pipe->pipelineLayout;

						context.named_pipelines.emplace("vatte", pci);
					}


					auto swapimages = vkb::get_swapchain_images(*vkswapchain);
					auto swapimageviews = *vkb::get_swapchain_image_views(*vkswapchain, *swapimages);

					using glm::vec3;
					float angle = 0.f;


				
					while (!glfwWindowShouldClose(window)) {
						glfwPollEvents();
						auto ifc = context.begin();
						auto ptc = ifc.begin();

						auto render_complete = ptc.semaphore_pool.acquire(1)[0];
						auto present_rdy = ptc.semaphore_pool.acquire(1)[0];
						auto acq_result = device.acquireNextImageKHR(swapchain, UINT64_MAX, present_rdy, vk::Fence{});
						auto index = acq_result.value;

						auto box = generate_cube();
						
						auto [verts, stub1] = ptc.create_scratch_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eVertexBuffer, gsl::span(&box.first[0], box.first.size()));
						auto [inds, stub2] = ptc.create_scratch_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eIndexBuffer, gsl::span(&box.second[0], box.second.size()));
						struct vp {
							glm::mat4 view;
							glm::mat4 proj;
						} vp;
						vp.view = glm::lookAt(vec3(0, 1.5, 3.5), vec3(0), vec3(0, 1, 0));
						vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 0.1f, 10.f);

						auto [ubo, stub3] = ptc.create_scratch_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eUniformBuffer, gsl::span(&vp, 1));

						auto model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), vec3(0.f, 1.f, 0.f)));
						angle += 1.f;
						auto [ubom, stub4] = ptc.create_scratch_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eUniformBuffer, gsl::span(&model, 1));

						auto [img, iv, stub5] = ptc.create_image(vk::Format::eR8G8B8A8Srgb, vk::Extent3D(x,y,1), doge_image);
						ptc.destroy(img);
						ptc.destroy(iv);
						ptc.wait_all_transfers();

						vuk::RenderGraph rg;
						rg.add_pass({
							.color_attachments = {{"SWAPCHAIN"}}, 
							.depth_attachment = Attachment{"depth"},
							.execute = [&](vuk::CommandBuffer& command_buffer) {
								command_buffer
								  .set_viewport(vk::Viewport(0, 480, 640, -1.f * 480, 0.f, 1.f))
								  .set_scissor(vk::Rect2D({ 0,0 }, { 640, 480 }))
								  .bind_pipeline("vatte")
								  .bind_uniform_buffer(0, 0, ubo)
								  .bind_uniform_buffer(0, 1, ubom)
								  .bind_sampled_image(0, 2, iv, vk::SamplerCreateInfo{})
								  .bind_vertex_buffer(verts)
								  .bind_index_buffer(inds)
								  .draw_indexed(box.second.size(), 1, 0, 0, 0);
								}
							}
						);
						rg.add_pass({
							.color_attachments = {{"SWAPCHAIN"}},
							.depth_attachment = Attachment{"depth"},
							.execute = [&](vuk::CommandBuffer& command_buffer) {
								command_buffer
								  .set_viewport(vk::Viewport(0, 100, 100, -1.f * 100, 0.f, 1.f))
								  .set_scissor(vk::Rect2D({ 0,0 }, { 100, 100 }))
								  .bind_pipeline("triangle")
								  .draw(3, 1, 0, 0);
								}
							}
						);

						rg.add_pass({
							.color_attachments = {{"SWAPCHAIN"}},
							.depth_attachment = Attachment{"depth"},
							.execute = [&](vuk::CommandBuffer& command_buffer) {
								command_buffer
								  .set_viewport(vk::Viewport(540, 100, 100, -1.f * 100, 0.f, 1.f))
								  .set_scissor(vk::Rect2D({ 540,0 }, { 100, 100 }))
								  .bind_pipeline("triangle")
								  .draw(3, 1, 0, 0);
								}
							}
						);

						rg.build();
						rg.bind_attachment_to_swapchain("SWAPCHAIN", vk::Format(vkswapchain->image_format), vkswapchain->extent, swapimageviews[index]);
						rg.mark_attachment_internal("depth", vk::Format::eD32Sfloat, vkswapchain->extent);
						rg.build(ptc);
						auto cb = rg.execute(ptc);

						vk::SubmitInfo si;
						si.commandBufferCount = 1;
						si.pCommandBuffers = &cb;
						si.pSignalSemaphores = &render_complete;
						si.signalSemaphoreCount = 1;
						si.waitSemaphoreCount = 1;
						si.pWaitSemaphores = &present_rdy;
						vk::PipelineStageFlags flags = vk::PipelineStageFlagBits::eColorAttachmentOutput;
						si.pWaitDstStageMask = &flags;
						graphics_queue.submit(si, {});

						vk::PresentInfoKHR pi;
						pi.swapchainCount = 1;
						pi.pSwapchains = &swapchain;
						pi.pImageIndices = &acq_result.value;
						pi.waitSemaphoreCount = 1;
						pi.pWaitSemaphores = &render_complete;
						graphics_queue.presentKHR(pi);
					}
					context.device.waitIdle();
					for (auto& swiv : swapimageviews) {
						device.destroy(swiv);
					}
				}
			}
			vkb::destroy_swapchain(*vkswapchain);
			vkDestroySurfaceKHR(inst.instance, surface, nullptr);
			destroy_window_glfw(window);
			vkb::destroy_device(*dev_ret);
			vkb::destroy_instance(inst);
}

int main() {
	device_init();
}