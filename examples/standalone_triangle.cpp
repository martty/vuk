#pragma once

#include "glfw.hpp"
#include "utils.hpp"
#include "vuk/Allocator.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/CommandBuffer.hpp"
#include "vuk/Context.hpp"
#include "vuk/Partials.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/SampledImage.hpp"
#include "vuk/resources/DeviceFrameResource.hpp"
#include "vuk/runtime/ThisThreadExecutor.hpp"
#include <VkBootstrap.h>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <tracy/TracyVulkan.hpp>

std::filesystem::path root;

using namespace vuk;

int main(int argc, char** argv) {
	auto path_to_root = std::filesystem::relative(VUK_EX_PATH_ROOT, VUK_EX_PATH_TGT);
	root = std::filesystem::canonical(std::filesystem::path(argv[0]).parent_path() / path_to_root);

	VkDevice device;
	VkPhysicalDevice physical_device;
	VkQueue graphics_queue;
	std::optional<Context> context;
	std::optional<DeviceSuperFrameResource> superframe_resource;
	std::optional<Allocator> superframe_allocator;
	bool suspend = false;
	std::optional<vuk::Swapchain> swapchain;
	GLFWwindow* window;
	VkSurfaceKHR surface;
	vkb::Instance vkbinstance;
	vkb::Device vkbdevice;
	double old_time = 0;
	uint32_t num_frames = 0;

	vkb::InstanceBuilder builder;
	builder.request_validation_layers()
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
	    .require_api_version(1, 2, 0)
	    .set_app_version(0, 1, 0);
	auto inst_ret = builder.build();
	if (!inst_ret) {
		throw std::runtime_error("Couldn't initialise instance");
	}

	vkbinstance = inst_ret.value();
	auto instance = vkbinstance.instance;
	vkb::PhysicalDeviceSelector selector{ vkbinstance };
	window = create_window_glfw("Vuk example", false);
	surface = create_surface_glfw(vkbinstance.instance, window);
	selector.set_surface(surface)
	    .set_minimum_version(1, 0)
	    .add_required_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
	    .add_desired_extension(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
	auto phys_ret = selector.select();
	vkb::PhysicalDevice vkbphysical_device;
	vkbphysical_device = phys_ret.value();

	physical_device = vkbphysical_device.physical_device;
	vkb::DeviceBuilder device_builder{ vkbphysical_device };
	VkPhysicalDeviceVulkan12Features vk12features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
	vk12features.timelineSemaphore = true;
	vk12features.descriptorBindingPartiallyBound = true;
	vk12features.descriptorBindingUpdateUnusedWhilePending = true;
	vk12features.shaderSampledImageArrayNonUniformIndexing = true;
	vk12features.runtimeDescriptorArray = true;
	vk12features.descriptorBindingVariableDescriptorCount = true;
	vk12features.hostQueryReset = true;
	vk12features.bufferDeviceAddress = true;
	vk12features.shaderOutputLayer = true;
	VkPhysicalDeviceVulkan11Features vk11features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
	vk11features.shaderDrawParameters = true;
	VkPhysicalDeviceFeatures2 vk10features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR };
	vk10features.features.shaderInt64 = true;
	VkPhysicalDeviceSynchronization2FeaturesKHR sync_feat{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR, .synchronization2 = true };
	device_builder = device_builder.add_pNext(&vk12features).add_pNext(&vk11features).add_pNext(&sync_feat).add_pNext(&vk10features);

	auto dev_ret = device_builder.build();
	if (!dev_ret) {
		throw std::runtime_error("Couldn't create device");
	}
	vkbdevice = dev_ret.value();
	graphics_queue = vkbdevice.get_queue(vkb::QueueType::graphics).value();
	auto graphics_queue_family_index = vkbdevice.get_queue_index(vkb::QueueType::graphics).value();
	device = vkbdevice.device;
	vuk::rtvk::FunctionPointers fps;
	fps.vkGetInstanceProcAddr = vkbinstance.fp_vkGetInstanceProcAddr;
	fps.vkGetDeviceProcAddr = vkbinstance.fp_vkGetDeviceProcAddr;
	fps.load_pfns(instance, device, true);
	std::vector<std::unique_ptr<Executor>> executors;

	executors.push_back(rtvk::create_vkqueue_executor(fps, device, graphics_queue, graphics_queue_family_index, DomainFlagBits::eGraphicsQueue));
	executors.push_back(std::make_unique<ThisThreadExecutor>());

	context.emplace(ContextCreateParameters{ instance, device, physical_device, std::move(executors), fps });
	const unsigned num_inflight_frames = 3;
	superframe_resource.emplace(*context, num_inflight_frames);
	superframe_allocator.emplace(*superframe_resource);
	swapchain = util::make_swapchain(*superframe_allocator, vkbdevice, {});

	Compiler compiler;

	vuk::PipelineBaseCreateInfo pci;
	pci.add_glsl(util::read_entire_file((root / "examples/triangle.vert").generic_string()), (root / "examples/triangle.vert").generic_string());
	pci.add_glsl(util::read_entire_file((root / "examples/triangle.frag").generic_string()), (root / "examples/triangle.frag").generic_string());
	// The pipeline is stored with a user give name for simplicity
	context->create_named_pipeline("triangle", pci);

	// our main loop
	while (!glfwWindowShouldClose(window)) {
		// pump the message loop
		glfwPollEvents();
		while (suspend) {
			glfwWaitEvents();
		}
		// advance the frame for the allocators and caches used by vuk
		auto& frame_resource = superframe_resource->get_next_frame();
		context->next_frame();
		// create a frame allocator - we can allocate objects for the duration of the frame from this allocator
		// all of the objects allocated from this allocator last for this frame, and get recycled automatically, so for this specific allocator, deallocation is
		// optional
		Allocator frame_allocator(frame_resource);
		// create a rendergraph we will use to prepare a swapchain image for the example to render into
		auto imported_swapchain = declare_swapchain(*swapchain);
		// acquire an image on the swapchain
		auto swapchain_image = acquire_next_image("swp_img", std::move(imported_swapchain));

		// clear the swapchain image
		Future<ImageAttachment> cleared_image_to_render_into = clear_image(std::move(swapchain_image), vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
		
		 auto pass = vuk::make_pass("01_triangle", [](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eColorWrite) color_rt) {
			command_buffer.set_viewport(0, vuk::Rect2D::framebuffer());
			// Set the scissor area to cover the entire framebuffer
			command_buffer.set_scissor(0, vuk::Rect2D::framebuffer());
			command_buffer
			    .set_rasterization({})              // Set the default rasterization state
			    .set_color_blend(color_rt, {})      // Set the default color blend state
			    .bind_graphics_pipeline("triangle") // Recall pipeline for "triangle" and bind
			    .draw(3, 1, 0, 0);                  // Draw 3 vertices
			return color_rt;
		});

		auto drawn = pass(std::move(cleared_image_to_render_into));

		// compile the IRModule that contains all the rendering of the example
		// submit and present the results to the swapchain we imported previously
		auto entire_thing = enqueue_presentation(std::move(drawn));

		entire_thing.wait(frame_allocator, compiler, {});
	}

	context->wait_idle();
	superframe_resource.reset();
	context.reset();
	auto vkDestroySurfaceKHR = (PFN_vkDestroySurfaceKHR)vkbinstance.fp_vkGetInstanceProcAddr(vkbinstance.instance, "vkDestroySurfaceKHR");
	vkDestroySurfaceKHR(vkbinstance.instance, surface, nullptr);
	destroy_window_glfw(window);
	vkb::destroy_device(vkbdevice);
	vkb::destroy_instance(vkbinstance);
}