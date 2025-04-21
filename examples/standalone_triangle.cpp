#include "utils.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/runtime/CommandBuffer.hpp"
#include "vuk/runtime/ThisThreadExecutor.hpp"
#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/runtime/vk/DeviceFrameResource.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"
#include "vuk/vsl/Core.hpp"
#include <VkBootstrap.h>
#include <filesystem>
#include <optional>
#include <vector>

#include <GLFW/glfw3.h>

using namespace vuk;

int main(int argc, char** argv) {
	std::filesystem::path root =
	    std::filesystem::canonical(std::filesystem::path(argv[0]).parent_path() / std::filesystem::relative(VUK_EX_PATH_ROOT, VUK_EX_PATH_TGT));

	vkb::Instance vkbinstance = vkb::InstanceBuilder{}.set_app_name("vuk_example").require_api_version(1, 2, 0).build().value();
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	GLFWwindow* window = glfwCreateWindow(1024, 1024, "Vuk example", NULL, NULL);
	VkSurfaceKHR surface;
	glfwCreateWindowSurface(vkbinstance.instance, window, NULL, &surface);
	vkb::PhysicalDeviceSelector selector{ vkbinstance };
	selector.set_surface(surface).set_minimum_version(1, 0).add_required_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
	vkb::PhysicalDevice vkbphysical_device = selector.select().value();
	VkPhysicalDeviceVulkan12Features vk12features{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .hostQueryReset = true, .timelineSemaphore = true, .bufferDeviceAddress = true
	};
	VkPhysicalDeviceSynchronization2FeaturesKHR sync_feat{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR, .synchronization2 = true };
	vkb::Device vkbdevice = vkb::DeviceBuilder{ vkbphysical_device }.add_pNext(&vk12features).add_pNext(&sync_feat).build().value();

	VkQueue graphics_queue = vkbdevice.get_queue(vkb::QueueType::graphics).value();
	uint32_t graphics_queue_family_index = vkbdevice.get_queue_index(vkb::QueueType::graphics).value();
	FunctionPointers fps{ .vkGetInstanceProcAddr = vkbinstance.fp_vkGetInstanceProcAddr, .vkGetDeviceProcAddr = vkbinstance.fp_vkGetDeviceProcAddr };
	fps.load_pfns(vkbinstance.instance, vkbdevice.device, true);
	std::vector<std::unique_ptr<Executor>> executors;
	executors.push_back(create_vkqueue_executor(fps, vkbdevice.device, graphics_queue, graphics_queue_family_index, DomainFlagBits::eGraphicsQueue));
	executors.push_back(std::make_unique<ThisThreadExecutor>());

	std::unique_ptr<Runtime> runtime = std::make_unique<Runtime>(
	    RuntimeCreateParameters{ vkbinstance.instance, vkbdevice.device, vkbphysical_device.physical_device, std::move(executors), fps });
	runtime->shader_compiler_target_version = VK_API_VERSION_1_2;
	std::unique_ptr<DeviceSuperFrameResource> superframe_resource = std::make_unique<DeviceSuperFrameResource>(*runtime, 3);
	Allocator superframe_allocator(*superframe_resource);

	vkb::SwapchainBuilder swb(vkbdevice, surface);
	swb.set_desired_format(SurfaceFormatKHR{ Format::eR8G8B8A8Srgb, ColorSpaceKHR::eSrgbNonlinear })
	    .add_fallback_format(SurfaceFormatKHR{ Format::eB8G8R8A8Srgb, ColorSpaceKHR::eSrgbNonlinear })
	    .set_image_usage_flags(VkImageUsageFlagBits::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	auto vkswapchain = swb.build();

	std::optional<Swapchain> swapchain = Swapchain(superframe_allocator,
	                                               vkswapchain->image_count,
	                                               vkswapchain->swapchain,
	                                               surface,
	                                               vkswapchain->extent,
	                                               vkswapchain->image_format,
	                                               *vkswapchain->get_images(),
	                                               *vkswapchain->get_image_views());

	Compiler compiler;

	PipelineBaseCreateInfo pci;
	pci.add_glsl(util::read_entire_file((root / "examples/triangle.vert").generic_string()), (root / "examples/triangle.vert").generic_string());
	pci.add_glsl(util::read_entire_file((root / "examples/triangle.frag").generic_string()), (root / "examples/triangle.frag").generic_string());
	// The pipeline is stored with a user give name for simplicity
	runtime->create_named_pipeline("triangle", pci);

	// our main loop
	while (!glfwWindowShouldClose(window)) {
		// pump the message loop
		glfwPollEvents();
		// advance the frame for the allocators and caches used by vuk
		auto& frame_resource = superframe_resource->get_next_frame();
		runtime->next_frame();
		// create a frame allocator - we can allocate objects for the duration of the frame from this allocator
		// all of the objects allocated from this allocator last for this frame, and get recycled automatically, so for this specific allocator, deallocation is
		// optional
		Allocator frame_allocator(frame_resource);
		// create a rendergraph we will use to prepare a swapchain image for the example to render into
		auto imported_swapchain = acquire_swapchain(*swapchain);
		// acquire an image on the swapchain
		auto swapchain_image = acquire_next_image("swp_img", std::move(imported_swapchain));

		// clear the swapchain image
		Value<ImageAttachment> cleared_image_to_render_into = clear_image(std::move(swapchain_image), ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });

		auto pass = make_pass("draw", [](CommandBuffer& command_buffer, VUK_IA(eColorWrite) color_rt) {
			command_buffer.set_viewport(0, Rect2D::framebuffer()); // Set the viewport area to cover the entire framebuffer
			command_buffer.set_scissor(0, Rect2D::framebuffer()); // Set the scissor area to cover the entire framebuffer
			command_buffer
			    .set_rasterization({})              // Set the default rasterization state
			    .set_color_blend(color_rt, {})      // Set the default color blend state
			    .bind_graphics_pipeline("triangle") // Recall pipeline for "triangle" and bind
			    .draw(3, 1, 0, 0);                  // Draw 3 vertices
			return color_rt;
		});

		auto drawn = pass(std::move(cleared_image_to_render_into));

		// submit and present the results to the swapchain we imported previously
		auto entire_thing = enqueue_presentation(std::move(drawn));

		entire_thing.submit(frame_allocator, compiler);
	}

	runtime->wait_idle();
	swapchain.reset();
	superframe_resource.reset();
	runtime.reset();
	vkb::destroy_surface(vkbinstance, surface);
	glfwDestroyWindow(window);
	glfwTerminate();
	vkb::destroy_device(vkbdevice);
	vkb::destroy_instance(vkbinstance);
}