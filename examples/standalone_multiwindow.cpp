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
#include <string>
#include <string_view>
#include <vector>

std::filesystem::path root;

using namespace vuk;

const int main_win_size_x = 512;
const int main_win_size_y = 512;
const int small_win_size_x = 128;
const int small_win_size_y = 128;

const int vp_size_x = main_win_size_x + 2 * small_win_size_x;
const int vp_size_y = main_win_size_y + small_win_size_y;

int main_win_x, main_win_y;

struct SmallWindow {
	SmallWindow(GLFWwindow* window, Swapchain swapchain, int offset) : window(window), swapchain(std::move(swapchain)), offset(offset) {}

	GLFWwindow* window;
	Swapchain swapchain;

	int offset;
	int vpx, vpy;
};

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
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	GLFWwindow* main_window = glfwCreateWindow(main_win_size_x, main_win_size_y, "Main window", NULL, NULL);
	glfwGetWindowPos(main_window, &main_win_x, &main_win_y);
	glfwSetWindowPosCallback(main_window, [](GLFWwindow*, int x, int y) {
		main_win_x = x;
		main_win_y = y;
	});
	VkSurfaceKHR surface = create_surface_glfw(vkbinstance.instance, main_window);
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

	// schmol windows
	std::vector<SmallWindow> small_windows;
	for (int i = 0; i < 5; i++) {
		glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
		auto s_window = glfwCreateWindow(128, 128, "Schmol window", NULL, NULL);
		VkSurfaceKHR s_surface = create_surface_glfw(vkbinstance.instance, s_window);
		auto s_swapchain = util::make_swapchain(*superframe_allocator, vkbdevice, s_surface, {});
		small_windows.emplace_back(s_window, s_swapchain, i * (340));
	}
	swapchain = util::make_swapchain(*superframe_allocator, vkbdevice, surface, {});

	Compiler compiler;

	{
		vuk::PipelineBaseCreateInfo pci;
		pci.add_glsl(util::read_entire_file((root / "examples/particle_points.vert").generic_string()), (root / "examples/large_triangle.vert").generic_string());
		pci.add_glsl(util::read_entire_file((root / "examples/point.frag").generic_string()), (root / "examples/triangle.frag").generic_string());
		context->create_named_pipeline("triangle", pci);
	}
	{
		vuk::PipelineBaseCreateInfo pci;
		pci.add_glsl(util::read_entire_file((root / "examples/particle_sim.comp").generic_string()), (root / "examples/particle_sim.comp").generic_string());
		context->create_named_pipeline("particle_sim", pci);
	}

	{
		vuk::PipelineBaseCreateInfo pci;
		pci.add_glsl(util::read_entire_file((root / "examples/particle_sim_init.comp").generic_string()), (root / "examples/particle_sim.comp").generic_string());
		context->create_named_pipeline("particle_sim_init", pci);
	}

	auto particle_count = 1000000;
	struct Particle {
		float x, y;
		float vx, vy;
	};

	auto clear_buffer = make_pass("clear buffer", [](CommandBuffer& command_buffer, VUK_BA(Access::eComputeRW) particles) {
		command_buffer.bind_compute_pipeline("particle_sim_init");
		command_buffer.bind_buffer(0, 1, particles);
		command_buffer.dispatch_invocations_per_element(particles, sizeof(Particle));
		return particles;
	});

	struct Attractors {
		struct Attractor {
			float x, y;
			float strength;
			float _pad;
		} attractors[32];
		unsigned int count;
	};

	auto particles_buf = allocate_buffer(*superframe_allocator, { .mem_usage = MemoryUsage::eGPUonly, .size = sizeof(Particle) * particle_count });
	auto particles = clear_buffer(declare_buf("particles", **particles_buf));
	particles.wait(*superframe_allocator, compiler, {});

	// our main loop
	while (!glfwWindowShouldClose(main_window)) {
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

		auto render = [particle_count](vuk::Rect2D viewport) {
			return vuk::make_pass("01_triangle", [=](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eColorWrite) color_rt, VUK_BA(vuk::eAttributeRead) particles) {
				command_buffer.set_viewport(0, viewport);
				// Set the scissor area to cover the entire framebuffer
				command_buffer.set_scissor(0, vuk::Rect2D::framebuffer());
				command_buffer
				    .set_rasterization({}) // Set the default rasterization state
				    .set_primitive_topology(PrimitiveTopology::ePointList)
				    .set_color_blend(color_rt, BlendPreset::eAlphaBlend) // Set the default color blend state
				    .bind_graphics_pipeline("triangle")                  // Recall pipeline for "triangle" and bind
				    .bind_vertex_buffer(0, particles, 0, Packed{ Format::eR32G32Sfloat, Format::eR32G32Sfloat })
				    .draw(particle_count, 1, 0, 0);
				return color_rt;
			});
		};

		std::vector<UntypedFuture> futs;
		{
			auto imported_swapchain = declare_swapchain(*swapchain);
			auto swapchain_image = acquire_next_image("swp_img", std::move(imported_swapchain));
			Future<ImageAttachment> cleared_image_to_render_into = clear_image(std::move(swapchain_image), vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });

			auto drawn = render(vuk::Rect2D::absolute(-small_win_size_x, 0, vp_size_x, vp_size_y))(std::move(cleared_image_to_render_into), particles);

			// compile the RG that contains all the rendering of the example
			// submit and present the results to the swapchain we imported previously
			futs.push_back(enqueue_presentation(std::move(drawn)));
		}
		for (auto& sw : small_windows) {
			int x, y;
			if (sw.offset < main_win_size_y) {
				x = main_win_size_x;
				y = sw.offset;
			} else if (sw.offset >= main_win_size_y && sw.offset < (main_win_size_x + main_win_size_y + 128)) {
				x = main_win_size_y - sw.offset + main_win_size_x;
				y = main_win_size_y;
			} else if (sw.offset >= (main_win_size_x + main_win_size_y + 128) && sw.offset < (2 * main_win_size_x + main_win_size_y + 128)) {
				x = -128;
				y = (main_win_size_x + main_win_size_y + 128) - sw.offset + main_win_size_y;
			} else {
				x = main_win_size_x;
				y = 0;
				sw.offset = -1;
			}
			sw.vpx = x;
			sw.vpy = y;
			x += main_win_x;
			y += main_win_y;
			glfwSetWindowPos(sw.window, x, y);
			sw.offset++;
			auto imported_swapchain = declare_swapchain(sw.swapchain);
			auto swapchain_image = acquire_next_image("swp_img", std::move(imported_swapchain));
			Future<ImageAttachment> cleared_image_to_render_into = clear_image(std::move(swapchain_image), vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });

			auto drawn = render(vuk::Rect2D::absolute(-(sw.vpx + 128), -sw.vpy, vp_size_x, vp_size_y))(std::move(cleared_image_to_render_into), particles);

			// compile the RG that contains all the rendering of the example
			// submit and present the results to the swapchain we imported previously
			futs.push_back(enqueue_presentation(std::move(drawn)));
		}

		// particle sim
		auto sim = vuk::make_pass("particle sim", [=](vuk::CommandBuffer& command_buffer, VUK_BA(vuk::eComputeRW) particles) {
			command_buffer.bind_compute_pipeline("particle_sim");
			Attractors attr{ .count = (uint32_t)small_windows.size() + 1 };
			for (int i = 0; i < small_windows.size(); i++) {
				auto& sw = small_windows[i];
				attr.attractors[i].x = ((float)(sw.vpx + 128 + 64) / vp_size_x) * 2.f - 1.f;
				attr.attractors[i].y = ((float)(sw.vpy + 64) / vp_size_y) * 2.f - 1.f;
				attr.attractors[i].strength = 0.001f;
			}
			attr.attractors[5].strength = 0.005f;
			*command_buffer.scratch_buffer<Attractors>(0, 0) = attr;
			command_buffer.bind_buffer(0, 1, particles);
			command_buffer.dispatch_invocations_per_element(particles, sizeof(Particle));
			return particles;
		});

		auto sim_step = sim(particles);
		sim_step.wait(frame_allocator, compiler, {});
		wait_for_futures_explicit(frame_allocator, compiler, futs);
		particles = declare_buf("particles", **particles_buf);
	}

	context->wait_idle();
	particles_buf->reset();
	superframe_resource.reset();
	context.reset();
	auto vkDestroySurfaceKHR = (PFN_vkDestroySurfaceKHR)vkbinstance.fp_vkGetInstanceProcAddr(vkbinstance.instance, "vkDestroySurfaceKHR");
	vkDestroySurfaceKHR(vkbinstance.instance, surface, nullptr);
	destroy_window_glfw(main_window);
	vkb::destroy_device(vkbdevice);
	vkb::destroy_instance(vkbinstance);
}