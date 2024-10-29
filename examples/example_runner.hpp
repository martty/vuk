#pragma once

#include "glfw.hpp"
#include "utils.hpp"
#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/runtime/CommandBuffer.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"
#include "vuk/vsl/Core.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/runtime/vk/DeviceFrameResource.hpp"
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

#include "backends/imgui_impl_glfw.h"

#include <tracy/TracyVulkan.hpp>

inline std::filesystem::path root;

namespace vuk {
	struct ExampleRunner;

	struct Example {
		std::string_view name;

		std::function<void(ExampleRunner&, vuk::Allocator&)> setup;
		std::function<vuk::Value<vuk::ImageAttachment>(ExampleRunner&, vuk::Allocator&, vuk::Value<vuk::ImageAttachment>)> render;
		std::function<void(ExampleRunner&, vuk::Allocator&)> cleanup;
	};

	struct ExampleRunner {
		VkDevice device;
		VkPhysicalDevice physical_device;
		VkQueue graphics_queue;
		VkQueue transfer_queue;
		std::optional<Runtime> runtime;
		std::optional<DeviceSuperFrameResource> superframe_resource;
		std::optional<Allocator> superframe_allocator;
		bool suspend = false;
		std::optional<vuk::Swapchain> swapchain;
		GLFWwindow* window;
		VkSurfaceKHR surface;
		vkb::Instance vkbinstance;
		vkb::Device vkbdevice;
		util::ImGuiData imgui_data;
		std::vector<UntypedValue> futures;
		std::mutex setup_lock;
		double old_time = 0;
		uint32_t num_frames = 0;
		bool has_rt;
		vuk::Unique<std::array<VkSemaphore, 3>> present_ready;
		vuk::Unique<std::array<VkSemaphore, 3>> render_complete;
		// one tracy::VkCtx per domain
		tracy::VkCtx* tracy_graphics_ctx;
		tracy::VkCtx* tracy_transfer_ctx;
		// command buffer and pool for Tracy to do init & collect
		vuk::Unique<vuk::CommandPool> tracy_cpool;
		vuk::Unique<vuk::CommandBufferAllocation> tracy_cbufai;

		// when called during setup, enqueues a device-side operation to be completed before rendering begins
		void enqueue_setup(UntypedValue&& fut) {
			std::scoped_lock _(setup_lock);
			futures.emplace_back(std::move(fut));
		}

		std::vector<Value<SampledImage>> sampled_images;
		std::vector<Example*> examples;

		ExampleRunner();

		void setup() {
			// Setup Dear ImGui runtime
			IMGUI_CHECKVERSION();
			ImGui::CreateContext();
			// Setup Dear ImGui style
			ImGui::StyleColorsDark();
			// Setup Platform/Renderer bindings
			ImGui_ImplGlfw_InitForVulkan(window, true);
			imgui_data = util::ImGui_ImplVuk_Init(*superframe_allocator);
			{
				std::vector<std::jthread> threads;
				for (auto& ex : examples) {
					threads.emplace_back(std::jthread([&] { ex->setup(*this, *superframe_allocator); }));
				}
			}
			glfwSetWindowSizeCallback(window, [](GLFWwindow* window, int width, int height) {
				ExampleRunner& runner = *reinterpret_cast<ExampleRunner*>(glfwGetWindowUserPointer(window));
				if (width == 0 && height == 0) {
					runner.suspend = true;
				} else {
					runner.swapchain = util::make_swapchain(*runner.superframe_allocator, runner.vkbdevice, runner.swapchain->surface, std::move(runner.swapchain));
					for (auto& iv : runner.swapchain->images) {
						runner.runtime->set_name(iv.image_view.payload, "Swapchain ImageView");
					}
					runner.suspend = false;
				}
			});
		}

		void render();

		void cleanup() {
			runtime->wait_idle();
			for (auto& ex : examples) {
				if (ex->cleanup) {
					ex->cleanup(*this, *superframe_allocator);
				}
			}
		}

		void set_window_title(std::string title) {
			glfwSetWindowTitle(window, title.c_str());
		}

		double get_time() {
			return glfwGetTime();
		}

		~ExampleRunner() {
			TracyVkDestroy(tracy_graphics_ctx);
			TracyVkDestroy(tracy_transfer_ctx);
			tracy_cbufai.reset();
			tracy_cpool.reset();
			present_ready.reset();
			render_complete.reset();
			imgui_data.font_image.reset();
			imgui_data.font_image_view.reset();
			swapchain.reset();
			superframe_resource.reset();
			runtime.reset();
			auto vkDestroySurfaceKHR = (PFN_vkDestroySurfaceKHR)vkbinstance.fp_vkGetInstanceProcAddr(vkbinstance.instance, "vkDestroySurfaceKHR");
			vkDestroySurfaceKHR(vkbinstance.instance, surface, nullptr);
			destroy_window_glfw(window);
			vkb::destroy_device(vkbdevice);
			vkb::destroy_instance(vkbinstance);
		}

		static ExampleRunner& get_runner() {
			static ExampleRunner runner;
			return runner;
		}
	};

	inline vuk::ExampleRunner::ExampleRunner() {
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

		has_rt = true;

		vkbinstance = inst_ret.value();
		auto instance = vkbinstance.instance;
		vkb::PhysicalDeviceSelector selector{ vkbinstance };
		window = create_window_glfw("Vuk example", true);
		glfwSetWindowUserPointer(window, this);
		surface = create_surface_glfw(vkbinstance.instance, window);
		selector.set_surface(surface)
		    .set_minimum_version(1, 0)
		    .add_required_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
		    .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
		    .add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
		    .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
		    .add_desired_extension(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
		auto phys_ret = selector.select();
		vkb::PhysicalDevice vkbphysical_device;
		if (!phys_ret) {
			has_rt = false;
			vkb::PhysicalDeviceSelector selector2{ vkbinstance };
			selector2.set_surface(surface)
			    .set_minimum_version(1, 0)
			    .add_required_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
			    .add_desired_extension(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
			auto phys_ret2 = selector2.select();
			if (!phys_ret2) {
				throw std::runtime_error("Couldn't create physical device");
			} else {
				vkbphysical_device = phys_ret2.value();
			}
		} else {
			vkbphysical_device = phys_ret.value();
		}

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
		VkPhysicalDeviceSynchronization2FeaturesKHR sync_feat{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
			                                                     .synchronization2 = true };
		VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
			                                                             .accelerationStructure = true };
		VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
			                                                               .rayTracingPipeline = true };
		device_builder = device_builder.add_pNext(&vk12features).add_pNext(&vk11features).add_pNext(&sync_feat).add_pNext(&vk10features);
		if (has_rt) {
			device_builder = device_builder.add_pNext(&rtPipelineFeature).add_pNext(&accelFeature);
		}
		auto dev_ret = device_builder.build();
		if (!dev_ret) {
			throw std::runtime_error("Couldn't create device");
		}
		vkbdevice = dev_ret.value();
		graphics_queue = vkbdevice.get_queue(vkb::QueueType::graphics).value();
		auto graphics_queue_family_index = vkbdevice.get_queue_index(vkb::QueueType::graphics).value();
		transfer_queue = vkbdevice.get_queue(vkb::QueueType::transfer).value();
		auto transfer_queue_family_index = vkbdevice.get_queue_index(vkb::QueueType::transfer).value();
		device = vkbdevice.device;
		vuk::FunctionPointers fps;
		fps.vkGetInstanceProcAddr = vkbinstance.fp_vkGetInstanceProcAddr;
		fps.load_pfns(instance, device, true);
		std::vector<std::unique_ptr<Executor>> executors;

		executors.push_back(vuk::create_vkqueue_executor(fps, device, graphics_queue, graphics_queue_family_index, DomainFlagBits::eGraphicsQueue));
		executors.push_back(vuk::create_vkqueue_executor(fps, device, transfer_queue, transfer_queue_family_index, DomainFlagBits::eTransferQueue));
		executors.push_back(std::make_unique<ThisThreadExecutor>());

		runtime.emplace(RuntimeCreateParameters{ instance, device, physical_device, std::move(executors), fps });
		const unsigned num_inflight_frames = 3;
		superframe_resource.emplace(*runtime, num_inflight_frames);
		superframe_allocator.emplace(*superframe_resource);
		swapchain = util::make_swapchain(*superframe_allocator, vkbdevice, surface, {});
		present_ready = vuk::Unique<std::array<VkSemaphore, 3>>(*superframe_allocator);
		render_complete = vuk::Unique<std::array<VkSemaphore, 3>>(*superframe_allocator);

		// match shader compilation target version to the vk version we request
		runtime->set_shader_target_version(VK_API_VERSION_1_2);

		superframe_allocator->allocate_semaphores(*present_ready);
		superframe_allocator->allocate_semaphores(*render_complete);

		// set up the example Tracy integration
		VkCommandPoolCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
		cpci.queueFamilyIndex = graphics_queue_family_index;
		tracy_cpool = Unique<CommandPool>(*superframe_allocator);
		superframe_allocator->allocate_command_pools(std::span{ &*tracy_cpool, 1 }, std::span{ &cpci, 1 });
		vuk::CommandBufferAllocationCreateInfo ci{ .command_pool = *tracy_cpool };
		tracy_cbufai = Unique<CommandBufferAllocation>(*superframe_allocator);
		superframe_allocator->allocate_command_buffers(std::span{ &*tracy_cbufai, 1 }, std::span{ &ci, 1 });
		tracy_graphics_ctx = TracyVkContextCalibrated(
		    instance, physical_device, device, graphics_queue, tracy_cbufai->command_buffer, fps.vkGetInstanceProcAddr, fps.vkGetDeviceProcAddr);
		tracy_transfer_ctx = TracyVkContextCalibrated(
		    instance, physical_device, device, graphics_queue, tracy_cbufai->command_buffer, fps.vkGetInstanceProcAddr, fps.vkGetDeviceProcAddr);
	}
} // namespace vuk

namespace util {
	struct Register {
		Register(vuk::Example& x) {
			vuk::ExampleRunner::get_runner().examples.push_back(&x);
		}
	};
} // namespace util

#define CONCAT_IMPL(x, y)   x##y
#define MACRO_CONCAT(x, y)  CONCAT_IMPL(x, y)
#define REGISTER_EXAMPLE(x) util::Register MACRO_CONCAT(_reg_, __LINE__)(x)
