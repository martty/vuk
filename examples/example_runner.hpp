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
#include <VkBootstrap.h>
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

namespace vuk {
	struct ExampleRunner;

	struct Example {
		std::string_view name;

		std::function<void(ExampleRunner&, vuk::Allocator&)> setup;
		std::function<vuk::Future(ExampleRunner&, vuk::Allocator&, vuk::Future)> render;
		std::function<void(ExampleRunner&, vuk::Allocator&)> cleanup;
	};

	struct ExampleRunner {
		VkDevice device;
		VkPhysicalDevice physical_device;
		VkQueue graphics_queue;
		VkQueue transfer_queue;
		std::optional<Context> context;
		std::optional<DeviceSuperFrameResource> xdev_rf_alloc;
		std::optional<Allocator> global;
		vuk::SwapchainRef swapchain;
		GLFWwindow* window;
		VkSurfaceKHR surface;
		vkb::Instance vkbinstance;
		vkb::Device vkbdevice;
		util::ImGuiData imgui_data;
		std::vector<Future> futures;
		std::mutex setup_lock;
		double old_time = 0;
		uint32_t num_frames = 0;
		bool has_rt;
		vuk::Unique<std::array<VkSemaphore, 3>> present_ready;
		vuk::Unique<std::array<VkSemaphore, 3>> render_complete;

		// when called during setup, enqueues a device-side operation to be completed before rendering begins
		void enqueue_setup(Future&& fut) {
			std::scoped_lock _(setup_lock);
			futures.emplace_back(std::move(fut));
		}

		plf::colony<vuk::SampledImage> sampled_images;
		std::vector<Example*> examples;

		ExampleRunner();

		void setup() {
			// Setup Dear ImGui context
			IMGUI_CHECKVERSION();
			ImGui::CreateContext();
			// Setup Dear ImGui style
			ImGui::StyleColorsDark();
			// Setup Platform/Renderer bindings
			ImGui_ImplGlfw_InitForVulkan(window, true);
			imgui_data = util::ImGui_ImplVuk_Init(*global);
			{
				std::vector<std::jthread> threads;
				for (auto& ex : examples) {
					threads.emplace_back(std::jthread([&] { ex->setup(*this, *global); }));
				}
			}
			glfwSetWindowSizeCallback(window, [](GLFWwindow* window, int width, int height) {

			});
		}

		void render();

		void cleanup() {
			context->wait_idle();
			for (auto& ex : examples) {
				if (ex->cleanup) {
					ex->cleanup(*this, *global);
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
			present_ready.reset();
			render_complete.reset();
			imgui_data.font_texture.view.reset();
			imgui_data.font_texture.image.reset();
			xdev_rf_alloc.reset();
			context.reset();
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
		window = create_window_glfw("Vuk example", false);
		surface = create_surface_glfw(vkbinstance.instance, window);
		selector.set_surface(surface)
		    .set_minimum_version(1, 0)
		    .add_required_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
		    .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
		    .add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
		    .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
		auto phys_ret = selector.select();
		vkb::PhysicalDevice vkbphysical_device;
		if (!phys_ret) {
			has_rt = false;
			vkb::PhysicalDeviceSelector selector2{ vkbinstance };
			selector2.set_surface(surface).set_minimum_version(1, 0).add_required_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
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
		device_builder = device_builder.add_pNext(&vk12features).add_pNext(&vk11features).add_pNext(&sync_feat).add_pNext(&accelFeature).add_pNext(&vk10features);
		if (has_rt) {
			device_builder = device_builder.add_pNext(&rtPipelineFeature);
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
		ContextCreateParameters::FunctionPointers fps;
#define VUK_EX_LOAD_FP(name) fps.name = (PFN_##name)vkGetDeviceProcAddr(device, #name);
		VUK_EX_LOAD_FP(vkSetDebugUtilsObjectNameEXT);
		VUK_EX_LOAD_FP(vkCmdBeginDebugUtilsLabelEXT);
		VUK_EX_LOAD_FP(vkCmdEndDebugUtilsLabelEXT);
		if (has_rt) {
			VUK_EX_LOAD_FP(vkCmdBuildAccelerationStructuresKHR);
			VUK_EX_LOAD_FP(vkGetAccelerationStructureBuildSizesKHR);
			VUK_EX_LOAD_FP(vkCmdTraceRaysKHR);
			VUK_EX_LOAD_FP(vkCreateAccelerationStructureKHR);
			VUK_EX_LOAD_FP(vkDestroyAccelerationStructureKHR);
			VUK_EX_LOAD_FP(vkGetRayTracingShaderGroupHandlesKHR);
			VUK_EX_LOAD_FP(vkCreateRayTracingPipelinesKHR);
		}
		context.emplace(ContextCreateParameters{ instance,
		                                         device,
		                                         physical_device,
		                                         graphics_queue,
		                                         graphics_queue_family_index,
		                                         VK_NULL_HANDLE,
		                                         VK_QUEUE_FAMILY_IGNORED,
		                                         transfer_queue,
		                                         transfer_queue_family_index,
		                                         fps });
		const unsigned num_inflight_frames = 3;
		xdev_rf_alloc.emplace(*context, num_inflight_frames);
		global.emplace(*xdev_rf_alloc);
		swapchain = context->add_swapchain(util::make_swapchain(vkbdevice));
		present_ready = vuk::Unique<std::array<VkSemaphore, 3>>(*global);
		render_complete = vuk::Unique<std::array<VkSemaphore, 3>>(*global);
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
