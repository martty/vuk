#pragma once

#include <VkBootstrap.h>
#include <vulkan/vulkan.hpp>
#include "Context.hpp"
#include <optional>
#include "utils.hpp"
#include "glfw.hpp"
#include <string_view>
#include <functional>
#include <string>
#include <stdio.h>
#include <vector>
#include "RenderGraph.hpp"
#include "CommandBuffer.hpp"


namespace vuk {
	struct ExampleRunner;

	struct Example {
		std::string_view name;

		std::function<void(ExampleRunner&)> setup;
		std::function<RenderGraph(ExampleRunner&, vuk::InflightContext&)> render;
	};
}

namespace vuk {
	struct ExampleRunner {
		vk::Device device;
		vk::PhysicalDevice physical_device;
		vk::Queue graphics_queue;
		std::optional<Context> context;
		vuk::SwapchainRef swapchain;
		GLFWwindow* window;
		vk::SurfaceKHR surface;
		vkb::Instance vkbinstance;
		vkb::Device vkbdevice;

		std::vector<Example*> examples;

		ExampleRunner() {
			vkb::InstanceBuilder builder;
			builder
				.request_validation_layers()
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
					vkbinstance = inst_ret.value();

					vkb::PhysicalDeviceSelector selector{ vkbinstance };
					window = create_window_glfw();
					surface = create_surface_glfw(vkbinstance.instance, window);
					selector.set_surface(surface)
						.set_minimum_version(1, 0);
					auto phys_ret = selector.select();
					if (!phys_ret.has_value()) {
						// error
					}
					vkb::PhysicalDevice vkbphysical_device = phys_ret.value();
					physical_device = vkbphysical_device.phys_device;

					vkb::DeviceBuilder device_builder{ vkbphysical_device };
					auto dev_ret = device_builder.build();
					if (!dev_ret.has_value()) {
						// error
					}
					vkbdevice = dev_ret.value();
					graphics_queue = vkb::get_graphics_queue(vkbdevice).value();
					device = vkbdevice.device;

					context.emplace(device, physical_device);
					context->graphics_queue = graphics_queue;

					swapchain = context->add_swapchain(util::make_swapchain(vkbdevice));
		}

		void setup() {
			examples[0]->setup(*this);
		}

		void render() {
			while (!glfwWindowShouldClose(window)) {
				glfwPollEvents();

				auto ifc = context->begin();
				auto rg = examples[0]->render(*this, ifc);
				rg.build();
				std::string attachment_name = std::string(examples[0]->name) + "_final";
				rg.bind_attachment_to_swapchain(attachment_name, swapchain, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
				auto ptc = ifc.begin();
				rg.build(ptc);
				execute_submit_and_present_to_one(ptc, rg, swapchain);
			}
		}

		~ExampleRunner() {
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
}

namespace util {
	struct Register {
		Register(vuk::Example& x) {
			vuk::ExampleRunner::get_runner().examples.push_back(&x);
		}
	};
}

#define CONCAT_IMPL( x, y ) x##y
#define MACRO_CONCAT( x, y ) CONCAT_IMPL( x, y )
#define REGISTER_EXAMPLE(x) util::Register MACRO_CONCAT(_reg_, __LINE__) (x)
