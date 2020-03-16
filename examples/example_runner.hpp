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
#include "examples/imgui_impl_glfw.h"


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
		util::ImGuiData imgui_data;

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
			{
				auto ifc = context->begin();
				auto ptc = ifc.begin();
				imgui_data = util::ImGui_ImplVuk_Init(ptc);
				ptc.wait_all_transfers();
			}

			for(auto& ex : examples)
				ex->setup(*this);
		}

		void render();

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
