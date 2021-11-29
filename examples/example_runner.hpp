#pragma once

#include <VkBootstrap.h>
#include "vuk/Context.hpp"
#include <optional>
#include "utils.hpp"
#include "glfw.hpp"
#include <string_view>
#include <functional>
#include <string>
#include <stdio.h>
#include <vector>
#include "vuk/RenderGraph.hpp"
#include "vuk/CommandBuffer.hpp"
#include "examples/imgui_impl_glfw.h"


namespace vuk {
	struct ExampleRunner;

	struct Example {
		std::string_view name;

		std::function<void(ExampleRunner&, vuk::NAllocator&)> setup;
		std::function<RenderGraph(ExampleRunner&, vuk::NAllocator&)> render;
		std::function<void(ExampleRunner&, vuk::NAllocator&)> cleanup;
	};
}

namespace vuk {
	struct ExampleRunner {
		VkDevice device;
		VkPhysicalDevice physical_device;
		VkQueue graphics_queue;
		std::optional<Context> context;
		std::optional<RingFrame> rf_alloc;
		vuk::SwapchainRef swapchain;
		GLFWwindow* window;
		VkSurfaceKHR surface;
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
				imgui_data = util::ImGui_ImplVuk_Init(context->get_direct_allocator());
				context->wait_all_transfers();
			}
			for (auto& ex : examples) {
				ex->setup(*this, context->get_direct_allocator());
			}
		}

		void render();

		void cleanup() {
			context->wait_idle();
			imgui_data.font_texture.view.reset();
			imgui_data.font_texture.image.reset();
			for (auto& ex : examples) {
				if (ex->cleanup) {
					ex->cleanup(*this, context->get_direct_allocator());
				}
			}
		}

		~ExampleRunner() {
			rf_alloc.reset();
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
