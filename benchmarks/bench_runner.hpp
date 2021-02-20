#pragma once

#include <VkBootstrap.h>
#include "vuk/Context.hpp"
#include <optional>
#include "../examples/utils.hpp"
#include "../examples/glfw.hpp"
#include <string_view>
#include <functional>
#include <string>
#include <stdio.h>
#include <vector>
#include "vuk/RenderGraph.hpp"
#include "vuk/CommandBuffer.hpp"
#include "examples/imgui_impl_glfw.h"

namespace vuk {
	struct BenchRunner;

	struct Bench {
		std::string_view name;

		std::function<void(BenchRunner&, vuk::InflightContext&)> setup;
		std::function<RenderGraph(BenchRunner&, vuk::InflightContext&)> render;
		std::function<void(BenchRunner&, vuk::InflightContext&)> gui;
		std::function<void(BenchRunner&, vuk::InflightContext&)> cleanup;
	};
}

namespace vuk {
	struct BenchRunner {
		VkDevice device;
		VkPhysicalDevice physical_device;
		VkQueue graphics_queue;
		std::optional<Context> context;
		vuk::SwapchainRef swapchain;
		GLFWwindow* window;
		VkSurfaceKHR surface;
		vkb::Instance vkbinstance;
		vkb::Device vkbdevice;
		util::ImGuiData imgui_data;

		std::vector<Bench*> benches;

		BenchRunner();

		void setup() {
			// Setup Dear ImGui context
			IMGUI_CHECKVERSION();
			ImGui::CreateContext();
			// Setup Dear ImGui style
			ImGui::StyleColorsDark();
			// Setup Platform/Renderer bindings
			ImGui_ImplGlfw_InitForVulkan(window, true);
			auto ifc = context->begin();
			{
				auto ptc = ifc.begin();
				imgui_data = util::ImGui_ImplVuk_Init(ptc);
				ptc.wait_all_transfers();
			}
			for(auto& ex : benches)
				ex->setup(*this, ifc);
		}

		void gui(vuk::InflightContext& ifc) {
			benches[0]->gui(*this, ifc);
		}

		void render();

		void cleanup() {
			context->wait_idle();
			imgui_data.font_texture.view.reset();
			imgui_data.font_texture.image.reset();
			auto ifc = context->begin();
			for (auto& ex : benches) {
				if (ex->cleanup) {
					ex->cleanup(*this, ifc);
				}
			}
			// this performs cleanups for all inflight frames
			for (auto i = 0; i < vuk::Context::FC; i++) {
				context->begin();
			}
		}

		~BenchRunner() {
			context.reset();
			vkDestroySurfaceKHR(vkbinstance.instance, surface, nullptr);
			destroy_window_glfw(window);
			vkb::destroy_device(vkbdevice);
			vkb::destroy_instance(vkbinstance);
		}

		static BenchRunner& get_runner() {
			static BenchRunner runner;
			return runner;
		}
	};
}

namespace util {
	struct Register {
		Register(vuk::Bench& x) {
			vuk::BenchRunner::get_runner().benches.push_back(&x);
		}
	};
}

#define CONCAT_IMPL( x, y ) x##y
#define MACRO_CONCAT( x, y ) CONCAT_IMPL( x, y )
#define REGISTER_BENCH(x) util::Register MACRO_CONCAT(_reg_, __LINE__) (x)
