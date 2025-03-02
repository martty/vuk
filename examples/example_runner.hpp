#pragma once

#include <filesystem>
#include <format>
#include <functional>
#include <mutex>
#include <thread>

#include "utils.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/runtime/CommandBuffer.hpp"
#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/runtime/vk/DeviceFrameResource.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"
#include "vuk/vsl/Core.hpp"

#include "vuk/extra/GlfwWindow.hpp"
#include "vuk/extra/ImGuiIntegration.hpp"
#include "vuk/extra/SimpleInit.hpp"
#include "vuk/extra/TracyIntegration.hpp"
#include <backends/imgui_impl_glfw.h>

inline std::filesystem::path root;

namespace vuk {
	struct ExampleRunner;

	struct Example {
		std::string_view name;

		std::function<void(ExampleRunner&, vuk::Allocator&, vuk::Runtime&)> setup;
		std::function<vuk::Value<vuk::ImageAttachment>(ExampleRunner&, vuk::Allocator&, vuk::Value<vuk::ImageAttachment>)> render;
		std::function<void(ExampleRunner&, vuk::Allocator&)> cleanup;
	};

	struct ExampleRunner {
		bool suspend = false;

		GLFWwindow* window;

		extra::ImGuiData imgui_data;

		// multithread initialization & lock
		std::mutex setup_lock;
		std::vector<UntypedValue> futures;

		// FPS counter
		double old_time = 0;
		uint32_t num_frames = 0;

		std::vector<Example*> examples;

#ifdef TRACY_ENABLE
		std::unique_ptr<extra::TracyContext> tracy_context;
#endif // TRACY_ENABLE

		std::unique_ptr<extra::SimpleApp> app;

		void setup() {
			// we initialize the Vulkan instance
			vkb::InstanceBuilder instance_builder =
			    extra::make_instance_builder(1, 2, true).set_app_name("vuk_example").set_engine_name("vuk").set_app_version(0, 1, 0);
			auto inst_ret = instance_builder.build();
			if (!inst_ret) {
				auto err_msg = std::string("ERROR: couldn't initialise instance - ") + inst_ret.error().message();
				printf("ERROR: %s\n", err_msg.c_str());
				throw std::runtime_error(err_msg);
			}
			vkb::Instance instance = inst_ret.value();

			// we initialize GLFW and create a window
			glfwInit();
			glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
			window = glfwCreateWindow(1024, 1024, "Vuk example", NULL, NULL);
			glfwSetWindowUserPointer(window, this);
			// we create a surface from the window
			auto surface = create_surface_glfw(instance.instance, window);
			// and select a physical device that can present to this surface and meets the minimum requirements for vuk
			auto physical_device = extra::select_physical_device(instance, surface);
			// we then build a vulkan device with required and recommended features
			// then we build a SimpleApp with a swapchain and 3 frames-in-flight
			app = extra::make_device_builder(physical_device).set_recommended_features().build_app(true, 3);

			// set up the example Tracy integration
#ifdef TRACY_ENABLE
			tracy_context = extra::init_Tracy(*app->superframe_allocator);
#endif // TRACY_ENABLE

			// Setup Dear ImGui runtime
			IMGUI_CHECKVERSION();
			ImGui::CreateContext();
			// Setup Dear ImGui style
			ImGui::StyleColorsDark();

			// Setup Platform/Renderer bindings
			ImGui_ImplGlfw_InitForVulkan(window, true);
			// Init ImGui for Vuk
			imgui_data = extra::ImGui_ImplVuk_Init(*app->superframe_allocator);
			{
				std::vector<std::thread> threads;
				for (auto& ex : examples) {
					threads.emplace_back(std::thread([&] { ex->setup(*this, *app->superframe_allocator, *app->runtime); }));
				}

				for (std::thread& t : threads) {
					t.join();
				}
			}
			glfwSetWindowSizeCallback(window, [](GLFWwindow* window, int width, int height) {
				ExampleRunner& runner = *reinterpret_cast<ExampleRunner*>(glfwGetWindowUserPointer(window));
				if (width == 0 && height == 0) {
					runner.suspend = true;
				} else {
					runner.app->update_swapchain();
					runner.suspend = false;
				}
			});
		}
		// when called during setup, enqueues a device-side operation to be completed before rendering begins
		void enqueue_setup(UntypedValue&& fut) {
			std::scoped_lock _(setup_lock);
			futures.emplace_back(std::move(fut));
		}

		void render();

		void cleanup() {
			app->runtime->wait_idle();
			for (auto& ex : examples) {
				if (ex->cleanup) {
					ex->cleanup(*this, *app->superframe_allocator);
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
			imgui_data.font_image.reset();
			imgui_data.font_image_view.reset();
			tracy_context.reset();
			app.reset();
			destroy_window_glfw(window);
		}

		static ExampleRunner& get_runner() {
			static ExampleRunner runner;
			return runner;
		}
	};
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