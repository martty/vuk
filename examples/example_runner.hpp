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
#include <optional>
#include <stdio.h>
#include <string>
#include <string_view>
#include <vector>

#include "backends/imgui_impl_glfw.h"

namespace vuk {
	struct ExampleRunner;

	struct Example {
		std::string_view name;

		std::function<void(ExampleRunner&, vuk::Allocator&)> setup;
		std::function<vuk::Future<vuk::ImageAttachment>(ExampleRunner&, vuk::Allocator&)> render;
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
		std::vector<Future<ImageAttachment>> ia_futures;
		std::vector<Future<Buffer>> buf_futures;

		// when called during setup, enqueues a device-side operation to be completed before rendering begins
		void enqueue_setup(Future<ImageAttachment>&& fut) {
			ia_futures.emplace_back(std::move(fut));
		}
		
		// when called during setup, enqueues a device-side operation to be completed before rendering begins
		void enqueue_setup(Future<Buffer>&& fut) {
			buf_futures.emplace_back(std::move(fut));
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
			{ imgui_data = util::ImGui_ImplVuk_Init(*global); }
			for (auto& ex : examples) {
				ex->setup(*this, *global);
			}
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

		~ExampleRunner() {
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
