#include "bench_runner.hpp"
#include "RenderGraphUtil.hpp"

std::vector<std::string> chosen_resource;

vuk::BenchRunner::BenchRunner() {
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
		.set_app_name("vuk_bench")
				.set_engine_name("vuk")
				.require_api_version(1, 2, 0)
				.set_app_version(0, 1, 0);
			auto inst_ret = builder.build();
			if (!inst_ret.has_value()) {
				// error
			}
			vkbinstance = inst_ret.value();
			auto instance = vkbinstance.instance;
			vkb::PhysicalDeviceSelector selector{ vkbinstance };
			window = create_window_glfw("vuk-benchmarker", false);
			surface = create_surface_glfw(vkbinstance.instance, window);
			selector.set_surface(surface)
				.set_minimum_version(1, 0);
			auto phys_ret = selector.select();
			if (!phys_ret.has_value()) {
				// error
			}
			vkb::PhysicalDevice vkbphysical_device = phys_ret.value();
			physical_device = vkbphysical_device.physical_device;

			vkb::DeviceBuilder device_builder{ vkbphysical_device };
			VkPhysicalDeviceVulkan12Features vk12features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
			vk12features.descriptorBindingPartiallyBound = true;
			vk12features.descriptorBindingUpdateUnusedWhilePending = true;
			vk12features.shaderSampledImageArrayNonUniformIndexing = true;
			vk12features.runtimeDescriptorArray = true;
			vk12features.descriptorBindingVariableDescriptorCount = true;
			vk12features.hostQueryReset = true;
			auto dev_ret = device_builder.add_pNext(&vk12features).build();
			if (!dev_ret.has_value()) {
				// error
			}
			vkbdevice = dev_ret.value();
			graphics_queue = vkbdevice.get_queue(vkb::QueueType::graphics).value();
			auto graphics_queue_family_index = vkbdevice.get_queue_index(vkb::QueueType::graphics).value();
			device = vkbdevice.device;

			context.emplace(ContextCreateParameters{ instance, device, physical_device, graphics_queue, graphics_queue_family_index });

			swapchain = context->add_swapchain(util::make_swapchain(vkbdevice));
}

void vuk::BenchRunner::render() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		auto ifc = context->begin();

		ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 352.f, 2));
		ImGui::SetNextWindowSize(ImVec2(350, 0));
		ImGui::Begin("Benchmark", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize);

		vuk::BenchRunner::get_runner().gui(ifc);
		
		ImGui::End();

		auto rg = benches[0]->render(*this, ifc);
		ImGui::Render();
		auto ptc = ifc.begin();
		std::string attachment_name = std::string(benches[0]->name) + "_final";
		util::ImGui_ImplVuk_Render(ptc, rg, attachment_name, "SWAPCHAIN", imgui_data, ImGui::GetDrawData());
		rg.attach_swapchain(attachment_name, swapchain, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
		execute_submit_and_present_to_one(ptc, std::move(rg).link(ptc), swapchain);
	}
}

int main() {
	vuk::BenchRunner::get_runner().setup();
	vuk::BenchRunner::get_runner().render();
	vuk::BenchRunner::get_runner().cleanup();
}