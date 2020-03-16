#include "example_runner.hpp"

vuk::ExampleRunner::ExampleRunner() {
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

void vuk::ExampleRunner::render() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		auto ifc = context->begin();

		ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 252.f, 2));
		ImGui::SetNextWindowSize(ImVec2(250, 0));
		ImGui::Begin("Example selector", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize);
		static vuk::Example* item_current = examples[0];            // Here our selection is a single pointer stored outside the object.
        if (ImGui::BeginCombo("Examples", item_current->name.data(), ImGuiComboFlags_None)) {
            for (int n = 0; n < examples.size(); n++)
            {
                bool is_selected = (item_current == examples[n]);
                if (ImGui::Selectable(examples[n]->name.data(), is_selected))
                    item_current = examples[n];
                if (is_selected)
                    ImGui::SetItemDefaultFocus();   // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }
		ImGui::End();
		auto rg = item_current->render(*this, ifc);

		ImGui::Render();

		auto ptc = ifc.begin();
		std::string attachment_name = std::string(item_current->name) + "_final";
		rg.add_pass(util::ImGui_ImplVuk_Render(ptc, attachment_name, "SWAPCHAIN", imgui_data, ImGui::GetDrawData()));
		rg.build();
		rg.bind_attachment_to_swapchain(attachment_name, swapchain, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
		rg.build(ptc);
		execute_submit_and_present_to_one(ptc, rg, swapchain);
	}
}

int main() {
	vuk::ExampleRunner::get_runner().setup();
	vuk::ExampleRunner::get_runner().render();
}