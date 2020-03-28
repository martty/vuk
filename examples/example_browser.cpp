#include "example_runner.hpp"

std::vector<std::string> chosen_resource;

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
				.require_api_version(1, 1, 0)
				.set_app_version(0, 1, 0);
			auto inst_ret = builder.build();
			if (!inst_ret.has_value()) {
				// error
			}
			vkbinstance = inst_ret.value();
			auto instance = vkbinstance.instance;
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
			physical_device = vkbphysical_device.physical_device;

			vkb::DeviceBuilder device_builder{ vkbphysical_device };
			auto dev_ret = device_builder.build();
			if (!dev_ret.has_value()) {
				// error
			}
			vkbdevice = dev_ret.value();
			graphics_queue = vkbdevice.get_queue(vkb::QueueType::graphics).value();
			device = vkbdevice.device;
			
			context.emplace(instance, device, physical_device, graphics_queue);

			swapchain = context->add_swapchain(util::make_swapchain(vkbdevice));
}

bool render_all = false;

void vuk::ExampleRunner::render() {
	chosen_resource.resize(examples.size());

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		auto ifc = context->begin();

		ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 352.f, 2));
		ImGui::SetNextWindowSize(ImVec2(350, 0));
		ImGui::Begin("Example selector", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize);
		ImGui::Checkbox("All", &render_all); 
		ImGui::SameLine();
		
		static vuk::Example* item_current = examples[0];            // Here our selection is a single pointer stored outside the object.
		if (!render_all) {
			if (ImGui::BeginCombo("Examples", item_current->name.data(), ImGuiComboFlags_None)) {
				for (int n = 0; n < examples.size(); n++) {
					bool is_selected = (item_current == examples[n]);
					if (ImGui::Selectable(examples[n]->name.data(), is_selected))
						item_current = examples[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus();   // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
				}
				ImGui::EndCombo();
			}
		}
		ImGui::End();

		if (!render_all) { // render a single full window example
			auto rg = item_current->render(*this, ifc);
			ImGui::Render();
			auto ptc = ifc.begin();
			std::string attachment_name = std::string(item_current->name) + "_final";
			rg.add_pass(util::ImGui_ImplVuk_Render(ptc, attachment_name, "SWAPCHAIN", imgui_data, ImGui::GetDrawData()));
			rg.build();
			rg.bind_attachment_to_swapchain(attachment_name, swapchain, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
			rg.build(ptc);
			execute_submit_and_present_to_one(ptc, rg, swapchain);
		} else { // render all examples as imgui windows
			RenderGraph rg;
			auto ptc = ifc.begin();
			plf::colony<std::string> attachment_names;

			size_t i = 0;
			for (auto& ex : examples) {
				auto rg_frag = ex->render(*this, ifc);
				rg_frag.build();
				rg.passes.insert(rg.passes.end(), rg_frag.passes.begin(), rg_frag.passes.end());
				rg.bound_attachments.insert(rg_frag.bound_attachments.begin(), rg_frag.bound_attachments.end());
				auto& attachment_name = *attachment_names.emplace(std::string(ex->name) + "_final");

				rg.mark_attachment_internal(attachment_name, vk::Format::eR8G8B8A8Srgb, vk::Extent2D(300, 300), vuk::Samples::e1, vuk::ClearColor(0.1f, 0.2f, 0.3f, 1.f));
				ImGui::Begin(ex->name.data());
				if (rg_frag.use_chains.size() > 1) {
					bool disable = false;
					for (auto& c : rg_frag.use_chains) {
						std::string btn_id = "";
						if (c.first == attachment_name) {
							disable = false;
							btn_id = "F";
						} else {
							auto usage = rg_frag.compute_usage(c.second);
							auto bound_it = rg_frag.bound_attachments.find(c.first);
							auto samples = vk::SampleCountFlagBits::e1;
							if (bound_it != rg_frag.bound_attachments.end()) {
								if (!bound_it->second.samples.infer)
									samples = bound_it->second.samples.count;
								else if (disable) {
									samples = vk::SampleCountFlagBits::e2; // hack: disable potentially MS attachments
								}
							}
							disable = samples != vk::SampleCountFlagBits::e1;
							if (usage & vk::ImageUsageFlagBits::eColorAttachment) {
								btn_id += "C";
							} else if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
								btn_id += "D";
							}
						}
						if (disable) {
							btn_id += " (MS)";
						} else {
							btn_id += "##" + std::string(c.first);
						}
						if (disable) {
							ImGui::TextDisabled("%s", btn_id.c_str());
						} else {
							if (ImGui::Button(btn_id.c_str())) {
								chosen_resource[i] = c.first;
							}
						}
						if (ImGui::IsItemHovered())
							ImGui::SetTooltip("%s", c.first.data());
						ImGui::SameLine();
					}
					ImGui::NewLine();
				}
				if (chosen_resource[i].empty())
					chosen_resource[i] = attachment_name;
				ImGui::Image(&ptc.make_sampled_image(chosen_resource[i], imgui_data.font_sci), ImVec2(200, 200));
				ImGui::End();
				i++;
			}

			ImGui::Render();
			rg.add_pass(util::ImGui_ImplVuk_Render(ptc, "SWAPCHAIN", "SWAPCHAIN", imgui_data, ImGui::GetDrawData()));
			rg.build();
			rg.bind_attachment_to_swapchain("SWAPCHAIN", swapchain, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
			rg.build(ptc);
			execute_submit_and_present_to_one(ptc, rg, swapchain);
		}
	}
}

int main() {
	vuk::ExampleRunner::get_runner().setup();
	vuk::ExampleRunner::get_runner().render();
	vuk::ExampleRunner::get_runner().cleanup();
}