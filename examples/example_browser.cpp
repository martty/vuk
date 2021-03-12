#include "example_runner.hpp"
#include "RenderGraphUtil.hpp"

std::vector<vuk::Name> chosen_resource;

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
				.require_api_version(1, 2, 0)
				.set_app_version(0, 1, 0);
			auto inst_ret = builder.build();
			if (!inst_ret.has_value()) {
				// error
			}
			vkbinstance = inst_ret.value();
			auto instance = vkbinstance.instance;
			vkb::PhysicalDeviceSelector selector{ vkbinstance };
			window = create_window_glfw("Vuk All Examples", false);
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
			VkPhysicalDeviceVulkan11Features vk11features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
			vk11features.shaderDrawParameters = true;
			auto dev_ret = device_builder.add_pNext(&vk12features).add_pNext(&vk11features).build();
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
			vuk::Name attachment_name = vuk::Name(std::string(item_current->name) + "_final");
			util::ImGui_ImplVuk_Render(ptc, rg, attachment_name, "SWAPCHAIN", imgui_data, ImGui::GetDrawData());
			rg.attach_swapchain(attachment_name, swapchain, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
			execute_submit_and_present_to_one(ptc, std::move(rg).link(ptc), swapchain);
		} else { // render all examples as imgui windows
			RenderGraph rg;
			auto ptc = ifc.begin();
			plf::colony<vuk::Name> attachment_names;

			size_t i = 0;
			for (auto& ex : examples) {
				auto rg_frag = ex->render(*this, ifc);
				auto& attachment_name = *attachment_names.emplace(std::string(ex->name) + "_final");
				rg_frag.attach_managed(attachment_name, swapchain->format, vuk::Dimension2D::absolute( 300, 300 ), vuk::Samples::e1, vuk::ClearColor(0.1f, 0.2f, 0.3f, 1.f));
				rg_frag.compile();
				ImGui::Begin(ex->name.data());
				if (rg_frag.get_use_chains().size() > 1) {
					const auto& bound_attachments = rg_frag.get_bound_attachments();
					bool disable = false;
					for (const auto [key, use_refs] : rg_frag.get_use_chains()) {
						auto bound_it = bound_attachments.find(key);
						if (bound_it == bound_attachments.end())
							continue;
						auto samples = vuk::SampleCountFlagBits::e1;
						if ((*bound_it).second.samples != vuk::Samples::eInfer)
							samples = (*bound_it).second.samples.count;
						disable = disable || (samples != vuk::SampleCountFlagBits::e1);
					}

					for (const auto [key, use_refs] : rg_frag.get_use_chains()) {
						auto bound_it = bound_attachments.find(key);
						if (bound_it == bound_attachments.end())
							continue;
						std::string btn_id = "";
						bool prevent_disable = false;
						if (key.to_sv() == attachment_name) {
							prevent_disable = true;
							btn_id = "F";
						} else {
							auto usage = rg_frag.compute_usage(use_refs);
							if (usage & vuk::ImageUsageFlagBits::eColorAttachment) {
								btn_id += "C";
							} else if (usage & vuk::ImageUsageFlagBits::eDepthStencilAttachment) {
								btn_id += "D";
							} else if (usage & (vuk::ImageUsageFlagBits::eTransferSrc | vuk::ImageUsageFlagBits::eTransferDst)) {
								btn_id += "X";
							}
						}
						if (disable && !prevent_disable) {
							btn_id += " (MS)";
						} else {
							btn_id += "##" + std::string(key.to_sv());
						}
						if (disable && !prevent_disable) {
							ImGui::TextDisabled("%s", btn_id.c_str());
						} else {
							if (ImGui::Button(btn_id.c_str())) {
								chosen_resource[i] = key.to_sv();
							}
						}
						if (ImGui::IsItemHovered())
							ImGui::SetTooltip("%s", key.c_str());
						ImGui::SameLine();
					}
					ImGui::NewLine();
				}
				rg.append(std::move(rg_frag));

				if (chosen_resource[i].is_invalid())
					chosen_resource[i] = attachment_name;
				ImGui::Image(&ptc.make_sampled_image(chosen_resource[i], imgui_data.font_sci), ImVec2(200, 200));
				ImGui::End();
				i++;
			}

			ImGui::Render();
			util::ImGui_ImplVuk_Render(ptc, rg, "SWAPCHAIN", "SWAPCHAIN", imgui_data, ImGui::GetDrawData());
			rg.attach_swapchain("SWAPCHAIN", swapchain, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
			execute_submit_and_present_to_one(ptc, std::move(rg).link(ptc), swapchain);
		}
	}
}

int main() {
	vuk::ExampleRunner::get_runner().setup();
	vuk::ExampleRunner::get_runner().render();
	vuk::ExampleRunner::get_runner().cleanup();
}