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
				.require_api_version(1, 2, 0)
				.set_app_version(0, 1, 0);
			auto inst_ret = builder.build();
			if (!inst_ret.has_value()) {
				// error
			}
			vkbinstance = inst_ret.value();
			auto instance = vkbinstance.instance;
			vkb::PhysicalDeviceSelector selector{ vkbinstance };
			window = create_window_glfw("Vuk example", false);
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
			rf_alloc.emplace(context->device, vuk::Context::FC);
			swapchain = context->add_swapchain(util::make_swapchain(vkbdevice));
}

void vuk::ExampleRunner::render() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		auto& frame_alloc = rf_alloc->get_next_frame();
		auto ifc = context->begin();
		auto rg = examples[0]->render(*this, ifc);
		auto attachment_name = vuk::Name(std::string(examples[0]->name) + "_final");
		rg.attach_swapchain(attachment_name, swapchain, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
		auto erg = std::move(rg).link(*context, vuk::RenderGraph::CompileOptions{});
		vuk::NLinear nl(context->device, frame_alloc);
		NAllocator alloc(nl);
		execute_submit_and_present_to_one(*context, alloc, std::move(erg), swapchain);
		nl.subsume();
	}
}


int main() {
	vuk::ExampleRunner::get_runner().setup();
	vuk::ExampleRunner::get_runner().render();
	vuk::ExampleRunner::get_runner().cleanup();
}