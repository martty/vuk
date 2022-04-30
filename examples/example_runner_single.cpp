#include "example_runner.hpp"

vuk::ExampleRunner::ExampleRunner() {
	vkb::InstanceBuilder builder;
	builder.request_validation_layers()
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
	selector.set_surface(surface).set_minimum_version(1, 0).add_required_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
	auto phys_ret = selector.select();
	if (!phys_ret.has_value()) {
		// error
	}
	vkb::PhysicalDevice vkbphysical_device = phys_ret.value();
	physical_device = vkbphysical_device.physical_device;

	vkb::DeviceBuilder device_builder{ vkbphysical_device };
	VkPhysicalDeviceVulkan12Features vk12features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
	vk12features.timelineSemaphore = true;
	vk12features.descriptorBindingPartiallyBound = true;
	vk12features.descriptorBindingUpdateUnusedWhilePending = true;
	vk12features.shaderSampledImageArrayNonUniformIndexing = true;
	vk12features.runtimeDescriptorArray = true;
	vk12features.descriptorBindingVariableDescriptorCount = true;
	vk12features.hostQueryReset = true;
	VkPhysicalDeviceVulkan11Features vk11features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
	vk11features.shaderDrawParameters = true;
	VkPhysicalDeviceSynchronization2FeaturesKHR sync_feat{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR, .synchronization2 = true };
	auto dev_ret = device_builder.add_pNext(&vk12features).add_pNext(&vk11features).add_pNext(&sync_feat).build();
	if (!dev_ret.has_value()) {
		// error
	}
	vkbdevice = dev_ret.value();
	graphics_queue = vkbdevice.get_queue(vkb::QueueType::graphics).value();
	auto graphics_queue_family_index = vkbdevice.get_queue_index(vkb::QueueType::graphics).value();
	transfer_queue = vkbdevice.get_queue(vkb::QueueType::transfer).value();
	auto transfer_queue_family_index = vkbdevice.get_queue_index(vkb::QueueType::transfer).value();
	device = vkbdevice.device;

	context.emplace(ContextCreateParameters{ instance,
	                                         device,
	                                         physical_device,
	                                         graphics_queue,
	                                         graphics_queue_family_index,
	                                         VK_NULL_HANDLE,
	                                         VK_QUEUE_FAMILY_IGNORED,
	                                         transfer_queue,
	                                         transfer_queue_family_index });
	const unsigned num_inflight_frames = 3;
	xdev_rf_alloc.emplace(*context, num_inflight_frames);
	global.emplace(*xdev_rf_alloc);
	swapchain = context->add_swapchain(util::make_swapchain(vkbdevice));
}

void vuk::ExampleRunner::render() {
	std::vector<FutureBase*> controls;
	std::vector<RenderGraph*> rendergraphs;
	for (auto& f : ia_futures) {
		controls.emplace_back(f.get_control());
		rendergraphs.emplace_back(f.get_render_graph());
	}
	for (auto& f : buf_futures) {
		controls.emplace_back(f.get_control());
		rendergraphs.emplace_back(f.get_render_graph());
	}
	vuk::wait_for_futures_explicit(*global, controls, rendergraphs);
	ia_futures.clear();

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		auto& xdev_frame_resource = xdev_rf_alloc->get_next_frame();
		context->next_frame();
		Allocator frame_allocator(xdev_frame_resource);
		auto fut = examples[0]->render(*this, frame_allocator);
		auto attachment_name = vuk::Name(examples[0]->name);
		fut.get_render_graph()->attach_swapchain(attachment_name, swapchain, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
		RenderGraph rg;
		rg.attach_in("result", std::move(fut));
		auto erg = std::move(rg).link(*context, vuk::RenderGraph::CompileOptions{});
		execute_submit_and_present_to_one(frame_allocator, std::move(erg), swapchain);
	}
}

int main() {
	vuk::ExampleRunner::get_runner().setup();
	vuk::ExampleRunner::get_runner().render();
	vuk::ExampleRunner::get_runner().cleanup();
}
