#include "vuk/extra/SimpleInit.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/runtime/ThisThreadExecutor.hpp"

namespace vuk::extra {
	vkb::InstanceBuilder make_instance_builder(uint32_t vulkan_major_version, uint32_t vulkan_minor_version, bool with_default_callback) {
		vkb::InstanceBuilder builder;
		assert(vulkan_major_version >= 1);
		assert(vulkan_minor_version >= 2 && "vuk needs at least Vulkan 1.2!");
		builder.request_validation_layers().require_api_version(vulkan_major_version, vulkan_minor_version, 0);
		if (with_default_callback) {
			builder.set_debug_callback([](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			                              VkDebugUtilsMessageTypeFlagsEXT messageType,
			                              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
			                              void* pUserData) -> VkBool32 {
				auto ms = vkb::to_string_message_severity(messageSeverity);
				auto mt = vkb::to_string_message_type(messageType);
				printf("[%s: %s]\n%s\n", ms, mt, pCallbackData->pMessage);
				return VK_FALSE;
			});
		}
		return builder;
	}

	PhysicalDevice select_physical_device(vkb::Instance vkbinstance, VkSurfaceKHR surface) {
		auto api_version = vkbinstance.api_version;
		uint32_t rq_major_version = VK_API_VERSION_MAJOR(api_version);
		uint32_t rq_minor_version = VK_API_VERSION_MINOR(api_version);
		vkb::PhysicalDeviceSelector selector{ vkbinstance };
		// vuk requires at least vulkan 1.2 and the synchronization2 extension
		selector.set_surface(surface).set_minimum_version(rq_major_version, rq_minor_version).add_required_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);

		auto phys_ret = selector.select();
		if (!phys_ret) {
			auto err_msg = std::string("ERROR: couldn't find suitable physical device - ") + phys_ret.error().message();
			printf("ERROR: %s\n", err_msg.c_str());
			throw std::runtime_error(err_msg);
		}
		vkb::PhysicalDevice vkbphysical_device = phys_ret.value();

		// enable ray tracing extensions if available
		vkbphysical_device.enable_extension_if_present(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
		vkbphysical_device.enable_extension_if_present(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
		vkbphysical_device.enable_extension_if_present(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
		// enable mesh shader extensions if available
		vkbphysical_device.enable_extension_if_present(VK_EXT_MESH_SHADER_EXTENSION_NAME);
		// enable calibrated timestamps if avaliable (useful for profiling)
		vkbphysical_device.enable_extension_if_present(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
		// enable push descriptor if available (useful for some optimisations)
		vkbphysical_device.enable_extension_if_present(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
#ifdef __APPLE__
		// enable portability if available (for Apple systems)
		vkbphysical_device.enable_extension_if_present(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif // __APPLE__

		return { vkbphysical_device, vkbinstance, rq_major_version, rq_minor_version };
	}

	DeviceBuilder make_device_builder(PhysicalDevice physical_device) {
		return DeviceBuilder{ physical_device };
	}

	DeviceBuilder::DeviceBuilder(PhysicalDevice vkbphysical_device) : vkb::DeviceBuilder(vkbphysical_device), physical_device(vkbphysical_device) {
		vk12features.timelineSemaphore = true;
		vk12features.hostQueryReset = true;
		vk12features.bufferDeviceAddress = true;
	}

	DeviceBuilder& DeviceBuilder::set_recommended_features() {
		vk12features.descriptorIndexing = true;
		vk12features.descriptorBindingPartiallyBound = true;
		vk12features.descriptorBindingUpdateUnusedWhilePending = true;
		vk12features.descriptorBindingSampledImageUpdateAfterBind = true;
		vk12features.descriptorBindingStorageImageUpdateAfterBind = true;
		vk12features.shaderSampledImageArrayNonUniformIndexing = true;
		vk12features.runtimeDescriptorArray = true;
		vk12features.descriptorBindingVariableDescriptorCount = true;

		vk12features.shaderOutputLayer = true;
		vk11features.shaderDrawParameters = true;
		vk10features.features.shaderInt64 = true;
		vk10features.features.tessellationShader = true;
		vk10features.features.fillModeNonSolid = true;

		return *this;
	}

	vkb::Result<vkb::Device> DeviceBuilder::build_device_only() {
		VkPhysicalDeviceSynchronization2FeaturesKHR sync_feat{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
			                                                     .synchronization2 = true };
		VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
			                                                             .accelerationStructure = true };
		VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
			                                                               .rayTracingPipeline = true };
		VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeature{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
			                                                       .taskShader = VK_TRUE,
			                                                       .meshShader = VK_TRUE };

		auto& device_builder = add_pNext(&vk12features).add_pNext(&vk11features).add_pNext(&vk10features);
		// add ray tracing features if available
		if (physical_device.is_extension_present(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
		    physical_device.is_extension_present(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) {
			device_builder.add_pNext(&rtPipelineFeature).add_pNext(&accelFeature);
		}
		// add mesh shader features if available
		if (physical_device.is_extension_present(VK_EXT_MESH_SHADER_EXTENSION_NAME)) {
			device_builder.add_pNext(&meshShaderFeature);
		}

		assert(physical_device.rq_major_version == 1 && "This code needs updating.");
#ifdef VK_VERSION_1_3
		if (physical_device.rq_minor_version >= 3) {
			device_builder.add_pNext(&vk13features);
		} else {
			device_builder.add_pNext(&sync_feat);
		}
#else
		device_builder.add_pNext(&sync_feat);
#endif
#ifdef VK_VERSION_1_4
		if (physical_device.rq_minor_version >= 4) {
			device_builder.add_pNext(&vk14features);
		}
#endif
		return device_builder.build();
	}

	Swapchain make_swapchain(Allocator allocator, vkb::Device vkbdevice, VkSurfaceKHR surface, std::optional<Swapchain> old_swapchain) {
		vkb::SwapchainBuilder swb(vkbdevice, surface);
		swb.set_desired_format(SurfaceFormatKHR{ Format::eR8G8B8A8Srgb, ColorSpaceKHR::eSrgbNonlinear });
		swb.add_fallback_format(SurfaceFormatKHR{ Format::eB8G8R8A8Srgb, ColorSpaceKHR::eSrgbNonlinear });
		swb.set_desired_present_mode((VkPresentModeKHR)PresentModeKHR::eImmediate);
		swb.set_image_usage_flags(VkImageUsageFlagBits::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_DST_BIT);

		bool is_recycle = false;
		vkb::Result<vkb::Swapchain> vkswapchain = { vkb::Swapchain{} };
		if (!old_swapchain) {
			vkswapchain = swb.build();
			old_swapchain.emplace(allocator, vkswapchain->image_count);
		} else {
			is_recycle = true;
			swb.set_old_swapchain(old_swapchain->swapchain);
			vkswapchain = swb.build();
		}

		if (is_recycle) {
			allocator.deallocate(std::span{ &old_swapchain->swapchain, 1 });
			for (auto& iv : old_swapchain->images) {
				allocator.deallocate(std::span{ &iv.image_view, 1 });
			}
		}

		auto images = *vkswapchain->get_images();
		auto views = *vkswapchain->get_image_views();

		old_swapchain->images.clear();

		for (auto i = 0; i < images.size(); i++) {
			ImageAttachment ia;
			ia.extent = { vkswapchain->extent.width, vkswapchain->extent.height, 1 };
			ia.format = (Format)vkswapchain->image_format;
			ia.image = Image{ images[i], nullptr };
			ia.image_view = ImageView{ { 0 }, views[i] };
			ia.view_type = ImageViewType::e2D;
			ia.sample_count = Samples::e1;
			ia.base_level = ia.base_layer = 0;
			ia.level_count = ia.layer_count = 1;
			old_swapchain->images.push_back(ia);
		}

		old_swapchain->swapchain = vkswapchain->swapchain;
		old_swapchain->surface = surface;

		return std::move(*old_swapchain);
	}

	std::unique_ptr<SimpleApp> DeviceBuilder::build_app(bool with_swapchain, unsigned num_inflight_frames) {
		auto dev_ret = build_device_only();
		if (!dev_ret) {
			auto err_msg = std::string("ERROR: couldn't create device - ") + dev_ret.error().message();
			printf("ERROR: %s\n", err_msg.c_str());
			throw std::runtime_error(err_msg);
		}
		std::unique_ptr<SimpleApp> sr = std::make_unique<SimpleApp>();
		sr->vkbdevice = dev_ret.value();
		sr->surface = physical_device.surface;
		sr->vkbinstance = physical_device.instance;
		sr->vk_api_major_version = physical_device.rq_major_version;
		sr->vk_api_minor_version = physical_device.rq_minor_version;
		sr->vk_device_version = physical_device.properties.apiVersion;
		auto instance = sr->vkbinstance.instance;
		FunctionPointers fps;
		fps.vkGetInstanceProcAddr = sr->vkbinstance.fp_vkGetInstanceProcAddr;
		fps.load_pfns(instance, sr->vkbdevice.device, true);
		std::vector<std::unique_ptr<Executor>> executors;

		// create an executor for the graphics queue
		auto graphics_queue = sr->vkbdevice.get_queue(vkb::QueueType::graphics).value();
		auto graphics_queue_family_index = sr->vkbdevice.get_queue_index(vkb::QueueType::graphics).value();
		executors.push_back(vuk::create_vkqueue_executor(fps, sr->vkbdevice.device, graphics_queue, graphics_queue_family_index, DomainFlagBits::eGraphicsQueue));

		// create an executor for a transfer queue
		// this is an optional executor
		if (sr->vkbdevice.get_queue(vkb::QueueType::transfer)) {
			auto transfer_queue = sr->vkbdevice.get_queue(vkb::QueueType::transfer).value();
			auto transfer_queue_family_index = sr->vkbdevice.get_queue_index(vkb::QueueType::transfer).value();
			executors.push_back(vuk::create_vkqueue_executor(fps, sr->vkbdevice.device, transfer_queue, transfer_queue_family_index, DomainFlagBits::eTransferQueue));
		}

		// create an executor for the main thread
		executors.push_back(std::make_unique<ThisThreadExecutor>());
		// create the runtime
		sr->runtime.emplace(RuntimeCreateParameters{ instance, sr->vkbdevice.device, physical_device.physical_device, std::move(executors), fps });

		// create a superframe resource and allocator
		sr->superframe_resource.emplace(*sr->runtime, num_inflight_frames);
		sr->superframe_allocator.emplace(*sr->superframe_resource);

		// match shader compilation target version to the vk version we request
		sr->runtime->set_shader_target_version(VK_MAKE_API_VERSION(0, sr->vk_api_major_version, sr->vk_api_minor_version, 0));

		// create a swapchain if requested
		if (with_swapchain) {
			sr->update_swapchain();
		}

		return sr;
	}

	void SimpleApp::update_swapchain() {
		swapchain = make_swapchain(*superframe_allocator, vkbdevice, surface, std::move(swapchain));
		for (auto& iv : swapchain->images) {
			runtime->set_name(iv.image_view.payload, "Swapchain ImageView");
		}
	}

	void SimpleApp::wait_idle() {
		runtime->wait_idle();
	}

	void SimpleApp::next_frame() {
		runtime->next_frame();
	}

	SimpleApp::~SimpleApp() {
		swapchain.reset();
		superframe_resource.reset();
		runtime.reset();
		if (surface != VK_NULL_HANDLE) {
			vkb::destroy_surface(vkbinstance, surface);
		}
		vkb::destroy_device(vkbdevice);
		vkb::destroy_instance(vkbinstance);
	}
} // namespace vuk::extra
