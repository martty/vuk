#pragma once
#include "renderdoc_app.h"
#include "vuk/Allocator.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/CommandBuffer.hpp"
#include "vuk/Config.hpp"
#include "vuk/Context.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/resources/DeviceFrameResource.hpp"
#ifdef WIN32
#include <Windows.h>
#endif
#include <VkBootstrap.h>
#include <mutex>

namespace vuk {
	struct TestContext {
		Compiler compiler;
		bool has_rt;
		VkDevice device;
		VkPhysicalDevice physical_device;
		VkQueue graphics_queue;
		VkQueue transfer_queue;
		std::optional<Context> context;
		vkb::Instance vkbinstance;
		vkb::Device vkbdevice;
		std::optional<DeviceSuperFrameResource> sfa_resource;
		std::optional<Allocator> allocator;
		RENDERDOC_API_1_6_0* rdoc_api = NULL;

		void bringup() {
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
			    .set_app_version(0, 1, 0)
			    .set_headless();
			auto inst_ret = builder.build();
			if (!inst_ret) {
				throw std::runtime_error("Couldn't initialise instance");
			}

			has_rt = true;

			vkbinstance = inst_ret.value();
			auto instance = vkbinstance.instance;
			vkb::PhysicalDeviceSelector selector{ vkbinstance };
			selector.set_minimum_version(1, 0)
			    .add_required_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
			    .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
			    .add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
			    .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
			auto phys_ret = selector.select();
			vkb::PhysicalDevice vkbphysical_device;
			if (!phys_ret) {
				has_rt = false;
				vkb::PhysicalDeviceSelector selector2{ vkbinstance };
				selector2.set_minimum_version(1, 0).add_required_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
				auto phys_ret2 = selector2.select();
				if (!phys_ret2) {
					throw std::runtime_error("Couldn't create physical device");
				} else {
					vkbphysical_device = phys_ret2.value();
				}
			} else {
				vkbphysical_device = phys_ret.value();
			}

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
			vk12features.bufferDeviceAddress = true;
			vk12features.shaderOutputLayer = true;
			VkPhysicalDeviceVulkan11Features vk11features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
			vk11features.shaderDrawParameters = true;
			VkPhysicalDeviceFeatures2 vk10features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR };
			vk10features.features.shaderInt64 = true;
			VkPhysicalDeviceSynchronization2FeaturesKHR sync_feat{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
				                                                     .synchronization2 = true };
			VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
				                                                             .accelerationStructure = true };
			VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
				                                                               .rayTracingPipeline = true };
			device_builder = device_builder.add_pNext(&vk12features).add_pNext(&vk11features).add_pNext(&sync_feat).add_pNext(&accelFeature).add_pNext(&vk10features);
			if (has_rt) {
				device_builder = device_builder.add_pNext(&rtPipelineFeature);
			}
			auto dev_ret = device_builder.build();
			if (!dev_ret) {
				throw std::runtime_error("Couldn't create device");
			}
			vkbdevice = dev_ret.value();
			graphics_queue = vkbdevice.get_queue(vkb::QueueType::graphics).value();
			auto graphics_queue_family_index = vkbdevice.get_queue_index(vkb::QueueType::graphics).value();
			transfer_queue = vkbdevice.get_queue(vkb::QueueType::transfer).value();
			auto transfer_queue_family_index = vkbdevice.get_queue_index(vkb::QueueType::transfer).value();
			device = vkbdevice.device;
			ContextCreateParameters::FunctionPointers fps;
			fps.vkGetInstanceProcAddr = vkbinstance.fp_vkGetInstanceProcAddr;
			fps.vkGetDeviceProcAddr = vkbinstance.fp_vkGetDeviceProcAddr;
			context.emplace(ContextCreateParameters{ instance,
			                                         device,
			                                         physical_device,
			                                         graphics_queue,
			                                         graphics_queue_family_index,
			                                         VK_NULL_HANDLE,
			                                         VK_QUEUE_FAMILY_IGNORED,
			                                         transfer_queue,
			                                         transfer_queue_family_index,
			                                         fps });
			needs_bringup = false;
			needs_teardown = true;
#ifdef WIN32
			// At init, on windows
			if (HMODULE mod = GetModuleHandleA("renderdoc.dll")) {
				pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
				int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, (void**)&rdoc_api);
				if (ret != 1) {
					rdoc_api = nullptr;
				}
			}
#endif // WIN32
		}

		void start(const char* name) {
			if (needs_bringup) {
				bringup();
			}

			const unsigned num_inflight_frames = 3;
			sfa_resource.emplace(*context, num_inflight_frames);
			allocator.emplace(*sfa_resource);

			if (rdoc_api) {
				rdoc_api->StartFrameCapture(NULL, NULL);
				rdoc_api->SetCaptureTitle(name);
			}
		}

		void finish() {
			context->wait_idle();
			sfa_resource.reset();
			if (rdoc_api)
				rdoc_api->EndFrameCapture(NULL, NULL);
#ifdef VUK_TEST_FULL_ISOLATION
			teardown();
#endif
		}

		void teardown() {
			context.reset();
			vkb::destroy_device(vkbdevice);
			vkb::destroy_instance(vkbinstance);
			needs_bringup = true;
			needs_teardown = false;
		}

		bool needs_teardown = false;
		bool needs_bringup = true;

		~TestContext() {
			if (needs_teardown) {
				teardown();
			}
		}
	};

	extern TestContext test_context;
} // namespace vuk

// helpers for testing

constexpr bool operator==(const std::span<uint32_t>& lhs, const std::span<uint32_t>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<uint32_t>& lhs, const std::span<const uint32_t>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<const uint32_t>& lhs, const std::span<const uint32_t>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<float>& lhs, const std::span<float>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<float>& lhs, const std::span<const float>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<const float>& lhs, const std::span<const float>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}
