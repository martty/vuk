#pragma once

#include <optional>

#include <VkBootstrap.h>

#include "vuk/runtime/vk/DeviceFrameResource.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"

namespace vuk::extra {
	/// @brief Wrapper around vuk::Runtime that manages Vulkan instance, device, swapchain and superframe resources
	struct SimpleApp {
		vkb::Instance vkbinstance;
		vkb::Device vkbdevice;

		uint32_t vk_api_major_version;
		uint32_t vk_api_minor_version;
		uint32_t vk_device_version;
		VkSurfaceKHR surface = VK_NULL_HANDLE;
		std::optional<Runtime> runtime;
		std::optional<DeviceSuperFrameResource> superframe_resource;
		std::optional<Allocator> superframe_allocator;
		std::optional<vuk::Swapchain> swapchain = {};

		/// @brief Create or recreate the swapchain
		void update_swapchain();
		/// @brief Wait for the device to be idle
		void wait_idle();
		/// @brief Advance the frame for the allocators and caches used by vuk
		void next_frame();

		~SimpleApp();
	};

	/// @brief Wrapper around vk-bootstrap's PhysicalDevice
	struct PhysicalDevice : vkb::PhysicalDevice {
		vkb::Instance instance;
		uint32_t rq_major_version;
		uint32_t rq_minor_version;
	};

	/// @brief Wrapper around vk-bootstrap's DeviceBuilder
	struct DeviceBuilder : vkb::DeviceBuilder {
		explicit DeviceBuilder(PhysicalDevice physical_device);

		VkPhysicalDeviceFeatures2 vk10features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR };
		VkPhysicalDeviceVulkan11Features vk11features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
		VkPhysicalDeviceVulkan12Features vk12features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
#ifdef VK_VERSION_1_3
		VkPhysicalDeviceVulkan13Features vk13features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
#endif
#ifdef VK_VERSION_1_4
		VkPhysicalDeviceVulkan14Features vk14features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES };
#endif

		/// @brief set all recommended features:
		/// descriptor indexing, shader output layer, shader draw parameters, shader int64, ...
		DeviceBuilder& set_recommended_features();

		/// @brief Build the Vulkan device only, use this if you want to create a vuk::Runtime yourself
		vkb::Result<vkb::Device> build_device_only();
		/// @brief Build the Vulkan device, vuk::Runtime and SimpleApp
		std::unique_ptr<SimpleApp> build_app(bool with_swapchain = true, unsigned num_inflight_frames = 3);

	private:
		PhysicalDevice physical_device;
	};

	/// @brief Prepopulated vkb::InstanceBuilder with default settings
	///		   Use the returned builder to set additional metadata, validation features, etc.
	/// @param vulkan_major_version Vulkan major version requested
	/// @param vulkan_minor_version Vulkan minor version requested
	/// @param with_default_callback If true, a default debug callback will be added
	vkb::InstanceBuilder make_instance_builder(uint32_t vulkan_major_version, uint32_t vulkan_minor_version, bool with_default_callback);

	/// @brief Build instance and select the first physical device vuk is compatible with and can present to the surface
	///		   You can enable or check for additional extensions on the returned PhysicalDevice
	/// @param instance Vulkan instance returned by building the instance builder
	///	@param surface Surface to create the swapchain for, can be VK_NULL_HANDLE if headless
	PhysicalDevice select_physical_device(vkb::Instance instance, VkSurfaceKHR surface);

	/// @brief Make a DeviceBuilder for the given physical device
	///		   You can chain additional pNexts for the VkDeviceCreateInfo on the returned builder
	///		   Use set_recommended_features() to set all recommended features, or set them manually on the returned DeviceBuilder member variables
	///        Use build_device_only() to build the device only, or build_runtime() to build a SimpleRuntime
	/// @param physical_device physical device to make the DeviceBuilder for
	DeviceBuilder make_device_builder(PhysicalDevice physical_device);

	/// @brief Helper function to create a swapchain
	/// @param allocator Allocator to allocate resources from
	/// @param vkbdevice Device made by vk-bootstrap
	/// @param surface Surface to create swapchain for
	/// @param old_swapchain An optional old swapchain to recycle
	/// @return New swapchain
	Swapchain make_swapchain(Allocator allocator, vkb::Device vkbdevice, VkSurfaceKHR surface, std::optional<Swapchain> old_swapchain);
} // namespace vuk::extra