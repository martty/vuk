#pragma once

#include "vuk/Config.hpp"
#include <GLFW/glfw3.h>

inline VkSurfaceKHR create_surface_glfw(VkInstance instance, GLFWwindow* window) {
	VkSurfaceKHR surface = nullptr;
	VkResult err = glfwCreateWindowSurface(instance, window, NULL, &surface);
	if (err) {
		const char* error_msg;
		int ret = glfwGetError(&error_msg);
		if (ret != 0) {
			throw error_msg;
		}
		surface = nullptr;
	}
	return surface;
}
