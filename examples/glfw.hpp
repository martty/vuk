#pragma once
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

inline GLFWwindow* create_window_glfw(const char* title, bool resize = true) {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	if (!resize) glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	return glfwCreateWindow(1024, 768, title, NULL, NULL);
}

inline void destroy_window_glfw(GLFWwindow* window) {
	glfwDestroyWindow(window);
	glfwTerminate();
}

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
