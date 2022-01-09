#pragma once

#include <exception>
#include <string>

namespace vuk {
	struct Exception : std::exception {
		std::string error_message;

		Exception() {}
		Exception(std::string message) : error_message(std::move(message)) {}

		const char* what() const noexcept override {
			return error_message.c_str();
		}
	};

	struct ShaderCompilationException : Exception {
		using Exception::Exception;
	};

	struct RenderGraphException : Exception {
		using Exception::Exception;
	};

	struct PresentException : Exception {
		PresentException(VkResult res) {
			switch (res) {
			case VK_SUBOPTIMAL_KHR: {
				error_message = "Suboptimal.";
				break;
			}
			case VK_ERROR_OUT_OF_DATE_KHR: {
				error_message = "Out of date.";
				break;
			}
			default:
				assert(0 && "Unimplemented error.");
				break;
			}
		}
	};

	struct AllocateException : Exception {
		AllocateException(VkResult res) {
			switch (res) {
			case VK_ERROR_OUT_OF_HOST_MEMORY: {
				error_message = "Out of host memory.";
				break;
			}
			case VK_ERROR_OUT_OF_DEVICE_MEMORY: {
				error_message = "Out of device memory.";
				break;
			}
			case VK_ERROR_INITIALIZATION_FAILED: {
				error_message = "Initialization failed.";
				break;
			}
			case VK_ERROR_DEVICE_LOST: {
				error_message = "Device lost.";
				break;
			}
			case VK_ERROR_MEMORY_MAP_FAILED: {
				error_message = "Memory map failed.";
				break;
			}
			case VK_ERROR_LAYER_NOT_PRESENT: {
				error_message = "Layer not present.";
				break;
			}
			case VK_ERROR_EXTENSION_NOT_PRESENT: {
				error_message = "Extension not present.";
				break;
			}
			case VK_ERROR_FEATURE_NOT_PRESENT: {
				error_message = "Feature not present.";
				break;
			}
			case VK_ERROR_INCOMPATIBLE_DRIVER: {
				error_message = "Incompatible driver.";
				break;
			}
			case VK_ERROR_TOO_MANY_OBJECTS: {
				error_message = "Too many objects.";
				break;
			}
			case VK_ERROR_FORMAT_NOT_SUPPORTED: {
				error_message = "Format not supported.";
				break;
			}
			default:
				assert(0 && "Unimplemented error.");
				break;
			}
		}
	};

	struct VkException : Exception {
		VkException(VkResult res) {
			switch (res) {
			case VK_ERROR_OUT_OF_HOST_MEMORY: {
				error_message = "Out of host memory.";
				break;
			}
			case VK_ERROR_OUT_OF_DEVICE_MEMORY: {
				error_message = "Out of device memory.";
				break;
			}
			case VK_ERROR_INITIALIZATION_FAILED: {
				error_message = "Initialization failed.";
				break;
			}
			case VK_ERROR_DEVICE_LOST: {
				error_message = "Device lost.";
				break;
			}
			case VK_ERROR_MEMORY_MAP_FAILED: {
				error_message = "Memory map failed.";
				break;
			}
			case VK_ERROR_LAYER_NOT_PRESENT: {
				error_message = "Layer not present.";
				break;
			}
			case VK_ERROR_EXTENSION_NOT_PRESENT: {
				error_message = "Extension not present.";
				break;
			}
			case VK_ERROR_FEATURE_NOT_PRESENT: {
				error_message = "Feature not present.";
				break;
			}
			case VK_ERROR_INCOMPATIBLE_DRIVER: {
				error_message = "Incompatible driver.";
				break;
			}
			case VK_ERROR_TOO_MANY_OBJECTS: {
				error_message = "Too many objects.";
				break;
			}
			case VK_ERROR_FORMAT_NOT_SUPPORTED: {
				error_message = "Format not supported.";
				break;
			}
			case VK_ERROR_UNKNOWN: {
				error_message = "Error unknown.";
				break;
			}
			default:
				assert(0 && "Unimplemented error.");
				break;
			}
		}
	};
} // namespace vuk
