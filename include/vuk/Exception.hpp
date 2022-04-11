#pragma once

#include <exception>
#include <string>
#include <cassert>
#include "vuk/Config.hpp"

namespace vuk {
	struct Exception : std::exception {
		std::string error_message;

		Exception() {}
		Exception(std::string message) : error_message(std::move(message)) {}

		const char* what() const noexcept override {
			return error_message.c_str();
		}

		virtual void throw_this() = 0;
	};

	struct ShaderCompilationException : Exception {
		using Exception::Exception;

		void throw_this() override {
			throw *this;
		}
	};

	struct RenderGraphException : Exception {
		using Exception::Exception;

		void throw_this() override {
			throw *this;
		}
	};

	struct VkException : Exception {
		VkResult error_code;
		
		using Exception::Exception;
		
		VkException(VkResult res) {
			error_code = res;
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
		
		VkResult code() const { return error_code; }

		void throw_this() override {
			throw *this;
		}
	};

	struct PresentException : VkException {
		PresentException(VkResult res) {
			error_code = res;
			
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

		void throw_this() override {
			throw *this;
		}
	};

	struct AllocateException : VkException {
		AllocateException(VkResult res) {
			error_code = res;
			
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

		void throw_this() override {
			throw *this;
		}
	};
} // namespace vuk
