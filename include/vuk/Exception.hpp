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
			case VK_SUBOPTIMAL_KHR:
			{
				error_message = "Suboptimal."; break;
			}
			case VK_ERROR_OUT_OF_DATE_KHR:
			{
				error_message = "Out of date."; break;
			}
			default:
				assert(0 && "Unimplemented error."); break;
			}
		}
	};
}
