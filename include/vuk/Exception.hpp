#pragma once

#include <exception>
#include <string>

namespace vuk {
	struct Exception : std::exception {
		std::string error_message;

		Exception(std::string message) : error_message(std::move(message)) {}

		const char* what() const override {
			return error_message.c_str();
		}
	};

	struct ShaderCompilationException : Exception {
		using Exception::Exception;
	};

	struct RenderGraphException : Exception {
		using Exception::Exception;
	};
}
