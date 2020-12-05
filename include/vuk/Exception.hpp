#pragma once

#include <exception>

namespace vuk {
	struct ShaderCompilationException {
		std::string error_message;

		const char* what() const {
			return error_message.c_str();
		}
	};
}
