#include "vuk/ir/ICEHandler.hpp"
#include <cstdio>

namespace vuk {
	void ICEHandler::trigger(const char* expression, const char* file, int line) {
		// Prevent cascading ICEs
		if (triggered_) {
			std::fprintf(stderr, "\n=== CASCADING ICE DETECTED (already triggered) ===\n");
			std::fprintf(stderr, "Expression: %s\n", expression);
			std::fprintf(stderr, "File: %s\n", file);
			std::fprintf(stderr, "Line: %d\n", line);
			std::fprintf(stderr, "===================================================\n\n");
			return;
		}

		triggered_ = true;

		// Print error message
		std::fprintf(stderr, "\n=== INTERNAL COMPILER ERROR ===\n");
		std::fprintf(stderr, "Expression: %s\n", expression);
		std::fprintf(stderr, "File: %s\n", file);
		std::fprintf(stderr, "Line: %d\n", line);
		std::fprintf(stderr, "================================\n\n");

		// Execute dump callback if set
		if (dump_callback_) {
			std::fprintf(stderr, "Dumping debug information...\n");
			try {
				dump_callback_();
				std::fprintf(stderr, "Debug dump complete.\n\n");
			} catch (...) {
				std::fprintf(stderr, "Exception thrown during debug dump!\n\n");
			}
		}
	}

	ICEHandler& ICEHandler::get_instance() {
		thread_local ICEHandler instance;
		return instance;
	}
} // namespace vuk
