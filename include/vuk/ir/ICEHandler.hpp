#pragma once

#include <functional>
#include <string>

namespace vuk {
	/// @brief Handler for Internal Compiler Errors (ICE)
	/// Provides a mechanism to dump debug information when an assertion fails
	/// Thread-local to allow different compilation contexts to have different handlers
	class ICEHandler {
	public:
		using DumpCallback = std::function<void()>;

		/// @brief Set a callback to be executed when an ICE occurs
		/// @param callback Function to call on ICE (typically dumps graphs/snapshots)
		void set_dump_callback(DumpCallback callback) {
			dump_callback_ = std::move(callback);
			triggered_ = false;
		}

		/// @brief Clear the dump callback
		void clear_dump_callback() {
			dump_callback_ = nullptr;
			triggered_ = false;
		}

		/// @brief Trigger the ICE handler
		/// This is called by the VUK_ICE macro when an assertion fails
		/// @param expression The expression that failed (as string)
		/// @param file The source file where the assertion failed
		/// @param line The line number where the assertion failed
		void trigger(const char* expression, const char* file, int line);

		/// @brief Get the thread-local instance of the ICE handler
		static ICEHandler& get_instance();

	private:
		DumpCallback dump_callback_;
		bool triggered_ = false;
	};
} // namespace vuk
