#pragma once

#include "vuk/Config.hpp"
#include "vuk/Executor.hpp"
#include "vuk/Result.hpp"

#include <span>
#include <vector>

namespace vuk {
	/// @brief Abstraction of execution on the current thread
	struct ThisThreadExecutor : Executor {
		ThisThreadExecutor() : Executor(Executor::Type::eThisThread, DomainFlagBits::eHost, 0) {}

		ThisThreadExecutor(ThisThreadExecutor&&) = delete;
		ThisThreadExecutor& operator=(ThisThreadExecutor&&) = delete;

		// scheduling on the current thread is lock-free
		void lock() override {}
		void unlock() override {}
		Result<void> wait_idle() override {
			return { expected_value };
		}
	};
} // namespace vuk