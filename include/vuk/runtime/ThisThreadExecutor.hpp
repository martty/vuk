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

		ThisThreadExecutor(ThisThreadExecutor&&) = default;
		ThisThreadExecutor& operator=(ThisThreadExecutor&&) = default;

		// scheduling on the current thread is lock-free
		void lock() override {}
		void unlock() override {}
		Result<void> wait_idle() {
			return { expected_value };
		}
	};
} // namespace vuk