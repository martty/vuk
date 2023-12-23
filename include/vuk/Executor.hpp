#pragma once

#include "vuk/Types.hpp"

namespace vuk {
	/// @brief Base class for high level execution
	struct Executor {
		enum class Type { eVulkanDeviceQueue, eThisThread } type;
		struct Tag {
			DomainFlagBits domain;
			size_t executor_id;

			std::strong_ordering operator<=>(const Tag&) const = default;
		} tag;

		Executor(Type type, DomainFlagBits domain, size_t executor_id) : type(type), tag{domain, executor_id} {}
		virtual ~Executor() {}
		Executor(const Executor&) = delete;
		Executor& operator=(const Executor&) = delete;

		// lock this executor
		virtual void lock() = 0;
		// unlock this executor
		virtual void unlock() = 0;

		virtual Result<void> wait_idle() = 0;
	};

} // namespace vuk