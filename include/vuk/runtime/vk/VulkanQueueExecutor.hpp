#pragma once

#include "vuk/Config.hpp"
#include "vuk/Executor.hpp"

#include <vector>
#include <span>

namespace vuk {
	struct SubmitInfo;
	struct TimelineSemaphore;
}

namespace vuk::rtvk {
	/// @brief Abstraction of a device queue in Vulkan
	struct QueueExecutor : Executor {
		QueueExecutor(VkDevice device, DomainFlagBits domain, const struct FunctionPointers& fps, VkQueue queue, uint32_t queue_family_index, TimelineSemaphore ts);
		~QueueExecutor();

		QueueExecutor(QueueExecutor&&) noexcept;
		QueueExecutor& operator=(QueueExecutor&&) noexcept;

		Result<void> submit_batch(std::vector<SubmitInfo> batch);
		uint64_t get_sync_value();
		VkSemaphore get_semaphore();
		uint32_t get_queue_family_index();

		void lock() override;
		void unlock() override;
		Result<void> wait_idle() override;

		Result<void> submit(std::span<VkSubmitInfo> submit_infos, VkFence fence);
		Result<void> submit(std::span<VkSubmitInfo2KHR> submit_infos, VkFence fence);

		Result<VkResult> queue_present(VkPresentInfoKHR pi);

		struct QueueImpl* impl;
	};
} // namespace vuk::rtvk