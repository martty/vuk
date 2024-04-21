#pragma once

#include "vuk/Config.hpp"
#include "vuk/Executor.hpp"
#include "vuk/SyncPoint.hpp"

#include <span>
#include <vector>

namespace vuk {
	struct SubmitInfo {
		std::vector<VkCommandBuffer> command_buffers;
		std::vector<std::pair<DomainFlagBits, uint64_t>> relative_waits;
		std::vector<Signal*> waits;
		std::vector<Signal*> signals;
		std::vector<VkSemaphore> pres_wait;
		std::vector<VkSemaphore> pres_signal;
	};

	/// @brief Abstraction of a device queue in Vulkan
	struct QueueExecutor : Executor {
		QueueExecutor(VkDevice device, DomainFlagBits domain, const struct FunctionPointers& fps, VkQueue queue, uint32_t queue_family_index, VkSemaphore ts);
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

	private:
		struct QueueImpl* impl;

		std::vector<VkSubmitInfo2KHR> sis;
		std::vector<VkCommandBufferSubmitInfoKHR> cbufsis;
		std::vector<VkSemaphoreSubmitInfoKHR> wait_semas;
		std::vector<VkSemaphoreSubmitInfoKHR> signal_semas;
	};
} // namespace vuk