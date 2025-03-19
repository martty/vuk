#include "vuk/runtime/vk/VkQueueExecutor.hpp"
#include "vuk/Exception.hpp"
#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"

#include <mutex>
#include <vector>

namespace vuk {
	struct QueueImpl {
		VkDevice device;
		// TODO: this recursive mutex should be changed to better queue handling
		std::recursive_mutex queue_lock;
		PFN_vkQueueSubmit queueSubmit;
		PFN_vkQueueSubmit2KHR queueSubmit2KHR;
		PFN_vkQueueWaitIdle queueWaitIdle;
		PFN_vkDestroySemaphore destroySemaphore;
		PFN_vkQueuePresentKHR queuePresentKHR;
		PFN_vkGetSemaphoreCounterValue getSemaphoreCounterValue;
		VkSemaphore submit_sync;
		uint64_t sync_value = 0;
		VkQueue queue;
		uint32_t family_index;

		QueueImpl(VkDevice device, const FunctionPointers& fps, VkQueue queue, uint32_t queue_family_index, VkSemaphore sema) :
		    device(device),
		    queueSubmit(fps.vkQueueSubmit),
		    queueSubmit2KHR(fps.vkQueueSubmit2KHR),
		    queueWaitIdle(fps.vkQueueWaitIdle),
		    destroySemaphore(fps.vkDestroySemaphore),
		    queuePresentKHR(fps.vkQueuePresentKHR),
		    getSemaphoreCounterValue(fps.vkGetSemaphoreCounterValue),
		    submit_sync(sema),
		    queue(queue),
		    family_index(queue_family_index) {}

		~QueueImpl() {
			destroySemaphore(device, submit_sync, nullptr);
		}
	};

	std::unique_ptr<Executor>
	create_vkqueue_executor(const FunctionPointers& fps, VkDevice device, VkQueue queue, uint32_t queue_family_index, DomainFlagBits domain) {
		VkSemaphore sema;
		VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
		VkSemaphoreTypeCreateInfo stci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
		stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
		stci.initialValue = 0;
		sci.pNext = &stci;
		VkResult res = fps.vkCreateSemaphore(device, &sci, nullptr, &sema);
		if (res != VK_SUCCESS) {
			return { nullptr };
		}

		if (fps.vkSetDebugUtilsObjectNameEXT) {
			VkDebugUtilsObjectNameInfoEXT info = { .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
			switch (domain) {
			case DomainFlagBits::eGraphicsQueue:
				info.pObjectName = "Graphics Queue";
				break;
			case DomainFlagBits::eComputeQueue:
				info.pObjectName = "Compute Queue";
				break;
			case DomainFlagBits::eTransferQueue:
				info.pObjectName = "Transfer Queue";
				break;
			default:
				break;
			}
			info.objectType = VK_OBJECT_TYPE_QUEUE;
			info.objectHandle = reinterpret_cast<uint64_t>(queue);
			fps.vkSetDebugUtilsObjectNameEXT(device, &info);
		}

		return std::make_unique<QueueExecutor>(device, domain, fps, queue, queue_family_index, sema);
	}

	QueueExecutor::QueueExecutor(VkDevice device,
	                             DomainFlagBits domain,
	                             const FunctionPointers& fps,
	                             VkQueue queue,
	                             uint32_t queue_family_index,
	                             VkSemaphore sema) :
	    Executor(Executor::Type::eVulkanDeviceQueue, domain, reinterpret_cast<uint64_t>(queue)),
	    impl(new QueueImpl(device, fps, queue, queue_family_index, sema)) {}

	QueueExecutor::~QueueExecutor() {
		delete impl;
	}

	QueueExecutor::QueueExecutor(QueueExecutor&& o) noexcept : Executor(o.type, o.tag.domain, o.tag.executor_id), impl(std::exchange(o.impl, nullptr)) {}

	QueueExecutor& QueueExecutor::operator=(QueueExecutor&& o) noexcept {
		impl = std::exchange(o.impl, nullptr);
		type = o.type;
		tag.domain = o.tag.domain;
		tag.executor_id = o.tag.executor_id;
		return *this;
	}

	Result<void> QueueExecutor::submit(std::span<VkSubmitInfo2KHR> sis, VkFence fence) {
		std::lock_guard _(impl->queue_lock);
		VkResult result = impl->queueSubmit2KHR(impl->queue, (uint32_t)sis.size(), sis.data(), fence);
		if (result != VK_SUCCESS) {
			return { expected_error, VkException{ result } };
		}
		return { expected_value };
	}

	Result<void> QueueExecutor::submit(std::span<VkSubmitInfo> sis, VkFence fence) {
		std::lock_guard _(impl->queue_lock);
		VkResult result = impl->queueSubmit(impl->queue, (uint32_t)sis.size(), sis.data(), fence);
		if (result != VK_SUCCESS) {
			return { expected_error, VkException{ result } };
		}
		return { expected_value };
	}

	Result<VkResult> QueueExecutor::queue_present(VkPresentInfoKHR pi) {
		std::lock_guard _(impl->queue_lock);
		auto present_result = impl->queuePresentKHR(impl->queue, &pi);
		if (present_result != VK_SUCCESS && present_result != VK_SUBOPTIMAL_KHR) {
			return { expected_error, VkException{ present_result } };
		}
		if (present_result == VK_SUBOPTIMAL_KHR) {
			return { expected_value, present_result };
		}
		return { expected_value, VK_SUCCESS };
	}

	Result<uint64_t> QueueExecutor::get_sync_value() {
		uint64_t value;
		auto res = impl->getSemaphoreCounterValue(impl->device, impl->submit_sync, &value);
		if (res != VK_SUCCESS) {
			return { expected_error, VkException{ res } };
		}
		return { expected_value, value };
	}

	VkSemaphore QueueExecutor::get_semaphore() {
		return impl->submit_sync;
	}

	uint32_t QueueExecutor::get_queue_family_index() {
		return impl->family_index;
	}

	VkQueue QueueExecutor::get_underlying() {
		return impl->queue;
	}

	Result<void> QueueExecutor::submit_batch(std::vector<SubmitInfo> batch) {
		std::unique_lock _(*this);

		sis.clear();
		cbufsis.clear();
		wait_semas.clear();
		signal_semas.clear();

		uint64_t num_cbufs = 0;
		uint64_t num_waits = 0;
		for (uint64_t i = 0; i < batch.size(); i++) {
			SubmitInfo& submit_info = batch[i];
			num_cbufs += submit_info.command_buffers.size();
			num_waits += submit_info.waits.size() + submit_info.pres_wait.size();
		}

		cbufsis.reserve(num_cbufs);
		wait_semas.reserve(num_waits);
		signal_semas.reserve(batch.size() + 1); // 1 extra for render_complete

		for (uint64_t i = 0; i < batch.size(); i++) {
			SubmitInfo& submit_info = batch[i];

			for (auto& fut : submit_info.signals) {
				fut->status = Signal::Status::eSynchronizable;
			}

			if (submit_info.command_buffers.size() == 0) {
				continue;
			}

			for (uint64_t i = 0; i < submit_info.command_buffers.size(); i++) {
				cbufsis.emplace_back(
				    VkCommandBufferSubmitInfoKHR{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR, .commandBuffer = submit_info.command_buffers[i] });
			}

			uint32_t wait_sema_count = 0;
			for (auto& w : submit_info.waits) {
				assert(w->source.executor->type == Executor::Type::eVulkanDeviceQueue);
				QueueExecutor* executor = static_cast<QueueExecutor*>(w->source.executor);
				VkSemaphoreSubmitInfoKHR ssi{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR };
				ssi.semaphore = executor->get_semaphore();
				ssi.value = w->source.visibility;
				ssi.stageMask = (VkPipelineStageFlagBits2KHR)PipelineStageFlagBits::eAllCommands; // TODO: w now has stage info
				wait_semas.emplace_back(ssi);
				wait_sema_count++;
			}

			for (auto& w : submit_info.pres_wait) {
				VkSemaphoreSubmitInfoKHR ssi{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR };
				ssi.semaphore = w;
				ssi.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR;
				wait_semas.emplace_back(ssi);
				wait_sema_count++;
			}

			VkSemaphoreSubmitInfoKHR ssi{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR };
			ssi.semaphore = impl->submit_sync;
			ssi.value = ++impl->sync_value;

			ssi.stageMask = (VkPipelineStageFlagBits2KHR)PipelineStageFlagBits::eAllCommands;

			for (auto& fut : submit_info.signals) {
				fut->status = Signal::Status::eSynchronizable;
				fut->source = { this, ssi.value };
			}

			uint32_t signal_sema_count = 1;
			signal_semas.emplace_back(ssi);

			for (auto& w : submit_info.pres_signal) {
				ssi.semaphore = w;
				ssi.value = 0; // binary sema
				signal_semas.emplace_back(ssi);
				signal_sema_count++;
			}

			VkSubmitInfo2KHR& si = sis.emplace_back(VkSubmitInfo2KHR{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR });
			VkCommandBufferSubmitInfoKHR* p_cbuf_infos = &cbufsis.back() - (submit_info.command_buffers.size() - 1);
			VkSemaphoreSubmitInfoKHR* p_wait_semas = wait_sema_count > 0 ? &wait_semas.back() - (wait_sema_count - 1) : nullptr;
			VkSemaphoreSubmitInfoKHR* p_signal_semas = &signal_semas.back() - (signal_sema_count - 1);

			si.pWaitSemaphoreInfos = p_wait_semas;
			si.waitSemaphoreInfoCount = wait_sema_count;
			si.pCommandBufferInfos = p_cbuf_infos;
			si.commandBufferInfoCount = (uint32_t)submit_info.command_buffers.size();
			si.pSignalSemaphoreInfos = p_signal_semas;
			si.signalSemaphoreInfoCount = signal_sema_count;
		}
		VUK_DO_OR_RETURN(submit(std::span{ sis }, VK_NULL_HANDLE));

		return { expected_value };
	}

	void QueueExecutor::lock() {
		impl->queue_lock.lock();
	}
	void QueueExecutor::unlock() {
		impl->queue_lock.unlock();
	}

	Result<void> QueueExecutor::wait_idle() {
		std::scoped_lock _{ *this };

		auto result = impl->queueWaitIdle(impl->queue);
		if (result < 0) {
			return { expected_error, VkException(result) };
		} else {
			return { expected_value };
		}
	}
} // namespace vuk