#pragma once

#include "vuk/runtime/vk/VkQueueExecutor.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"
#include <tracy/TracyVulkan.hpp>

namespace vuk::extra {
	struct TracyContext {
		std::vector<tracy::VkCtx*> contexts;

		// command buffer and pool for Tracy to do init & collect
		Unique<CommandPool> tracy_cpool;
		Unique<CommandBufferAllocation> tracy_cbufai;

		std::vector<Executor*> executors;

		~TracyContext() {
			for (auto ctx : contexts) {
				TracyVkDestroy(ctx);
			}
			tracy_cbufai.reset();
			tracy_cpool.reset();
		}
	};

	/// @brief Initialize Tracy for Vulkan/vuk
	/// @param allocator Allocator to use for Tracy init
	inline std::unique_ptr<TracyContext> init_Tracy(Allocator& allocator) {
		std::unique_ptr<TracyContext> tracy_context = std::make_unique<TracyContext>();
		Runtime& runtime = allocator.get_context();
		auto graphics_queue_executor = static_cast<QueueExecutor*>(runtime.get_executor(DomainFlagBits::eGraphicsQueue));
		VkCommandPoolCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
		cpci.queueFamilyIndex = graphics_queue_executor->get_queue_family_index();
		tracy_context->tracy_cpool = Unique<CommandPool>(allocator);
		allocator.allocate_command_pools(std::span{ &*tracy_context->tracy_cpool, 1 }, std::span{ &cpci, 1 });
		vuk::CommandBufferAllocationCreateInfo ci{ .command_pool = *tracy_context->tracy_cpool };
		tracy_context->tracy_cbufai = Unique<CommandBufferAllocation>(allocator);
		allocator.allocate_command_buffers(std::span{ &*tracy_context->tracy_cbufai, 1 }, std::span{ &ci, 1 });

		auto graphics_queue = graphics_queue_executor->get_underlying();
		tracy_context->executors = runtime.get_executors();
		for (size_t i = 0; i < tracy_context->executors.size(); i++) {
			auto ctx = TracyVkContextCalibrated(runtime.instance,
			                                    runtime.physical_device,
			                                    runtime.device,
			                                    graphics_queue,
			                                    tracy_context->tracy_cbufai->command_buffer,
			                                    runtime.vkGetInstanceProcAddr,
			                                    runtime.vkGetDeviceProcAddr);
			tracy_context->contexts.push_back(ctx);
		}

		return tracy_context;
	}

	/// @brief Make profiling callbacks for Tracy that can be passed to submits
	/// @param context A TracyContext previously made via init_Tracy
	inline ProfilingCallbacks make_Tracy_callbacks(TracyContext& context) {
		ProfilingCallbacks cbs;
		cbs.user_data = &context;
		cbs.on_begin_command_buffer = [](void* user_data, ExecutorTag tag, VkCommandBuffer cbuf) -> void* {
			TracyContext& tracy_ctx = *reinterpret_cast<TracyContext*>(user_data);
			if ((tag.domain & DomainFlagBits::eQueueMask) != DomainFlagBits::eTransferQueue) {
				for (auto& ctx : tracy_ctx.contexts) {
					TracyVkCollect(ctx, cbuf);
				}
			}
			return nullptr;
		};
		// runs whenever entering a new vuk::Pass
		// we start a GPU zone and then keep it open
		cbs.on_begin_pass = [](void* user_data, Name pass_name, CommandBuffer& cbuf, DomainFlagBits domain) {
			TracyContext& tracy_ctx = *reinterpret_cast<TracyContext*>(user_data);
			void* pass_data = nullptr;
			for (size_t i = 0; i < tracy_ctx.executors.size(); i++) {
				auto& exe = tracy_ctx.executors[i];
				if (exe->tag.domain == domain) {
					pass_data = new char[sizeof(tracy::VkCtxScope)];
					new (pass_data) TracyVkZoneTransient(tracy_ctx.contexts[i], , cbuf.get_underlying(), pass_name.c_str(), true);
					break;
				}
			}

			return pass_data;
		};
		// runs whenever a pass has ended, we end the GPU zone we started
		cbs.on_end_pass = [](void* user_data, void* pass_data, CommandBuffer&) {
			auto tracy_scope = reinterpret_cast<tracy::VkCtxScope*>(pass_data);
			if (tracy_scope) {
				tracy_scope->~VkCtxScope();
				delete reinterpret_cast<char*>(pass_data);
			}
		};

		return cbs;
	}
} // namespace vuk::extra