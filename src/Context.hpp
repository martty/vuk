#pragma once

#include <atomic>
#include <gsl/span>

#include "Pool.hpp"
#include "Cache.hpp"

namespace vuk {
	class Context {
	public:
		constexpr static size_t FC = 3;

		vk::Device device;
		Pool<vk::CommandBuffer, FC> cbuf_pools;
		Pool<vk::Semaphore, FC> semaphore_pools;
		vk::UniquePipelineCache vk_pipeline_cache;
		Cache<vk::Pipeline> pipeline_cache;
		Cache<vk::RenderPass> renderpass_cache;


		Context(vk::Device device) : device(device),
			cbuf_pools(*this),
			semaphore_pools(*this),
			pipeline_cache(*this),
			renderpass_cache(*this)
		{
			vk_pipeline_cache = device.createPipelineCacheUnique({});
		}


		std::atomic<size_t> frame_counter = 0;
		InflightContext begin();
	};

	inline unsigned prev_(unsigned frame, unsigned amt, unsigned FC) {
		return ((frame - amt) % FC) + ((frame >= amt) ? 0 : FC - 1);
	}

	class InflightContext {
	public:
		Context& ctx;
		unsigned frame;
		Pool<vk::CommandBuffer, Context::FC>::PFView commandbuffer_pools;
		Pool<vk::Semaphore, Context::FC>::PFView semaphore_pools;
		Cache<vk::Pipeline>::View pipeline_cache;
		Cache<vk::RenderPass>::View renderpass_cache;


		InflightContext(Context& ctx, unsigned frame) : ctx(ctx), frame(frame),
			commandbuffer_pools(ctx.cbuf_pools.get_view(*this)),
			semaphore_pools(ctx.semaphore_pools.get_view(*this)),
			pipeline_cache(*this, ctx.pipeline_cache),
			renderpass_cache(*this, ctx.renderpass_cache)
		{
			auto prev_frame = prev_(frame, 1, Context::FC);
			ctx.cbuf_pools.reset(prev_frame);
			ctx.semaphore_pools.reset(prev_frame);
		}

		PerThreadContext begin();
	};

	inline InflightContext Context::begin() {
		return InflightContext(*this, frame_counter++ % FC);
	}

	class PerThreadContext {
	public:
		Context& ctx;
		InflightContext& ifc;
		unsigned tid;
		Pool<vk::CommandBuffer, Context::FC>::PFPTView commandbuffer_pool;
		Pool<vk::Semaphore, Context::FC>::PFPTView semaphore_pool;

		PerThreadContext(InflightContext& ifc, unsigned tid) : ctx(ifc.ctx), ifc(ifc), tid(tid),
			commandbuffer_pool(ifc.commandbuffer_pools.get_view(*this)),
			semaphore_pool(ifc.semaphore_pools.get_view(*this))
		{}

	};

	inline PerThreadContext InflightContext::begin() {
		return PerThreadContext{ *this, 0 };
	}

	template<class T, size_t FC>
	typename Pool<T, FC>::PFView Pool<T, FC>::get_view(InflightContext& ctx) {
		return { ctx, *this, per_frame_storage[ctx.frame] };
	}

	template<class T>
	T create(Context& ctx, create_info_t<T> cinfo) {
		if constexpr (std::is_same_v<T, vk::Pipeline>) {
			return ctx.device.createGraphicsPipeline(*ctx.vk_pipeline_cache, cinfo);
		} else if constexpr (std::is_same_v<T, vk::RenderPass>) {
			return ctx.device.createRenderPass(cinfo);
		}
	}

}
