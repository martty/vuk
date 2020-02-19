#pragma once

#include <atomic>
#include <gsl/span>

#include "Pool.hpp"

namespace vuk {
	class Context {
	public:
		constexpr static size_t FC = 3;

		vk::Device device;
		Pool<vk::CommandBuffer, FC> cbuf_pools;
		Pool<vk::Semaphore, FC> semaphore_pools;

		Context(vk::Device device) : device(device),
			cbuf_pools(*this),
			semaphore_pools(*this)
		{}

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
		PFView<vk::CommandBuffer, Context::FC> commandbuffer_pools;
		PFView<vk::Semaphore, Context::FC> semaphore_pools;

		InflightContext(Context& ctx, unsigned frame) : ctx(ctx), frame(frame),
			commandbuffer_pools(ctx.cbuf_pools.get_view(*this)),
			semaphore_pools(ctx.semaphore_pools.get_view(*this))
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
		PFPTView<vk::CommandBuffer> commandbuffer_pool;
		PFPTView<vk::Semaphore> semaphore_pool;

		PerThreadContext(InflightContext& ifc, unsigned tid) : ctx(ifc.ctx), ifc(ifc), tid(tid),
			commandbuffer_pool(ifc.commandbuffer_pools.get_view(*this)),
			semaphore_pool(ifc.semaphore_pools.get_view(*this))
		{}

	};

	inline PerThreadContext InflightContext::begin() {
		return PerThreadContext{ *this, 0 };
	}

	template<class T, size_t FC>
	PFView<T, FC> Pool<T, FC>::get_view(InflightContext& ctx) {
		return PFView<T, FC>(ctx, *this, per_frame_storage[ctx.frame]);
	}
}
