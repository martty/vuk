#include "Pool.hpp"
#include "Context.hpp"

namespace vuk {
	// pools

	gsl::span<vk::Semaphore> PooledType<vk::Semaphore>::acquire(PerThreadContext& ptc, size_t count) {
		if (values.size() < (needle + count)) {
			auto remaining = values.size() - needle;
			for (auto i = 0; i < (count - remaining); i++) {
				auto nalloc = ptc.ctx.device.createSemaphore({});
				values.push_back(nalloc);
			}
		}
		gsl::span<vk::Semaphore> ret{ &*values.begin() + needle, (ptrdiff_t)count };
		needle += count;
		return ret;
	}

	template<>
	gsl::span<vk::Fence> PooledType<vk::Fence>::acquire(PerThreadContext& ptc, size_t count) {
		if (values.size() < (needle + count)) {
			auto remaining = values.size() - needle;
			for (auto i = 0; i < (count - remaining); i++) {
				auto nalloc = ptc.ctx.device.createFence({});
				values.push_back(nalloc);
			}
		}
		gsl::span<vk::Fence> ret{ &*values.begin() + needle, (ptrdiff_t)count };
		needle += count;
		return ret;
	}

	template<class T>
	void PooledType<T>::free(Context& ctx) {
		for (auto& v : values) {
			ctx.device.destroy(v);
		}
	}

	template struct PooledType<vk::Semaphore>;
	template struct PooledType<vk::Fence>;

	void PooledType<vk::Fence>::reset(Context& ctx) {
		if (needle > 0) {
			ctx.device.waitForFences((uint32_t)needle, values.data(), true, UINT64_MAX);
			ctx.device.resetFences((uint32_t)needle, values.data());
		}
		needle = 0;
	}

	// vk::CommandBuffer pool
	PooledType<vk::CommandBuffer>::PooledType(Context& ctx) {
		pool = ctx.device.createCommandPoolUnique({});
	}

	gsl::span<vk::CommandBuffer> PooledType<vk::CommandBuffer>::acquire(PerThreadContext& ptc, size_t count) {
		if (values.size() < (needle + count)) {
			auto remaining = values.size() - needle;
			vk::CommandBufferAllocateInfo cbai;
			cbai.commandBufferCount = (unsigned)(count - remaining);
			cbai.commandPool = *pool;
			cbai.level = vk::CommandBufferLevel::ePrimary;
			auto nalloc = ptc.ctx.device.allocateCommandBuffers(cbai);
			values.insert(values.end(), nalloc.begin(), nalloc.end());
		}
		gsl::span<vk::CommandBuffer> ret{ &*values.begin() + needle, (ptrdiff_t)count };
		needle += count;
		return ret;
	}
	void PooledType<vk::CommandBuffer>::reset(Context& ctx) {
		vk::CommandPoolResetFlags flags = {};
		ctx.device.resetCommandPool(*pool, flags);
		needle = 0;
	}

	void PooledType<vk::CommandBuffer>::free(Context& ctx) {
		ctx.device.freeCommandBuffers(*pool, values);
		pool.reset();
	}

}
