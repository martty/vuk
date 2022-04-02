#include "vuk/resources/DeviceFrameResource.hpp"
#include "../src/LegacyGPUAllocator.hpp"
#include "vuk/Context.hpp"
#include "vuk/Query.hpp"
#include "vuk/Descriptor.hpp"
#include "RenderPass.hpp"

#include <atomic>

namespace vuk {
	struct DeviceSuperFrameResourceImpl {
		std::mutex new_frame_mutex;
		std::atomic<uint64_t> frame_counter;
		std::atomic<uint64_t> local_frame;

		std::unique_ptr<char[]> frames_storage;
		DeviceFrameResource* frames;

		std::mutex command_pool_mutex;
		std::array<std::vector<VkCommandPool>, 3> command_pools;

		DeviceSuperFrameResourceImpl(DeviceSuperFrameResource& sfr, size_t frames_in_flight) {
			frames_storage = std::unique_ptr<char[]>(new char[sizeof(DeviceFrameResource) * frames_in_flight]);
			for (uint64_t i = 0; i < frames_in_flight; i++) {
				new (frames_storage.get() + i * sizeof(DeviceFrameResource)) DeviceFrameResource(sfr.direct.device, sfr);
			}
			frames = reinterpret_cast<DeviceFrameResource*>(frames_storage.get());
		}
	};

	struct DeviceFrameResourceImpl {
		std::mutex sema_mutex;
		std::vector<VkSemaphore> semaphores;

		std::mutex fence_mutex;
		std::vector<VkFence> fences;
		std::mutex cbuf_mutex;
		std::vector<CommandBufferAllocation> cmdbuffers_to_free;
		std::vector<CommandPool> cmdpools_to_free;
		std::mutex framebuffer_mutex;
		std::vector<VkFramebuffer> framebuffers;
		std::mutex images_mutex;
		std::vector<Image> images;
		std::mutex image_views_mutex;
		std::vector<ImageView> image_views;
		std::mutex pds_mutex;
		std::vector<PersistentDescriptorSet> persistent_descriptor_sets;
		std::mutex ds_mutex;
		std::vector<DescriptorSet> descriptor_sets;
		// only for use via SuperframeAllocator
		std::mutex buffers_mutex;
		std::vector<BufferGPU> buffer_gpus;
		std::vector<BufferCrossDevice> buffer_cross_devices;

		std::vector<TimestampQueryPool> ts_query_pools;
		std::mutex query_pool_mutex;
		std::mutex ts_query_mutex;
		uint64_t query_index = 0;
		uint64_t current_ts_pool = 0;
		std::mutex tsema_mutex;
		std::vector<TimelineSemaphore> tsemas;
		std::mutex swapchain_mutex;
		std::vector<VkSwapchainKHR> swapchains;

		LegacyLinearAllocator linear_cpu_only;
		LegacyLinearAllocator linear_cpu_gpu;
		LegacyLinearAllocator linear_gpu_cpu;
		LegacyLinearAllocator linear_gpu_only;

		DeviceFrameResourceImpl(VkDevice device, DeviceSuperFrameResource& upstream) :
		    linear_cpu_only(upstream.direct.legacy_gpu_allocator->allocate_linear(vuk::MemoryUsage::eCPUonly, LegacyGPUAllocator::all_usage)),
		    linear_cpu_gpu(upstream.direct.legacy_gpu_allocator->allocate_linear(vuk::MemoryUsage::eCPUtoGPU, LegacyGPUAllocator::all_usage)),
		    linear_gpu_cpu(upstream.direct.legacy_gpu_allocator->allocate_linear(vuk::MemoryUsage::eGPUtoCPU, LegacyGPUAllocator::all_usage)),
		    linear_gpu_only(upstream.direct.legacy_gpu_allocator->allocate_linear(vuk::MemoryUsage::eGPUonly, LegacyGPUAllocator::all_usage)) {}
	};

	DeviceFrameResource::DeviceFrameResource(VkDevice device, DeviceSuperFrameResource& upstream) :
	    DeviceNestedResource(&upstream),
	    device(device),
	    impl(new DeviceFrameResourceImpl(device, upstream)) {}

	DeviceFrameResource::~DeviceFrameResource() {
		delete impl;
	}

	Result<void, AllocateException> DeviceFrameResource::allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_semaphores(dst, loc));
		std::unique_lock _(impl->sema_mutex);
		auto& vec = impl->semaphores;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_semaphores(std::span<const VkSemaphore> src) {} // noop

	Result<void, AllocateException> DeviceFrameResource::allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_fences(dst, loc));
		std::unique_lock _(impl->fence_mutex);
		auto& vec = impl->fences;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_fences(std::span<const VkFence> src) {} // noop

	Result<void, AllocateException> DeviceFrameResource::allocate_command_buffers(std::span<CommandBufferAllocation> dst,
	                                                                              std::span<const CommandBufferAllocationCreateInfo> cis,
	                                                                              SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_command_buffers(dst, cis, loc));
		std::unique_lock _(impl->cbuf_mutex);
		auto& vec = impl->cmdbuffers_to_free;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_command_buffers(std::span<const CommandBufferAllocation> src) {} // no-op, deallocated with pools

	Result<void, AllocateException>
	DeviceFrameResource::allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_command_pools(dst, cis, loc));
		std::unique_lock _(impl->cbuf_mutex);
		auto& vec = impl->cmdpools_to_free;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_command_pools(std::span<const CommandPool> dst) {} // no-op

	Result<void, AllocateException>
	DeviceFrameResource::allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		auto& rf = *static_cast<DeviceSuperFrameResource*>(upstream);
		auto& legacy = *rf.direct.legacy_gpu_allocator;

		// TODO: legacy allocator can't signal errors
		// TODO: legacy linear allocators don't nest
		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			if (ci.mem_usage == MemoryUsage::eCPUonly) {
				dst[i] = BufferCrossDevice{ legacy.allocate_buffer(impl->linear_cpu_only, ci.size, ci.alignment, true) };
			} else if (ci.mem_usage == MemoryUsage::eCPUtoGPU) {
				dst[i] = BufferCrossDevice{ legacy.allocate_buffer(impl->linear_cpu_gpu, ci.size, ci.alignment, true) };
			} else if (ci.mem_usage == MemoryUsage::eGPUtoCPU) {
				dst[i] = BufferCrossDevice{ legacy.allocate_buffer(impl->linear_gpu_cpu, ci.size, ci.alignment, true) };
			} else {
				return { expected_error, AllocateException{ VK_ERROR_FEATURE_NOT_PRESENT } }; // tried to allocate gpu only buffer as BufferCrossDevice
			}
		}

		return { expected_value };
	}

	void DeviceFrameResource::deallocate_buffers(std::span<const BufferCrossDevice> src) {} // no-op, linear

	Result<void, AllocateException>
	DeviceFrameResource::allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		auto& rf = *static_cast<DeviceSuperFrameResource*>(upstream);
		auto& legacy = *rf.direct.legacy_gpu_allocator;

		// TODO: legacy allocator can't signal errors
		// TODO: legacy linear allocators don't nest
		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			if (ci.mem_usage == MemoryUsage::eGPUonly) {
				dst[i] = BufferGPU{ legacy.allocate_buffer(impl->linear_gpu_only, ci.size, ci.alignment, false) };
			} else {
				return { expected_error, AllocateException{ VK_ERROR_FEATURE_NOT_PRESENT } }; // tried to allocate xdev buffer as BufferGPU
			}
		}

		return { expected_value };
	}

	void DeviceFrameResource::deallocate_buffers(std::span<const BufferGPU> src) {} // no-op, linear

	Result<void, AllocateException>
	DeviceFrameResource::allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_framebuffers(dst, cis, loc));
		std::unique_lock _(impl->framebuffer_mutex);
		auto& vec = impl->framebuffers;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_framebuffers(std::span<const VkFramebuffer> src) {} // noop

	Result<void, AllocateException> DeviceFrameResource::allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_images(dst, cis, loc));
		std::unique_lock _(impl->images_mutex);
		auto& vec = impl->images;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_images(std::span<const Image> src) {} // noop

	Result<void, AllocateException>
	DeviceFrameResource::allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_image_views(dst, cis, loc));
		std::unique_lock _(impl->image_views_mutex);

		auto& vec = impl->image_views;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_image_views(std::span<const ImageView> src) {} // noop

	Result<void, AllocateException> DeviceFrameResource::allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
	                                                                                         std::span<const PersistentDescriptorSetCreateInfo> cis,
	                                                                                         SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_persistent_descriptor_sets(dst, cis, loc));
		std::unique_lock _(impl->pds_mutex);

		auto& vec = impl->persistent_descriptor_sets;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) {} // noop

	Result<void, AllocateException>
	DeviceFrameResource::allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_descriptor_sets(dst, cis, loc));

		std::unique_lock _(impl->ds_mutex);

		auto& vec = impl->descriptor_sets;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_descriptor_sets(std::span<const DescriptorSet> src) {} // noop

	Result<void, AllocateException> DeviceFrameResource::allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst,
	                                                                                    std::span<const VkQueryPoolCreateInfo> cis,
	                                                                                    SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_timestamp_query_pools(dst, cis, loc));
		std::unique_lock _(impl->query_pool_mutex);

		auto& vec = impl->ts_query_pools;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) {} // noop

	Result<void, AllocateException>
	DeviceFrameResource::allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) {
		std::unique_lock _(impl->ts_query_mutex);
		assert(dst.size() == cis.size());

		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];

			if (ci.pool) { // use given pool to allocate query
				ci.pool->queries[ci.pool->count++] = ci.query;
				dst[i].id = ci.pool->count;
				dst[i].pool = ci.pool->pool;
			} else { // allocate a pool on demand
				std::unique_lock _(impl->query_pool_mutex);
				if (impl->query_index % TimestampQueryPool::num_queries == 0) {
					VkQueryPoolCreateInfo qpci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
					qpci.queryCount = TimestampQueryPool::num_queries;
					qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
					TimestampQueryPool p;
					VUK_DO_OR_RETURN(upstream->allocate_timestamp_query_pools(std::span{ &p, 1 }, std::span{ &qpci, 1 }, loc));

					auto& vec = impl->ts_query_pools;
					vec.emplace_back(p);
					impl->current_ts_pool = vec.size() - 1;
				}

				auto& pool = impl->ts_query_pools[impl->current_ts_pool];
				pool.queries[pool.count++] = ci.query;
				dst[i].id = pool.count - 1;
				dst[i].pool = pool.pool;

				impl->query_index++;
			}
		}

		return { expected_value };
	}

	void DeviceFrameResource::deallocate_timestamp_queries(std::span<const TimestampQuery> src) {} // noop

	Result<void, AllocateException> DeviceFrameResource::allocate_timeline_semaphores(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_timeline_semaphores(dst, loc));
		std::unique_lock _(impl->tsema_mutex);

		auto& vec = impl->tsemas;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_timeline_semaphores(std::span<const TimelineSemaphore> src) {} // noop

	void DeviceFrameResource::deallocate_swapchains(std::span<const VkSwapchainKHR> src) {
		std::scoped_lock _(impl->swapchain_mutex);

		auto& vec = impl->swapchains;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	void DeviceFrameResource::wait() {
		if (impl->fences.size() > 0) {
			vkWaitForFences(device, (uint32_t)impl->fences.size(), impl->fences.data(), true, UINT64_MAX);
		}
		if (impl->tsemas.size() > 0) {
			VkSemaphoreWaitInfo swi{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };

			std::vector<VkSemaphore> semas(impl->tsemas.size());
			std::vector<uint64_t> values(impl->tsemas.size());

			for (uint64_t i = 0; i < impl->tsemas.size(); i++) {
				semas[i] = impl->tsemas[i].semaphore;
				values[i] = *impl->tsemas[i].value;
			}
			swi.pSemaphores = semas.data();
			swi.pValues = values.data();
			swi.semaphoreCount = (uint32_t)impl->tsemas.size();
			vkWaitSemaphores(device, &swi, UINT64_MAX);
		}
	}

	DeviceSuperFrameResource::DeviceSuperFrameResource(Context& ctx, uint64_t frames_in_flight) :
	    frames_in_flight(frames_in_flight),
	    direct(ctx, ctx.get_legacy_gpu_allocator()),
	    impl(new DeviceSuperFrameResourceImpl(*this, frames_in_flight)) {}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
		return direct.allocate_semaphores(dst, loc);
	}

	void DeviceSuperFrameResource::deallocate_semaphores(std::span<const VkSemaphore> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->sema_mutex);
		auto& vec = get_last_frame().impl->semaphores;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) {
		return direct.allocate_fences(dst, loc);
	}

	void DeviceSuperFrameResource::deallocate_fences(std::span<const VkFence> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->fence_mutex);
		auto& vec = f.impl->fences;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_command_buffers(std::span<CommandBufferAllocation> dst,
	                                                                                   std::span<const CommandBufferAllocationCreateInfo> cis,
	                                                                                   SourceLocationAtFrame loc) {
		return direct.allocate_command_buffers(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_command_buffers(std::span<const CommandBufferAllocation> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->cbuf_mutex);
		auto& vec = f.impl->cmdbuffers_to_free;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException>
	DeviceSuperFrameResource::allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		std::scoped_lock _(impl->command_pool_mutex);
		assert(cis.size() == dst.size());
		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			auto& source = impl->command_pools[ci.queueFamilyIndex];
			if (source.size() > 0) {
				dst[i] = { source.back(), ci.queueFamilyIndex };
				source.pop_back();
			} else {
				VUK_DO_OR_RETURN(direct.allocate_command_pools(std::span{ &dst[i], 1 }, std::span{ &ci, 1 }, loc));
			}
		}
		return { expected_value };
	}

	void DeviceSuperFrameResource::deallocate_command_pools(std::span<const CommandPool> src) {
		std::scoped_lock _(impl->command_pool_mutex);
		for (auto& p : src) {
			impl->command_pools[p.queue_family_index].push_back(p.command_pool);
		}
	}

	Result<void, AllocateException>
	DeviceSuperFrameResource::allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_buffers(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_buffers(std::span<const BufferCrossDevice> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->buffers_mutex);
		auto& vec = f.impl->buffer_cross_devices;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException>
	DeviceSuperFrameResource::allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_buffers(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_buffers(std::span<const BufferGPU> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->buffers_mutex);
		auto& vec = f.impl->buffer_gpus;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException>
	DeviceSuperFrameResource::allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_framebuffers(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_framebuffers(std::span<const VkFramebuffer> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->framebuffer_mutex);
		auto& vec = f.impl->framebuffers;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException>
	DeviceSuperFrameResource::allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_images(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_images(std::span<const Image> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->images_mutex);
		auto& vec = f.impl->images;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException>
	DeviceSuperFrameResource::allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_image_views(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_image_views(std::span<const ImageView> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->image_views_mutex);
		auto& vec = f.impl->image_views;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
	                                                                                              std::span<const PersistentDescriptorSetCreateInfo> cis,
	                                                                                              SourceLocationAtFrame loc) {
		return direct.allocate_persistent_descriptor_sets(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->pds_mutex);
		auto& vec = f.impl->persistent_descriptor_sets;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException>
	DeviceSuperFrameResource::allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		return direct.allocate_descriptor_sets(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_descriptor_sets(std::span<const DescriptorSet> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->ds_mutex);
		auto& vec = f.impl->descriptor_sets;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst,
	                                                                                         std::span<const VkQueryPoolCreateInfo> cis,
	                                                                                         SourceLocationAtFrame loc) {
		return direct.allocate_timestamp_query_pools(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->query_pool_mutex);
		auto& vec = f.impl->ts_query_pools;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_timestamp_queries(std::span<TimestampQuery> dst,
	                                                                                     std::span<const TimestampQueryCreateInfo> cis,
	                                                                                     SourceLocationAtFrame loc) {
		return direct.allocate_timestamp_queries(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_timestamp_queries(std::span<const TimestampQuery> src) {} // noop

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_timeline_semaphores(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc) {
		return direct.allocate_timeline_semaphores(dst, loc);
	}

	void DeviceSuperFrameResource::deallocate_timeline_semaphores(std::span<const TimelineSemaphore> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->tsema_mutex);
		auto& vec = f.impl->tsemas;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	void DeviceSuperFrameResource::deallocate_swapchains(std::span<const VkSwapchainKHR> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->swapchain_mutex);
		auto& vec = f.impl->swapchains;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	DeviceFrameResource& DeviceSuperFrameResource::get_last_frame() {
		return impl->frames[impl->frame_counter.load() % frames_in_flight];
	}

	DeviceFrameResource& DeviceSuperFrameResource::get_next_frame() {
		std::unique_lock _(impl->new_frame_mutex);

		impl->frame_counter++;
		impl->local_frame = impl->frame_counter % frames_in_flight;

		auto& f = impl->frames[impl->local_frame];
		f.wait();
		deallocate_frame(f);
		f.current_frame = impl->frame_counter.load();

		return f;
	}

	void DeviceSuperFrameResource::deallocate_frame(DeviceFrameResource& frame) {
		auto& f = *frame.impl;
		direct.deallocate_semaphores(f.semaphores);
		direct.deallocate_fences(f.fences);
		direct.deallocate_command_buffers(f.cmdbuffers_to_free);
		for (auto& pool : f.cmdpools_to_free) {
			vkResetCommandPool(direct.device, pool.command_pool, {});
		}
		deallocate_command_pools(f.cmdpools_to_free);
		direct.deallocate_buffers(f.buffer_gpus);
		direct.deallocate_buffers(f.buffer_cross_devices);
		direct.deallocate_framebuffers(f.framebuffers);
		direct.deallocate_images(f.images);
		direct.deallocate_image_views(f.image_views);
		direct.deallocate_persistent_descriptor_sets(f.persistent_descriptor_sets);
		direct.deallocate_descriptor_sets(f.descriptor_sets);
		direct.ctx->make_timestamp_results_available(f.ts_query_pools);
		direct.deallocate_timestamp_query_pools(f.ts_query_pools);
		direct.deallocate_timeline_semaphores(f.tsemas);
		direct.deallocate_swapchains(f.swapchains);

		f.semaphores.clear();
		f.fences.clear();
		f.buffer_cross_devices.clear();
		f.buffer_gpus.clear();
		f.cmdbuffers_to_free.clear();
		f.cmdpools_to_free.clear();
		auto& legacy = direct.legacy_gpu_allocator;
		legacy->reset_pool(f.linear_cpu_only);
		legacy->reset_pool(f.linear_cpu_gpu);
		legacy->reset_pool(f.linear_gpu_cpu);
		legacy->reset_pool(f.linear_gpu_only);
		f.framebuffers.clear();
		f.images.clear();
		f.image_views.clear();
		f.persistent_descriptor_sets.clear();
		f.descriptor_sets.clear();
		f.ts_query_pools.clear();
		f.query_index = 0;
		f.tsemas.clear();
		f.swapchains.clear();
	}

	DeviceSuperFrameResource::~DeviceSuperFrameResource() {
		for (auto i = 0; i < frames_in_flight; i++) {
			auto lframe = (impl->frame_counter + i) % frames_in_flight;
			auto& f = impl->frames[lframe];
			f.wait();
			deallocate_frame(f);
			direct.legacy_gpu_allocator->destroy(f.impl->linear_cpu_only);
			direct.legacy_gpu_allocator->destroy(f.impl->linear_cpu_gpu);
			direct.legacy_gpu_allocator->destroy(f.impl->linear_gpu_cpu);
			direct.legacy_gpu_allocator->destroy(f.impl->linear_gpu_only);
			f.DeviceFrameResource::~DeviceFrameResource();
		}
		for (uint32_t i = 0; i < (uint32_t)impl->command_pools.size(); i++) {
			for (auto& cpool : impl->command_pools[i]) {
				CommandPool p{ cpool, i };
				direct.deallocate_command_pools(std::span{ &p, 1 });
			}
		}
		delete impl;
	}
} // namespace vuk