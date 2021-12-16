#include "vuk/Context.hpp"
#include "vuk/resources/DeviceFrameResource.hpp"

namespace vuk {
	DeviceFrameResource::DeviceFrameResource(VkDevice device, DeviceSuperFrameResource& upstream) : device(device), DeviceNestedResource(&upstream),
		linear_cpu_only(upstream.direct.legacy_gpu_allocator->allocate_linear(vuk::MemoryUsage::eCPUonly, LegacyGPUAllocator::all_usage)),
		linear_cpu_gpu(upstream.direct.legacy_gpu_allocator->allocate_linear(vuk::MemoryUsage::eCPUtoGPU, LegacyGPUAllocator::all_usage)),
		linear_gpu_cpu(upstream.direct.legacy_gpu_allocator->allocate_linear(vuk::MemoryUsage::eGPUtoCPU, LegacyGPUAllocator::all_usage)),
		linear_gpu_only(upstream.direct.legacy_gpu_allocator->allocate_linear(vuk::MemoryUsage::eGPUonly, LegacyGPUAllocator::all_usage)) {}

	Result<void, AllocateException> DeviceFrameResource::allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_semaphores(dst, loc));
		std::unique_lock _(sema_mutex);
		auto& vec = semaphores;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_semaphores(std::span<const VkSemaphore> src) {} // noop

	Result<void, AllocateException> DeviceFrameResource::allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_fences(dst, loc));
		std::unique_lock _(fence_mutex);
		auto& vec = fences;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_fences(std::span<const VkFence> src) {} // noop

	Result<void, AllocateException> DeviceFrameResource::allocate_command_buffers(std::span<CommandBufferAllocation> dst, std::span<const CommandBufferAllocationCreateInfo> cis, SourceLocationAtFrame loc) {
		return upstream->allocate_command_buffers(dst, cis, loc);
	}

	void DeviceFrameResource::deallocate_command_buffers(std::span<const CommandBufferAllocation> src) {} // no-op, deallocated with pools

	Result<void, AllocateException> DeviceFrameResource::allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_commandpools(dst, cis, loc));
		std::unique_lock _(cbuf_mutex);
		auto& vec = cmdpools_to_free;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_commandpools(std::span<const VkCommandPool> dst) {} // no-op

	Result<void, AllocateException> DeviceFrameResource::allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		auto& rf = *static_cast<DeviceSuperFrameResource*>(upstream);
		auto& legacy = *rf.direct.legacy_gpu_allocator;

		// TODO: legacy allocator can't signal errors
		// TODO: legacy linear allocators don't nest
		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			if (ci.mem_usage == MemoryUsage::eCPUonly) {
				dst[i] = BufferCrossDevice{ legacy.allocate_buffer(linear_cpu_only, ci.size, ci.alignment, true) };
			} else if (ci.mem_usage == MemoryUsage::eCPUtoGPU) {
				dst[i] = BufferCrossDevice{ legacy.allocate_buffer(linear_cpu_gpu, ci.size, ci.alignment, true) };
			} else if (ci.mem_usage == MemoryUsage::eGPUtoCPU) {
				dst[i] = BufferCrossDevice{ legacy.allocate_buffer(linear_gpu_cpu, ci.size, ci.alignment, true) };
			} else {
				return { expected_error, AllocateException{VK_ERROR_FEATURE_NOT_PRESENT} }; // tried to allocate gpu only buffer as BufferCrossDevice
			}
		}

		return { expected_value };
	}

	void DeviceFrameResource::deallocate_buffers(std::span<const BufferCrossDevice> src) {} // no-op, linear

	Result<void, AllocateException> DeviceFrameResource::allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		auto& rf = *static_cast<DeviceSuperFrameResource*>(upstream);
		auto& legacy = *rf.direct.legacy_gpu_allocator;

		// TODO: legacy allocator can't signal errors
		// TODO: legacy linear allocators don't nest
		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			if (ci.mem_usage == MemoryUsage::eGPUonly) {
				dst[i] = BufferGPU{ legacy.allocate_buffer(linear_gpu_only, ci.size, ci.alignment, false) };
			} else {
				return { expected_error, AllocateException{VK_ERROR_FEATURE_NOT_PRESENT} }; // tried to allocate xdev buffer as BufferGPU
			}
		}

		return { expected_value };
	}

	void DeviceFrameResource::deallocate_buffers(std::span<const BufferGPU> src) {} // no-op, linear

	Result<void, AllocateException> DeviceFrameResource::allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_framebuffers(dst, cis, loc));
		std::unique_lock _(framebuffer_mutex);
		auto& vec = framebuffers;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_framebuffers(std::span<const VkFramebuffer> src) {} // noop

	Result<void, AllocateException> DeviceFrameResource::allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_images(dst, cis, loc));
		std::unique_lock _(images_mutex);
		auto& vec = images;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_images(std::span<const Image> src) {} // noop

	Result<void, AllocateException> DeviceFrameResource::allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_image_views(dst, cis, loc));
		std::unique_lock _(image_views_mutex);

		auto& vec = image_views;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_image_views(std::span<const ImageView> src) {} // noop

	Result<void, AllocateException> DeviceFrameResource::allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_persistent_descriptor_sets(dst, cis, loc));
		std::unique_lock _(pds_mutex);

		auto& vec = persistent_descriptor_sets;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) {} // noop

	Result<void, AllocateException> DeviceFrameResource::allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_descriptor_sets(dst, cis, loc));

		std::unique_lock _(ds_mutex);

		auto& vec = descriptor_sets;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_descriptor_sets(std::span<const DescriptorSet> src) {} // noop


	void deallocate(DeviceFrameResource& res, DescriptorSet& ds) {
		std::unique_lock _{ res.ds_mutex };

		auto& vec = res.descriptor_sets;
		vec.emplace_back(ds);
	}

	template<class T>
	T& DeviceFrameResource::Cache<T>::acquire(uint64_t current_frame, const create_info_t<T>& ci) {
		if (auto it = lru_map.find(ci); it != lru_map.end()) {
			it->second.last_use_frame = current_frame;
			return it->second.value;
		} else {
			// if the value is not in the cache, we look in our per thread buffers
			// if it doesn't exist there either, we add it
			auto& ptv = per_thread_append_v[0 /*ptc.tid*/];
			auto& ptk = per_thread_append_k[0 /*ptc.tid*/]; // TODO: restore TIDs
			auto pit = std::find(ptk.begin(), ptk.end(), ci);
			if (pit == ptk.end()) {
				ptv.emplace_back(allocate(ci));
				pit = ptk.insert(ptk.end(), ci);
			}
			auto index = std::distance(ptk.begin(), pit);
			return ptv[index];
		}
	}

	template<class T>
	void DeviceFrameResource::Cache<T>::collect(uint64_t current_frame, size_t threshold) {
		std::unique_lock _(cache_mtx);
		for (auto it = lru_map.begin(); it != lru_map.end();) {
			if (current_frame - it->second.last_use_frame > threshold) {
				deallocate(it->second.value);
				it = lru_map.erase(it);
			} else {
				++it;
			}
		}

		for (size_t tid = 0; tid < per_thread_append_v.size(); tid++) {
			auto& vs = per_thread_append_v[tid];
			auto& ks = per_thread_append_k[tid];
			for (size_t i = 0; i < vs.size(); i++) {
				if (lru_map.find(ks[i]) == lru_map.end()) {
					lru_map.emplace(ks[i], LRUEntry{ std::move(vs[i]), current_frame });
				} else {
					deallocate(vs[i]);
				}
			}
			vs.clear();
			ks.clear();
		}
	}

	Result<void, AllocateException> DeviceFrameResource::allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_timestamp_query_pools(dst, cis, loc));
		std::unique_lock _(query_pool_mutex);

		auto& vec = ts_query_pools;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) {} // noop

	Result<void, AllocateException> DeviceFrameResource::allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) {
		std::unique_lock _(ts_query_mutex);
		assert(dst.size() == cis.size());

		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];

			if (ci.pool) { // use given pool to allocate query
				ci.pool->queries[ci.pool->count++] = ci.query;
				dst[i].id = ci.pool->count;
				dst[i].pool = ci.pool->pool;
			} else { // allocate a pool on demand
				std::unique_lock _(query_pool_mutex);
				if (query_index % TimestampQueryPool::num_queries == 0) {
					VkQueryPoolCreateInfo qpci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
					qpci.queryCount = TimestampQueryPool::num_queries;
					qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
					TimestampQueryPool p;
					VUK_DO_OR_RETURN(upstream->allocate_timestamp_query_pools(std::span{ &p, 1 }, std::span{ &qpci, 1 }, loc));

					auto& vec = ts_query_pools;
					vec.emplace_back(p);
					current_ts_pool = vec.size() - 1;
				}

				auto& pool = ts_query_pools[current_ts_pool];
				pool.queries[pool.count++] = ci.query;
				dst[i].id = pool.count - 1;
				dst[i].pool = pool.pool;

				query_index++;
			}
		}

		return { expected_value };
	}

	void DeviceFrameResource::deallocate_timestamp_queries(std::span<const TimestampQuery> src) {} // noop

	Result<void, AllocateException> DeviceFrameResource::allocate_timeline_semaphores(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_timeline_semaphores(dst, loc));
		std::unique_lock _(tsema_mutex);

		auto& vec = tsemas;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_timeline_semaphores(std::span<const TimelineSemaphore> src) {} // noop

	void DeviceFrameResource::wait() {
		if (fences.size() > 0) {
			vkWaitForFences(device, (uint32_t)fences.size(), fences.data(), true, UINT64_MAX);
		}
		if (tsemas.size() > 0) {
			VkSemaphoreWaitInfo swi{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };

			std::vector<VkSemaphore> semas(tsemas.size());
			std::vector<uint64_t> values(tsemas.size());

			for (uint64_t i = 0; i < tsemas.size(); i++) {
				semas[i] = tsemas[i].semaphore;
				values[i] = *tsemas[i].value;
			}
			swi.pSemaphores = semas.data();
			swi.pValues = values.data();
			swi.semaphoreCount = (uint32_t)tsemas.size();
			vkWaitSemaphores(device, &swi, UINT64_MAX);
		}
	}

	DeviceSuperFrameResource::DeviceSuperFrameResource(Context& ctx, uint64_t frames_in_flight) : direct(ctx, ctx.get_legacy_gpu_allocator()), frames_in_flight(frames_in_flight) {
		frames_storage = std::unique_ptr<char[]>(new char[sizeof(DeviceFrameResource) * frames_in_flight]);
		for (uint64_t i = 0; i < frames_in_flight; i++) {
			new(frames_storage.get() + i * sizeof(DeviceFrameResource)) DeviceFrameResource(direct.device, *this);
		}
		frames = reinterpret_cast<DeviceFrameResource*>(frames_storage.get());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
		return direct.allocate_semaphores(dst, loc);
	}

	void DeviceSuperFrameResource::deallocate_semaphores(std::span<const VkSemaphore> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.sema_mutex);
		auto& vec = get_last_frame().semaphores;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) {
		return direct.allocate_fences(dst, loc);
	}

	void DeviceSuperFrameResource::deallocate_fences(std::span<const VkFence> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.fence_mutex);
		auto& vec = f.fences;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_command_buffers(std::span<CommandBufferAllocation> dst, std::span<const CommandBufferAllocationCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_command_buffers(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_command_buffers(std::span<const CommandBufferAllocation> src) {} // noop, deallocate pools

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_commandpools(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_commandpools(std::span<const VkCommandPool> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.cbuf_mutex);
		auto& vec = f.cmdpools_to_free;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_buffers(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_buffers(std::span<const BufferCrossDevice> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.buffers_mutex);
		auto& vec = f.buffer_cross_devices;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_buffers(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_buffers(std::span<const BufferGPU> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.buffers_mutex);
		auto& vec = f.buffer_gpus;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_framebuffers(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_framebuffers(std::span<const VkFramebuffer> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.framebuffer_mutex);
		auto& vec = f.framebuffers;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_images(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_images(std::span<const Image> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.images_mutex);
		auto& vec = f.images;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_image_views(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_image_views(std::span<const ImageView> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.image_views_mutex);
		auto& vec = f.image_views;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_persistent_descriptor_sets(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.pds_mutex);
		auto& vec = f.persistent_descriptor_sets;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		return direct.allocate_descriptor_sets(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_descriptor_sets(std::span<const DescriptorSet> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.ds_mutex);
		auto& vec = f.descriptor_sets;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_timestamp_query_pools(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.query_pool_mutex);
		auto& vec = f.ts_query_pools;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) {
		return direct.allocate_timestamp_queries(dst, cis, loc);
	}

	void DeviceSuperFrameResource::deallocate_timestamp_queries(std::span<const TimestampQuery> src) {} // noop

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_timeline_semaphores(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc) {
		return direct.allocate_timeline_semaphores(dst, loc);
	}

	void DeviceSuperFrameResource::deallocate_timeline_semaphores(std::span<const TimelineSemaphore> src) {
		auto& f = get_last_frame();
		std::unique_lock _(f.tsema_mutex);
		auto& vec = f.tsemas;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	DeviceFrameResource& DeviceSuperFrameResource::get_last_frame() {
		return frames[frame_counter.load() % frames_in_flight];
	}

	DeviceFrameResource& DeviceSuperFrameResource::get_next_frame() {
		std::unique_lock _(new_frame_mutex);
		auto& ctx = direct.get_context();
		frame_counter++;
		local_frame = frame_counter % frames_in_flight;

		auto& f = frames[local_frame];
		f.wait();
		deallocate_frame(f);
		f.current_frame = frame_counter.load();

		return f;
	}

	void DeviceSuperFrameResource::deallocate_frame(DeviceFrameResource& f) {
		//f.descriptor_set_cache.collect(frame_counter.load(), 16);
		direct.deallocate_semaphores(f.semaphores);
		direct.deallocate_fences(f.fences);
		direct.deallocate_commandpools(f.cmdpools_to_free);
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

		f.semaphores.clear();
		f.fences.clear();
		f.buffer_cross_devices.clear();
		f.buffer_gpus.clear();
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
	}

	DeviceSuperFrameResource::~DeviceSuperFrameResource() {
		for (auto i = 0; i < frames_in_flight; i++) {
			auto lframe = (frame_counter + i) % frames_in_flight;
			auto& f = frames[lframe];
			f.wait();
			deallocate_frame(f);
			direct.legacy_gpu_allocator->destroy(f.linear_cpu_only);
			direct.legacy_gpu_allocator->destroy(f.linear_cpu_gpu);
			direct.legacy_gpu_allocator->destroy(f.linear_gpu_cpu);
			direct.legacy_gpu_allocator->destroy(f.linear_gpu_only);
			f.~DeviceFrameResource();
		}
	}
}