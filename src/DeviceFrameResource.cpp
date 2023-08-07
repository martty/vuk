#include "vuk/resources/DeviceFrameResource.hpp"
#include "BufferAllocator.hpp"
#include "Cache.hpp"
#include "RenderPass.hpp"
#include "vuk/Context.hpp"
#include "vuk/Descriptor.hpp"
#include "vuk/PipelineInstance.hpp"
#include "vuk/Query.hpp"

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <numeric>
#include <plf_colony.h>

namespace vuk {
	struct DeviceSuperFrameResourceImpl {
		DeviceSuperFrameResource* sfr;

		std::shared_mutex new_frame_mutex;
		std::atomic<uint64_t> frame_counter;
		std::atomic<uint64_t> local_frame;

		std::unique_ptr<char[]> frames_storage;
		DeviceFrameResource* frames;
		plf::colony<DeviceMultiFrameResource> multi_frames;

		std::mutex command_pool_mutex;
		std::array<std::vector<VkCommandPool>, 3> command_pools;
		std::mutex ds_pool_mutex;
		std::vector<VkDescriptorPool> ds_pools;

		std::mutex images_mutex;
		std::unordered_map<ImageCreateInfo, uint32_t> image_identity;
		Cache<ImageWithIdentity> image_cache;
		Cache<ImageView> image_view_cache;

		Cache<GraphicsPipelineInfo> graphics_pipeline_cache;
		Cache<ComputePipelineInfo> compute_pipeline_cache;
		Cache<RayTracingPipelineInfo> ray_tracing_pipeline_cache;

		BufferSubAllocator suballocators[4];

		DeviceSuperFrameResourceImpl(DeviceSuperFrameResource& sfr, size_t frames_in_flight) :
		    sfr(&sfr),
		    image_cache(
		        this,
		        +[](void* allocator, const CachedImageIdentifier& cii) {
			        ImageWithIdentity i;
			        reinterpret_cast<DeviceSuperFrameResourceImpl*>(allocator)->sfr->allocate_images({ &i.image, 1 }, { &cii.ici, 1 }, {}); // TODO: dropping error
			        return i;
		        },
		        +[](void* allocator, const ImageWithIdentity& i) {
			        reinterpret_cast<DeviceSuperFrameResourceImpl*>(allocator)->sfr->deallocate_images({ &i.image, 1 });
		        }),
		    image_view_cache(
		        this,
		        +[](void* allocator, const CompressedImageViewCreateInfo& civci) {
			        ImageView iv;
			        ImageViewCreateInfo ivci = static_cast<ImageViewCreateInfo>(civci);
			        reinterpret_cast<DeviceSuperFrameResourceImpl*>(allocator)->sfr->allocate_image_views({ &iv, 1 }, { &ivci, 1 }, {}); // TODO: dropping error
			        return iv;
		        },
		        +[](void* allocator, const ImageView& iv) {
			        reinterpret_cast<DeviceSuperFrameResourceImpl*>(allocator)->sfr->deallocate_image_views({ &iv, 1 });
		        }),
		    graphics_pipeline_cache(
		        this,
		        +[](void* allocator, const GraphicsPipelineInstanceCreateInfo& ci) {
			        GraphicsPipelineInfo dst;
			        reinterpret_cast<DeviceSuperFrameResourceImpl*>(allocator)->sfr->allocate_graphics_pipelines({ &dst, 1 }, { &ci, 1 }, {});
			        return dst;
		        },
		        +[](void* allocator, const GraphicsPipelineInfo& v) {
			        reinterpret_cast<DeviceSuperFrameResourceImpl*>(allocator)->sfr->deallocate_graphics_pipelines({ &v, 1 });
		        }),
		    compute_pipeline_cache(
		        this,
		        +[](void* allocator, const ComputePipelineInstanceCreateInfo& ci) {
			        ComputePipelineInfo dst;
			        reinterpret_cast<DeviceSuperFrameResourceImpl*>(allocator)->sfr->allocate_compute_pipelines({ &dst, 1 }, { &ci, 1 }, {});
			        return dst;
		        },
		        +[](void* allocator, const ComputePipelineInfo& v) {
			        reinterpret_cast<DeviceSuperFrameResourceImpl*>(allocator)->sfr->deallocate_compute_pipelines({ &v, 1 });
		        }),
		    ray_tracing_pipeline_cache(
		        this,
		        +[](void* allocator, const RayTracingPipelineInstanceCreateInfo& ci) {
			        RayTracingPipelineInfo dst;
			        reinterpret_cast<DeviceSuperFrameResourceImpl*>(allocator)->sfr->allocate_ray_tracing_pipelines({ &dst, 1 }, { &ci, 1 }, {});
			        return dst;
		        },
		        +[](void* allocator, const RayTracingPipelineInfo& v) {
			        reinterpret_cast<DeviceSuperFrameResourceImpl*>(allocator)->sfr->deallocate_ray_tracing_pipelines({ &v, 1 });
		        }),
		    suballocators{ { *sfr.upstream, vuk::MemoryUsage::eGPUonly, all_buffer_usage_flags, 64 * 1024 * 1024 },
			                 { *sfr.upstream, vuk::MemoryUsage::eCPUonly, all_buffer_usage_flags, 64 * 1024 * 1024 },
			                 { *sfr.upstream, vuk::MemoryUsage::eCPUtoGPU, all_buffer_usage_flags, 64 * 1024 * 1024 },
			                 { *sfr.upstream, vuk::MemoryUsage::eGPUtoCPU, all_buffer_usage_flags, 64 * 1024 * 1024 } } {
			frames_storage = std::unique_ptr<char[]>(new char[sizeof(DeviceFrameResource) * frames_in_flight]);
			for (uint64_t i = 0; i < frames_in_flight; i++) {
				new (frames_storage.get() + i * sizeof(DeviceFrameResource)) DeviceFrameResource(sfr.get_context().device, sfr);
			}
			frames = reinterpret_cast<DeviceFrameResource*>(frames_storage.get());
		}
	};

	struct DeviceFrameResourceImpl {
		Context* ctx;
		std::mutex sema_mutex;
		std::vector<VkSemaphore> semaphores;
		std::mutex buf_mutex;
		std::vector<Buffer> buffers;
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
		std::atomic<VkDescriptorPool*> last_ds_pool;
		plf::colony<VkDescriptorPool> ds_pools;
		std::vector<VkDescriptorPool> ds_pools_to_destroy;

		// only for use via SuperframeAllocator
		std::mutex buffers_mutex;
		std::vector<Buffer> buffer_gpus;

		std::vector<TimestampQueryPool> ts_query_pools;
		std::mutex query_pool_mutex;
		std::mutex ts_query_mutex;
		uint64_t query_index = 0;
		uint64_t current_ts_pool = 0;
		std::mutex tsema_mutex;
		std::vector<TimelineSemaphore> tsemas;
		std::mutex as_mutex;
		std::vector<VkAccelerationStructureKHR> ass;
		std::mutex swapchain_mutex;
		std::vector<VkSwapchainKHR> swapchains;

		BufferLinearAllocator linear_cpu_only;
		BufferLinearAllocator linear_cpu_gpu;
		BufferLinearAllocator linear_gpu_cpu;
		BufferLinearAllocator linear_gpu_only;

		DeviceFrameResourceImpl(VkDevice device, DeviceSuperFrameResource& upstream) :
		    ctx(&upstream.get_context()),
		    linear_cpu_only(upstream, vuk::MemoryUsage::eCPUonly, all_buffer_usage_flags),
		    linear_cpu_gpu(upstream, vuk::MemoryUsage::eCPUtoGPU, all_buffer_usage_flags),
		    linear_gpu_cpu(upstream, vuk::MemoryUsage::eGPUtoCPU, all_buffer_usage_flags),
		    linear_gpu_only(upstream, vuk::MemoryUsage::eGPUonly, all_buffer_usage_flags) {}
	};

	DeviceFrameResource::DeviceFrameResource(VkDevice device, DeviceSuperFrameResource& pstream) :
	    DeviceNestedResource(static_cast<DeviceResource&>(pstream)),
	    device(device),
	    impl(new DeviceFrameResourceImpl(device, pstream)) {}

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
	DeviceFrameResource::allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());

		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			Result<Buffer, AllocateException> result{ expected_value };
			auto alignment = std::lcm(ci.alignment, get_context().min_buffer_alignment);
			if (ci.mem_usage == MemoryUsage::eGPUonly) {
				result = impl->linear_gpu_only.allocate_buffer(ci.size, alignment, loc);
			} else if (ci.mem_usage == MemoryUsage::eCPUonly) {
				result = impl->linear_cpu_only.allocate_buffer(ci.size, alignment, loc);
			} else if (ci.mem_usage == MemoryUsage::eCPUtoGPU) {
				result = impl->linear_cpu_gpu.allocate_buffer(ci.size, alignment, loc);
			} else if (ci.mem_usage == MemoryUsage::eGPUtoCPU) {
				result = impl->linear_gpu_cpu.allocate_buffer(ci.size, alignment, loc);
			}
			if (!result) {
				deallocate_buffers({ dst.data(), (uint64_t)i });
				return result;
			}
			dst[i] = result.value();
		}
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_buffers(std::span<const Buffer> src) {} // no-op, linear

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
		VUK_DO_OR_RETURN(static_cast<DeviceSuperFrameResource*>(upstream)->allocate_cached_images(dst, cis, loc));
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_images(std::span<const Image> src) {} // noop

	Result<void, AllocateException>
	DeviceFrameResource::allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(static_cast<DeviceSuperFrameResource*>(upstream)->allocate_cached_image_views(dst, cis, loc));
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
	DeviceFrameResource::allocate_descriptor_sets_with_value(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_descriptor_sets_with_value(dst, cis, loc));

		std::unique_lock _(impl->ds_mutex);

		auto& vec = impl->descriptor_sets;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceFrameResource::deallocate_descriptor_sets(std::span<const DescriptorSet> src) {} // noop

	Result<void, AllocateException>
	DeviceFrameResource::allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const DescriptorSetLayoutAllocInfo> cis, SourceLocationAtFrame loc) {
		VkDescriptorPoolCreateInfo dpci{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		dpci.maxSets = 1000;
		std::array<VkDescriptorPoolSize, 12> descriptor_counts = {};
		size_t count = get_context().vkCmdBuildAccelerationStructuresKHR ? descriptor_counts.size() : descriptor_counts.size() - 1;
		for (size_t i = 0; i < count; i++) {
			auto& d = descriptor_counts[i];
			d.type = i == 11 ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR : VkDescriptorType(i);
			d.descriptorCount = 1000;
		}
		dpci.pPoolSizes = descriptor_counts.data();
		dpci.poolSizeCount = (uint32_t)count;

		if (impl->ds_pools.size() == 0) {
			std::unique_lock _(impl->ds_mutex);
			if (impl->ds_pools.size() == 0) { // this assures only 1 thread gets to do this
				VkDescriptorPool pool;
				VUK_DO_OR_RETURN(upstream->allocate_descriptor_pools({ &pool, 1 }, { &dpci, 1 }, loc));
				impl->last_ds_pool = &*impl->ds_pools.emplace(pool);
			}
		}

		// look at last stored pool
		VkDescriptorPool* last_pool = impl->last_ds_pool.load();

		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			// attempt to allocate a set
			VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
			dsai.descriptorPool = *last_pool;
			dsai.descriptorSetCount = 1;
			dsai.pSetLayouts = &ci.layout;
			dst[i].layout_info = ci;
			auto result = impl->ctx->vkAllocateDescriptorSets(device, &dsai, &dst[i].descriptor_set);
			// if we fail, we allocate another pool from upstream
			if (result == VK_ERROR_OUT_OF_POOL_MEMORY ||
			    result == VK_ERROR_FRAGMENTED_POOL) { // we potentially run this from multiple threads which results in additional pool allocs
				{
					std::unique_lock _(impl->ds_mutex);
					VkDescriptorPool pool;
					VUK_DO_OR_RETURN(upstream->allocate_descriptor_pools({ &pool, 1 }, { &dpci, 1 }, loc));
					last_pool = &*impl->ds_pools.emplace(pool);
				}
				dsai.descriptorPool = *last_pool;
				result = impl->ctx->vkAllocateDescriptorSets(device, &dsai, &dst[i].descriptor_set);
				if (result != VK_SUCCESS) {
					return { expected_error, AllocateException{ result } };
				}
			}
		}
		return { expected_value };
	}

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

	Result<void, AllocateException> DeviceFrameResource::allocate_graphics_pipelines(std::span<GraphicsPipelineInfo> dst,
	                                                                                 std::span<const GraphicsPipelineInstanceCreateInfo> cis,
	                                                                                 SourceLocationAtFrame loc) {
		auto& sfr = *static_cast<DeviceSuperFrameResource*>(upstream);
		assert(dst.size() == cis.size());

		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			dst[i] = sfr.impl->graphics_pipeline_cache.acquire(ci, construction_frame);
		}

		return { expected_value };
	}
	void DeviceFrameResource::deallocate_graphics_pipelines(std::span<const GraphicsPipelineInfo> src) {}

	Result<void, AllocateException> DeviceFrameResource::allocate_compute_pipelines(std::span<ComputePipelineInfo> dst,
	                                                                                std::span<const ComputePipelineInstanceCreateInfo> cis,
	                                                                                SourceLocationAtFrame loc) {
		auto& sfr = *static_cast<DeviceSuperFrameResource*>(upstream);
		assert(dst.size() == cis.size());

		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			dst[i] = sfr.impl->compute_pipeline_cache.acquire(ci, construction_frame);
		}

		return { expected_value };
	}
	void DeviceFrameResource::deallocate_compute_pipelines(std::span<const ComputePipelineInfo> src) {}

	Result<void, AllocateException> DeviceFrameResource::allocate_ray_tracing_pipelines(std::span<RayTracingPipelineInfo> dst,
	                                                                                    std::span<const RayTracingPipelineInstanceCreateInfo> cis,
	                                                                                    SourceLocationAtFrame loc) {
		auto& sfr = *static_cast<DeviceSuperFrameResource*>(upstream);
		assert(dst.size() == cis.size());

		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			dst[i] = sfr.impl->ray_tracing_pipeline_cache.acquire(ci, construction_frame);
		}

		return { expected_value };
	}
	void DeviceFrameResource::deallocate_ray_tracing_pipelines(std::span<const RayTracingPipelineInfo> src) {}

	void DeviceFrameResource::wait() {
		if (impl->fences.size() > 0) {
			if (impl->fences.size() > 64) {
				int i = 0;
				for (; i < impl->fences.size() - 64; i += 64) {
					impl->ctx->vkWaitForFences(device, 64, impl->fences.data() + i, true, UINT64_MAX);
				}
				impl->ctx->vkWaitForFences(device, (uint32_t)impl->fences.size() - i, impl->fences.data() + i, true, UINT64_MAX);
			} else {
				impl->ctx->vkWaitForFences(device, (uint32_t)impl->fences.size(), impl->fences.data(), true, UINT64_MAX);
			}
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
			impl->ctx->vkWaitSemaphores(device, &swi, UINT64_MAX);
		}
	}

	DeviceMultiFrameResource::DeviceMultiFrameResource(VkDevice device, DeviceSuperFrameResource& upstream, uint32_t frame_lifetime) :
	    DeviceFrameResource(device, upstream),
	    frame_lifetime(frame_lifetime),
	    remaining_lifetime(frame_lifetime),
	    multiframe_id((uint32_t)(construction_frame % frame_lifetime)) {}

	Result<void, AllocateException>
	DeviceMultiFrameResource::allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(static_cast<DeviceSuperFrameResource*>(upstream)->allocate_cached_images(dst, cis, loc));
		return { expected_value };
	}

	DeviceSuperFrameResource::DeviceSuperFrameResource(Context& ctx, uint64_t frames_in_flight) :
	    DeviceNestedResource(ctx.get_vk_resource()),
	    frames_in_flight(frames_in_flight),
	    direct(static_cast<DeviceVkResource*>(upstream)),
	    impl(new DeviceSuperFrameResourceImpl(*this, frames_in_flight)) {}

	DeviceSuperFrameResource::DeviceSuperFrameResource(DeviceResource& upstream, uint64_t frames_in_flight) :
	    DeviceNestedResource(upstream),
	    frames_in_flight(frames_in_flight),
	    direct(dynamic_cast<DeviceVkResource*>(this->upstream)),
	    impl(new DeviceSuperFrameResourceImpl(*this, frames_in_flight)) {}

	void DeviceSuperFrameResource::deallocate_semaphores(std::span<const VkSemaphore> src) {
		std::shared_lock _s(impl->new_frame_mutex);
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->sema_mutex);
		auto& vec = get_last_frame().impl->semaphores;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	void DeviceSuperFrameResource::deallocate_fences(std::span<const VkFence> src) {
		std::shared_lock _s(impl->new_frame_mutex);
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->fence_mutex);
		auto& vec = f.impl->fences;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	void DeviceSuperFrameResource::deallocate_command_buffers(std::span<const CommandBufferAllocation> src) {
		std::shared_lock _s(impl->new_frame_mutex);
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
				VUK_DO_OR_RETURN(upstream->allocate_command_pools(std::span{ &dst[i], 1 }, std::span{ &ci, 1 }, loc));
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

	void DeviceSuperFrameResource::deallocate_buffers(std::span<const Buffer> src) {
		std::shared_lock _s(impl->new_frame_mutex);
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->buffers_mutex);
		auto& vec = f.impl->buffer_gpus;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	void DeviceSuperFrameResource::deallocate_framebuffers(std::span<const VkFramebuffer> src) {
		std::shared_lock _s(impl->new_frame_mutex);
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->framebuffer_mutex);
		auto& vec = f.impl->framebuffers;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	void DeviceSuperFrameResource::deallocate_images(std::span<const Image> src) {
		std::shared_lock _s(impl->new_frame_mutex);
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->images_mutex);
		auto& vec = f.impl->images;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException>
	DeviceSuperFrameResource::allocate_cached_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		std::unique_lock _(impl->images_mutex);
		assert(dst.size() == cis.size());
		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			auto index = impl->image_identity[ci]++;
			CachedImageIdentifier iici = { ci, index, 0 };
			dst[i] = impl->image_cache.acquire(iici, impl->frame_counter).image;
		}
		return { expected_value };
	}

	Result<void, AllocateException>
	DeviceSuperFrameResource::allocate_cached_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			CompressedImageViewCreateInfo civci(ci);
			dst[i] = impl->image_view_cache.acquire(civci, impl->frame_counter);
		}
		return { expected_value };
	}

	Result<void, AllocateException>
	DeviceSuperFrameResource::allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			auto& alloc = impl->suballocators[(int)ci.mem_usage - 1];
			auto alignment = std::lcm(ci.alignment, get_context().min_buffer_alignment);
			auto res = alloc.allocate_buffer(ci.size, alignment, loc);
			if (!res) {
				deallocate_buffers({ dst.data(), (uint64_t)i });
				return res;
			}
			dst[i] = *res;
		}
		return { expected_value };
	}

	void DeviceSuperFrameResource::deallocate_image_views(std::span<const ImageView> src) {
		std::shared_lock _s(impl->new_frame_mutex);
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->image_views_mutex);
		auto& vec = f.impl->image_views;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	void DeviceSuperFrameResource::deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) {
		std::shared_lock _s(impl->new_frame_mutex);
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->pds_mutex);
		auto& vec = f.impl->persistent_descriptor_sets;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	void DeviceSuperFrameResource::deallocate_descriptor_sets(std::span<const DescriptorSet> src) {
		std::shared_lock _s(impl->new_frame_mutex);
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->ds_mutex);
		auto& vec = f.impl->descriptor_sets;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	Result<void, AllocateException> DeviceSuperFrameResource::allocate_descriptor_pools(std::span<VkDescriptorPool> dst,
	                                                                                    std::span<const VkDescriptorPoolCreateInfo> cis,
	                                                                                    SourceLocationAtFrame loc) {
		std::scoped_lock _(impl->ds_pool_mutex);
		assert(cis.size() == dst.size());
		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			auto& source = impl->ds_pools;
			if (source.size() > 0) {
				dst[i] = source.back();
				source.pop_back();
			} else {
				VUK_DO_OR_RETURN(upstream->allocate_descriptor_pools(std::span{ &dst[i], 1 }, std::span{ &ci, 1 }, loc));
			}
		}
		return { expected_value };
	}

	void DeviceSuperFrameResource::deallocate_descriptor_pools(std::span<const VkDescriptorPool> src) {
		std::shared_lock _s(impl->new_frame_mutex);
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->ds_mutex);
		auto& vec = f.impl->ds_pools_to_destroy;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	void DeviceSuperFrameResource::deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) {
		std::shared_lock _s(impl->new_frame_mutex);
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->query_pool_mutex);
		auto& vec = f.impl->ts_query_pools;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	void DeviceSuperFrameResource::deallocate_timestamp_queries(std::span<const TimestampQuery> src) {} // noop

	void DeviceSuperFrameResource::deallocate_timeline_semaphores(std::span<const TimelineSemaphore> src) {
		std::shared_lock _s(impl->new_frame_mutex);
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->tsema_mutex);
		auto& vec = f.impl->tsemas;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	void DeviceSuperFrameResource::deallocate_acceleration_structures(std::span<const VkAccelerationStructureKHR> src) {
		std::shared_lock _s(impl->new_frame_mutex);
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->as_mutex);
		auto& vec = f.impl->ass;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	void DeviceSuperFrameResource::deallocate_swapchains(std::span<const VkSwapchainKHR> src) {
		std::shared_lock _s(impl->new_frame_mutex);
		auto& f = get_last_frame();
		std::unique_lock _(f.impl->swapchain_mutex);
		auto& vec = f.impl->swapchains;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	DeviceFrameResource& DeviceSuperFrameResource::get_last_frame() {
		return impl->frames[impl->frame_counter.load() % frames_in_flight];
	}

	DeviceFrameResource& DeviceSuperFrameResource::get_next_frame() {
		std::unique_lock _s(impl->new_frame_mutex);

		impl->frame_counter++;
		impl->local_frame = impl->frame_counter % frames_in_flight;

		// handle FrameResource
		auto& f = impl->frames[impl->local_frame];
		f.wait();
		deallocate_frame(f);
		f.construction_frame = impl->frame_counter.load();

		// handle MultiFrameResources
		for (auto it = impl->multi_frames.begin(); it != impl->multi_frames.end();) {
			auto& multi_frame = *it;
			multi_frame.remaining_lifetime--;
			if (multi_frame.remaining_lifetime == 0) {
				multi_frame.wait();
				deallocate_frame(multi_frame);
				it = impl->multi_frames.erase(it);
			} else {
				++it;
			}
		}

		impl->image_identity.clear();
		_s.unlock();
		// garbage collect caches
		impl->image_cache.collect(impl->frame_counter, 16);
		impl->image_view_cache.collect(impl->frame_counter, 16);
		impl->graphics_pipeline_cache.collect(impl->frame_counter, 16);
		impl->compute_pipeline_cache.collect(impl->frame_counter, 16);
		impl->ray_tracing_pipeline_cache.collect(impl->frame_counter, 16);

		return f;
	}

	DeviceMultiFrameResource& DeviceSuperFrameResource::get_multiframe_allocator(uint32_t frame_lifetime_count) {
		std::unique_lock _s(impl->new_frame_mutex);

		auto it = impl->multi_frames.emplace(DeviceMultiFrameResource(get_context().device, *this, frame_lifetime_count));
		return *it;
	}

	template<class T>
	void DeviceSuperFrameResource::deallocate_frame(T& frame) {
		auto& f = *frame.impl;
		upstream->deallocate_semaphores(f.semaphores);
		upstream->deallocate_fences(f.fences);
		upstream->deallocate_command_buffers(f.cmdbuffers_to_free);
		for (auto& pool : f.cmdpools_to_free) {
			direct->ctx->vkResetCommandPool(get_context().device, pool.command_pool, {});
		}
		deallocate_command_pools(f.cmdpools_to_free);
		for (Buffer& buf : f.buffer_gpus) {
			impl->suballocators[(int)buf.memory_usage - 1].deallocate_buffer(buf);
		}
		upstream->deallocate_framebuffers(f.framebuffers);
		upstream->deallocate_images(f.images);
		upstream->deallocate_image_views(f.image_views);
		upstream->deallocate_persistent_descriptor_sets(f.persistent_descriptor_sets);
		upstream->deallocate_descriptor_sets(f.descriptor_sets);
		get_context().make_timestamp_results_available(f.ts_query_pools);
		upstream->deallocate_timestamp_query_pools(f.ts_query_pools);
		upstream->deallocate_timeline_semaphores(f.tsemas);
		upstream->deallocate_acceleration_structures(f.ass);
		upstream->deallocate_swapchains(f.swapchains);
		upstream->deallocate_buffers(f.buffers);

		for (auto& p : f.ds_pools) {
			direct->ctx->vkResetDescriptorPool(get_context().device, p, {});
			impl->ds_pools.push_back(p);
		}

		upstream->deallocate_descriptor_pools(f.ds_pools_to_destroy);

		f.semaphores.clear();
		f.fences.clear();
		f.buffer_gpus.clear();
		f.cmdbuffers_to_free.clear();
		f.cmdpools_to_free.clear();
		f.ds_pools.clear();
		if (direct) {
			if (frame.construction_frame % 16 == 0) {
				f.linear_cpu_only.trim();
				f.linear_cpu_gpu.trim();
				f.linear_gpu_cpu.trim();
				f.linear_gpu_only.trim();
			}
			f.linear_cpu_only.reset();
			f.linear_cpu_gpu.reset();
			f.linear_gpu_cpu.reset();
			f.linear_gpu_only.reset();
		}
		f.framebuffers.clear();
		f.images.clear();
		f.image_views.clear();
		f.persistent_descriptor_sets.clear();
		f.descriptor_sets.clear();
		f.ts_query_pools.clear();
		f.query_index = 0;
		f.tsemas.clear();
		f.ass.clear();
		f.swapchains.clear();
		f.buffers.clear();
		f.ds_pools_to_destroy.clear();
	}

	void DeviceSuperFrameResource::force_collect() {
		impl->image_cache.collect(impl->frame_counter, 0);
		impl->image_view_cache.collect(impl->frame_counter, 0);
		impl->graphics_pipeline_cache.collect(impl->frame_counter, 0);
		impl->compute_pipeline_cache.collect(impl->frame_counter, 0);
		impl->ray_tracing_pipeline_cache.collect(impl->frame_counter, 0);
	}

	DeviceSuperFrameResource::~DeviceSuperFrameResource() {
		impl->image_cache.clear();
		impl->image_view_cache.clear();
		impl->graphics_pipeline_cache.clear();
		impl->compute_pipeline_cache.clear();
		impl->ray_tracing_pipeline_cache.clear();

		for (auto i = 0; i < frames_in_flight; i++) {
			auto lframe = (impl->frame_counter + i) % frames_in_flight;
			auto& f = impl->frames[lframe];
			f.wait();
			// free the resources manually, because we are destroying the individual FAs
			f.impl->linear_cpu_gpu.free();
			f.impl->linear_gpu_cpu.free();
			f.impl->linear_cpu_only.free();
			f.impl->linear_gpu_only.free();
		}

		for (auto i = 0; i < frames_in_flight; i++) {
			auto lframe = (impl->frame_counter + i) % frames_in_flight;
			auto& f = impl->frames[lframe];
			deallocate_frame(f);
			f.DeviceFrameResource::~DeviceFrameResource();
		}
		for (uint32_t i = 0; i < (uint32_t)impl->command_pools.size(); i++) {
			for (auto& cpool : impl->command_pools[i]) {
				CommandPool p{ cpool, i };
				upstream->deallocate_command_pools(std::span{ &p, 1 });
			}
		}
		for (auto& p : impl->ds_pools) {
			direct->deallocate_descriptor_pools(std::span{ &p, 1 });
		}
		delete impl;
	}
} // namespace vuk