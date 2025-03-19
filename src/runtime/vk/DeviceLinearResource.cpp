#include "vuk/runtime/vk/DeviceLinearResource.hpp"
#include "vuk/runtime/vk/BufferAllocator.hpp"
#include "vuk/runtime/vk/Descriptor.hpp"
#include "vuk/runtime/vk/Query.hpp"
#include "vuk/runtime/vk/RenderPass.hpp"
#include "vuk/runtime/vk/VkQueueExecutor.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"

#include <numeric>
#include <plf_colony.h>

namespace vuk {
	struct DeviceLinearResourceImpl {
		Runtime* ctx;
		VkDevice device;
		std::vector<VkSemaphore> semaphores;
		std::vector<Buffer> buffers;
		std::vector<VkFence> fences;
		std::vector<CommandBufferAllocation> cmdbuffers_to_free;
		std::vector<CommandPool> cmdpools_to_free;
		std::vector<VkFramebuffer> framebuffers;
		std::vector<Image> images;
		std::vector<ImageView> image_views;
		std::vector<PersistentDescriptorSet> persistent_descriptor_sets;
		std::vector<DescriptorSet> descriptor_sets;
		VkDescriptorPool* last_ds_pool;
		plf::colony<VkDescriptorPool> ds_pools;
		std::vector<TimestampQueryPool> ts_query_pools;
		uint64_t query_index = 0;
		uint64_t current_ts_pool = 0;
		std::vector<SyncPoint> syncpoints;
		std::vector<VkAccelerationStructureKHR> ass;

		BufferUsageFlags all_buffer_usage_flags;

		BufferLinearAllocator linear_cpu_only;
		BufferLinearAllocator linear_cpu_gpu;
		BufferLinearAllocator linear_gpu_cpu;
		BufferLinearAllocator linear_gpu_only;

		DeviceLinearResourceImpl(DeviceResource& upstream) :
		    ctx(&upstream.get_context()),
		    device(ctx->device),
		    all_buffer_usage_flags(upstream.get_all_buffer_usage_flags(*ctx)),
		    linear_cpu_only(upstream, vuk::MemoryUsage::eCPUonly, all_buffer_usage_flags),
		    linear_cpu_gpu(upstream, vuk::MemoryUsage::eCPUtoGPU, all_buffer_usage_flags),
		    linear_gpu_cpu(upstream, vuk::MemoryUsage::eGPUtoCPU, all_buffer_usage_flags),
		    linear_gpu_only(upstream, vuk::MemoryUsage::eGPUonly, all_buffer_usage_flags) {}
	};

	DeviceLinearResource::DeviceLinearResource(DeviceResource& pstream) : DeviceNestedResource(pstream), impl(new DeviceLinearResourceImpl(pstream)) {}

	DeviceLinearResource::~DeviceLinearResource() {
		if (impl) {
			free();
		}
	}

	Result<void, AllocateException> DeviceLinearResource::allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_semaphores(dst, loc));
		auto& vec = impl->semaphores;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceLinearResource::deallocate_semaphores(std::span<const VkSemaphore> src) {} // noop

	Result<void, AllocateException> DeviceLinearResource::allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_fences(dst, loc));
		auto& vec = impl->fences;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceLinearResource::deallocate_fences(std::span<const VkFence> src) {} // noop

	Result<void, AllocateException> DeviceLinearResource::allocate_command_buffers(std::span<CommandBufferAllocation> dst,
	                                                                               std::span<const CommandBufferAllocationCreateInfo> cis,
	                                                                               SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_command_buffers(dst, cis, loc));
		auto& vec = impl->cmdbuffers_to_free;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceLinearResource::deallocate_command_buffers(std::span<const CommandBufferAllocation> src) {} // no-op, deallocated with pools

	Result<void, AllocateException>
	DeviceLinearResource::allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_command_pools(dst, cis, loc));
		auto& vec = impl->cmdpools_to_free;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceLinearResource::deallocate_command_pools(std::span<const CommandPool> dst) {} // no-op

	Result<void, AllocateException>
	DeviceLinearResource::allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
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

	void DeviceLinearResource::deallocate_buffers(std::span<const Buffer> src) {} // no-op, linear

	Result<void, AllocateException>
	DeviceLinearResource::allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_framebuffers(dst, cis, loc));
		auto& vec = impl->framebuffers;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceLinearResource::deallocate_framebuffers(std::span<const VkFramebuffer> src) {} // noop

	Result<void, AllocateException> DeviceLinearResource::allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_images(dst, cis, loc));
		auto& vec = impl->images;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceLinearResource::deallocate_images(std::span<const Image> src) {} // noop

	Result<void, AllocateException>
	DeviceLinearResource::allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_image_views(dst, cis, loc));
		auto& vec = impl->image_views;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceLinearResource::deallocate_image_views(std::span<const ImageView> src) {} // noop

	Result<void, AllocateException> DeviceLinearResource::allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
	                                                                                          std::span<const PersistentDescriptorSetCreateInfo> cis,
	                                                                                          SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_persistent_descriptor_sets(dst, cis, loc));
		auto& vec = impl->persistent_descriptor_sets;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceLinearResource::deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) {} // noop

	Result<void, AllocateException>
	DeviceLinearResource::allocate_descriptor_sets_with_value(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_descriptor_sets_with_value(dst, cis, loc));
		auto& vec = impl->descriptor_sets;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceLinearResource::deallocate_descriptor_sets(std::span<const DescriptorSet> src) {} // noop

	Result<void, AllocateException>
	DeviceLinearResource::allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const DescriptorSetLayoutAllocInfo> cis, SourceLocationAtFrame loc) {
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
			VkDescriptorPool pool;
			VUK_DO_OR_RETURN(upstream->allocate_descriptor_pools({ &pool, 1 }, { &dpci, 1 }, loc));
			impl->last_ds_pool = &*impl->ds_pools.emplace(pool);
		}

		// look at last stored pool
		VkDescriptorPool* last_pool = impl->last_ds_pool;

		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			// attempt to allocate a set
			VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
			dsai.descriptorPool = *last_pool;
			dsai.descriptorSetCount = 1;
			dsai.pSetLayouts = &ci.layout;
			dst[i].layout_info = ci;
			auto result = impl->ctx->vkAllocateDescriptorSets(impl->device, &dsai, &dst[i].descriptor_set);
			// if we fail, we allocate another pool from upstream
			if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
				{
					VkDescriptorPool pool;
					VUK_DO_OR_RETURN(upstream->allocate_descriptor_pools({ &pool, 1 }, { &dpci, 1 }, loc));
					last_pool = &*impl->ds_pools.emplace(pool);
				}
				dsai.descriptorPool = *last_pool;
				result = impl->ctx->vkAllocateDescriptorSets(impl->device, &dsai, &dst[i].descriptor_set);
				if (result != VK_SUCCESS) {
					return { expected_error, AllocateException{ result } };
				}
			}
		}
		return { expected_value };
	}

	Result<void, AllocateException> DeviceLinearResource::allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst,
	                                                                                     std::span<const VkQueryPoolCreateInfo> cis,
	                                                                                     SourceLocationAtFrame loc) {
		VUK_DO_OR_RETURN(upstream->allocate_timestamp_query_pools(dst, cis, loc));
		auto& vec = impl->ts_query_pools;
		vec.insert(vec.end(), dst.begin(), dst.end());
		return { expected_value };
	}

	void DeviceLinearResource::deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) {} // noop

	Result<void, AllocateException>
	DeviceLinearResource::allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());

		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];

			if (ci.pool) { // use given pool to allocate query
				ci.pool->queries[ci.pool->count++] = ci.query;
				dst[i].id = ci.pool->count;
				dst[i].pool = ci.pool->pool;
			} else { // allocate a pool on demand
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

	void DeviceLinearResource::deallocate_timestamp_queries(std::span<const TimestampQuery> src) {} // noop

	void DeviceLinearResource::wait_sync_points(std::span<const SyncPoint> src) {
		auto& vec = impl->syncpoints;
		vec.insert(vec.end(), src.begin(), src.end());
	}

	void DeviceLinearResource::wait() {
		if (impl->fences.size() > 0) {
			if (impl->fences.size() > 64) {
				int i = 0;
				for (; i < impl->fences.size() - 64; i += 64) {
					impl->ctx->vkWaitForFences(impl->device, 64, impl->fences.data() + i, true, UINT64_MAX);
				}
				impl->ctx->vkWaitForFences(impl->device, (uint32_t)impl->fences.size() - i, impl->fences.data() + i, true, UINT64_MAX);
			} else {
				impl->ctx->vkWaitForFences(impl->device, (uint32_t)impl->fences.size(), impl->fences.data(), true, UINT64_MAX);
			}
		}
		if (impl->syncpoints.size() > 0) {
			VkSemaphoreWaitInfo swi{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };

			std::vector<VkSemaphore> semas;
			semas.reserve(impl->syncpoints.size());
			std::vector<uint64_t> values;
			values.reserve(impl->syncpoints.size());

			for (uint64_t i = 0; i < impl->syncpoints.size(); i++) {
				auto& sp = impl->syncpoints[i];
				if (sp.executor->type == Executor::Type::eVulkanDeviceQueue) {
					auto dev_queue = static_cast<QueueExecutor*>(sp.executor);
					semas.push_back(dev_queue->get_semaphore());
					values.push_back(sp.visibility);
				}
			}
			swi.pSemaphores = semas.data();
			swi.pValues = values.data();
			swi.semaphoreCount = (uint32_t)semas.size();
			impl->ctx->vkWaitSemaphores(impl->device, &swi, UINT64_MAX);
		}
	}

	void DeviceLinearResource::free() {
		auto& f = *impl;
		upstream->deallocate_semaphores(f.semaphores);
		upstream->deallocate_fences(f.fences);
		upstream->deallocate_command_buffers(f.cmdbuffers_to_free);
		for (auto& pool : f.cmdpools_to_free) {
			f.ctx->vkResetCommandPool(f.device, pool.command_pool, {});
		}
		upstream->deallocate_command_pools(f.cmdpools_to_free);
		upstream->deallocate_framebuffers(f.framebuffers);
		upstream->deallocate_images(f.images);
		upstream->deallocate_image_views(f.image_views);
		upstream->deallocate_buffers(f.buffers);
		upstream->deallocate_persistent_descriptor_sets(f.persistent_descriptor_sets);
		upstream->deallocate_descriptor_sets(f.descriptor_sets);
		f.ctx->make_timestamp_results_available(f.ts_query_pools);
		upstream->deallocate_timestamp_query_pools(f.ts_query_pools);
		upstream->deallocate_acceleration_structures(f.ass);

		f.syncpoints.clear();

		for (auto& p : f.ds_pools) {
			upstream->deallocate_descriptor_pools({ &p, 1 });
		}
	}
} // namespace vuk