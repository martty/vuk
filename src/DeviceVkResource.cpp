#include "vuk/resources/DeviceVkResource.hpp"
#include "../src/LegacyGPUAllocator.hpp"
#include "../src/RenderPass.hpp"
#include "vuk/Buffer.hpp"
#include "vuk/Context.hpp"
#include "vuk/Exception.hpp"
#include "vuk/Query.hpp"

namespace vuk {
	DeviceVkResource::DeviceVkResource(Context& ctx, LegacyGPUAllocator& allocator) : ctx(&ctx), device(ctx.device), legacy_gpu_allocator(&allocator) {}

	Result<void, AllocateException> DeviceVkResource::allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
		VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkResult res = vkCreateSemaphore(device, &sci, nullptr, &dst[i]);
			if (res != VK_SUCCESS) {
				deallocate_semaphores({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_semaphores(std::span<const VkSemaphore> src) {
		for (auto& v : src) {
			if (v != VK_NULL_HANDLE) {
				vkDestroySemaphore(device, v, nullptr);
			}
		}
	}

	Result<void, AllocateException> DeviceVkResource::allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) {
		VkFenceCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkResult res = vkCreateFence(device, &sci, nullptr, &dst[i]);
			if (res != VK_SUCCESS) {
				deallocate_fences({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_fences(std::span<const VkFence> src) {
		for (auto& v : src) {
			if (v != VK_NULL_HANDLE) {
				vkDestroyFence(device, v, nullptr);
			}
		}
	}

	Result<void, AllocateException> DeviceVkResource::allocate_command_buffers(std::span<CommandBufferAllocation> dst,
	                                                                           std::span<const CommandBufferAllocationCreateInfo> cis,
	                                                                           SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());

		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];

			VkCommandBufferAllocateInfo cbai{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
			cbai.commandBufferCount = 1;
			cbai.commandPool = ci.command_pool.command_pool;
			cbai.level = ci.level;

			VkResult res = vkAllocateCommandBuffers(device, &cbai, &dst[i].command_buffer);
			if (res != VK_SUCCESS) {
				return { expected_error, AllocateException{ res } };
			}
			dst[i].command_pool = ci.command_pool;
		}

		return { expected_value };
	}

	void DeviceVkResource::deallocate_command_buffers(std::span<const CommandBufferAllocation> dst) {
		for (auto& c : dst) {
			vkFreeCommandBuffers(device, c.command_pool.command_pool, 1, &c.command_buffer);
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkResult res = vkCreateCommandPool(device, &cis[i], nullptr, &dst[i].command_pool);
			dst[i].queue_family_index = cis[i].queueFamilyIndex;
			if (res != VK_SUCCESS) {
				deallocate_command_pools({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_command_pools(std::span<const CommandPool> src) {
		for (auto& v : src) {
			if (v.command_pool != VK_NULL_HANDLE) {
				vkDestroyCommandPool(device, v.command_pool, nullptr);
			}
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkResult res = vkCreateFramebuffer(device, &cis[i], nullptr, &dst[i]);
			if (res != VK_SUCCESS) {
				deallocate_framebuffers({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_framebuffers(std::span<const VkFramebuffer> src) {
		for (auto& v : src) {
			if (v != VK_NULL_HANDLE) {
				vkDestroyFramebuffer(device, v, nullptr);
			}
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			auto& ci = cis[i];
			if (ci.mem_usage != MemoryUsage::eCPUonly && ci.mem_usage != MemoryUsage::eCPUtoGPU && ci.mem_usage != MemoryUsage::eGPUtoCPU) {
				deallocate_buffers(std::span{ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ VK_ERROR_FEATURE_NOT_PRESENT } }; // tried to allocate gpu only buffer as BufferCrossDevice
			}
			// TODO: legacy buffer alloc can't signal errors
			dst[i] = BufferCrossDevice{ legacy_gpu_allocator->allocate_buffer(ci.mem_usage, LegacyGPUAllocator::all_usage, ci.size, ci.alignment, true) };
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_buffers(std::span<const BufferCrossDevice> src) {
		for (auto& v : src) {
			if (v) {
				legacy_gpu_allocator->free_buffer(v);
			}
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			auto& ci = cis[i];
			// TODO: legacy buffer alloc can't signal errors
			if (ci.mem_usage != MemoryUsage::eGPUonly) {
				deallocate_buffers(std::span{ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ VK_ERROR_FEATURE_NOT_PRESENT } }; // tried to allocate cross device buffer as BufferGPU
			}
			dst[i] = BufferGPU{ legacy_gpu_allocator->allocate_buffer(ci.mem_usage, LegacyGPUAllocator::all_usage, ci.size, ci.alignment, false) };
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_buffers(std::span<const BufferGPU> src) {
		for (auto& v : src) {
			if (v) {
				legacy_gpu_allocator->free_buffer(v);
			}
		}
	}

	Result<void, AllocateException> DeviceVkResource::allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			// TODO: legacy image alloc can't signal errors

			dst[i] = legacy_gpu_allocator->create_image(cis[i]);
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_images(std::span<const Image> src) {
		for (auto& v : src) {
			if (v != VK_NULL_HANDLE) {
				legacy_gpu_allocator->destroy_image(v);
			}
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkImageViewCreateInfo ci = cis[i];
			VkImageView iv;
			VkResult res = vkCreateImageView(device, &ci, nullptr, &iv);
			if (res != VK_SUCCESS) {
				deallocate_image_views({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
			dst[i] = ctx->wrap(iv, cis[i]);
		}
		return { expected_value };
	}

	Result<void, AllocateException> DeviceVkResource::allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
	                                                                                      std::span<const PersistentDescriptorSetCreateInfo> cis,
	                                                                                      SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			auto& ci = cis[i];
			auto& dslai = ci.dslai;
			PersistentDescriptorSet& tda = dst[i];
			auto dsl = dslai.layout;
			VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
			dpci.maxSets = 1;
			std::array<VkDescriptorPoolSize, 12> descriptor_counts = {};
			uint32_t used_idx = 0;
			for (auto i = 0; i < descriptor_counts.size(); i++) {
				bool used = false;
				// create non-variable count descriptors
				if (dslai.descriptor_counts[i] > 0) {
					auto& d = descriptor_counts[used_idx];
					d.type = VkDescriptorType(i);
					d.descriptorCount = dslai.descriptor_counts[i];
					used = true;
				}
				// create variable count descriptors
				if (dslai.variable_count_binding != (unsigned)-1 && dslai.variable_count_binding_type == DescriptorType(i)) {
					auto& d = descriptor_counts[used_idx];
					d.type = VkDescriptorType(i);
					d.descriptorCount += ci.num_descriptors;
					used = true;
				}
				if (used) {
					used_idx++;
				}
			}

			dpci.pPoolSizes = descriptor_counts.data();
			dpci.poolSizeCount = used_idx;
			VkResult result = vkCreateDescriptorPool(device, &dpci, nullptr, &tda.backing_pool);
			if (result != VK_SUCCESS) {
				deallocate_persistent_descriptor_sets({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ result } };
			}
			VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
			dsai.descriptorPool = tda.backing_pool;
			dsai.descriptorSetCount = 1;
			dsai.pSetLayouts = &dsl;
			VkDescriptorSetVariableDescriptorCountAllocateInfo dsvdcai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO };
			dsvdcai.descriptorSetCount = 1;
			dsvdcai.pDescriptorCounts = &ci.num_descriptors;
			dsai.pNext = &dsvdcai;

			vkAllocateDescriptorSets(device, &dsai, &tda.backing_set);
			if (result != VK_SUCCESS) {
				deallocate_persistent_descriptor_sets({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ result } };
			}
			// TODO: we need more information here to handled arrayed bindings properly
			// for now we assume no arrayed bindings outside of the variable count one
			for (auto& bindings : tda.descriptor_bindings) {
				bindings.resize(1);
			}
			if (dslai.variable_count_binding != (unsigned)-1) {
				tda.descriptor_bindings[dslai.variable_count_binding].resize(ci.num_descriptors);
			}
		}

		return { expected_value };
	}

	void DeviceVkResource::deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) {
		for (auto& v : src) {
			vkDestroyDescriptorPool(ctx->device, v.backing_pool, nullptr);
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			auto& cinfo = cis[i];
			auto& pool = ctx->acquire_descriptor_pool(cinfo.layout_info, ctx->frame_counter);
			auto ds = pool.acquire(*ctx, cinfo.layout_info);
			auto mask = cinfo.used.to_ulong();
			uint32_t leading_ones = num_leading_ones(mask);
			std::array<VkWriteDescriptorSet, VUK_MAX_BINDINGS> writes = {};
			int j = 0;
			for (uint32_t i = 0; i < leading_ones; i++, j++) {
				if (!cinfo.used.test(i)) {
					j--;
					continue;
				}
				auto& write = writes[j];
				write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
				auto& binding = cinfo.bindings[i];
				write.descriptorType = (VkDescriptorType)binding.type;
				write.dstArrayElement = 0;
				write.descriptorCount = 1;
				write.dstBinding = i;
				write.dstSet = ds;
				switch (binding.type) {
				case DescriptorType::eUniformBuffer:
				case DescriptorType::eStorageBuffer:
					write.pBufferInfo = &binding.buffer;
					break;
				case DescriptorType::eSampledImage:
				case DescriptorType::eSampler:
				case DescriptorType::eCombinedImageSampler:
				case DescriptorType::eStorageImage:
					write.pImageInfo = &binding.image.dii;
					break;
				default:
					assert(0);
				}
			}
			vkUpdateDescriptorSets(device, j, writes.data(), 0, nullptr);
			dst[i] = { ds, cinfo.layout_info };
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_descriptor_sets(std::span<const DescriptorSet> src) {
		for (int64_t i = 0; i < (int64_t)src.size(); i++) {
			DescriptorPool& pool = ctx->acquire_descriptor_pool(src[i].layout_info, ctx->frame_counter);
			pool.release(src[i].descriptor_set);
		}
	}

	void DeviceVkResource::deallocate_image_views(std::span<const ImageView> src) {
		for (auto& v : src) {
			if (v.payload != VK_NULL_HANDLE) {
				vkDestroyImageView(device, v.payload, nullptr);
			}
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkResult res = vkCreateQueryPool(device, &cis[i], nullptr, &dst[i].pool);
			if (res != VK_SUCCESS) {
				deallocate_timestamp_query_pools({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
			vkResetQueryPool(device, dst[i].pool, 0, cis[i].queryCount);
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) {
		for (auto& v : src) {
			if (v.pool != VK_NULL_HANDLE) {
				vkDestroyQueryPool(device, v.pool, nullptr);
			}
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());

		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];

			ci.pool->queries[ci.pool->count++] = ci.query;
			dst[i].id = ci.pool->count;
			dst[i].pool = ci.pool->pool;
		}

		return { expected_value };
	}

	void DeviceVkResource::deallocate_timestamp_queries(std::span<const TimestampQuery> src) {}

	Result<void, AllocateException> DeviceVkResource::allocate_timeline_semaphores(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc) {
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
			VkSemaphoreTypeCreateInfo stci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
			stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
			stci.initialValue = 0;
			sci.pNext = &stci;
			VkResult res = vkCreateSemaphore(device, &sci, nullptr, &dst[i].semaphore);
			if (res != VK_SUCCESS) {
				deallocate_timeline_semaphores({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
			dst[i].value = new uint64_t{ 0 }; // TODO: more sensibly
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_timeline_semaphores(std::span<const TimelineSemaphore> src) {
		for (auto& v : src) {
			if (v.semaphore != VK_NULL_HANDLE) {
				vkDestroySemaphore(device, v.semaphore, nullptr);
				delete v.value;
			}
		}
	}

	void DeviceVkResource::deallocate_swapchains(std::span<const VkSwapchainKHR> src) {
		for (auto& v : src) {
			if (v != VK_NULL_HANDLE) {
				vkDestroySwapchainKHR(device, v, nullptr);
			}
		}
	}
} // namespace vuk