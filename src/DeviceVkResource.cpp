#include "vuk/resources/DeviceVkResource.hpp"
#include "../src/RenderPass.hpp"
#include "vuk/Buffer.hpp"
#include "vuk/Context.hpp"
#include "vuk/Exception.hpp"
#include "vuk/PipelineInstance.hpp"
#include "vuk/Query.hpp"
#include "vuk/resources/DeviceNestedResource.hpp"
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#if VUK_DEBUG_ALLOCATIONS
#define VMA_DEBUG_LOG_FORMAT(format, ...)                                                                                                                      \
	do {                                                                                                                                                         \
		printf((format), __VA_ARGS__);                                                                                                                             \
		printf("\n");                                                                                                                                              \
	} while (false)
#endif
#include <mutex>
#include <numeric>
#include <sstream>
#include <vk_mem_alloc.h>

namespace vuk {
	std::string to_string(SourceLocationAtFrame loc) {
		std::stringstream sstream;

		sstream << loc.location.file_name() << '(' << loc.location.line() << ':' << loc.location.column() << "): " << loc.location.function_name();
		if (loc.absolute_frame != -1) {
			sstream << "@" << loc.absolute_frame;
		}
		return sstream.str();
	}

	std::string to_human_readable(uint64_t in) {
		/*       k       M      G */
		if (in >= 1024 * 1024 * 1024) {
			return std::to_string(in / (1024 * 1024 * 1024)) + " GiB";
		} else if (in >= 1024 * 1024) {
			return std::to_string(in / (1024 * 1024)) + " MiB";
		} else if (in >= 1024) {
			return std::to_string(in / (1024)) + " kiB";
		} else {
			return std::to_string(in) + " B";
		}
	}

	struct DeviceVkResourceImpl {
		std::mutex mutex;
		VmaAllocator allocator;
		VkPhysicalDeviceProperties properties;
		std::vector<uint32_t> all_queue_families;
		uint32_t queue_family_count;
	};

	DeviceVkResource::DeviceVkResource(Context& ctx) : ctx(&ctx), impl(new DeviceVkResourceImpl), device(ctx.device) {
		VmaAllocatorCreateInfo allocatorInfo = {};
		allocatorInfo.instance = ctx.instance;
		allocatorInfo.physicalDevice = ctx.physical_device;
		allocatorInfo.device = device;
		allocatorInfo.flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT | VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

		VmaVulkanFunctions vulkanFunctions = {};
		vulkanFunctions.vkGetPhysicalDeviceProperties = ctx.vkGetPhysicalDeviceProperties;
		vulkanFunctions.vkGetPhysicalDeviceMemoryProperties = ctx.vkGetPhysicalDeviceMemoryProperties;
		vulkanFunctions.vkAllocateMemory = ctx.vkAllocateMemory;
		vulkanFunctions.vkFreeMemory = ctx.vkFreeMemory;
		vulkanFunctions.vkMapMemory = ctx.vkMapMemory;
		vulkanFunctions.vkUnmapMemory = ctx.vkUnmapMemory;
		vulkanFunctions.vkFlushMappedMemoryRanges = ctx.vkFlushMappedMemoryRanges;
		vulkanFunctions.vkInvalidateMappedMemoryRanges = ctx.vkInvalidateMappedMemoryRanges;
		vulkanFunctions.vkBindBufferMemory = ctx.vkBindBufferMemory;
		vulkanFunctions.vkBindImageMemory = ctx.vkBindImageMemory;
		vulkanFunctions.vkGetBufferMemoryRequirements = ctx.vkGetBufferMemoryRequirements;
		vulkanFunctions.vkGetImageMemoryRequirements = ctx.vkGetImageMemoryRequirements;
		vulkanFunctions.vkCreateBuffer = ctx.vkCreateBuffer;
		vulkanFunctions.vkDestroyBuffer = ctx.vkDestroyBuffer;
		vulkanFunctions.vkCreateImage = ctx.vkCreateImage;
		vulkanFunctions.vkDestroyImage = ctx.vkDestroyImage;
		vulkanFunctions.vkCmdCopyBuffer = ctx.vkCmdCopyBuffer;
		allocatorInfo.pVulkanFunctions = &vulkanFunctions;

		vmaCreateAllocator(&allocatorInfo, &impl->allocator);
		ctx.vkGetPhysicalDeviceProperties(ctx.physical_device, &impl->properties);

		impl->all_queue_families = ctx.all_queue_families;
		impl->queue_family_count = (uint32_t)impl->all_queue_families.size();
	}

	DeviceVkResource::~DeviceVkResource() {
		vmaDestroyAllocator(impl->allocator);
		delete impl;
	}

	Result<void, AllocateException> DeviceVkResource::allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
		VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkResult res = ctx->vkCreateSemaphore(device, &sci, nullptr, &dst[i]);
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
				ctx->vkDestroySemaphore(device, v, nullptr);
			}
		}
	}

	Result<void, AllocateException> DeviceVkResource::allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) {
		VkFenceCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkResult res = ctx->vkCreateFence(device, &sci, nullptr, &dst[i]);
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
				ctx->vkDestroyFence(device, v, nullptr);
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

			VkResult res = ctx->vkAllocateCommandBuffers(device, &cbai, &dst[i].command_buffer);
			if (res != VK_SUCCESS) {
				return { expected_error, AllocateException{ res } };
			}
			dst[i].command_pool = ci.command_pool;
		}

		return { expected_value };
	}

	void DeviceVkResource::deallocate_command_buffers(std::span<const CommandBufferAllocation> dst) {
		for (auto& c : dst) {
			ctx->vkFreeCommandBuffers(device, c.command_pool.command_pool, 1, &c.command_buffer);
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkResult res = ctx->vkCreateCommandPool(device, &cis[i], nullptr, &dst[i].command_pool);
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
				ctx->vkDestroyCommandPool(device, v.command_pool, nullptr);
			}
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkResult res = ctx->vkCreateFramebuffer(device, &cis[i], nullptr, &dst[i]);
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
				ctx->vkDestroyFramebuffer(device, v, nullptr);
			}
		}
	}

	Result<void, AllocateException> DeviceVkResource::allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			std::lock_guard _(impl->mutex);
			auto& ci = cis[i];
			VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
			bci.size = ci.size;
			bci.usage = (VkBufferUsageFlags)all_buffer_usage_flags;
			bci.queueFamilyIndexCount = impl->queue_family_count;
			bci.sharingMode = bci.queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
			bci.pQueueFamilyIndices = impl->all_queue_families.data();

			VmaAllocationCreateInfo aci = {};
			aci.usage = VmaMemoryUsage(to_integral(ci.mem_usage));
			aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VkBuffer buffer;
			VmaAllocation allocation;
			VmaAllocationInfo allocation_info;
			// ignore alignment: we get a fresh VkBuffer which satisfies all alignments inside the VkBfufer
			auto res = vmaCreateBuffer(impl->allocator, &bci, &aci, &buffer, &allocation, &allocation_info);
			if (res != VK_SUCCESS) {
				deallocate_buffers({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
#if VUK_DEBUG_ALLOCATIONS
			vmaSetAllocationName(impl->allocator, allocation, to_string(loc).c_str());
#endif
			VkBufferDeviceAddressInfo bdai{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, buffer };
			uint64_t device_address = ctx->vkGetBufferDeviceAddress(device, &bdai);
			dst[i] = Buffer{ allocation, buffer, 0, ci.size, device_address, static_cast<std::byte*>(allocation_info.pMappedData), ci.mem_usage };
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_buffers(std::span<const Buffer> src) {
		for (auto& v : src) {
			if (v) {
				vmaDestroyBuffer(impl->allocator, v.buffer, static_cast<VmaAllocation>(v.allocation));
			}
		}
	}

	Result<void, AllocateException> DeviceVkResource::allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			std::lock_guard _(impl->mutex);
			VmaAllocationCreateInfo aci{};
			aci.usage = VMA_MEMORY_USAGE_GPU_ONLY;

			VkImage vkimg;
			VmaAllocation allocation;
			VkImageCreateInfo vkici = cis[i];
			if (cis[i].usage & (vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eDepthStencilAttachment)) {
				// this is a rendertarget, put it into the dedicated memory
				aci.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
			}

			auto res = vmaCreateImage(impl->allocator, &vkici, &aci, &vkimg, &allocation, nullptr);

			if (res != VK_SUCCESS) {
				deallocate_images({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
#if VUK_DEBUG_ALLOCATIONS
			vmaSetAllocationName(impl->allocator, allocation, to_string(loc).c_str());
#endif

			dst[i] = Image{ vkimg, allocation };
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_images(std::span<const Image> src) {
		for (auto& v : src) {
			if (v) {
				vmaDestroyImage(impl->allocator, v.image, static_cast<VmaAllocation>(v.allocation));
			}
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkImageViewCreateInfo ci = cis[i];
			VkImageViewUsageCreateInfo uvci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO };
			uvci.usage = (VkImageUsageFlags)cis[i].view_usage;
			if (uvci.usage != 0) {
				ci.pNext = &uvci;
			}
			ci.subresourceRange.layerCount = ci.subresourceRange.layerCount == 65535 ? VK_REMAINING_ARRAY_LAYERS : ci.subresourceRange.layerCount;
			ci.subresourceRange.levelCount = ci.subresourceRange.levelCount == 65535 ? VK_REMAINING_MIP_LEVELS : ci.subresourceRange.levelCount;
			VkImageView iv;
			VkResult res = ctx->vkCreateImageView(device, &ci, nullptr, &iv);
			if (res != VK_SUCCESS) {
				deallocate_image_views({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
			dst[i] = ctx->wrap(iv);
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
			size_t count = get_context().vkCmdBuildAccelerationStructuresKHR ? descriptor_counts.size() : descriptor_counts.size() - 1;
			uint32_t used_idx = 0;
			for (size_t i = 0; i < count; i++) {
				bool used = false;
				// create non-variable count descriptors
				if (dslai.descriptor_counts[i] > 0) {
					auto& d = descriptor_counts[used_idx];
					d.type = i == 11 ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR : VkDescriptorType(i);
					d.descriptorCount = dslai.descriptor_counts[i];
					used = true;
				}
				// create variable count descriptors
				if (dslai.variable_count_binding != (unsigned)-1 && dslai.variable_count_binding_type == DescriptorType(i)) {
					auto& d = descriptor_counts[used_idx];
					d.type = i == 11 ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR : VkDescriptorType(i);
					d.descriptorCount += ci.num_descriptors;
					used = true;
				}
				if (used) {
					used_idx++;
				}
			}

			dpci.pPoolSizes = descriptor_counts.data();
			dpci.poolSizeCount = used_idx;
			VkResult result = ctx->vkCreateDescriptorPool(device, &dpci, nullptr, &tda.backing_pool);
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
			if (dslai.variable_count_binding != (unsigned)-1) {
				dsai.pNext = &dsvdcai;
			}

			ctx->vkAllocateDescriptorSets(device, &dsai, &tda.backing_set);
			if (result != VK_SUCCESS) {
				deallocate_persistent_descriptor_sets({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ result } };
			}

			for (unsigned i = 0; i < ci.dslci.bindings.size(); i++) {
				tda.descriptor_bindings[i].resize(ci.dslci.bindings[i].descriptorCount);
			}
			if (dslai.variable_count_binding != (unsigned)-1) {
				tda.descriptor_bindings[dslai.variable_count_binding].resize(ci.num_descriptors);
			}
			tda.set_layout_create_info = ci.dslci;
			tda.set_layout = dsl;
		}

		return { expected_value };
	}

	void DeviceVkResource::deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) {
		for (auto& v : src) {
			ctx->vkDestroyDescriptorPool(ctx->device, v.backing_pool, nullptr);
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_descriptor_sets_with_value(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			auto& cinfo = cis[i];
			auto& pool = ctx->acquire_descriptor_pool(*cinfo.layout_info, ctx->get_frame_count());
			auto ds = pool.acquire(*ctx, *cinfo.layout_info);
			auto mask = cinfo.used.to_ulong();
			uint32_t leading_ones = num_leading_ones((uint32_t)mask);
			std::array<VkWriteDescriptorSet, VUK_MAX_BINDINGS> writes = {};
			std::array<VkWriteDescriptorSetAccelerationStructureKHR, VUK_MAX_BINDINGS> as_writes = {};
			int j = 0;
			for (uint32_t i = 0; i < leading_ones; i++, j++) {
				if (!cinfo.used.test(i)) {
					j--;
					continue;
				}
				auto& write = writes[j];
				auto& as_write = as_writes[j];
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
				case DescriptorType::eAccelerationStructureKHR:
					as_write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
					as_write.pAccelerationStructures = &binding.as.as;
					as_write.accelerationStructureCount = 1;
					write.pNext = &as_write;
					break;
				default:
					assert(0);
				}
			}
			ctx->vkUpdateDescriptorSets(device, j, writes.data(), 0, nullptr);
			dst[i] = { ds, *cinfo.layout_info };
		}
		return { expected_value };
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const DescriptorSetLayoutAllocInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			auto& cinfo = cis[i];
			auto& pool = ctx->acquire_descriptor_pool(cinfo, ctx->get_frame_count());
			dst[i] = { pool.acquire(*ctx, cinfo), cinfo };
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_descriptor_sets(std::span<const DescriptorSet> src) {
		for (int64_t i = 0; i < (int64_t)src.size(); i++) {
			DescriptorPool& pool = ctx->acquire_descriptor_pool(src[i].layout_info, ctx->get_frame_count());
			pool.release(src[i].descriptor_set);
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_descriptor_pools(std::span<VkDescriptorPool> dst, std::span<const VkDescriptorPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkDescriptorPoolCreateInfo ci = cis[i];
			VkResult res = ctx->vkCreateDescriptorPool(device, &ci, nullptr, &dst[i]);
			if (res != VK_SUCCESS) {
				deallocate_descriptor_pools({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_descriptor_pools(std::span<const VkDescriptorPool> src) {
		for (int64_t i = 0; i < (int64_t)src.size(); i++) {
			ctx->vkDestroyDescriptorPool(device, src[i], nullptr);
		}
	}

	void DeviceVkResource::deallocate_image_views(std::span<const ImageView> src) {
		for (auto& v : src) {
			if (v.payload != VK_NULL_HANDLE) {
				ctx->vkDestroyImageView(device, v.payload, nullptr);
			}
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkResult res = ctx->vkCreateQueryPool(device, &cis[i], nullptr, &dst[i].pool);
			if (res != VK_SUCCESS) {
				deallocate_timestamp_query_pools({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
			ctx->vkResetQueryPool(device, dst[i].pool, 0, cis[i].queryCount);
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) {
		for (auto& v : src) {
			if (v.pool != VK_NULL_HANDLE) {
				ctx->vkDestroyQueryPool(device, v.pool, nullptr);
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
			VkResult res = ctx->vkCreateSemaphore(device, &sci, nullptr, &dst[i].semaphore);
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
				ctx->vkDestroySemaphore(device, v.semaphore, nullptr);
				delete v.value;
			}
		}
	}

	Result<void, AllocateException> DeviceVkResource::allocate_acceleration_structures(std::span<VkAccelerationStructureKHR> dst,
	                                                                                   std::span<const VkAccelerationStructureCreateInfoKHR> cis,
	                                                                                   SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkAccelerationStructureCreateInfoKHR ci = cis[i];
			VkResult res = ctx->vkCreateAccelerationStructureKHR(device, &ci, nullptr, &dst[i]);
			if (res != VK_SUCCESS) {
				deallocate_acceleration_structures({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_acceleration_structures(std::span<const VkAccelerationStructureKHR> src) {
		for (auto& v : src) {
			if (v != VK_NULL_HANDLE) {
				ctx->vkDestroyAccelerationStructureKHR(device, v, nullptr);
			}
		}
	}

	void DeviceVkResource::deallocate_swapchains(std::span<const VkSwapchainKHR> src) {
		for (auto& v : src) {
			if (v != VK_NULL_HANDLE) {
				ctx->vkDestroySwapchainKHR(device, v, nullptr);
			}
		}
	}

	template<class T>
	T read(const std::byte*& data_ptr) {
		T t;
		memcpy(&t, data_ptr, sizeof(T));
		data_ptr += sizeof(T);
		return t;
	};

	Result<void, AllocateException> DeviceVkResource::allocate_graphics_pipelines(std::span<GraphicsPipelineInfo> dst,
	                                                                              std::span<const GraphicsPipelineInstanceCreateInfo> cis,
	                                                                              SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			GraphicsPipelineInstanceCreateInfo cinfo = cis[i];
			// create gfx pipeline
			VkGraphicsPipelineCreateInfo gpci{ .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
			gpci.renderPass = cinfo.render_pass;
			gpci.layout = cinfo.base->pipeline_layout;
			auto psscis = cinfo.base->psscis;
			for (auto i = 0; i < psscis.size(); i++) {
				psscis[i].pName = cinfo.base->entry_point_names[i].c_str();
			}
			gpci.pStages = psscis.data();
			gpci.stageCount = (uint32_t)psscis.size();

			// read variable sized data
			const std::byte* data_ptr = cinfo.is_inline() ? cinfo.inline_data : cinfo.extended_data;

			// subpass
			if (cinfo.records.nonzero_subpass) {
				gpci.subpass = read<uint8_t>(data_ptr);
			}

			// INPUT ASSEMBLY
			VkPipelineInputAssemblyStateCreateInfo input_assembly_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				                                                           .topology = static_cast<VkPrimitiveTopology>(cinfo.topology),
				                                                           .primitiveRestartEnable = cinfo.primitive_restart_enable };
			gpci.pInputAssemblyState = &input_assembly_state;
			// VERTEX INPUT
			fixed_vector<VkVertexInputBindingDescription, VUK_MAX_ATTRIBUTES> vibds;
			fixed_vector<VkVertexInputAttributeDescription, VUK_MAX_ATTRIBUTES> viads;
			VkPipelineVertexInputStateCreateInfo vertex_input_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
			if (cinfo.records.vertex_input) {
				viads.resize(cinfo.base->reflection_info.attributes.size());
				for (auto& viad : viads) {
					auto compressed = read<GraphicsPipelineInstanceCreateInfo::VertexInputAttributeDescription>(data_ptr);
					viad.binding = compressed.binding;
					viad.location = compressed.location;
					viad.format = (VkFormat)compressed.format;
					viad.offset = compressed.offset;
				}
				vertex_input_state.pVertexAttributeDescriptions = viads.data();
				vertex_input_state.vertexAttributeDescriptionCount = (uint32_t)viads.size();

				vibds.resize(read<uint8_t>(data_ptr));
				for (auto& vibd : vibds) {
					auto compressed = read<GraphicsPipelineInstanceCreateInfo::VertexInputBindingDescription>(data_ptr);
					vibd.binding = compressed.binding;
					vibd.inputRate = (VkVertexInputRate)compressed.inputRate;
					vibd.stride = compressed.stride;
				}
				vertex_input_state.pVertexBindingDescriptions = vibds.data();
				vertex_input_state.vertexBindingDescriptionCount = (uint32_t)vibds.size();
			}
			gpci.pVertexInputState = &vertex_input_state;
			// PIPELINE COLOR BLEND ATTACHMENTS
			VkPipelineColorBlendStateCreateInfo color_blend_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
				                                                     .attachmentCount = cinfo.attachmentCount };
			auto default_writemask = ColorComponentFlagBits::eR | ColorComponentFlagBits::eG | ColorComponentFlagBits::eB | ColorComponentFlagBits::eA;
			std::vector<VkPipelineColorBlendAttachmentState> pcbas(
			    cinfo.attachmentCount, VkPipelineColorBlendAttachmentState{ .blendEnable = false, .colorWriteMask = (VkColorComponentFlags)default_writemask });
			if (cinfo.records.color_blend_attachments) {
				if (!cinfo.records.broadcast_color_blend_attachment_0) {
					for (auto& pcba : pcbas) {
						auto compressed = read<GraphicsPipelineInstanceCreateInfo::PipelineColorBlendAttachmentState>(data_ptr);
						pcba = { compressed.blendEnable,
							       (VkBlendFactor)compressed.srcColorBlendFactor,
							       (VkBlendFactor)compressed.dstColorBlendFactor,
							       (VkBlendOp)compressed.colorBlendOp,
							       (VkBlendFactor)compressed.srcAlphaBlendFactor,
							       (VkBlendFactor)compressed.dstAlphaBlendFactor,
							       (VkBlendOp)compressed.alphaBlendOp,
							       compressed.colorWriteMask };
					}
				} else { // handle broadcast
					auto compressed = read<GraphicsPipelineInstanceCreateInfo::PipelineColorBlendAttachmentState>(data_ptr);
					for (auto& pcba : pcbas) {
						pcba = { compressed.blendEnable,
							       (VkBlendFactor)compressed.srcColorBlendFactor,
							       (VkBlendFactor)compressed.dstColorBlendFactor,
							       (VkBlendOp)compressed.colorBlendOp,
							       (VkBlendFactor)compressed.srcAlphaBlendFactor,
							       (VkBlendFactor)compressed.dstAlphaBlendFactor,
							       (VkBlendOp)compressed.alphaBlendOp,
							       compressed.colorWriteMask };
					}
				}
			}
			if (cinfo.records.logic_op) {
				auto compressed = read<GraphicsPipelineInstanceCreateInfo::BlendStateLogicOp>(data_ptr);
				color_blend_state.logicOpEnable = true;
				color_blend_state.logicOp = static_cast<VkLogicOp>(compressed.logic_op);
			}
			if (cinfo.records.blend_constants) {
				memcpy(&color_blend_state.blendConstants, data_ptr, sizeof(float) * 4);
				data_ptr += sizeof(float) * 4;
			}

			color_blend_state.pAttachments = pcbas.data();
			color_blend_state.attachmentCount = (uint32_t)pcbas.size();
			gpci.pColorBlendState = &color_blend_state;

			// SPECIALIZATION CONSTANTS
			fixed_vector<VkSpecializationInfo, graphics_stage_count> specialization_infos;
			fixed_vector<VkSpecializationMapEntry, VUK_MAX_SPECIALIZATIONCONSTANT_RANGES> specialization_map_entries;
			uint16_t specialization_constant_data_size = 0;
			const std::byte* specialization_constant_data = nullptr;
			if (cinfo.records.specialization_constants) {
				Bitset<VUK_MAX_SPECIALIZATIONCONSTANT_RANGES> set_constants = {};
				set_constants = read<Bitset<VUK_MAX_SPECIALIZATIONCONSTANT_RANGES>>(data_ptr);
				specialization_constant_data = data_ptr;

				for (unsigned i = 0; i < cinfo.base->reflection_info.spec_constants.size(); i++) {
					auto& sc = cinfo.base->reflection_info.spec_constants[i];
					uint16_t size = sc.type == Program::Type::edouble ? (uint16_t)sizeof(double) : 4;
					if (set_constants.test(i)) {
						specialization_constant_data_size += size;
					}
				}
				data_ptr += specialization_constant_data_size;

				uint16_t entry_offset = 0;
				for (uint32_t i = 0; i < psscis.size(); i++) {
					auto& pssci = psscis[i];
					uint16_t data_offset = 0;
					uint16_t current_entry_offset = entry_offset;
					for (unsigned i = 0; i < cinfo.base->reflection_info.spec_constants.size(); i++) {
						auto& sc = cinfo.base->reflection_info.spec_constants[i];
						auto size = sc.type == Program::Type::edouble ? sizeof(double) : 4;
						if (sc.stage & pssci.stage) {
							specialization_map_entries.emplace_back(VkSpecializationMapEntry{ sc.binding, data_offset, size });
							data_offset += (uint16_t)size;
							entry_offset++;
						}
					}

					VkSpecializationInfo si;
					si.pMapEntries = specialization_map_entries.data() + current_entry_offset;
					si.mapEntryCount = (uint32_t)specialization_map_entries.size() - current_entry_offset;
					si.pData = specialization_constant_data;
					si.dataSize = specialization_constant_data_size;
					if (si.mapEntryCount > 0) {
						specialization_infos.push_back(si);
						pssci.pSpecializationInfo = &specialization_infos.back();
					}
				}
			}

			// RASTER STATE
			VkPipelineRasterizationStateCreateInfo rasterization_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				                                                          .polygonMode = VK_POLYGON_MODE_FILL,
				                                                          .cullMode = cinfo.cullMode,
				                                                          .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
				                                                          .lineWidth = 1.f };

			if (cinfo.records.non_trivial_raster_state) {
				auto rs = read<GraphicsPipelineInstanceCreateInfo::RasterizationState>(data_ptr);
				rasterization_state = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
					                      .depthClampEnable = rs.depthClampEnable,
					                      .rasterizerDiscardEnable = rs.rasterizerDiscardEnable,
					                      .polygonMode = (VkPolygonMode)rs.polygonMode,
					                      .cullMode = cinfo.cullMode,
					                      .frontFace = (VkFrontFace)rs.frontFace,
					                      .lineWidth = 1.f };
			}
			rasterization_state.depthBiasEnable = cinfo.records.depth_bias_enable;
			if (cinfo.records.depth_bias) {
				auto db = read<GraphicsPipelineInstanceCreateInfo::DepthBias>(data_ptr);
				rasterization_state.depthBiasClamp = db.depthBiasClamp;
				rasterization_state.depthBiasConstantFactor = db.depthBiasConstantFactor;
				rasterization_state.depthBiasSlopeFactor = db.depthBiasSlopeFactor;
			}
			if (cinfo.records.line_width_not_1) {
				rasterization_state.lineWidth = read<float>(data_ptr);
			}
			VkPipelineRasterizationConservativeStateCreateInfoEXT conservative_state{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT
			};
			if (cinfo.records.conservative_rasterization_enabled) {
				auto cs = read<GraphicsPipelineInstanceCreateInfo::ConservativeState>(data_ptr);
				conservative_state = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT,
					                     .conservativeRasterizationMode = (VkConservativeRasterizationModeEXT)cs.conservativeMode,
					                     .extraPrimitiveOverestimationSize = cs.overestimationAmount };
				rasterization_state.pNext = &conservative_state;
			}
			gpci.pRasterizationState = &rasterization_state;

			// DEPTH - STENCIL STATE
			VkPipelineDepthStencilStateCreateInfo depth_stencil_state{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
			if (cinfo.records.depth_stencil) {
				auto d = read<GraphicsPipelineInstanceCreateInfo::Depth>(data_ptr);
				depth_stencil_state.depthTestEnable = d.depthTestEnable;
				depth_stencil_state.depthWriteEnable = d.depthWriteEnable;
				depth_stencil_state.depthCompareOp = (VkCompareOp)d.depthCompareOp;
				if (cinfo.records.depth_bounds) {
					auto db = read<GraphicsPipelineInstanceCreateInfo::DepthBounds>(data_ptr);
					depth_stencil_state.depthBoundsTestEnable = true;
					depth_stencil_state.minDepthBounds = db.minDepthBounds;
					depth_stencil_state.maxDepthBounds = db.maxDepthBounds;
				}
				if (cinfo.records.stencil_state) {
					auto s = read<GraphicsPipelineInstanceCreateInfo::Stencil>(data_ptr);
					depth_stencil_state.stencilTestEnable = true;
					depth_stencil_state.front = s.front;
					depth_stencil_state.back = s.back;
				}
				gpci.pDepthStencilState = &depth_stencil_state;
			}

			// MULTISAMPLE STATE
			VkPipelineMultisampleStateCreateInfo multisample_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				                                                      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
			if (cinfo.records.more_than_one_sample) {
				auto ms = read<GraphicsPipelineInstanceCreateInfo::Multisample>(data_ptr);
				multisample_state.rasterizationSamples = static_cast<VkSampleCountFlagBits>(ms.rasterization_samples);
				multisample_state.alphaToCoverageEnable = ms.alpha_to_coverage_enable;
				multisample_state.alphaToOneEnable = ms.alpha_to_one_enable;
				multisample_state.minSampleShading = ms.min_sample_shading;
				multisample_state.sampleShadingEnable = ms.sample_shading_enable;
				multisample_state.pSampleMask = nullptr; // not yet supported
			}
			gpci.pMultisampleState = &multisample_state;

			// VIEWPORTS
			const VkViewport* viewports = nullptr;
			uint8_t num_viewports = 1;
			if (cinfo.records.viewports) {
				num_viewports = read<uint8_t>(data_ptr);
				if (!(static_cast<vuk::DynamicStateFlags>(cinfo.dynamic_state_flags) & vuk::DynamicStateFlagBits::eViewport)) {
					viewports = reinterpret_cast<const VkViewport*>(data_ptr);
					data_ptr += num_viewports * sizeof(VkViewport);
				}
			}

			// SCISSORS
			const VkRect2D* scissors = nullptr;
			uint8_t num_scissors = 1;
			if (cinfo.records.scissors) {
				num_scissors = read<uint8_t>(data_ptr);
				if (!(static_cast<vuk::DynamicStateFlags>(cinfo.dynamic_state_flags) & vuk::DynamicStateFlagBits::eScissor)) {
					scissors = reinterpret_cast<const VkRect2D*>(data_ptr);
					data_ptr += num_scissors * sizeof(VkRect2D);
				}
			}

			VkPipelineViewportStateCreateInfo viewport_state{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
			viewport_state.pViewports = viewports;
			viewport_state.viewportCount = num_viewports;
			viewport_state.pScissors = scissors;
			viewport_state.scissorCount = num_scissors;
			gpci.pViewportState = &viewport_state;

			VkPipelineDynamicStateCreateInfo dynamic_state{ .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
			dynamic_state.dynamicStateCount = std::popcount(cinfo.dynamic_state_flags);
			fixed_vector<VkDynamicState, VkDynamicState::VK_DYNAMIC_STATE_DEPTH_BOUNDS> dyn_states;
			uint64_t dyn_state_cnt = 0;
			uint16_t mask = cinfo.dynamic_state_flags;
			while (mask > 0) {
				bool set = mask & 0x1;
				if (set) {
					dyn_states.push_back((VkDynamicState)dyn_state_cnt); // TODO: we will need a switch here instead of a cast when handling EXT
				}
				mask >>= 1;
				dyn_state_cnt++;
			}
			dynamic_state.pDynamicStates = dyn_states.data();
			gpci.pDynamicState = &dynamic_state;

			VkPipeline pipeline;
			VkResult res = ctx->vkCreateGraphicsPipelines(device, ctx->vk_pipeline_cache, 1, &gpci, nullptr, &pipeline);
			if (res != VK_SUCCESS) {
				deallocate_graphics_pipelines({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}

			ctx->set_name(pipeline, cinfo.base->pipeline_name);
			dst[i] = { cinfo.base, pipeline, gpci.layout, cinfo.base->layout_info };
		}

		return { expected_value };
	}
	void DeviceVkResource::deallocate_graphics_pipelines(std::span<const GraphicsPipelineInfo> src) {
		for (auto& v : src) {
			ctx->vkDestroyPipeline(device, v.pipeline, nullptr);
		}
	}

	Result<void, AllocateException> DeviceVkResource::allocate_compute_pipelines(std::span<ComputePipelineInfo> dst,
	                                                                             std::span<const ComputePipelineInstanceCreateInfo> cis,
	                                                                             SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			ComputePipelineInstanceCreateInfo cinfo = cis[i];
			// create compute pipeline
			VkComputePipelineCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
			cpci.layout = cinfo.base->pipeline_layout;
			cpci.stage = cinfo.base->psscis[0];
			cpci.stage.pName = cinfo.base->entry_point_names[0].c_str();

			VkPipeline pipeline;
			VkResult res = ctx->vkCreateComputePipelines(device, ctx->vk_pipeline_cache, 1, &cpci, nullptr, &pipeline);
			if (res != VK_SUCCESS) {
				deallocate_compute_pipelines({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}

			ctx->set_name(pipeline, cinfo.base->pipeline_name);
			dst[i] = { { cinfo.base, pipeline, cpci.layout, cinfo.base->layout_info }, cinfo.base->reflection_info.local_size };
		}

		return { expected_value };
	}

	void DeviceVkResource::deallocate_compute_pipelines(std::span<const ComputePipelineInfo> src) {
		for (auto& v : src) {
			ctx->vkDestroyPipeline(device, v.pipeline, nullptr);
		}
	}

	Result<void, AllocateException> DeviceVkResource::allocate_ray_tracing_pipelines(std::span<RayTracingPipelineInfo> dst,
	                                                                                 std::span<const RayTracingPipelineInstanceCreateInfo> cis,
	                                                                                 SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			RayTracingPipelineInstanceCreateInfo cinfo = cis[i];
			// create ray tracing pipeline
			VkRayTracingPipelineCreateInfoKHR cpci{ .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
			cpci.layout = cinfo.base->pipeline_layout;

			std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;
			VkRayTracingShaderGroupCreateInfoKHR group{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
			group.anyHitShader = VK_SHADER_UNUSED_KHR;
			group.closestHitShader = VK_SHADER_UNUSED_KHR;
			group.generalShader = VK_SHADER_UNUSED_KHR;
			group.intersectionShader = VK_SHADER_UNUSED_KHR;

			uint32_t miss_count = 0;
			uint32_t hit_count = 0;
			uint32_t callable_count = 0;

			auto psscis = cinfo.base->psscis;
			for (auto i = 0; i < psscis.size(); i++) {
				psscis[i].pName = cinfo.base->entry_point_names[i].c_str();
			}

			for (size_t i = 0; i < cinfo.base->psscis.size(); i++) {
				auto& stage = cinfo.base->psscis[i];
				if (stage.stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR) {
					group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
					group.generalShader = (uint32_t)i;
					groups.push_back(group);
				} else if (stage.stage == VK_SHADER_STAGE_MISS_BIT_KHR) {
					group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
					group.generalShader = (uint32_t)i;
					groups.push_back(group);
					miss_count++;
				} else if (stage.stage == VK_SHADER_STAGE_CALLABLE_BIT_KHR) {
					group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
					group.generalShader = (uint32_t)i;
					groups.push_back(group);
					callable_count++;
				}
			}
			for (auto& hg : cinfo.base->hit_groups) {
				group.type = (VkRayTracingShaderGroupTypeKHR)hg.type;
				group.generalShader = VK_SHADER_UNUSED_KHR;
				group.anyHitShader = hg.any_hit;
				group.intersectionShader = hg.intersection;
				group.closestHitShader = hg.closest_hit;
				groups.push_back(group);
				hit_count++;
			}

			cpci.groupCount = (uint32_t)groups.size();
			cpci.pGroups = groups.data();

			cpci.maxPipelineRayRecursionDepth = cinfo.base->max_ray_recursion_depth;
			cpci.pStages = psscis.data();
			cpci.stageCount = (uint32_t)psscis.size();

			VkPipeline pipeline;
			VkResult res = ctx->vkCreateRayTracingPipelinesKHR(device, {}, ctx->vk_pipeline_cache, 1, &cpci, nullptr, &pipeline);

			if (res != VK_SUCCESS) {
				deallocate_ray_tracing_pipelines({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}

			auto handleCount = 1 + miss_count + hit_count + callable_count;
			uint32_t handleSize = ctx->rt_properties.shaderGroupHandleSize;
			// The SBT (buffer) need to have starting groups to be aligned and handles in the group to be aligned.
			uint32_t handleSizeAligned = vuk::align_up(handleSize, ctx->rt_properties.shaderGroupHandleAlignment);

			VkStridedDeviceAddressRegionKHR rgen_region{};
			VkStridedDeviceAddressRegionKHR miss_region{};
			VkStridedDeviceAddressRegionKHR hit_region{};
			VkStridedDeviceAddressRegionKHR call_region{};

			rgen_region.stride = vuk::align_up(handleSizeAligned, ctx->rt_properties.shaderGroupBaseAlignment);
			rgen_region.size = rgen_region.stride; // The size member of pRayGenShaderBindingTable must be equal to its stride member
			miss_region.stride = handleSizeAligned;
			miss_region.size = vuk::align_up(miss_count * handleSizeAligned, ctx->rt_properties.shaderGroupBaseAlignment);
			hit_region.stride = handleSizeAligned;
			hit_region.size = vuk::align_up(hit_count * handleSizeAligned, ctx->rt_properties.shaderGroupBaseAlignment);
			call_region.stride = handleSizeAligned;
			call_region.size = vuk::align_up(callable_count * handleSizeAligned, ctx->rt_properties.shaderGroupBaseAlignment);

			// Get the shader group handles
			uint32_t dataSize = handleCount * handleSize;
			std::vector<uint8_t> handles(dataSize);
			auto result = ctx->vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, 0, handleCount, dataSize, handles.data());
			assert(result == VK_SUCCESS);

			VkDeviceSize sbt_size = rgen_region.size + miss_region.size + hit_region.size + call_region.size;
			Buffer SBT;
			BufferCreateInfo bci{ .mem_usage = vuk::MemoryUsage::eCPUtoGPU, .size = sbt_size, .alignment = ctx->rt_properties.shaderGroupBaseAlignment };
			auto buff_cr_result = allocate_buffers(std::span{ &SBT, 1 }, std::span{ &bci, 1 }, {});
			assert(buff_cr_result);

			// Helper to retrieve the handle data
			auto get_handle = [&](int i) {
				return handles.data() + i * handleSize;
			};
			std::byte* pData{ nullptr };
			uint32_t handleIdx{ 0 };
			// Raygen
			pData = SBT.mapped_ptr;
			memcpy(pData, get_handle(handleIdx++), handleSize);
			// Miss
			pData = SBT.mapped_ptr + rgen_region.size;
			for (uint32_t c = 0; c < miss_count; c++) {
				memcpy(pData, get_handle(handleIdx++), handleSize);
				pData += miss_region.stride;
			}
			// Hit
			pData = SBT.mapped_ptr + rgen_region.size + miss_region.size;
			for (uint32_t c = 0; c < hit_count; c++) {
				memcpy(pData, get_handle(handleIdx++), handleSize);
				pData += hit_region.stride;
			}
			// Call
			pData = SBT.mapped_ptr + rgen_region.size + miss_region.size + hit_region.size;
			for (uint32_t c = 0; c < callable_count; c++) {
				memcpy(pData, get_handle(handleIdx++), handleSize);
				pData += call_region.stride;
			}

			auto sbtAddress = SBT.device_address;

			rgen_region.deviceAddress = sbtAddress;
			miss_region.deviceAddress = sbtAddress + rgen_region.size;
			hit_region.deviceAddress = sbtAddress + rgen_region.size + miss_region.size;
			call_region.deviceAddress = sbtAddress + rgen_region.size + miss_region.size + hit_region.size;

			ctx->set_name(pipeline, cinfo.base->pipeline_name);
			dst[i] = { { cinfo.base, pipeline, cpci.layout, cinfo.base->layout_info }, rgen_region, miss_region, hit_region, call_region, SBT };
		}

		return { expected_value };
	}

	void DeviceVkResource::deallocate_ray_tracing_pipelines(std::span<const RayTracingPipelineInfo> src) {
		for (auto& v : src) {
			deallocate_buffers(std::span{ &v.sbt, 1 });
			ctx->vkDestroyPipeline(device, v.pipeline, nullptr);
		}
	}

	Result<void, AllocateException>
	DeviceVkResource::allocate_render_passes(std::span<VkRenderPass> dst, std::span<const RenderPassCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			auto cinfo = cis[i];
			VkResult res = ctx->vkCreateRenderPass(device, &cinfo, nullptr, &dst[i]);
			if (res != VK_SUCCESS) {
				deallocate_render_passes({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_render_passes(std::span<const VkRenderPass> src) {
		for (auto& v : src) {
			ctx->vkDestroyRenderPass(device, v, nullptr);
		}
	}

	Result<void, AllocateException> DeviceNestedResource::allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
		return upstream->allocate_semaphores(dst, loc);
	}

	void DeviceNestedResource::deallocate_semaphores(std::span<const VkSemaphore> sema) {
		upstream->deallocate_semaphores(sema);
	}

	Result<void, AllocateException> DeviceNestedResource::allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) {
		return upstream->allocate_fences(dst, loc);
	}

	void DeviceNestedResource::deallocate_fences(std::span<const VkFence> dst) {
		upstream->deallocate_fences(dst);
	}

	Result<void, AllocateException> DeviceNestedResource::allocate_command_buffers(std::span<CommandBufferAllocation> dst,
	                                                                               std::span<const CommandBufferAllocationCreateInfo> cis,
	                                                                               SourceLocationAtFrame loc) {
		return upstream->allocate_command_buffers(dst, cis, loc);
	}

	void DeviceNestedResource::deallocate_command_buffers(std::span<const CommandBufferAllocation> dst) {
		upstream->deallocate_command_buffers(dst);
	}

	Result<void, AllocateException>
	DeviceNestedResource::allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		return upstream->allocate_command_pools(dst, cis, loc);
	}

	void DeviceNestedResource::deallocate_command_pools(std::span<const CommandPool> dst) {
		upstream->deallocate_command_pools(dst);
	}

	Result<void, AllocateException>
	DeviceNestedResource::allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return upstream->allocate_buffers(dst, cis, loc);
	}

	void DeviceNestedResource::deallocate_buffers(std::span<const Buffer> src) {
		upstream->deallocate_buffers(src);
	}

	Result<void, AllocateException>
	DeviceNestedResource::allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return upstream->allocate_framebuffers(dst, cis, loc);
	}

	void DeviceNestedResource::deallocate_framebuffers(std::span<const VkFramebuffer> src) {
		upstream->deallocate_framebuffers(src);
	}

	Result<void, AllocateException> DeviceNestedResource::allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		return upstream->allocate_images(dst, cis, loc);
	}

	void DeviceNestedResource::deallocate_images(std::span<const Image> src) {
		upstream->deallocate_images(src);
	}

	Result<void, AllocateException>
	DeviceNestedResource::allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		return upstream->allocate_image_views(dst, cis, loc);
	}

	void DeviceNestedResource::deallocate_image_views(std::span<const ImageView> src) {
		upstream->deallocate_image_views(src);
	}

	Result<void, AllocateException> DeviceNestedResource::allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
	                                                                                          std::span<const PersistentDescriptorSetCreateInfo> cis,
	                                                                                          SourceLocationAtFrame loc) {
		return upstream->allocate_persistent_descriptor_sets(dst, cis, loc);
	}

	void DeviceNestedResource::deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) {
		upstream->deallocate_persistent_descriptor_sets(src);
	}

	Result<void, AllocateException>
	DeviceNestedResource::allocate_descriptor_sets_with_value(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		return upstream->allocate_descriptor_sets_with_value(dst, cis, loc);
	}

	Result<void, AllocateException>
	DeviceNestedResource::allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const DescriptorSetLayoutAllocInfo> cis, SourceLocationAtFrame loc) {
		return upstream->allocate_descriptor_sets(dst, cis, loc);
	}

	void DeviceNestedResource::deallocate_descriptor_sets(std::span<const DescriptorSet> src) {
		upstream->deallocate_descriptor_sets(src);
	}

	Result<void, AllocateException>
	DeviceNestedResource::allocate_descriptor_pools(std::span<VkDescriptorPool> dst, std::span<const VkDescriptorPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		return upstream->allocate_descriptor_pools(dst, cis, loc);
	}

	void DeviceNestedResource::deallocate_descriptor_pools(std::span<const VkDescriptorPool> src) {
		upstream->deallocate_descriptor_pools(src);
	}

	Result<void, AllocateException> DeviceNestedResource::allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst,
	                                                                                     std::span<const VkQueryPoolCreateInfo> cis,
	                                                                                     SourceLocationAtFrame loc) {
		return upstream->allocate_timestamp_query_pools(dst, cis, loc);
	}

	void DeviceNestedResource::deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) {
		upstream->deallocate_timestamp_query_pools(src);
	}

	Result<void, AllocateException>
	DeviceNestedResource::allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) {
		return upstream->allocate_timestamp_queries(dst, cis, loc);
	}

	void DeviceNestedResource::deallocate_timestamp_queries(std::span<const TimestampQuery> src) {
		upstream->deallocate_timestamp_queries(src);
	}

	Result<void, AllocateException> DeviceNestedResource::allocate_timeline_semaphores(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc) {
		return upstream->allocate_timeline_semaphores(dst, loc);
	}

	void DeviceNestedResource::deallocate_timeline_semaphores(std::span<const TimelineSemaphore> src) {
		upstream->deallocate_timeline_semaphores(src);
	}

	Result<void, AllocateException> DeviceNestedResource::allocate_acceleration_structures(std::span<VkAccelerationStructureKHR> dst,
	                                                                                       std::span<const VkAccelerationStructureCreateInfoKHR> cis,
	                                                                                       SourceLocationAtFrame loc) {
		return upstream->allocate_acceleration_structures(dst, cis, loc);
	}

	void DeviceNestedResource::deallocate_acceleration_structures(std::span<const VkAccelerationStructureKHR> src) {
		upstream->deallocate_acceleration_structures(src);
	}

	void DeviceNestedResource::deallocate_swapchains(std::span<const VkSwapchainKHR> src) {
		upstream->deallocate_swapchains(src);
	}

	Result<void, AllocateException> DeviceNestedResource::allocate_graphics_pipelines(std::span<GraphicsPipelineInfo> dst,
	                                                                                  std::span<const GraphicsPipelineInstanceCreateInfo> cis,
	                                                                                  SourceLocationAtFrame loc) {
		return upstream->allocate_graphics_pipelines(dst, cis, loc);
	}
	void DeviceNestedResource::deallocate_graphics_pipelines(std::span<const GraphicsPipelineInfo> src) {
		upstream->deallocate_graphics_pipelines(src);
	}

	Result<void, AllocateException> DeviceNestedResource::allocate_compute_pipelines(std::span<ComputePipelineInfo> dst,
	                                                                                 std::span<const ComputePipelineInstanceCreateInfo> cis,
	                                                                                 SourceLocationAtFrame loc) {
		return upstream->allocate_compute_pipelines(dst, cis, loc);
	}
	void DeviceNestedResource::deallocate_compute_pipelines(std::span<const ComputePipelineInfo> src) {
		upstream->deallocate_compute_pipelines(src);
	}

	Result<void, AllocateException> DeviceNestedResource::allocate_ray_tracing_pipelines(std::span<RayTracingPipelineInfo> dst,
	                                                                                     std::span<const RayTracingPipelineInstanceCreateInfo> cis,
	                                                                                     SourceLocationAtFrame loc) {
		return upstream->allocate_ray_tracing_pipelines(dst, cis, loc);
	}
	void DeviceNestedResource::deallocate_ray_tracing_pipelines(std::span<const RayTracingPipelineInfo> src) {
		upstream->deallocate_ray_tracing_pipelines(src);
	}

	Result<void, AllocateException>
	DeviceNestedResource::allocate_render_passes(std::span<VkRenderPass> dst, std::span<const RenderPassCreateInfo> cis, SourceLocationAtFrame loc) {
		return upstream->allocate_render_passes(dst, cis, loc);
	}
	void DeviceNestedResource::deallocate_render_passes(std::span<const VkRenderPass> src) {
		return upstream->deallocate_render_passes(src);
	}
} // namespace vuk