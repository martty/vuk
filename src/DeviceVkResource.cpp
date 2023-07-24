#include "vuk/resources/DeviceVkResource.hpp"
#include "../src/RenderPass.hpp"
#include "vuk/Buffer.hpp"
#include "vuk/Context.hpp"
#include "vuk/Exception.hpp"
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

		if (ctx.transfer_queue_family_index != ctx.graphics_queue_family_index && ctx.compute_queue_family_index != ctx.graphics_queue_family_index) {
			impl->all_queue_families = { ctx.graphics_queue_family_index, ctx.compute_queue_family_index, ctx.transfer_queue_family_index };
		} else if (ctx.transfer_queue_family_index != ctx.graphics_queue_family_index) {
			impl->all_queue_families = { ctx.graphics_queue_family_index, ctx.transfer_queue_family_index };
		} else if (ctx.compute_queue_family_index != ctx.graphics_queue_family_index) {
			impl->all_queue_families = { ctx.graphics_queue_family_index, ctx.compute_queue_family_index };
		} else {
			impl->all_queue_families = { ctx.graphics_queue_family_index };
		}
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
			auto alignment = std::lcm(ci.alignment, get_context().min_buffer_alignment);
			auto res = vmaCreateBufferWithAlignment(impl->allocator, &bci, &aci, alignment, &buffer, &allocation, &allocation_info);
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
			unsigned count = get_context().vkCmdBuildAccelerationStructuresKHR ? descriptor_counts.size() : descriptor_counts.size() - 1;
			uint32_t used_idx = 0;
			for (auto i = 0; i < count; i++) {
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
			uint32_t leading_ones = num_leading_ones(mask);
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
} // namespace vuk