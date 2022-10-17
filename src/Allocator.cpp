#include "vuk/Allocator.hpp"
#include "LegacyGPUAllocator.hpp"
#include "RenderPass.hpp"
#include "vuk/Context.hpp"
#include "vuk/Exception.hpp"
#include "vuk/Query.hpp"
#include "vuk/resources/DeviceFrameResource.hpp"
#include "vuk/resources/DeviceVkResource.hpp"

#include <numeric>
#include <string>
#include <utility>

namespace vuk {
	/****Allocator impls *****/

	Result<void, AllocateException> Allocator::allocate(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
		return device_resource->allocate_semaphores(dst, loc);
	}

	Result<void, AllocateException> Allocator::allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
		return device_resource->allocate_semaphores(dst, loc);
	}

	void Allocator::deallocate(std::span<const VkSemaphore> src) {
		device_resource->deallocate_semaphores(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<VkFence> dst, SourceLocationAtFrame loc) {
		return device_resource->allocate_fences(dst, loc);
	}

	Result<void, AllocateException> Allocator::allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) {
		return device_resource->allocate_fences(dst, loc);
	}

	void Allocator::deallocate(std::span<const VkFence> src) {
		device_resource->deallocate_fences(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_command_pools(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_command_pools(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const CommandPool> src) {
		device_resource->deallocate_command_pools(src);
	}

	Result<void, AllocateException>
	Allocator::allocate(std::span<CommandBufferAllocation> dst, std::span<const CommandBufferAllocationCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_command_buffers(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_command_buffers(std::span<CommandBufferAllocation> dst,
	                                                                    std::span<const CommandBufferAllocationCreateInfo> cis,
	                                                                    SourceLocationAtFrame loc) {
		return device_resource->allocate_command_buffers(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const CommandBufferAllocation> src) {
		device_resource->deallocate_command_buffers(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_buffers(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_buffers(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const BufferCrossDevice> src) {
		device_resource->deallocate_buffers(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_buffers(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_buffers(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const BufferGPU> src) {
		device_resource->deallocate_buffers(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_framebuffers(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_framebuffers(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const VkFramebuffer> src) {
		device_resource->deallocate_framebuffers(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_images(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_images(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const Image> src) {
		device_resource->deallocate_images(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_image_views(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_image_views(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const ImageView> src) {
		device_resource->deallocate_image_views(src);
	}

	Result<void, AllocateException>
	Allocator::allocate(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_persistent_descriptor_sets(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
	                                                                               std::span<const PersistentDescriptorSetCreateInfo> cis,
	                                                                               SourceLocationAtFrame loc) {
		return device_resource->allocate_persistent_descriptor_sets(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const PersistentDescriptorSet> src) {
		device_resource->deallocate_persistent_descriptor_sets(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_descriptor_sets_with_value(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_descriptor_sets_with_value(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_descriptor_sets_with_value(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate(std::span<DescriptorSet> dst, std::span<const DescriptorSetLayoutAllocInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_descriptor_sets(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const DescriptorSetLayoutAllocInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_descriptor_sets(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const DescriptorSet> src) {
		device_resource->deallocate_descriptor_sets(src);
	}

	Result<void, AllocateException>
	Allocator::allocate(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_timestamp_query_pools(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_timestamp_query_pools(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const TimestampQueryPool> src) {
		device_resource->deallocate_timestamp_query_pools(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_timestamp_queries(dst, cis, loc);
	}

	Result<void, AllocateException>
	Allocator::allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_timestamp_queries(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const TimestampQuery> src) {
		device_resource->deallocate_timestamp_queries(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc) {
		return device_resource->allocate_timeline_semaphores(dst, loc);
	}

	Result<void, AllocateException> Allocator::allocate_timeline_semaphores(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc) {
		return device_resource->allocate_timeline_semaphores(dst, loc);
	}

	void Allocator::deallocate(std::span<const TimelineSemaphore> src) {
		device_resource->deallocate_timeline_semaphores(src);
	}

	Result<void, AllocateException>
	Allocator::allocate(std::span<VkAccelerationStructureKHR> dst, std::span<const VkAccelerationStructureCreateInfoKHR> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_acceleration_structures(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_acceleration_structures(std::span<VkAccelerationStructureKHR> dst,
	                                                                            std::span<const VkAccelerationStructureCreateInfoKHR> cis,
	                                                                            SourceLocationAtFrame loc) {
		return device_resource->allocate_acceleration_structures(dst, cis, loc);
	}

	void Allocator::deallocate(std::span<const VkAccelerationStructureKHR> src) {
		device_resource->deallocate_acceleration_structures(src);
	}

	void Allocator::deallocate(std::span<const VkSwapchainKHR> src) {
		device_resource->deallocate_swapchains(src);
	}

	PFN_vmaAllocateDeviceMemoryFunction LegacyGPUAllocator::real_alloc_callback = nullptr;

	std::string to_string(BufferUsageFlags value) {
		if (!value)
			return "{}";
		std::string result;

		if (value & BufferUsageFlagBits::eTransferRead)
			result += "TransferSrc | ";
		if (value & BufferUsageFlagBits::eTransferWrite)
			result += "TransferDst | ";
		if (value & BufferUsageFlagBits::eUniformTexelBuffer)
			result += "UniformTexelBuffer | ";
		if (value & BufferUsageFlagBits::eStorageTexelBuffer)
			result += "StorageTexelBuffer | ";
		if (value & BufferUsageFlagBits::eUniformBuffer)
			result += "UniformBuffer | ";
		if (value & BufferUsageFlagBits::eStorageBuffer)
			result += "StorageBuffer | ";
		if (value & BufferUsageFlagBits::eIndexBuffer)
			result += "IndexBuffer | ";
		if (value & BufferUsageFlagBits::eVertexBuffer)
			result += "VertexBuffer | ";
		if (value & BufferUsageFlagBits::eIndirectBuffer)
			result += "IndirectBuffer | ";
		if (value & BufferUsageFlagBits::eShaderDeviceAddress)
			result += "ShaderDeviceAddress | ";
		if (value & BufferUsageFlagBits::eTransformFeedbackBufferEXT)
			result += "TransformFeedbackBufferEXT | ";
		if (value & BufferUsageFlagBits::eTransformFeedbackCounterBufferEXT)
			result += "TransformFeedbackCounterBufferEXT | ";
		if (value & BufferUsageFlagBits::eConditionalRenderingEXT)
			result += "ConditionalRenderingEXT | ";
		return "{ " + result.substr(0, result.size() - 3) + " }";
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

	void LegacyGPUAllocator::pool_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size, void* userdata) {
		auto& pags = *reinterpret_cast<PoolAllocHelper*>(userdata);
		pags.bci.size = size;
		VkBuffer buffer;
		vkCreateBuffer(pags.device, &pags.bci, nullptr, &buffer);
		vkBindBufferMemory(pags.device, buffer, memory, 0);
		pags.result = buffer;
		VkBufferDeviceAddressInfo bdai{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, buffer };
		pags.device_address = vkGetBufferDeviceAddress(pags.device, &bdai);
		std::string devmem_name = "DeviceMemory (Pool [" + std::to_string(memoryType) + "] " + to_human_readable(size) + ")";
		std::string buffer_name = "Buffer (Pool ";
		buffer_name += to_string(vuk::BufferUsageFlags(pags.bci.usage));
		buffer_name += ")";

		{
			VkDebugUtilsObjectNameInfoEXT info;
			info.pNext = nullptr;
			info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
			info.pObjectName = devmem_name.c_str();
			info.objectType = VK_OBJECT_TYPE_DEVICE_MEMORY;
			info.objectHandle = reinterpret_cast<uint64_t>(memory);
			if (pags.setDebugUtilsObjectNameEXT) {
				pags.setDebugUtilsObjectNameEXT(pags.device, &info);
			}
		}
		{
			VkDebugUtilsObjectNameInfoEXT info;
			info.pNext = nullptr;
			info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
			info.pObjectName = buffer_name.c_str();
			info.objectType = VK_OBJECT_TYPE_BUFFER;
			info.objectHandle = reinterpret_cast<uint64_t>((VkBuffer)buffer);
			if (pags.setDebugUtilsObjectNameEXT) {
				pags.setDebugUtilsObjectNameEXT(pags.device, &info);
			}
		}
	}

	void LegacyGPUAllocator::noop_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size, void* userdata) {
		auto& pags = *reinterpret_cast<PoolAllocHelper*>(userdata);
		std::string devmem_name = "DeviceMemory (Dedicated [" + std::to_string(memoryType) + "] " + to_human_readable(size) + ")";
		{
			VkDebugUtilsObjectNameInfoEXT info;
			info.pNext = nullptr;
			info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
			info.pObjectName = devmem_name.c_str();
			info.objectType = VK_OBJECT_TYPE_DEVICE_MEMORY;
			info.objectHandle = reinterpret_cast<uint64_t>(memory);
			if (pags.setDebugUtilsObjectNameEXT) {
				pags.setDebugUtilsObjectNameEXT(pags.device, &info);
			}
		}
	}

	LegacyGPUAllocator::LegacyGPUAllocator(VkInstance instance,
	                                       VkDevice device,
	                                       VkPhysicalDevice phys_dev,
	                                       uint32_t graphics_queue_family,
	                                       uint32_t compute_queue_family,
	                                       uint32_t transfer_queue_family,
	                                       PFN_vkSetDebugUtilsObjectNameEXT pfn_vkSetDebugUtilsObjectNameEXT) :
	    device(device) {
		VmaAllocatorCreateInfo allocatorInfo = {};
		allocatorInfo.instance = instance;
		allocatorInfo.physicalDevice = phys_dev;
		allocatorInfo.device = device;
		allocatorInfo.flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT | VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

		VmaVulkanFunctions vulkanFunctions = {};
		vulkanFunctions.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
		vulkanFunctions.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
		vulkanFunctions.vkAllocateMemory = vkAllocateMemory;
		vulkanFunctions.vkFreeMemory = vkFreeMemory;
		vulkanFunctions.vkMapMemory = vkMapMemory;
		vulkanFunctions.vkUnmapMemory = vkUnmapMemory;
		vulkanFunctions.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
		vulkanFunctions.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
		vulkanFunctions.vkBindBufferMemory = vkBindBufferMemory;
		vulkanFunctions.vkBindImageMemory = vkBindImageMemory;
		vulkanFunctions.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
		vulkanFunctions.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
		vulkanFunctions.vkCreateBuffer = vkCreateBuffer;
		vulkanFunctions.vkDestroyBuffer = vkDestroyBuffer;
		vulkanFunctions.vkCreateImage = vkCreateImage;
		vulkanFunctions.vkDestroyImage = vkDestroyImage;
		vulkanFunctions.vkCmdCopyBuffer = vkCmdCopyBuffer;
		allocatorInfo.pVulkanFunctions = &vulkanFunctions;

		VmaDeviceMemoryCallbacks cbs;
		cbs.pfnFree = nullptr;
		real_alloc_callback = noop_cb;
		cbs.pfnAllocate = allocation_cb;
		pool_helper = std::make_unique<PoolAllocHelper>();
		pool_helper->setDebugUtilsObjectNameEXT = pfn_vkSetDebugUtilsObjectNameEXT;
		cbs.pUserData = pool_helper.get();
		allocatorInfo.pDeviceMemoryCallbacks = &cbs;

		vmaCreateAllocator(&allocatorInfo, &allocator);
		vkGetPhysicalDeviceProperties(phys_dev, &properties);

		pool_helper->device = device;

		if (transfer_queue_family != graphics_queue_family && compute_queue_family != graphics_queue_family) {
			all_queue_families = { graphics_queue_family, compute_queue_family, transfer_queue_family };
		} else if (transfer_queue_family != graphics_queue_family) {
			all_queue_families = { graphics_queue_family, transfer_queue_family };
		} else if (compute_queue_family != graphics_queue_family) {
			all_queue_families = { graphics_queue_family, compute_queue_family };
		} else {
			all_queue_families = { graphics_queue_family };
		}
		queue_family_count = (uint32_t)all_queue_families.size();
	}

	// not locked, must be called from a locked fn

	VmaPool LegacyGPUAllocator::_create_pool(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage) {
		VmaPoolCreateInfo poolCreateInfo = {};
		VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bci.size = 1024; // Whatever.
		bci.usage = (VkBufferUsageFlags)buffer_usage;
		bci.queueFamilyIndexCount = queue_family_count;
		bci.sharingMode = bci.queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
		bci.pQueueFamilyIndices = all_queue_families.data();

		VmaAllocationCreateInfo allocCreateInfo = {};
		allocCreateInfo.usage = VmaMemoryUsage(to_integral(mem_usage));

		uint32_t memTypeIndex;
		vmaFindMemoryTypeIndexForBufferInfo(allocator, &bci, &allocCreateInfo, &memTypeIndex);
		poolCreateInfo.memoryTypeIndex = memTypeIndex;
		poolCreateInfo.blockSize = 0;
		poolCreateInfo.maxBlockCount = 0;
		poolCreateInfo.flags = VMA_POOL_CREATE_IGNORE_BUFFER_IMAGE_GRANULARITY_BIT;

		VmaPool pool;
		vmaCreatePool(allocator, &poolCreateInfo, &pool);
		return pool;
	}

	// Aligns given value down to nearest multiply of align value. For example: VmaAlignUp(11, 8) = 8.
	// Use types like uint32_t, uint64_t as T.
	template<typename T>
	static inline T VmaAlignDown(T val, T align) {
		return val / align * align;
	}

	// Aligns given value up to nearest multiply of align value. For example: VmaAlignUp(11, 8) = 16.
	// Use types like uint32_t, uint64_t as T.
	template<typename T>
	static inline T VmaAlignUp(T val, T align) {
		return (val + align - 1) / align * align;
	}

	void LegacyGPUAllocator::_grow(LegacyLinearAllocator& pool, size_t num_blocks) {
		VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bci.size = pool.block_size * num_blocks;
		bci.usage = (VkBufferUsageFlags)pool.usage;
		bci.queueFamilyIndexCount = queue_family_count;
		bci.sharingMode = bci.queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
		bci.pQueueFamilyIndices = all_queue_families.data();

		VmaAllocationCreateInfo vaci = {};
		vaci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
		vaci.usage = pool.mem_usage;

		VmaAllocation res;
		VmaAllocationInfo vai;

		VkBuffer vkbuffer;

		std::lock_guard _(mutex);

		int best_fit_block_size = 1024;
		int best_fit_index = -1;
		size_t actual_blocks = num_blocks;

		// find best fit allocation
		for (size_t i = 0; i < pool.available_allocation_count; i++) {
			int block_over = (int)pool.available_allocations[i].num_blocks - (int)num_blocks;
			if (block_over >= 0 && block_over < best_fit_block_size) {
				best_fit_block_size = block_over;
				best_fit_index = (int)i;
				if (block_over == 0) {
					break;
				}
			}
		}

		if (best_fit_index == -1) { // no allocation suitable, allocate new one
			auto result = vmaCreateBuffer(allocator, &bci, &vaci, &vkbuffer, &res, &vai);
			assert(result == VK_SUCCESS);
			VkBufferDeviceAddressInfo bdai{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, vkbuffer };
			auto bda = vkGetBufferDeviceAddress(device, &bdai);
			std::byte* mapped_ptr = (std::byte*)vai.pMappedData;
			for (auto i = 0; i < num_blocks; i++) {
				pool.used_allocations[pool.used_allocation_count + i] = { res, vai.deviceMemory,      vai.offset + i * pool.block_size, vkbuffer, mapped_ptr,
					                                                        bda, i > 0 ? 0 : num_blocks };
			}
			buffers.emplace(reinterpret_cast<uint64_t>(vai.deviceMemory), std::tuple(vkbuffer, bci.size, bda));
			pool.current_buffer += (int)num_blocks;
		} else { // we found one, we swap it into the used allocations and compact the available allocations
			std::swap(pool.used_allocations[pool.used_allocation_count], pool.available_allocations[best_fit_index]);
			std::swap(pool.available_allocations[best_fit_index], pool.available_allocations[pool.available_allocation_count - 1]);
			pool.available_allocation_count--;
			for (auto i = 1; i < pool.used_allocations[pool.used_allocation_count].num_blocks; i++) {
				auto& alloc = pool.used_allocations[pool.used_allocation_count];
				// create 1 entry per block in used_allocations
				pool.used_allocations[pool.used_allocation_count + i] = {
					alloc.allocation, alloc.device_memory, alloc.device_memory_offset + i * pool.block_size, alloc.buffer, alloc.mapped_ptr, alloc.bda, 0
				};
			}
			pool.current_buffer += (int)pool.used_allocations[pool.used_allocation_count].num_blocks;
			actual_blocks = pool.used_allocations[pool.used_allocation_count].num_blocks;
		}
		pool.used_allocations[0].base_address = 0;
		for (auto i = 0; i < actual_blocks; i++) {
			if (pool.used_allocation_count + i == 0) {
				continue;
			}
			if (i == 0) { // the first block will have its based_address calculated, the remaining blocks share this address
				for (int j = (int)pool.used_allocation_count - 1; j >= 0; j--) {
					if (pool.used_allocations[j].num_blocks > 0) {
						pool.used_allocations[pool.used_allocation_count].base_address =
						    pool.used_allocations[j].base_address + pool.used_allocations[j].num_blocks * pool.block_size;
						break;
					}
				}
			} else {
				pool.used_allocations[pool.used_allocation_count + i].base_address = pool.used_allocations[pool.used_allocation_count + i - 1].base_address;
			}
		}
		pool.used_allocation_count += actual_blocks;
	}

	// lock-free bump allocation if there is still space
	Buffer LegacyGPUAllocator::_allocate_buffer(LegacyLinearAllocator& pool, size_t size, size_t alignment, bool create_mapped) {
		if (size == 0) {
			return { .buffer = VK_NULL_HANDLE, .size = 0 };
		}

		alignment = std::lcm(pool.mem_reqs.alignment, alignment);
		if (pool.usage & vuk::BufferUsageFlagBits::eUniformBuffer) {
			alignment = std::lcm(alignment, properties.limits.minUniformBufferOffsetAlignment);
		}
		if (pool.usage & vuk::BufferUsageFlagBits::eStorageBuffer) {
			alignment = std::lcm(alignment, properties.limits.minStorageBufferOffsetAlignment);
		}

		uint64_t old_needle = pool.needle.load();
		uint64_t new_needle = VmaAlignUp(old_needle, alignment) + size;
		uint64_t low_buffer = old_needle / pool.block_size;
		uint64_t high_buffer = new_needle / pool.block_size;
		bool is_straddling = low_buffer != high_buffer;
		if (is_straddling) { // boost alignment to place on block start
			new_needle = VmaAlignUp(old_needle, pool.block_size) + size;
			low_buffer = old_needle / pool.block_size;
			high_buffer = new_needle / pool.block_size;
			is_straddling = low_buffer != high_buffer;
		}
		while (!std::atomic_compare_exchange_strong(&pool.needle, &old_needle, new_needle)) { // CAS loop
			old_needle = pool.needle.load();
			new_needle = VmaAlignUp(old_needle, alignment) + size;
			low_buffer = old_needle / pool.block_size;
			high_buffer = new_needle / pool.block_size;
			is_straddling = low_buffer != high_buffer;
			if (is_straddling) { // boost alignment to place on block start
				new_needle = VmaAlignUp(old_needle, pool.block_size) + size;
				low_buffer = old_needle / pool.block_size;
				high_buffer = new_needle / pool.block_size;
				is_straddling = low_buffer != high_buffer;
			}
		}

		uint64_t base = new_needle - size;
		int base_buffer = (int)(base / pool.block_size);
		bool needs_to_create = old_needle == 0 || is_straddling;
		if (needs_to_create) {
			size_t num_blocks = std::max(high_buffer - low_buffer + (old_needle == 0 ? 1 : 0), static_cast<uint64_t>(1));
			while (pool.current_buffer.load() < (int)high_buffer) {
				_grow(pool, num_blocks);
			}
			assert(base % pool.block_size == 0);
		}
		// wait for the buffer to be allocated
		while (pool.current_buffer.load() < (int)high_buffer) {
		};
		auto& current_alloc = pool.used_allocations[base_buffer];
		auto offset = base - current_alloc.base_address;
		Buffer b;
		b.buffer = current_alloc.buffer;
		b.offset = offset;
		b.size = size;
		b.mapped_ptr = current_alloc.mapped_ptr != nullptr ? current_alloc.mapped_ptr + offset : nullptr;
		b.device_address = current_alloc.bda != 0 ? current_alloc.bda + offset : 0;
		b.allocation_size = pool.block_size * current_alloc.num_blocks;

		return b;
	}

	Buffer LegacyGPUAllocator::_allocate_buffer(LegacyPoolAllocator& pool, size_t size, size_t alignment, bool create_mapped) {
		if (size == 0) {
			return { .buffer = VK_NULL_HANDLE, .size = 0 };
		}
		VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bci.size = 1024; // ignored
		bci.usage = (VkBufferUsageFlags)pool.usage;
		bci.queueFamilyIndexCount = queue_family_count;
		bci.sharingMode = bci.queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
		bci.pQueueFamilyIndices = all_queue_families.data();

		VmaAllocationCreateInfo vaci = {};
		vaci.pool = pool.pool;
		if (create_mapped)
			vaci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocation res;
		VmaAllocationInfo vai;
		real_alloc_callback = pool_cb;
		pool_helper->bci = bci;
		pool_helper->result = VK_NULL_HANDLE;
		auto mem_reqs = pool.mem_reqs;
		mem_reqs.size = size;
		mem_reqs.alignment = std::max(static_cast<size_t>(1), mem_reqs.alignment);
		alignment = std::max(static_cast<size_t>(1), alignment);
		mem_reqs.alignment = std::lcm(mem_reqs.alignment, alignment);
		VkMemoryRequirements vkmem_reqs = mem_reqs;
		auto result = vmaAllocateMemory(allocator, &vkmem_reqs, &vaci, &res, &vai);
		real_alloc_callback = noop_cb;
		if (result != VK_SUCCESS && size >= large_allocation_size) { // too large to fit in a pool, we allocate a dedicated devmem
			vaci.pool = VK_NULL_HANDLE;
			vaci.memoryTypeBits = pool.mem_reqs.memoryTypeBits;
			vaci.usage = (VmaMemoryUsage)to_integral(pool.mem_usage);
			vaci.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
			result = vmaAllocateMemory(allocator, &vkmem_reqs, &vaci, &res, &vai);
			assert(result == VK_SUCCESS);
			Buffer b;
			bci.size = size;
			vkCreateBuffer(device, &bci, nullptr, &b.buffer);
			vkBindBufferMemory(device, b.buffer, vai.deviceMemory, vai.offset);
			VkBufferDeviceAddressInfo bdai{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, b.buffer };
			uint64_t device_address = vkGetBufferDeviceAddress(device, &bdai);
			buffers.emplace(reinterpret_cast<uint64_t>(vai.deviceMemory), std::tuple(b.buffer, bci.size, device_address));
			b.offset = 0;
			b.size = bci.size;
			b.mapped_ptr = (std::byte*)vai.pMappedData;
			b.device_address = device_address + vai.offset;
			b.allocation_size = bci.size;
			buffer_allocations.emplace(BufferID{ reinterpret_cast<uint64_t>(b.buffer), b.offset }, res);
			return b;
		}
		assert(result == VK_SUCCESS);

		// record if new buffer was used
		if (pool_helper->result != VK_NULL_HANDLE) {
			buffers.emplace(reinterpret_cast<uint64_t>(vai.deviceMemory), std::tuple(pool_helper->result, pool_helper->bci.size, pool_helper->device_address));
			pool.buffers.emplace_back(pool_helper->result);
		}
		Buffer b;
		auto [vkbuffer, allocation_size, device_address] = buffers.at(reinterpret_cast<uint64_t>(vai.deviceMemory));
		b.buffer = vkbuffer;
		b.offset = vai.offset;
		b.size = vai.size;
		b.mapped_ptr = (std::byte*)vai.pMappedData;
		b.device_address = device_address + vai.offset;
		b.allocation_size = allocation_size;
		buffer_allocations.emplace(BufferID{ reinterpret_cast<uint64_t>(b.buffer), b.offset }, res);
		return b;
	}

	VkMemoryRequirements LegacyGPUAllocator::get_memory_requirements(VkBufferCreateInfo& bci) {
		VkBuffer testbuff;
		vkCreateBuffer(device, &bci, nullptr, &testbuff);
		VkMemoryRequirements mem_reqs;
		vkGetBufferMemoryRequirements(device, testbuff, &mem_reqs);
		vkDestroyBuffer(device, testbuff, nullptr);
		return mem_reqs;
	}

	// allocate an externally managed pool
	LegacyPoolAllocator LegacyGPUAllocator::allocate_pool(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage) {
		std::lock_guard _(mutex);

		VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bci.size = 1024; // ignored
		bci.usage = (VkBufferUsageFlags)buffer_usage;
		bci.queueFamilyIndexCount = queue_family_count;
		bci.sharingMode = bci.queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
		bci.pQueueFamilyIndices = all_queue_families.data();

		LegacyPoolAllocator pi;
		pi.mem_reqs = get_memory_requirements(bci);
		pi.pool = _create_pool(mem_usage, buffer_usage);
		pi.mem_usage = mem_usage;
		pi.usage = buffer_usage;
		return pi;
	}

	LegacyLinearAllocator LegacyGPUAllocator::allocate_linear(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage) {
		std::lock_guard _(mutex);

		VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bci.size = 1024; // ignored
		bci.usage = (VkBufferUsageFlags)buffer_usage;
		bci.queueFamilyIndexCount = queue_family_count;
		bci.sharingMode = bci.queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
		bci.pQueueFamilyIndices = all_queue_families.data();

		return LegacyLinearAllocator{ get_memory_requirements(bci), VmaMemoryUsage(to_integral(mem_usage)), buffer_usage };
	}

	// allocate buffer from an internally managed pool
	Buffer LegacyGPUAllocator::allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment, bool create_mapped) {
		std::lock_guard _(mutex);

		VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bci.size = 1024; // ignored
		bci.usage = (VkBufferUsageFlags)buffer_usage;
		bci.queueFamilyIndexCount = queue_family_count;
		bci.sharingMode = bci.queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
		bci.pQueueFamilyIndices = all_queue_families.data();

		auto pool_it = pools.find(PoolSelect{ mem_usage, buffer_usage });
		if (pool_it == pools.end()) {
			LegacyPoolAllocator pi;

			pi.mem_reqs = get_memory_requirements(bci);
			pi.pool = _create_pool(mem_usage, buffer_usage);
			pi.usage = buffer_usage;
			pi.mem_usage = mem_usage;
			pool_it = pools.emplace(PoolSelect{ mem_usage, buffer_usage }, pi).first;
		}

		return _allocate_buffer(pool_it->second, size, alignment, create_mapped);
	}

	// allocate a buffer from an externally managed pool
	Buffer LegacyGPUAllocator::allocate_buffer(LegacyPoolAllocator& pool, size_t size, size_t alignment, bool create_mapped) {
		std::lock_guard _(mutex);
		return _allocate_buffer(pool, size, alignment, create_mapped);
	}

	// allocate a buffer from an externally managed linear pool
	Buffer LegacyGPUAllocator::allocate_buffer(LegacyLinearAllocator& pool, size_t size, size_t alignment, bool create_mapped) {
		return _allocate_buffer(pool, size, alignment, create_mapped);
	}

	size_t LegacyGPUAllocator::get_allocation_size(const Buffer& b) {
		return b.allocation_size;
	}

	void LegacyGPUAllocator::reset_pool(LegacyPoolAllocator& pool) {
		std::lock_guard _(mutex);
		vmaResetPool(allocator, pool.pool);
	}

	void LegacyGPUAllocator::reset_pool(LegacyLinearAllocator& pool) {
		std::lock_guard _(mutex);
		for (size_t i = 0; i < pool.used_allocation_count;) {
			pool.available_allocations[pool.available_allocation_count++] = pool.used_allocations[i];
			i += pool.used_allocations[i].num_blocks;
		}
		pool.used_allocations = {};
		pool.used_allocation_count = 0;
		pool.current_buffer = -1;
		pool.needle = 0;
	}

	// we just destroy the buffers that we have left in the available allocations
	void LegacyGPUAllocator::trim_pool(LegacyLinearAllocator& pool) {
		std::lock_guard _(mutex);
		for (size_t i = 0; i < pool.available_allocation_count; i++) {
			auto& alloc = pool.available_allocations[i];
			vmaDestroyBuffer(allocator, alloc.buffer, alloc.allocation);
		}
		pool.available_allocation_count = 0;
	}

	void LegacyGPUAllocator::free_buffer(const Buffer& b) {
		std::lock_guard _(mutex);
		vuk::BufferID bufid{ reinterpret_cast<uint64_t>(b.buffer), b.offset };
		vmaFreeMemory(allocator, buffer_allocations.at(bufid));
		buffer_allocations.erase(bufid);
		if (b.offset == 0 && b.size > large_allocation_size) {
			vkDestroyBuffer(device, b.buffer, nullptr);
		}
	}

	void LegacyGPUAllocator::destroy(const LegacyPoolAllocator& pool) {
		std::lock_guard _(mutex);
		vmaResetPool(allocator, pool.pool);
		vmaForceUnmapPool(allocator, pool.pool);
		for (auto& buffer : pool.buffers) {
			vkDestroyBuffer(device, buffer, nullptr);
		}
		vmaDestroyPool(allocator, pool.pool);
	}

	void LegacyGPUAllocator::destroy(const LegacyLinearAllocator& pool) {
		std::lock_guard _(mutex);
		for (size_t i = 0; i < pool.used_allocation_count; i++) {
			auto& [va, mem, offset, buffer, map, bda, nb_blocks, ba] = pool.used_allocations[i];
			if (va) {
				vmaDestroyBuffer(allocator, buffer, va);
			}
		}
		for (size_t i = 0; i < pool.available_allocation_count; i++) {
			auto& [va, mem, offset, buffer, map, bda, nb_blocks, ba] = pool.available_allocations[i];
			if (va) {
				vmaDestroyBuffer(allocator, buffer, va);
			}
		}
	}

	vuk::Image LegacyGPUAllocator::create_image_for_rendertarget(vuk::ImageCreateInfo ici) {
		std::lock_guard _(mutex);
		VmaAllocationCreateInfo db{};
		db.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
		db.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		db.requiredFlags = 0;
		db.preferredFlags = 0;
		db.pool = nullptr;
		VkImage vkimg;
		VmaAllocation vout;
		VkImageCreateInfo vkici = ici;
		VmaAllocationInfo vai;
		auto result = vmaCreateImage(allocator, &vkici, &db, &vkimg, &vout, &vai);
		assert(result == VK_SUCCESS);
		images.emplace(reinterpret_cast<uint64_t>(vkimg), vout);
		return vkimg;
	}
	vuk::Image LegacyGPUAllocator::create_image(vuk::ImageCreateInfo ici) {
		std::lock_guard _(mutex);
		VmaAllocationCreateInfo db{};
		db.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
		db.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		db.requiredFlags = 0;
		db.preferredFlags = 0;
		db.pool = nullptr;
		VkImage vkimg;
		VmaAllocation vout;
		VkImageCreateInfo vkici = ici;
		auto result = vmaCreateImage(allocator, &vkici, &db, &vkimg, &vout, nullptr);
		assert(result == VK_SUCCESS);
		images.emplace(reinterpret_cast<uint64_t>(vkimg), vout);
		return vkimg;
	}
	void LegacyGPUAllocator::destroy_image(vuk::Image image) {
		std::lock_guard _(mutex);
		auto vkimg = (VkImage)image;
		vmaDestroyImage(allocator, image, images.at(reinterpret_cast<uint64_t>(vkimg)));
		images.erase(reinterpret_cast<uint64_t>(vkimg));
	}
	LegacyGPUAllocator::~LegacyGPUAllocator() {
		for (auto& [ps, p] : pools) {
			destroy(p);
		}
		vmaDestroyAllocator(allocator);
	}
} // namespace vuk
