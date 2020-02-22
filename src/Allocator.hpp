#pragma once

#include "vk_mem_alloc.h"
#include <vulkan/vulkan.hpp>
#include <mutex>
#include <unordered_map>

class Allocator {
	static std::mutex pool_mutex;
	struct PoolAllocGlobalState {
		vk::Device device;
		vk::Buffer buffer;
		vk::BufferCreateInfo bci;
	};
	static PoolAllocGlobalState pags;

	static void pool_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size) {
		pags.buffer = pags.device.createBuffer(pags.bci);
		pags.device.bindBufferMemory(pags.buffer, memory, 0);
		printf("Pool allocated\n");
	};

	static void noop_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size){}

	static PFN_vmaAllocateDeviceMemoryFunction real_alloc_callback;

	static void allocation_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size) {
		real_alloc_callback(allocator, memoryType, memory, size);
	}

public:
	Allocator(vk::Device device, vk::PhysicalDevice phys_dev) {
		VmaAllocatorCreateInfo allocatorInfo = {};
		allocatorInfo.physicalDevice = phys_dev;
		allocatorInfo.device = device;
		
		VmaDeviceMemoryCallbacks cbs;
		cbs.pfnFree = nullptr;
		real_alloc_callback = noop_cb;
		cbs.pfnAllocate = allocation_cb;
		allocatorInfo.pDeviceMemoryCallbacks = &cbs;

		vmaCreateAllocator(&allocatorInfo, &allocator);
	}

	void create_pool() {
		std::lock_guard _(pool_mutex);
		real_alloc_callback = pool_cb;
		// Create a pool that can have at most 2 blocks, 128 MiB each.
		VmaPoolCreateInfo poolCreateInfo = {};
		VkBufferCreateInfo exampleBufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		exampleBufCreateInfo.size = 1024; // Whatever.
		exampleBufCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT; // Change if needed.

		VmaAllocationCreateInfo allocCreateInfo = {};
		allocCreateInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU; // Change if needed.

		uint32_t memTypeIndex;
		vmaFindMemoryTypeIndexForBufferInfo(allocator, &exampleBufCreateInfo, &allocCreateInfo, &memTypeIndex);
		poolCreateInfo.memoryTypeIndex = memTypeIndex;
		poolCreateInfo.blockSize = 128ull * 1024 * 1024;
		poolCreateInfo.maxBlockCount = 2;
		// TODO: set up PAGS

		VmaPool pool;
		vmaCreatePool(allocator, &poolCreateInfo, &pool);

		auto testbuff = device.createBuffer(exampleBufCreateInfo);
		auto memrq = (VkMemoryRequirements)device.getBufferMemoryRequirements(testbuff);
		device.destroy(testbuff);
	}

	std::unordered_map<uint64_t, VmaAllocation> images;

	vk::Image create_image_for_rendertarget(vk::ImageCreateInfo ici) {
		VmaAllocationCreateInfo db;
		db.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
		db.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		db.requiredFlags = 0;
		db.preferredFlags = 0;
		db.pool = nullptr;
		VkImage vkimg;
		VmaAllocation vout;
		vmaCreateImage(allocator, &(VkImageCreateInfo)ici, &db, &vkimg, &vout, nullptr);
		images.emplace(reinterpret_cast<uint64_t>(vkimg), vout);
		return vkimg;
	}

	void destroy_image(vk::Image image) {
		auto vkimg = (VkImage)image;
		vmaDestroyImage(allocator, image, images.at(reinterpret_cast<uint64_t>(vkimg)));
	}
	
	vk::Device device;
	vk::PhysicalDevice physdev;

	VmaAllocator allocator;

	struct PoolInfo {
		VmaPool pool;
		vk::MemoryRequirements mem_reqs;
		vk::Buffer buffer;
	};
	std::vector<PoolInfo> pools;

	~Allocator() {
		for (auto& p : pools) {
			device.destroy(p.buffer);
			vmaDestroyPool(allocator, p.pool);
		}
		vmaDestroyAllocator(allocator);
	}
};
