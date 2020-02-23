#pragma once

#include "vk_mem_alloc.h"
#include <vulkan/vulkan.hpp>
#include <mutex>
#include <unordered_map>
#include "Hash.hpp"
#include "Cache.hpp" // for the hashes

enum class MemoryUsage {
	eGPUonly = VMA_MEMORY_USAGE_GPU_ONLY,
	eCPUtoGPU = VMA_MEMORY_USAGE_CPU_TO_GPU,
	eCPUonly = VMA_MEMORY_USAGE_CPU_ONLY,
};

struct PoolSelect {
	MemoryUsage mem_usage;
	vk::BufferUsageFlags buffer_usage;

	bool operator==(const PoolSelect& o) const {
		return std::tie(mem_usage, buffer_usage) == std::tie(o.mem_usage, o.buffer_usage);
	}
};

namespace std {
	template <>
	struct hash<PoolSelect> {
		size_t operator()(PoolSelect const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, to_integral(x.mem_usage), x.buffer_usage);
			return h;
		}
	};
};


class Allocator {
	static std::mutex mutex;
	struct PoolAllocGlobalState {
		vk::Device device;
		vk::Buffer buffer = {};
		vk::BufferCreateInfo bci;
	};
	static PoolAllocGlobalState pags;

	static void pool_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size) {
		pags.bci.size = size;
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
	Allocator(vk::Device device, vk::PhysicalDevice phys_dev) : device(device), physdev(phys_dev) {
		VmaAllocatorCreateInfo allocatorInfo = {};
		allocatorInfo.physicalDevice = phys_dev;
		allocatorInfo.device = device;
		allocatorInfo.flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;
		
		VmaDeviceMemoryCallbacks cbs;
		cbs.pfnFree = nullptr;
		real_alloc_callback = noop_cb;
		cbs.pfnAllocate = allocation_cb;
		allocatorInfo.pDeviceMemoryCallbacks = &cbs;

		vmaCreateAllocator(&allocatorInfo, &allocator);

		pags.device = device;
	}


private:
	// not locked, must be called from a locked fn
	VmaPool create_pool(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage) {
		real_alloc_callback = pool_cb;
		// Create a pool that can have at most 2 blocks, 128 MiB each.
		VmaPoolCreateInfo poolCreateInfo = {};
		VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bci.size = 1024; // Whatever.
		bci.usage = (VkBufferUsageFlags)buffer_usage; // Change if needed.

		VmaAllocationCreateInfo allocCreateInfo = {};
		allocCreateInfo.usage = VmaMemoryUsage(to_integral(mem_usage));

		uint32_t memTypeIndex;
		vmaFindMemoryTypeIndexForBufferInfo(allocator, &bci, &allocCreateInfo, &memTypeIndex);
		poolCreateInfo.memoryTypeIndex = memTypeIndex;
		poolCreateInfo.blockSize = 0;
		poolCreateInfo.maxBlockCount = 0;

		VmaPool pool;
		vmaCreatePool(allocator, &poolCreateInfo, &pool);
		return pool;
	}
public:
	struct PoolInfo {
		VmaPool pool;
		vk::MemoryRequirements mem_reqs;
		vk::Buffer buffer;
	};

	std::unordered_map<PoolSelect, PoolInfo> pools;

	struct Buffer {
		vk::DeviceMemory device_memory;
		vk::Buffer buffer;
		size_t offset;
		size_t size;
		void* mapped_ptr;
		VmaAllocation allocation;
	};

	Buffer allocate_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, size_t size, bool create_mapped) {
		std::lock_guard _(mutex);

		vk::BufferCreateInfo bci;
		bci.size = 1024; // ignored
		bci.usage = buffer_usage;

		auto pool_it = pools.find(PoolSelect{ mem_usage, buffer_usage });
		if (pool_it == pools.end()) {
			PoolInfo pi;
			auto testbuff = device.createBuffer(bci);
			pi.mem_reqs = (VkMemoryRequirements)device.getBufferMemoryRequirements(testbuff);
			device.destroy(testbuff);
			pi.pool = create_pool(mem_usage, buffer_usage);
			pool_it = pools.emplace(PoolSelect{ mem_usage, buffer_usage }, pi).first;
		}

		VmaAllocationCreateInfo vaci;
		vaci.pool = pool_it->second.pool;
		if(create_mapped)
			vaci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocation res;
		VmaAllocationInfo vai;
		real_alloc_callback = pool_cb;
		pags.bci = bci;
		auto mem_reqs = pool_it->second.mem_reqs;
		mem_reqs.size = size;
		vmaAllocateMemory(allocator, &(VkMemoryRequirements)mem_reqs, &vaci, &res, &vai);
		real_alloc_callback = noop_cb;
		if (pags.buffer != vk::Buffer{}) {
			// TODO: this breaks if we allocate multiple memories for a pool
			pool_it->second.buffer = pags.buffer;
			pags.buffer = vk::Buffer{};
		}
		Buffer b;
		b.buffer = pool_it->second.buffer;
		b.device_memory = vai.deviceMemory;
		b.offset = vai.offset;
		b.size = vai.size;
		b.mapped_ptr = vai.pMappedData;
		b.allocation = res;
		return b;
	}

	void free_buffer(const Buffer& b) {
		std::lock_guard _(mutex);
		vmaFreeMemory(allocator, b.allocation);
	}

	std::unordered_map<uint64_t, VmaAllocation> images;

	vk::Image create_image_for_rendertarget(vk::ImageCreateInfo ici) {
		std::lock_guard _(mutex);
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
		std::lock_guard _(mutex);
		auto vkimg = (VkImage)image;
		vmaDestroyImage(allocator, image, images.at(reinterpret_cast<uint64_t>(vkimg)));
		images.erase(reinterpret_cast<uint64_t>(vkimg));
	}
	
	vk::Device device;
	vk::PhysicalDevice physdev;

	VmaAllocator allocator;

	~Allocator() {
		for (auto& [ps, pi] : pools) {
			device.destroy(pi.buffer);
			vmaDestroyPool(allocator, pi.pool);
		}
		vmaDestroyAllocator(allocator);
	}
};
