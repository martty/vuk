#include "Allocator.hpp"
namespace vuk {
	PFN_vmaAllocateDeviceMemoryFunction Allocator::real_alloc_callback = nullptr;
	Allocator::PoolAllocGlobalState Allocator::pags;
	std::mutex Allocator::mutex;

	Allocator::Allocator(vk::Device device, vk::PhysicalDevice phys_dev) : device(device), physdev(phys_dev) {
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

	// not locked, must be called from a locked fn

	VmaPool Allocator::_create_pool(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage) {
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
		poolCreateInfo.flags = VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT;

		VmaPool pool;
		vmaCreatePool(allocator, &poolCreateInfo, &pool);
		return pool;
	}
	Allocator::Buffer Allocator::_allocate_buffer(Pool& pool, size_t size, bool create_mapped) {
		if (size == 0) {
			return { .buffer = vk::Buffer{}, .size = 0 };
		}
		vk::BufferCreateInfo bci;
		bci.size = 1024; // ignored
		bci.usage = pool.usage;

		VmaAllocationCreateInfo vaci = {};
		vaci.pool = pool.pool;
		if (create_mapped)
			vaci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocation res;
		VmaAllocationInfo vai;
		real_alloc_callback = pool_cb;
		pags.bci = bci;
		auto mem_reqs = pool.mem_reqs;
		mem_reqs.size = size;
		auto result = vmaAllocateMemory(allocator, &(VkMemoryRequirements)mem_reqs, &vaci, &res, &vai);
		assert(result == VK_SUCCESS);
		real_alloc_callback = noop_cb;
		if (pags.buffer != vk::Buffer{}) {
			// TODO: this breaks if we allocate multiple memories for a pool
			// we need a devicememory -> buffer mapping to figure out which vk::Buffer we got
			pool.buffer = pags.buffer;
			pags.buffer = vk::Buffer{};
		}
		Buffer b;
		b.buffer = pool.buffer;
		b.device_memory = vai.deviceMemory;
		b.offset = vai.offset;
		b.size = vai.size;
		b.mapped_ptr = vai.pMappedData;
		b.allocation = res;
		return b;
	}

	// allocate an externally managed pool

	Allocator::Pool Allocator::allocate_pool(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage) {
		std::lock_guard _(mutex);

		vk::BufferCreateInfo bci;
		bci.size = 1024; // ignored
		bci.usage = buffer_usage;

		Pool pi;
		auto testbuff = device.createBuffer(bci);
		pi.mem_reqs = (VkMemoryRequirements)device.getBufferMemoryRequirements(testbuff);
		device.destroy(testbuff);
		pi.pool = _create_pool(mem_usage, buffer_usage);
		pi.usage = buffer_usage;
		return pi;
	}

	// allocate buffer from an internally managed pool

	Allocator::Buffer Allocator::allocate_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, size_t size, bool create_mapped) {
		std::lock_guard _(mutex);

		vk::BufferCreateInfo bci;
		bci.size = 1024; // ignored
		bci.usage = buffer_usage;

		auto pool_it = pools.find(PoolSelect{ mem_usage, buffer_usage });
		if (pool_it == pools.end()) {
			Pool pi;
			auto testbuff = device.createBuffer(bci);
			pi.mem_reqs = (VkMemoryRequirements)device.getBufferMemoryRequirements(testbuff);
			device.destroy(testbuff);
			pi.pool = _create_pool(mem_usage, buffer_usage);
			pi.usage = buffer_usage;
			pool_it = pools.emplace(PoolSelect{ mem_usage, buffer_usage }, pi).first;
		}

		return _allocate_buffer(pool_it->second, size, create_mapped);
	}

	// allocate a buffer from an externally managed pool
	Allocator::Buffer Allocator::allocate_buffer(Pool& pool, size_t size, bool create_mapped) {
		std::lock_guard _(mutex);
		return _allocate_buffer(pool, size, create_mapped);
	}
	void Allocator::reset_pool(Pool pool) {
		std::lock_guard _(mutex);
		vmaResetPool(allocator, pool.pool);
	}
	void Allocator::free_buffer(const Buffer& b) {
		std::lock_guard _(mutex);
		vmaFreeMemory(allocator, b.allocation);
	}
	void Allocator::destroy_scratch_pool(Pool pool) {
		std::lock_guard _(mutex);
		vmaResetPool(allocator, pool.pool);
		vmaForceUnmapPool(allocator, pool.pool);
		device.destroy(pool.buffer);
		vmaDestroyPool(allocator, pool.pool);
	}
	vk::Image Allocator::create_image_for_rendertarget(vk::ImageCreateInfo ici) {
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
	vk::Image Allocator::create_image(vk::ImageCreateInfo ici) {
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
	void Allocator::destroy_image(vk::Image image) {
		std::lock_guard _(mutex);
		auto vkimg = (VkImage)image;
		vmaDestroyImage(allocator, image, images.at(reinterpret_cast<uint64_t>(vkimg)));
		images.erase(reinterpret_cast<uint64_t>(vkimg));
	}
	Allocator::~Allocator() {
		for (auto [img, alloc] : images) {
			vmaDestroyImage(allocator, (VkImage)img, alloc);
		}

		for (auto& [ps, pi] : pools) {
			device.destroy(pi.buffer);
			vmaDestroyPool(allocator, pi.pool);
		}
		vmaDestroyAllocator(allocator);
	}
}
