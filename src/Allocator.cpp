#include "Allocator.hpp"
#include <string>

namespace vuk {
	PFN_vmaAllocateDeviceMemoryFunction Allocator::real_alloc_callback = nullptr;

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

	void Allocator::pool_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size, void* userdata) {
		auto& pags = *reinterpret_cast<PoolAllocHelper*>(userdata);
		pags.bci.size = size;
		auto buffer = pags.device.createBuffer(pags.bci);
		pags.device.bindBufferMemory(buffer, memory, 0);
		pags.result = buffer;

		std::string devmem_name = "DeviceMemory (Pool [" + std::to_string(memoryType) + "] " + to_human_readable(size) + ")";
		std::string buffer_name = "Buffer (Pool ";
		buffer_name += vk::to_string(pags.bci.usage);
		buffer_name += ")";
		
		{
			VkDebugUtilsObjectNameInfoEXT info;
			info.pNext = nullptr;
			info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
			info.pObjectName = devmem_name.c_str();
			info.objectType = (VkObjectType)vk::DeviceMemory::objectType;
			info.objectHandle = reinterpret_cast<uint64_t>(memory);
			pags.setDebugUtilsObjectNameEXT(pags.device, &info);
		}
		{
			VkDebugUtilsObjectNameInfoEXT info;
			info.pNext = nullptr;
			info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
			info.pObjectName = buffer_name.c_str();
			info.objectType = (VkObjectType)vk::Buffer::objectType;
			info.objectHandle = reinterpret_cast<uint64_t>((VkBuffer)buffer);
			pags.setDebugUtilsObjectNameEXT(pags.device, &info);
		}
	}

	void Allocator::noop_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size, void* userdata) {
		auto& pags = *reinterpret_cast<PoolAllocHelper*>(userdata);
		std::string devmem_name = "DeviceMemory (Dedicated [" + std::to_string(memoryType) + "] " + to_human_readable(size) + ")";
		{
			VkDebugUtilsObjectNameInfoEXT info;
			info.pNext = nullptr;
			info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
			info.pObjectName = devmem_name.c_str();
			info.objectType = (VkObjectType)vk::DeviceMemory::objectType;
			info.objectHandle = reinterpret_cast<uint64_t>(memory);
			pags.setDebugUtilsObjectNameEXT(pags.device, &info);
		}

	}

	Allocator::Allocator(vk::Instance instance, vk::Device device, vk::PhysicalDevice phys_dev) : device(device), physdev(phys_dev) {
		VmaAllocatorCreateInfo allocatorInfo = {};
		allocatorInfo.instance = instance;
		allocatorInfo.physicalDevice = phys_dev;
		allocatorInfo.device = device;
		allocatorInfo.flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;

		VmaDeviceMemoryCallbacks cbs;
		cbs.pfnFree = nullptr;
		real_alloc_callback = noop_cb;
		cbs.pfnAllocate = allocation_cb;
		pool_helper = std::make_unique<PoolAllocHelper>();
		pool_helper->setDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT");
		cbs.pUserData = pool_helper.get();
		allocatorInfo.pDeviceMemoryCallbacks = &cbs;

		vmaCreateAllocator(&allocatorInfo, &allocator);

		pool_helper->device = device;
	}

	// not locked, must be called from a locked fn

	VmaPool Allocator::_create_pool(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage) {
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
	
	Buffer Allocator::_allocate_buffer(Pool& pool, size_t size, bool create_mapped) {
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
		pool_helper->bci = bci;
		pool_helper->result = vk::Buffer{};
		auto mem_reqs = pool.mem_reqs;
		mem_reqs.size = size;
		VkMemoryRequirements vkmem_reqs = mem_reqs;
		auto result = vmaAllocateMemory(allocator, &vkmem_reqs, &vaci, &res, &vai);
		assert(result == VK_SUCCESS);
		real_alloc_callback = noop_cb;

		// record if new buffer was used
		if (pool_helper->result != vk::Buffer{}) {
			buffers.emplace(reinterpret_cast<uint64_t>(vai.deviceMemory), pool_helper->result);
			pool.buffers.emplace_back(pool_helper->result);
		}
		Buffer b;
		b.buffer = buffers.at(reinterpret_cast<uint64_t>(vai.deviceMemory));
		b.device_memory = vai.deviceMemory;
		b.offset = vai.offset;
		b.size = vai.size;
		b.mapped_ptr = vai.pMappedData;
		buffer_allocations.emplace(BufferID{ reinterpret_cast<uint64_t>((VkBuffer)b.buffer), b.offset }, res);
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
	Buffer Allocator::allocate_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, size_t size, bool create_mapped) {
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
	Buffer Allocator::allocate_buffer(Pool& pool, size_t size, bool create_mapped) {
		std::lock_guard _(mutex);
		return _allocate_buffer(pool, size, create_mapped);
	}

	void Allocator::reset_pool(Pool pool) {
		std::lock_guard _(mutex);
		vmaResetPool(allocator, pool.pool);
	}

	void Allocator::free_buffer(const Buffer& b) {
		std::lock_guard _(mutex);
		vuk::BufferID bufid{ reinterpret_cast<uint64_t>((VkBuffer)b.buffer), b.offset };
		vmaFreeMemory(allocator, buffer_allocations.at(bufid));
		buffer_allocations.erase(bufid);
	}

	void Allocator::destroy_pool(Pool pool) {
		std::lock_guard _(mutex);
		vmaResetPool(allocator, pool.pool);
		vmaForceUnmapPool(allocator, pool.pool);
		for (auto& buffer : pool.buffers) {
			device.destroy(buffer);
		}
		vmaDestroyPool(allocator, pool.pool);
	}

	vk::Image Allocator::create_image_for_rendertarget(vk::ImageCreateInfo ici) {
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
		images.emplace(reinterpret_cast<uint64_t>(vkimg), vout);
		return vkimg;
	}
	vk::Image Allocator::create_image(vk::ImageCreateInfo ici) {
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
		vmaCreateImage(allocator, &vkici, &db, &vkimg, &vout, nullptr);
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
		for (auto& [ps, p] : pools) {
			destroy_pool(p);
		}
		vmaDestroyAllocator(allocator);
	}
}
