#include "Allocator.hpp"
#include "vuk/Allocator.hpp"
#include "vuk/Context.hpp"
#include <string>
#include <numeric>

namespace vuk {
	Direct::Direct(Context& ctx, Allocator& alloc) : ctx(&ctx), device(ctx.device), gpumem(&alloc) {}
	RingFrame::RingFrame(Context& ctx, uint64_t frames_in_flight) : direct(ctx, ctx.get_gpumem()), frames_in_flight(frames_in_flight) {
		frames.resize(frames_in_flight, FrameResource{ ctx.device, *this });
	}
	NLinear::NLinear(VkResource& upstream, SyncScope scope) : ctx(&upstream.get_context()), device(ctx->device), scope(scope), VkResourceNested(&upstream) {}

	PFN_vmaAllocateDeviceMemoryFunction Allocator::real_alloc_callback = nullptr;

	std::string to_string(BufferUsageFlags value) {
		if (!value) return "{}";
		std::string result;

		if (value & BufferUsageFlagBits::eTransferSrc) result += "TransferSrc | ";
		if (value & BufferUsageFlagBits::eTransferDst) result += "TransferDst | ";
		if (value & BufferUsageFlagBits::eUniformTexelBuffer) result += "UniformTexelBuffer | ";
		if (value & BufferUsageFlagBits::eStorageTexelBuffer) result += "StorageTexelBuffer | ";
		if (value & BufferUsageFlagBits::eUniformBuffer) result += "UniformBuffer | ";
		if (value & BufferUsageFlagBits::eStorageBuffer) result += "StorageBuffer | ";
		if (value & BufferUsageFlagBits::eIndexBuffer) result += "IndexBuffer | ";
		if (value & BufferUsageFlagBits::eVertexBuffer) result += "VertexBuffer | ";
		if (value & BufferUsageFlagBits::eIndirectBuffer) result += "IndirectBuffer | ";
		if (value & BufferUsageFlagBits::eShaderDeviceAddress) result += "ShaderDeviceAddress | ";
		if (value & BufferUsageFlagBits::eTransformFeedbackBufferEXT) result += "TransformFeedbackBufferEXT | ";
		if (value & BufferUsageFlagBits::eTransformFeedbackCounterBufferEXT) result += "TransformFeedbackCounterBufferEXT | ";
		if (value & BufferUsageFlagBits::eConditionalRenderingEXT) result += "ConditionalRenderingEXT | ";
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

	void Allocator::pool_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size, void* userdata) {
		auto& pags = *reinterpret_cast<PoolAllocHelper*>(userdata);
		pags.bci.size = size;
		VkBuffer buffer;
		vkCreateBuffer(pags.device, &pags.bci, nullptr, &buffer);
		vkBindBufferMemory(pags.device, buffer, memory, 0);
		pags.result = buffer;

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
			pags.setDebugUtilsObjectNameEXT(pags.device, &info);
		}
		{
			VkDebugUtilsObjectNameInfoEXT info;
			info.pNext = nullptr;
			info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
			info.pObjectName = buffer_name.c_str();
			info.objectType = VK_OBJECT_TYPE_BUFFER;
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
			info.objectType = VK_OBJECT_TYPE_DEVICE_MEMORY;
			info.objectHandle = reinterpret_cast<uint64_t>(memory);
			pags.setDebugUtilsObjectNameEXT(pags.device, &info);
		}

	}

	Allocator::Allocator(VkInstance instance, VkDevice device, VkPhysicalDevice phys_dev, uint32_t graphics_queue_family, uint32_t transfer_queue_family) : device(device) {
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
		vkGetPhysicalDeviceProperties(phys_dev, &properties);

		pool_helper->device = device;

		if(transfer_queue_family != graphics_queue_family) {
			all_queue_families = { graphics_queue_family, transfer_queue_family };
		} else {
			all_queue_families = { graphics_queue_family };
		}
		queue_family_count = (uint32_t)all_queue_families.size();
	}

	// not locked, must be called from a locked fn

	VmaPool Allocator::_create_pool(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage) {
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

	// lock-free bump allocation if there is still space
	Buffer Allocator::_allocate_buffer(LinearAllocator& pool, size_t size, size_t alignment, bool create_mapped) {
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

		if ((size + alignment) > pool.block_size) {
			// we are not handling sizes bigger than the block_size
			// we could allocate a buffer that is multiple block_sizes big
			// and fake the entries, but for now this is too much complexity
			return allocate_buffer((vuk::MemoryUsage)pool.mem_usage, pool.usage, size, alignment, create_mapped);
		}

		auto new_needle = pool.needle.fetch_add(size + alignment) + size + alignment;
		auto base_addr = new_needle - size - alignment;

		size_t buffer = new_needle / pool.block_size;
		bool needs_to_create = base_addr == 0 || (base_addr / pool.block_size != new_needle / pool.block_size);
		if (needs_to_create) {
			VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
			bci.size = pool.block_size;
			bci.usage = (VkBufferUsageFlags)pool.usage;
			bci.queueFamilyIndexCount = queue_family_count;
			bci.sharingMode = bci.queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
			bci.pQueueFamilyIndices = all_queue_families.data();

			VmaAllocationCreateInfo vaci = {};
			if (create_mapped)
				vaci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
			vaci.usage = pool.mem_usage;

			VmaAllocation res;
			VmaAllocationInfo vai;

			auto mem_reqs = pool.mem_reqs;
			mem_reqs.size = size;
			VkBuffer vkbuffer;

			auto next_index = pool.current_buffer.load() + 1;
			if (std::get<VkBuffer>(pool.allocations[next_index]) == VK_NULL_HANDLE) {
				std::lock_guard _(mutex);
				auto result = vmaCreateBuffer(allocator, &bci, &vaci, &vkbuffer, &res, &vai);
				assert(result == VK_SUCCESS);
				pool.allocations[next_index] =
					std::tuple(res, vai.deviceMemory, vai.offset, vkbuffer, (std::byte*)vai.pMappedData);
                buffers.emplace(reinterpret_cast<uint64_t>(vai.deviceMemory), std::pair(vkbuffer, bci.size));
			}
			pool.current_buffer++;
			if (base_addr > 0) {
				// there is no space in the beginning of this allocation, so we just retry
				return _allocate_buffer(pool, size, alignment, create_mapped);
			}
		}
		// wait for the buffer to be allocated
		while (pool.current_buffer.load() < buffer) {};
		auto offset = VmaAlignDown(new_needle - size, alignment) % pool.block_size;
		auto& current_alloc = pool.allocations[buffer];
		Buffer b;
		b.buffer = std::get<VkBuffer>(current_alloc);
		b.device_memory = std::get<VkDeviceMemory>(current_alloc);
		b.offset = offset;
		b.size = size;
		b.mapped_ptr = std::get<std::byte*>(current_alloc) + offset;
        b.allocation_size = pool.block_size;

		return b;
	}

	Buffer Allocator::_allocate_buffer(PoolAllocator& pool, size_t size, size_t alignment, bool create_mapped) {
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
		mem_reqs.alignment = std::max(1ull, mem_reqs.alignment);
		alignment = std::max(1ull, alignment);
		mem_reqs.alignment = std::lcm(mem_reqs.alignment, alignment);
		VkMemoryRequirements vkmem_reqs = mem_reqs;
		auto result = vmaAllocateMemory(allocator, &vkmem_reqs, &vaci, &res, &vai);
		assert(result == VK_SUCCESS);
		real_alloc_callback = noop_cb;

		// record if new buffer was used
		if (pool_helper->result != VK_NULL_HANDLE) {
			buffers.emplace(reinterpret_cast<uint64_t>(vai.deviceMemory), std::pair(pool_helper->result, pool_helper->bci.size));
			pool.buffers.emplace_back(pool_helper->result);
		}
		Buffer b;
		auto [vkbuffer, allocation_size] = buffers.at(reinterpret_cast<uint64_t>(vai.deviceMemory));
		b.buffer = vkbuffer;
		b.device_memory = vai.deviceMemory;
		b.offset = vai.offset;
		b.size = vai.size;
		b.mapped_ptr = (std::byte*)vai.pMappedData;
        b.allocation_size = allocation_size;
		buffer_allocations.emplace(BufferID{ reinterpret_cast<uint64_t>(b.buffer), b.offset }, res);
		return b;
	}

	VkMemoryRequirements Allocator::get_memory_requirements(VkBufferCreateInfo& bci) {
		VkBuffer testbuff;
		vkCreateBuffer(device, &bci, nullptr, &testbuff);
		VkMemoryRequirements mem_reqs;
		vkGetBufferMemoryRequirements(device, testbuff, &mem_reqs);
		vkDestroyBuffer(device, testbuff, nullptr);
		return mem_reqs;
	}

	// allocate an externally managed pool
	PoolAllocator Allocator::allocate_pool(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage) {
		std::lock_guard _(mutex);

		VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bci.size = 1024; // ignored
		bci.usage = (VkBufferUsageFlags)buffer_usage;
		bci.queueFamilyIndexCount = queue_family_count;
		bci.sharingMode = bci.queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
		bci.pQueueFamilyIndices = all_queue_families.data();

		PoolAllocator pi;
		pi.mem_reqs = get_memory_requirements(bci);
		pi.pool = _create_pool(mem_usage, buffer_usage);
		pi.usage = buffer_usage;
		return pi;
	}

	LinearAllocator Allocator::allocate_linear(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage) {
		std::lock_guard _(mutex);

		VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bci.size = 1024; // ignored
		bci.usage = (VkBufferUsageFlags)buffer_usage;
		bci.queueFamilyIndexCount = queue_family_count;
		bci.sharingMode = bci.queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
		bci.pQueueFamilyIndices = all_queue_families.data();

		return LinearAllocator{ get_memory_requirements(bci), VmaMemoryUsage(to_integral(mem_usage)), buffer_usage };
	}

	// allocate buffer from an internally managed pool
	Buffer Allocator::allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment, bool create_mapped) {
		std::lock_guard _(mutex);

		VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bci.size = 1024; // ignored
		bci.usage = (VkBufferUsageFlags)buffer_usage;
		bci.queueFamilyIndexCount = queue_family_count;
		bci.sharingMode = bci.queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
		bci.pQueueFamilyIndices = all_queue_families.data();

		auto pool_it = pools.find(PoolSelect{ mem_usage, buffer_usage });
		if (pool_it == pools.end()) {
			PoolAllocator pi;

			pi.mem_reqs = get_memory_requirements(bci);
			pi.pool = _create_pool(mem_usage, buffer_usage);
			pi.usage = buffer_usage;
			pool_it = pools.emplace(PoolSelect{ mem_usage, buffer_usage }, pi).first;
		}

		return _allocate_buffer(pool_it->second, size, alignment, create_mapped);
	}

	// allocate a buffer from an externally managed pool
	Buffer Allocator::allocate_buffer(PoolAllocator& pool, size_t size, size_t alignment, bool create_mapped) {
		std::lock_guard _(mutex);
		return _allocate_buffer(pool, size, alignment, create_mapped);
	}

	// allocate a buffer from an externally managed linear pool
	Buffer Allocator::allocate_buffer(LinearAllocator& pool, size_t size, size_t alignment, bool create_mapped) {
		return _allocate_buffer(pool, size, alignment, create_mapped);
	}

	size_t Allocator::get_allocation_size(const Buffer& b) {
        return b.allocation_size;
	}

	void Allocator::reset_pool(PoolAllocator& pool) {
		std::lock_guard _(mutex);
		vmaResetPool(allocator, pool.pool);
	}

	void Allocator::reset_pool(LinearAllocator& pool) {
		std::lock_guard _(mutex);
		pool.current_buffer = -1;
		pool.needle = 0;
	}

	void Allocator::free_buffer(const Buffer& b) {
		std::lock_guard _(mutex);
		vuk::BufferID bufid{ reinterpret_cast<uint64_t>(b.buffer), b.offset };
		vmaFreeMemory(allocator, buffer_allocations.at(bufid));
		buffer_allocations.erase(bufid);
	}

	void Allocator::destroy(const PoolAllocator& pool) {
		std::lock_guard _(mutex);
		vmaResetPool(allocator, pool.pool);
		vmaForceUnmapPool(allocator, pool.pool);
		for (auto& buffer : pool.buffers) {
			vkDestroyBuffer(device, buffer, nullptr);
		}
		vmaDestroyPool(allocator, pool.pool);
	}

	void Allocator::destroy(const LinearAllocator& pool) {
		std::lock_guard _(mutex);
		for (auto& [va, mem, offset, buffer, map] : pool.allocations) {
			vmaDestroyBuffer(allocator, buffer, va);
		}
	}


	vuk::Image Allocator::create_image_for_rendertarget(vuk::ImageCreateInfo ici) {
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
	vuk::Image Allocator::create_image(vuk::ImageCreateInfo ici) {
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
	void Allocator::destroy_image(vuk::Image image) {
		std::lock_guard _(mutex);
		auto vkimg = (VkImage)image;
		vmaDestroyImage(allocator, image, images.at(reinterpret_cast<uint64_t>(vkimg)));
		images.erase(reinterpret_cast<uint64_t>(vkimg));
	}
	Allocator::~Allocator() {
		for (auto& [ps, p] : pools) {
			destroy(p);
		}
		vmaDestroyAllocator(allocator);
	}
}
