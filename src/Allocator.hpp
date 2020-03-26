#pragma once

#include "vk_mem_alloc.h"
#include <vulkan/vulkan.hpp>
#include <mutex>
#include <unordered_map>
#include "Hash.hpp"
#include "Cache.hpp" // for the hashes
#include "CreateInfo.hpp"

namespace vuk {
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
};

namespace std {
	template <>
	struct hash<vuk::PoolSelect> {
		size_t operator()(vuk::PoolSelect const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, to_integral(x.mem_usage), x.buffer_usage);
			return h;
		}
	};
};

namespace vuk {
	class Allocator {
	public:
		struct Pool {
			VmaPool pool;
			vk::MemoryRequirements mem_reqs;
			vk::BufferUsageFlags usage;
			vk::Buffer buffer;
		};
		struct Buffer {
			vk::DeviceMemory device_memory;
			vk::Buffer buffer;
			size_t offset;
			size_t size;
			void* mapped_ptr;
			VmaAllocation allocation;
		};
	private:
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

		static void noop_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size) {}

		static PFN_vmaAllocateDeviceMemoryFunction real_alloc_callback;

		static void allocation_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size) {
			real_alloc_callback(allocator, memoryType, memory, size);
		}
		vk::Device device;
		vk::PhysicalDevice physdev;

		std::unordered_map<uint64_t, VmaAllocation> images;
		std::unordered_map<PoolSelect, Pool> pools;

		VmaAllocator allocator;
	public:
		Allocator(vk::Device device, vk::PhysicalDevice phys_dev);
		~Allocator();

		// allocate an externally managed pool
		Pool allocate_pool(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage);
		// allocate buffer from an internally managed pool
		Buffer allocate_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, size_t size, bool create_mapped);
		// allocate a buffer from an externally managed pool
		Buffer allocate_buffer(Pool& pool, size_t size, bool create_mapped);

		void reset_pool(Pool pool);
		void free_buffer(const Buffer& b);
		void destroy_scratch_pool(Pool pool);
		
		vk::Image create_image_for_rendertarget(vk::ImageCreateInfo ici);
		vk::Image create_image(vk::ImageCreateInfo ici);
		void destroy_image(vk::Image image);

	private:
		// not locked, must be called from a locked fn
		VmaPool _create_pool(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage);
		Buffer _allocate_buffer(Pool& pool, size_t size, bool create_mapped);
	};

	template<> struct create_info<Allocator::Pool> {
		using type = PoolSelect;
	};
};