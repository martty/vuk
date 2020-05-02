#pragma once

#include "vk_mem_alloc.h"
#include <vulkan/vulkan.hpp>
#include <mutex>
#include <unordered_map>
#include "Hash.hpp"
#include "Cache.hpp" // for the hashes
#include "CreateInfo.hpp"
#include "Types.hpp"
#include <atomic>

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

	struct BufferID {
		uint64_t vk_buffer;
		uint64_t offset;

		bool operator==(const BufferID& o) const noexcept {
			return ::memcmp(this, &o, sizeof(BufferID)) == 0;
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
	
	template <>
	struct hash<vuk::BufferID> {
		size_t operator()(vuk::BufferID const & x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.vk_buffer, x.offset); 
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
			std::vector<vk::Buffer> buffers;
		};

		struct Linear {
            std::atomic<int> current_buffer = -1;
            std::atomic<size_t> needle = 0;
			vk::MemoryRequirements mem_reqs;
            VmaMemoryUsage mem_usage;
			vk::BufferUsageFlags usage;
            std::array<std::tuple<VmaAllocation, vk::DeviceMemory, size_t, vk::Buffer, void*>, 32> allocations;

            size_t block_size = 1024 * 1024;

			Linear(vk::MemoryRequirements mem_reqs, VmaMemoryUsage mem_usage, vk::BufferUsageFlags buf_usage)
                : mem_reqs(mem_reqs), mem_usage(mem_usage), usage(buf_usage) {}

			Linear(Linear&& o) noexcept {
                current_buffer = o.current_buffer.load();
                needle = o.needle.load();
                mem_reqs = o.mem_reqs;
                mem_usage = o.mem_usage;
                usage = o.usage;
                allocations = o.allocations;
                block_size = o.block_size;
			}
		};
	private:
		std::mutex mutex;
		struct PoolAllocHelper {
			vk::Device device;
			vk::BufferCreateInfo bci;
			vk::Buffer result;
			PFN_vkSetDebugUtilsObjectNameEXT setDebugUtilsObjectNameEXT;
		};
		std::unique_ptr<PoolAllocHelper> pool_helper;

		static void pool_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size, void* userdata);

		static void noop_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size, void* userdata);

		static PFN_vmaAllocateDeviceMemoryFunction real_alloc_callback;

		static void allocation_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size, void* userdata) {
			real_alloc_callback(allocator, memoryType, memory, size, userdata);
		}
		vk::Device device;
		vk::PhysicalDevice physdev;

		std::unordered_map<uint64_t, VmaAllocation> images;
		std::unordered_map<BufferID, VmaAllocation> buffer_allocations;
		std::unordered_map<PoolSelect, Pool> pools;
		std::unordered_map<uint64_t, vk::Buffer> buffers;

		VmaAllocator allocator;
        vk::PhysicalDeviceProperties properties;
	public:
		Allocator(vk::Instance instance, vk::Device device, vk::PhysicalDevice phys_dev);
		~Allocator();

		// allocate an externally managed pool
		Pool allocate_pool(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage);
		// allocate an externally managed linear pool
		Linear allocate_linear(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage);
		// allocate buffer from an internally managed pool
		Buffer allocate_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, size_t size, bool create_mapped);
		// allocate a buffer from an externally managed pool
		Buffer allocate_buffer(Pool& pool, size_t size, bool create_mapped);
        // allocate a buffer from an externally managed linear pool
		Buffer allocate_buffer(Linear& pool, size_t size, bool create_mapped);

		void reset_pool(Pool& pool);
		void reset_pool(Linear& pool);

		void free_buffer(const Buffer& b);
		void destroy(const Pool& pool);
		void destroy(const Linear& pool);
		
		vk::Image create_image_for_rendertarget(vk::ImageCreateInfo ici);
		vk::Image create_image(vk::ImageCreateInfo ici);
		void destroy_image(vk::Image image);

	private:
		// not locked, must be called from a locked fn
		VmaPool _create_pool(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage);
		Buffer _allocate_buffer(Pool& pool, size_t size, bool create_mapped);
		Buffer _allocate_buffer(Linear& pool, size_t size, bool create_mapped);
	};

	template<> struct create_info<Allocator::Pool> {
		using type = PoolSelect;
	};

	template<> struct create_info<Allocator::Linear> {
		using type = PoolSelect;
	};

};
