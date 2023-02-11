#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/SourceLocation.hpp"
#include "vuk/Types.hpp"

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <vk_mem_alloc.h>

namespace vuk {
	struct DeviceResource;

	struct LinearAllocSegment {
		VmaAllocation allocation = nullptr;
		VkDeviceMemory device_memory;
		size_t device_memory_offset;
		VkBuffer buffer = VK_NULL_HANDLE;
		std::byte* mapped_ptr = nullptr;
		uint64_t bda;
		size_t num_blocks;
		uint64_t base_address = 0;
	};

	struct LinearSegment {
		Buffer buffer;
		size_t num_blocks;
		uint64_t base_address = 0;
	};

	struct LinearBufferAllocator {
		DeviceResource* upstream;
		std::mutex mutex;
		std::atomic<int> current_buffer = -1;
		std::atomic<uint64_t> needle = 0;
		MemoryUsage mem_usage;
		BufferUsageFlags usage;
		std::array<LinearSegment, 256> available_allocations; // up to 4 GB of allocations
		std::array<LinearSegment, 256> used_allocations;      // up to 4 GB of allocations
		size_t available_allocation_count = 0;
		size_t used_allocation_count = 0;

		size_t block_size;

		LinearBufferAllocator(DeviceResource& upstream, MemoryUsage mem_usage, BufferUsageFlags buf_usage, size_t block_size = 1024 * 1024 * 16) :
		    upstream(&upstream),
		    mem_usage(mem_usage),
		    usage(buf_usage),
		    block_size(block_size) {}
		~LinearBufferAllocator();

		Result<void, AllocateException> grow(size_t num_blocks, SourceLocationAtFrame source);
		Result<Buffer, AllocateException> allocate_buffer(size_t size, size_t alignment, SourceLocationAtFrame source);
		// trim the amount of memory to the currently used amount
		void trim();
		// return all resources to available
		void reset();
		// explicitly release resources
		void free();

		LinearBufferAllocator(LinearBufferAllocator&& o) noexcept {
			current_buffer = o.current_buffer.load();
			needle = o.needle.load();
			mem_usage = o.mem_usage;
			usage = o.usage;
			used_allocations = o.used_allocations;
			available_allocations = o.available_allocations;
			block_size = o.block_size;
			available_allocation_count = o.available_allocation_count;
			used_allocation_count = o.used_allocation_count;
		}
	};

	struct BufferBlock {
		Buffer buffer;
		VmaVirtualBlock block;
	};

	struct SubAllocation {
		VmaVirtualBlock block;
		VmaVirtualAllocation allocation;
	};

	struct BufferSubAllocator {
		DeviceResource* upstream;
		MemoryUsage mem_usage;
		BufferUsageFlags usage;
		std::vector<BufferBlock> blocks;

		size_t block_size;

		BufferSubAllocator(DeviceResource& upstream, MemoryUsage mem_usage, BufferUsageFlags buf_usage, size_t block_size) :
		    upstream(&upstream),
		    mem_usage(mem_usage),
		    usage(buf_usage),
		    block_size(block_size) {}
		~BufferSubAllocator();

		Result<void, AllocateException> grow(size_t num_blocks, size_t alignment, SourceLocationAtFrame source);
		Result<Buffer, AllocateException> allocate_buffer(size_t size, size_t alignment, SourceLocationAtFrame source);
		void deallocate_buffer(const Buffer& buf);

		void reset();
	};
}; // namespace vuk
