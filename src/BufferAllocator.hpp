#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/Config.hpp"
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

	struct LinearSegment {
		Buffer buffer;
		size_t num_blocks;
		uint64_t base_address = 0;
	};

	struct BufferLinearAllocator {
		DeviceResource* upstream;
		std::mutex mutex;
		std::atomic<int> current_buffer = -1;
		std::atomic<uint64_t> needle = 0;
		MemoryUsage mem_usage;
		BufferUsageFlags usage;
		// TODO: convert to deque
		std::array<LinearSegment, 256> available_allocations; // up to 4 GB of allocations with the default block_size
		std::array<LinearSegment, 256> used_allocations;      // up to 4 GB of allocations with the default block_size
		size_t available_allocation_count = 0;
		size_t used_allocation_count = 0;

		size_t block_size;

		BufferLinearAllocator(DeviceResource& upstream, MemoryUsage mem_usage, BufferUsageFlags buf_usage, size_t block_size = 1024 * 1024 * 16) :
		    upstream(&upstream),
		    mem_usage(mem_usage),
		    usage(buf_usage),
		    block_size(block_size) {}
		~BufferLinearAllocator();

		Result<void, AllocateException> grow(size_t num_blocks, SourceLocationAtFrame source);
		Result<Buffer, AllocateException> allocate_buffer(size_t size, size_t alignment, SourceLocationAtFrame source);
		// trim the amount of memory to the currently used amount
		void trim();
		// return all resources to available
		void reset();
		// explicitly release resources
		void free();
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
		// explicitly release resources
		void free();
	};
}; // namespace vuk
