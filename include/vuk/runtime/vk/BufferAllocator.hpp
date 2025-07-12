#pragma once

#include "vuk/Config.hpp"
#include "vuk/SourceLocation.hpp"
#include "vuk/Types.hpp"
#include "vuk/runtime/vk/Allocation.hpp"
#include "vuk/runtime/vk/VkTypes.hpp" // TODO: leaking vk

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <plf_colony.h>
#include <utility>
#include <vector>
#include <vk_mem_alloc.h>

namespace vuk {
	struct DeviceResource;

	struct LinearSegment {
		ptr_base buffer;
		size_t num_blocks;
		uint64_t base_address = 0;
		AllocationEntry entry;
	};

	struct BufferLinearAllocator {
		DeviceResource* upstream;
		std::mutex mutex;
		std::atomic<int> current_buffer = -1;
		std::atomic<size_t> needle = 0;
		MemoryUsage memory_usage;
		BufferUsageFlags usage;
		// TODO: convert to deque
		std::array<LinearSegment, 256> available_allocations; // up to 4 GB of allocations with the default block_size
		std::array<LinearSegment, 256> used_allocations;      // up to 4 GB of allocations with the default block_size
		size_t available_allocation_count = 0;
		size_t used_allocation_count = 0;

		size_t block_size;

		BufferLinearAllocator(DeviceResource& upstream, MemoryUsage memory_usage, BufferUsageFlags buf_usage, size_t block_size = 1024 * 1024 * 16) :
		    upstream(&upstream),
		    memory_usage(memory_usage),
		    usage(buf_usage),
		    block_size(block_size) {}
		~BufferLinearAllocator();

		Result<void, AllocateException> grow(size_t num_blocks, SourceLocationAtFrame source);
		Result<ptr_base, AllocateException> allocate_memory(size_t size, size_t alignment, SourceLocationAtFrame source);
		// trim the amount of memory to the currently used amount
		void trim();
		// return all resources to available
		void reset();
		// explicitly release resources
		void free();
	};

	struct BufferBlock {
		ptr_base buffer = {};
		size_t allocation_count = 0;
		AllocationEntry entry;
	};

	struct SubAllocation {
		size_t block_index;
		VmaVirtualAllocation allocation;
	};

	struct BufferSubAllocator {
		DeviceResource* upstream;
		MemoryUsage memory_usage;
		BufferUsageFlags usage;
		std::vector<BufferBlock> blocks;
		VmaVirtualBlock virtual_alloc;
		std::mutex mutex;
		size_t block_size;

		BufferSubAllocator(DeviceResource& upstream, MemoryUsage memory_usage, BufferUsageFlags buf_usage, size_t block_size);
		~BufferSubAllocator();

		Result<ptr_base, AllocateException> allocate_memory(size_t size, size_t alignment, SourceLocationAtFrame source);
		void deallocate_memory(const ptr_base& buf);
	};
}; // namespace vuk
