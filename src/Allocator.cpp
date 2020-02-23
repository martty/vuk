#include "Allocator.hpp"

PFN_vmaAllocateDeviceMemoryFunction Allocator::real_alloc_callback = nullptr;
Allocator::PoolAllocGlobalState Allocator::pags;
std::mutex Allocator::mutex;
