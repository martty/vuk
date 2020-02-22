#include "Allocator.hpp"

PFN_vmaAllocateDeviceMemoryFunction Allocator::real_alloc_callback = nullptr;
