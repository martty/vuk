#include "LegacyGPUAllocator.hpp"
#include "vuk/Allocator.hpp"
#include "vuk/resources/DeviceVkResource.hpp"
#include "vuk/resources/DeviceFrameResource.hpp"
#include "vuk/resources/DeviceLinearResource.hpp"
#include "vuk/Context.hpp"
#include "vuk/Exception.hpp"
#include <string>
#include <numeric>

namespace vuk {
	DeviceVkResource::DeviceVkResource(Context& ctx, LegacyGPUAllocator& alloc) : ctx(&ctx), device(ctx.device), legacy_gpu_allocator(&alloc) {}

	Result<void, AllocateException> DeviceVkResource::allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			auto& ci = cis[i];
			if (ci.mem_usage != MemoryUsage::eCPUonly && ci.mem_usage != MemoryUsage::eCPUtoGPU && ci.mem_usage != MemoryUsage::eGPUtoCPU) {
				// TODO: signal error, tried to allocate non-xdev buffer
			}
			// TODO: legacy buffer alloc can't signal errors
			dst[i] = BufferCrossDevice{ legacy_gpu_allocator->allocate_buffer(ci.mem_usage, LegacyGPUAllocator::all_usage, ci.size, ci.alignment, true) };
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_buffers(std::span<const BufferCrossDevice> src) {
		for (auto& v : src) {
			if (v) {
				legacy_gpu_allocator->free_buffer(v);
			}
		}
	}

	Result<void, AllocateException> DeviceVkResource::allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			// TODO: legacy image alloc can't signal errors

			dst[i] = legacy_gpu_allocator->create_image(cis[i]);
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_images(std::span<const Image> src) {
		for (auto& v : src) {
			if (v != VK_NULL_HANDLE) {
				legacy_gpu_allocator->destroy_image(v);
			}
		}
	}

	Result<void, AllocateException> DeviceVkResource::allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			auto& ci = cis[i];
			// TODO: legacy buffer alloc can't signal errors
			if (ci.mem_usage != MemoryUsage::eGPUonly) {
				// TODO: signal error, tried to allocate non-gpuonly buffer
			}
			dst[i] = BufferGPU{ legacy_gpu_allocator->allocate_buffer(ci.mem_usage, LegacyGPUAllocator::all_usage, ci.size, ci.alignment, false) };
		}
		return { expected_value };
	}

	void DeviceVkResource::deallocate_buffers(std::span<const BufferGPU> src) {
		for (auto& v : src) {
			if (v) {
				legacy_gpu_allocator->free_buffer(v);
			}
		}
	}

	Result<void, AllocateException> DeviceVkResource::allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
			VkImageViewCreateInfo ci = cis[i];
			VkImageView iv;
			VkResult res = vkCreateImageView(device, &ci, nullptr, &iv);
			if (res != VK_SUCCESS) {
				deallocate_image_views({ dst.data(), (uint64_t)i });
				return { expected_error, AllocateException{ res } };
			}
			dst[i] = ctx->wrap(iv, cis[i]);
		}
		return { expected_value };
	}

	DeviceSuperFrameResource::DeviceSuperFrameResource(Context& ctx, uint64_t frames_in_flight) : direct(ctx, ctx.get_legacy_gpu_allocator()), frames_in_flight(frames_in_flight) {
		frames_storage = std::unique_ptr<char[]>(new char[sizeof(DeviceFrameResource) * frames_in_flight]);
		for (uint64_t i = 0; i < frames_in_flight; i++) {
			new(frames_storage.get() + i * sizeof(DeviceFrameResource)) DeviceFrameResource(direct.device, *this);
		}
		frames = reinterpret_cast<DeviceFrameResource*>(frames_storage.get());
	}

	DeviceFrameResource& DeviceSuperFrameResource::get_next_frame() {
		std::unique_lock _(new_frame_mutex);
		auto& ctx = direct.get_context();
		frame_counter++;
		local_frame = frame_counter % frames_in_flight;

		auto& f = frames[local_frame];
		f.wait();
		deallocate_frame(f);
		f.current_frame = frame_counter.load();

		return f;
	}

	void deallocate(DeviceFrameResource& res, DescriptorSet& ds) {
		std::unique_lock _{ res.ds_mutex };

		auto& vec = res.descriptor_sets;
		vec.emplace_back(ds);
	}

	template<class T>
	T& DeviceFrameResource::Cache<T>::acquire(uint64_t current_frame, const create_info_t<T>& ci) {
		if (auto it = lru_map.find(ci); it != lru_map.end()) {
			it->second.last_use_frame = current_frame;
			return it->second.value;
		} else {
			// if the value is not in the cache, we look in our per thread buffers
			// if it doesn't exist there either, we add it
			auto& ptv = per_thread_append_v[0 /*ptc.tid*/];
			auto& ptk = per_thread_append_k[0 /*ptc.tid*/]; // TODO: restore TIDs
			auto pit = std::find(ptk.begin(), ptk.end(), ci);
			if (pit == ptk.end()) {
				ptv.emplace_back(allocate(ci));
				pit = ptk.insert(ptk.end(), ci);
			}
			auto index = std::distance(ptk.begin(), pit);
			return ptv[index];
		}
	}

	template<class T>
	void DeviceFrameResource::Cache<T>::collect(uint64_t current_frame, size_t threshold) {
		std::unique_lock _(cache_mtx);
		for (auto it = lru_map.begin(); it != lru_map.end();) {
			if (current_frame - it->second.last_use_frame > threshold) {
				deallocate(it->second.value);
				it = lru_map.erase(it);
			} else {
				++it;
			}
		}

		for (size_t tid = 0; tid < per_thread_append_v.size(); tid++) {
			auto& vs = per_thread_append_v[tid];
			auto& ks = per_thread_append_k[tid];
			for (size_t i = 0; i < vs.size(); i++) {
				if (lru_map.find(ks[i]) == lru_map.end()) {
					lru_map.emplace(ks[i], LRUEntry{ std::move(vs[i]), current_frame });
				} else {
					deallocate(vs[i]);
				}
			}
			vs.clear();
			ks.clear();
		}
	}

	DeviceFrameResource::DeviceFrameResource(VkDevice device, DeviceSuperFrameResource& upstream) : device(device), DeviceNestedResource(&upstream),
		linear_cpu_only(upstream.direct.legacy_gpu_allocator->allocate_linear(vuk::MemoryUsage::eCPUonly, LegacyGPUAllocator::all_usage)),
		linear_cpu_gpu(upstream.direct.legacy_gpu_allocator->allocate_linear(vuk::MemoryUsage::eCPUtoGPU, LegacyGPUAllocator::all_usage)),
		linear_gpu_cpu(upstream.direct.legacy_gpu_allocator->allocate_linear(vuk::MemoryUsage::eGPUtoCPU, LegacyGPUAllocator::all_usage)),
		linear_gpu_only(upstream.direct.legacy_gpu_allocator->allocate_linear(vuk::MemoryUsage::eGPUonly, LegacyGPUAllocator::all_usage)){
	}

	DeviceLinearResource::DeviceLinearResource(DeviceResource& upstream, SyncScope scope) : DeviceNestedResource(&upstream),
		ctx(&upstream.get_context()), device(ctx->device), scope(scope),
		linear_cpu_only(ctx->get_legacy_gpu_allocator().allocate_linear(vuk::MemoryUsage::eCPUonly, LegacyGPUAllocator::all_usage)),
		linear_cpu_gpu(ctx->get_legacy_gpu_allocator().allocate_linear(vuk::MemoryUsage::eCPUtoGPU, LegacyGPUAllocator::all_usage)),
		linear_gpu_cpu(ctx->get_legacy_gpu_allocator().allocate_linear(vuk::MemoryUsage::eGPUtoCPU, LegacyGPUAllocator::all_usage)),
		linear_gpu_only(ctx->get_legacy_gpu_allocator().allocate_linear(vuk::MemoryUsage::eGPUtoCPU, LegacyGPUAllocator::all_usage)) {
	}

	Result<void, AllocateException> DeviceFrameResource::allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		auto& rf = *static_cast<DeviceSuperFrameResource*>(upstream);
		auto& legacy = *rf.direct.legacy_gpu_allocator;

		// TODO: legacy allocator can't signal errors
		// TODO: legacy linear allocators don't nest
		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			if (ci.mem_usage == MemoryUsage::eCPUonly) {
				dst[i] = BufferCrossDevice{ legacy.allocate_buffer(linear_cpu_only, ci.size, ci.alignment, true) };
			} else if (ci.mem_usage == MemoryUsage::eCPUtoGPU) {
				dst[i] = BufferCrossDevice{ legacy.allocate_buffer(linear_cpu_gpu, ci.size, ci.alignment, true) };
			} else if (ci.mem_usage == MemoryUsage::eGPUtoCPU) {
				dst[i] = BufferCrossDevice{ legacy.allocate_buffer(linear_gpu_cpu, ci.size, ci.alignment, true) };
			} else {
				return { expected_error, AllocateException{VK_ERROR_FEATURE_NOT_PRESENT} }; // tried to allocate gpu only buffer as BufferCrossDevice
			}
		}

		return { expected_value };
	}

	Result<void, AllocateException> DeviceFrameResource::allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		assert(dst.size() == cis.size());
		auto& rf = *static_cast<DeviceSuperFrameResource*>(upstream);
		auto& legacy = *rf.direct.legacy_gpu_allocator;

		// TODO: legacy allocator can't signal errors
		// TODO: legacy linear allocators don't nest
		for (uint64_t i = 0; i < dst.size(); i++) {
			auto& ci = cis[i];
			if (ci.mem_usage == MemoryUsage::eGPUonly) {
				dst[i] = BufferGPU{ legacy.allocate_buffer(linear_gpu_only, ci.size, ci.alignment, false) };
			} else {
				return { expected_error, AllocateException{VK_ERROR_FEATURE_NOT_PRESENT} }; // tried to allocate xdev buffer as BufferGPU
			}
		}

		return { expected_value };
	}

	/****Allocator impls *****/

	Result<void, AllocateException> Allocator::allocate(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
		return device_resource->allocate_semaphores(dst, loc);
	}

	Result<void, AllocateException> Allocator::allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
		return device_resource->allocate_semaphores(dst, loc);
	}

	void Allocator::deallocate_impl(std::span<const VkSemaphore> src) {
		device_resource->deallocate_semaphores(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<VkFence> dst, SourceLocationAtFrame loc) {
		return device_resource->allocate_fences(dst, loc);
	}

	Result<void, AllocateException> Allocator::allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) {
		return device_resource->allocate_fences(dst, loc);
	}

	void Allocator::deallocate_impl(std::span<const VkFence> src) {
		device_resource->deallocate_fences(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_hl_commandbuffers(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_hl_commandbuffers(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_hl_commandbuffers(dst, cis, loc);
	}

	void Allocator::deallocate_impl(std::span<const HLCommandBuffer> src) {
		device_resource->deallocate_hl_commandbuffers(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_buffers(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_buffers(dst, cis, loc);
	}

	void Allocator::deallocate_impl(std::span<const BufferCrossDevice> src) {
		device_resource->deallocate_buffers(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_buffers(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_buffers(dst, cis, loc);
	}

	void Allocator::deallocate_impl(std::span<const BufferGPU> src) {
		device_resource->deallocate_buffers(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_framebuffers(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_framebuffers(dst, cis, loc);
	}

	void Allocator::deallocate_impl(std::span<const VkFramebuffer> src) {
		device_resource->deallocate_framebuffers(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_images(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_images(dst, cis, loc);
	}

	void Allocator::deallocate_impl(std::span<const Image> src) {
		device_resource->deallocate_images(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_image_views(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_image_views(dst, cis, loc);
	}

	void Allocator::deallocate_impl(std::span<const ImageView> src) {
		device_resource->deallocate_image_views(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_persistent_descriptor_sets(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_persistent_descriptor_sets(dst, cis, loc);
	}

	void Allocator::deallocate_impl(std::span<const PersistentDescriptorSet> src) {
		device_resource->deallocate_persistent_descriptor_sets(src);
	}

	Result<void, AllocateException> Allocator::allocate(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_descriptor_sets(dst, cis, loc);
	}

	Result<void, AllocateException> Allocator::allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) {
		return device_resource->allocate_descriptor_sets(dst, cis, loc);
	}

	void Allocator::deallocate_impl(std::span<const DescriptorSet> src) {
		device_resource->deallocate_descriptor_sets(src);
	}

	PFN_vmaAllocateDeviceMemoryFunction LegacyGPUAllocator::real_alloc_callback = nullptr;

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

	void LegacyGPUAllocator::pool_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size, void* userdata) {
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

	void LegacyGPUAllocator::noop_cb(VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory, VkDeviceSize size, void* userdata) {
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

	LegacyGPUAllocator::LegacyGPUAllocator(VkInstance instance, VkDevice device, VkPhysicalDevice phys_dev, uint32_t graphics_queue_family, uint32_t transfer_queue_family) : device(device) {
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

		if (transfer_queue_family != graphics_queue_family) {
			all_queue_families = { graphics_queue_family, transfer_queue_family };
		} else {
			all_queue_families = { graphics_queue_family };
		}
		queue_family_count = (uint32_t)all_queue_families.size();
	}

	// not locked, must be called from a locked fn

	VmaPool LegacyGPUAllocator::_create_pool(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage) {
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
	Buffer LegacyGPUAllocator::_allocate_buffer(LegacyLinearAllocator& pool, size_t size, size_t alignment, bool create_mapped) {
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

	Buffer LegacyGPUAllocator::_allocate_buffer(LegacyPoolAllocator& pool, size_t size, size_t alignment, bool create_mapped) {
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

	VkMemoryRequirements LegacyGPUAllocator::get_memory_requirements(VkBufferCreateInfo& bci) {
		VkBuffer testbuff;
		vkCreateBuffer(device, &bci, nullptr, &testbuff);
		VkMemoryRequirements mem_reqs;
		vkGetBufferMemoryRequirements(device, testbuff, &mem_reqs);
		vkDestroyBuffer(device, testbuff, nullptr);
		return mem_reqs;
	}

	// allocate an externally managed pool
	LegacyPoolAllocator LegacyGPUAllocator::allocate_pool(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage) {
		std::lock_guard _(mutex);

		VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bci.size = 1024; // ignored
		bci.usage = (VkBufferUsageFlags)buffer_usage;
		bci.queueFamilyIndexCount = queue_family_count;
		bci.sharingMode = bci.queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
		bci.pQueueFamilyIndices = all_queue_families.data();

		LegacyPoolAllocator pi;
		pi.mem_reqs = get_memory_requirements(bci);
		pi.pool = _create_pool(mem_usage, buffer_usage);
		pi.usage = buffer_usage;
		return pi;
	}

	LegacyLinearAllocator LegacyGPUAllocator::allocate_linear(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage) {
		std::lock_guard _(mutex);

		VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bci.size = 1024; // ignored
		bci.usage = (VkBufferUsageFlags)buffer_usage;
		bci.queueFamilyIndexCount = queue_family_count;
		bci.sharingMode = bci.queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
		bci.pQueueFamilyIndices = all_queue_families.data();

		return LegacyLinearAllocator{ get_memory_requirements(bci), VmaMemoryUsage(to_integral(mem_usage)), buffer_usage };
	}

	// allocate buffer from an internally managed pool
	Buffer LegacyGPUAllocator::allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment, bool create_mapped) {
		std::lock_guard _(mutex);

		VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bci.size = 1024; // ignored
		bci.usage = (VkBufferUsageFlags)buffer_usage;
		bci.queueFamilyIndexCount = queue_family_count;
		bci.sharingMode = bci.queueFamilyIndexCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
		bci.pQueueFamilyIndices = all_queue_families.data();

		auto pool_it = pools.find(PoolSelect{ mem_usage, buffer_usage });
		if (pool_it == pools.end()) {
			LegacyPoolAllocator pi;

			pi.mem_reqs = get_memory_requirements(bci);
			pi.pool = _create_pool(mem_usage, buffer_usage);
			pi.usage = buffer_usage;
			pool_it = pools.emplace(PoolSelect{ mem_usage, buffer_usage }, pi).first;
		}

		return _allocate_buffer(pool_it->second, size, alignment, create_mapped);
	}

	// allocate a buffer from an externally managed pool
	Buffer LegacyGPUAllocator::allocate_buffer(LegacyPoolAllocator& pool, size_t size, size_t alignment, bool create_mapped) {
		std::lock_guard _(mutex);
		return _allocate_buffer(pool, size, alignment, create_mapped);
	}

	// allocate a buffer from an externally managed linear pool
	Buffer LegacyGPUAllocator::allocate_buffer(LegacyLinearAllocator& pool, size_t size, size_t alignment, bool create_mapped) {
		return _allocate_buffer(pool, size, alignment, create_mapped);
	}

	size_t LegacyGPUAllocator::get_allocation_size(const Buffer& b) {
		return b.allocation_size;
	}

	void LegacyGPUAllocator::reset_pool(LegacyPoolAllocator& pool) {
		std::lock_guard _(mutex);
		vmaResetPool(allocator, pool.pool);
	}

	void LegacyGPUAllocator::reset_pool(LegacyLinearAllocator& pool) {
		std::lock_guard _(mutex);
		pool.current_buffer = -1;
		pool.needle = 0;
	}

	void LegacyGPUAllocator::free_buffer(const Buffer& b) {
		std::lock_guard _(mutex);
		vuk::BufferID bufid{ reinterpret_cast<uint64_t>(b.buffer), b.offset };
		vmaFreeMemory(allocator, buffer_allocations.at(bufid));
		buffer_allocations.erase(bufid);
	}

	void LegacyGPUAllocator::destroy(const LegacyPoolAllocator& pool) {
		std::lock_guard _(mutex);
		vmaResetPool(allocator, pool.pool);
		vmaForceUnmapPool(allocator, pool.pool);
		for (auto& buffer : pool.buffers) {
			vkDestroyBuffer(device, buffer, nullptr);
		}
		vmaDestroyPool(allocator, pool.pool);
	}

	void LegacyGPUAllocator::destroy(const LegacyLinearAllocator& pool) {
		std::lock_guard _(mutex);
		for (auto& [va, mem, offset, buffer, map] : pool.allocations) {
			vmaDestroyBuffer(allocator, buffer, va);
		}
	}


	vuk::Image LegacyGPUAllocator::create_image_for_rendertarget(vuk::ImageCreateInfo ici) {
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
	vuk::Image LegacyGPUAllocator::create_image(vuk::ImageCreateInfo ici) {
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
	void LegacyGPUAllocator::destroy_image(vuk::Image image) {
		std::lock_guard _(mutex);
		auto vkimg = (VkImage)image;
		vmaDestroyImage(allocator, image, images.at(reinterpret_cast<uint64_t>(vkimg)));
		images.erase(reinterpret_cast<uint64_t>(vkimg));
	}
	LegacyGPUAllocator::~LegacyGPUAllocator() {
		for (auto& [ps, p] : pools) {
			destroy(p);
		}
		vmaDestroyAllocator(allocator);
	}
}
