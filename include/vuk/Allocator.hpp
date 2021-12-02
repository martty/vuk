#pragma once

#include <vuk/vuk_fwd.hpp>
#include <vuk/Config.hpp>
#include <vuk/Result.hpp>
#include <vuk/Exception.hpp>
#include <vuk/Types.hpp>
#include <span>
#include <vector>
#include <atomic>
#include <../src/Allocator.hpp>

namespace vuk {
	struct SourceLocation {
		const char* file;
		unsigned line;
	};

	struct SourceLocationAtFrame {
		SourceLocation source_location;
		uint64_t absolute_frame;
	};

#define VUK_HERE_AND_NOW() SourceLocationAtFrame{SourceLocation{__FILE__, __LINE__}, (uint64_t)-1LL}
#define VUK_HERE_AT_FRAME(frame) SourceLocationAtFrame{SourceLocation{__FILE__, __LINE__}, frame}
#define VUK_DO_OR_RETURN(what) if(auto res = what; !res){ return { expected_error, res.error() }; }

	struct AllocateException : Exception {
		AllocateException(VkResult res) {
			switch (res) {
			case VK_ERROR_OUT_OF_HOST_MEMORY:
			{
				error_message = "Out of host memory."; break;
			}
			case VK_ERROR_OUT_OF_DEVICE_MEMORY:
			{
				error_message = "Out of device memory."; break;
			}
			case VK_ERROR_INITIALIZATION_FAILED:
			{
				error_message = "Initialization failed."; break;
			}
			case VK_ERROR_DEVICE_LOST:
			{
				error_message = "Device lost."; break;
			}
			case VK_ERROR_MEMORY_MAP_FAILED:
			{
				error_message = "Memory map failed."; break;
			}
			case VK_ERROR_LAYER_NOT_PRESENT:
			{
				error_message = "Layer not present."; break;
			}
			case VK_ERROR_EXTENSION_NOT_PRESENT:
			{
				error_message = "Extension not present."; break;
			}
			case VK_ERROR_FEATURE_NOT_PRESENT:
			{
				error_message = "Feature not present."; break;
			}
			case VK_ERROR_INCOMPATIBLE_DRIVER:
			{
				error_message = "Incompatible driver."; break;
			}
			case VK_ERROR_TOO_MANY_OBJECTS:
			{
				error_message = "Too many objects."; break;
			}
			case VK_ERROR_FORMAT_NOT_SUPPORTED:
			{
				error_message = "Format not supported."; break;
			}
			default:
				assert(0 && "Unimplemented error."); break;
			}
		}
	};

	struct CPUResource {
		virtual void* allocate(size_t bytes, size_t alignment, SourceLocationAtFrame loc) = 0;
		//virtual void allocate_at_least(size_t bytes, size_t alignment, SourceLocationAtFrame loc) = 0;
		virtual void deallocate(void* ptr, size_t bytes, size_t alignment) = 0;
	};

	struct CPUResourceNested : CPUResource {
		virtual void* allocate(size_t bytes, size_t alignment, SourceLocationAtFrame loc) {
			return upstream->allocate(bytes, alignment, loc);
		}
		virtual void deallocate(void* ptr, size_t bytes, size_t alignment) {
			return upstream->deallocate(ptr, bytes, alignment);
		}

		CPUResource* upstream = nullptr;
	};

	struct gvoid;

	struct GPUResource {
		virtual Result<gvoid*, AllocateException> allocate(size_t bytes, size_t alignment, SourceLocationAtFrame loc) = 0;
		virtual void deallocate(gvoid*) = 0;
	};

	struct HLCommandBufferCreateInfo {
		VkCommandBufferLevel level;
		uint32_t queue_family_index;
	};

	struct HLCommandBuffer {
		VkCommandBuffer command_buffer;
		VkCommandPool command_pool;

		operator VkCommandBuffer() {
			return command_buffer;
		}
	};

	struct BufferCreateInfo {
		MemoryUsage mem_usage;
		BufferUsageFlags buffer_usage;
		size_t size;
		size_t alignment;
	};

	struct PersistentDescriptorSetCreateInfo {
		DescriptorSetLayoutAllocInfo dslai; 
		uint32_t num_descriptors;
	};

	struct VkResource {
		virtual Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_semaphores(std::span<const VkSemaphore> src) = 0;

		virtual Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_fences(std::span<const VkFence> dst) = 0;

		virtual Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<const VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_commandbuffers(VkCommandPool pool, std::span<const VkCommandBuffer> dst) = 0;

		virtual Result<void, AllocateException> allocate_commandbuffers_hl(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_commandbuffers_hl(std::span<const HLCommandBuffer> dst) = 0;

		virtual Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_commandpools(std::span<const VkCommandPool> dst) = 0;

		virtual Result<void, AllocateException> allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_buffers(std::span<const Buffer> dst) = 0;

		virtual Result<void, AllocateException> allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_framebuffers(std::span<const VkFramebuffer> dst) = 0;

		virtual Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_images(std::span<const Image> dst) = 0;

		virtual Result<void, AllocateException> allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_image_views(std::span<const ImageView> src) = 0;

		virtual Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) = 0;

		virtual Context& get_context() = 0;
	};

	struct VkResourceNested : VkResource {
		VkResourceNested(VkResource* upstream) : upstream(upstream) {}

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override { return upstream->allocate_semaphores(dst, loc); }
		void deallocate_semaphores(std::span<const VkSemaphore> sema) override { upstream->deallocate_semaphores(sema); }

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override { return upstream->allocate_fences(dst, loc); }
		void deallocate_fences(std::span<const VkFence> dst) override { upstream->deallocate_fences(dst); }

		Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<const VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_commandbuffers(dst, cis, loc);
		}

		void deallocate_commandbuffers(VkCommandPool pool, std::span<const VkCommandBuffer> dst) override {
			upstream->deallocate_commandbuffers(pool, dst);
		}

		Result<void, AllocateException> allocate_commandbuffers_hl(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_commandbuffers_hl(dst, cis, loc);
		}

		void deallocate_commandbuffers_hl(std::span<const HLCommandBuffer> dst) override {
			upstream->deallocate_commandbuffers_hl(dst);
		}

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_commandpools(dst, cis, loc);
		}

		void deallocate_commandpools(std::span<const VkCommandPool> dst) override {
			upstream->deallocate_commandpools(dst);
		}

		Result<void, AllocateException> allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_buffers(dst, cis, loc);
		}

		void deallocate_buffers(std::span<const Buffer> src) override {
			upstream->deallocate_buffers(src);
		}

		Result<void, AllocateException> allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_framebuffers(dst, cis, loc);
		}

		void deallocate_framebuffers(std::span<const VkFramebuffer> src) override {
			upstream->deallocate_framebuffers(src);
		}

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_images(dst, cis, loc);
		}

		void deallocate_images(std::span<const Image> src) override {
			upstream->deallocate_images(src);
		}

		Result<void, AllocateException> allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_image_views(dst, cis, loc);
		}

		void deallocate_image_views(std::span<const ImageView> src) override {
			upstream->deallocate_image_views(src);
		}

		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_persistent_descriptor_sets(dst, cis, loc);
		}

		void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) override {
			upstream->deallocate_persistent_descriptor_sets(src);
		}
		/*
		virtual Result<void, AllocateException> allocate_timeline_semaphore(uint64_t initial_value, uint64_t frame, SourceLocation loc) { return upstream->allocate_timeline_semaphore(initial_value, frame, loc); }
		 }
		virtual void deallocate_timeline_semaphore(VkSemaphore sema) { upstream->deallocate_timeline_semaphore(sema); }
		*/

		VkResource* upstream = nullptr;
	};


	/*
	* HL cmdbuffers: 1:1 with pools
	*/
	struct Direct final : VkResource {
		Direct(Context& ctx, LegacyGPUAllocator& alloc);

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override {
			VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
			for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
				VkResult res = vkCreateSemaphore(device, &sci, nullptr, &dst[i]);
				if (res != VK_SUCCESS) {
					deallocate_semaphores({ dst.data(), (uint64_t)i });
					return { expected_error, AllocateException{res} };
				}
			}
			return { expected_value };
		}

		void deallocate_semaphores(std::span<const VkSemaphore> src) override {
			for (auto& v : src) {
				if (v != VK_NULL_HANDLE) {
					vkDestroySemaphore(device, v, nullptr);
				}
			}
		}

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override {
			VkFenceCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
			for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
				VkResult res = vkCreateFence(device, &sci, nullptr, &dst[i]);
				if (res != VK_SUCCESS) {
					deallocate_fences({ dst.data(), (uint64_t)i });
					return { expected_error, AllocateException{res} };
				}
			}
			return { expected_value };
		}

		void deallocate_fences(std::span<const VkFence> src) override {
			for (auto& v : src) {
				if (v != VK_NULL_HANDLE) {
					vkDestroyFence(device, v, nullptr);
				}
			}
		}

		Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<const VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) override {
			assert(dst.size() == cis.size());
			VkResult res = vkAllocateCommandBuffers(device, cis.data(), dst.data());
			if (res != VK_SUCCESS) {
				return { expected_error, AllocateException{res} };
			}
			return { expected_value };
		}

		void deallocate_commandbuffers(VkCommandPool pool, std::span<const VkCommandBuffer> dst) override {
			vkFreeCommandBuffers(device, pool, (uint32_t)dst.size(), dst.data());
		}

		Result<void, AllocateException> allocate_commandbuffers_hl(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			assert(dst.size() == cis.size());

			for (uint64_t i = 0; i < dst.size(); i++) {
				auto& ci = cis[i];
				VkCommandPoolCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
				cpci.queueFamilyIndex = ci.queue_family_index;
				cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
				allocate_commandpools(std::span{ &dst[i].command_pool, 1 }, std::span{ &cpci, 1 }, loc);

				VkCommandBufferAllocateInfo cbai{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
				cbai.commandBufferCount = 1;
				cbai.commandPool = dst[i].command_pool;
				cbai.level = ci.level;
				allocate_commandbuffers(std::span{ &dst[i].command_buffer, 1 }, std::span{ &cbai, 1 }, loc);
			}

			return { expected_value };
		}

		void deallocate_commandbuffers_hl(std::span<const HLCommandBuffer> dst) override {
			for (auto& c : dst) {
				deallocate_commandpools(std::span{ &c.command_pool, 1 });
			}
		}

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override {
			assert(dst.size() == cis.size());
			for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
				VkResult res = vkCreateCommandPool(device, &cis[i], nullptr, &dst[i]);
				if (res != VK_SUCCESS) {
					deallocate_commandpools({ dst.data(), (uint64_t)i });
					return { expected_error, AllocateException{res} };
				}
			}
			return { expected_value };
		}

		void deallocate_commandpools(std::span<const VkCommandPool> src) override {
			for (auto& v : src) {
				if (v != VK_NULL_HANDLE) {
					vkDestroyCommandPool(device, v, nullptr);
				}
			}
		}

		Result<void, AllocateException> allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			assert(dst.size() == cis.size());
			for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
				auto& ci = cis[i];
				bool create_mapped = ci.mem_usage == MemoryUsage::eCPUonly || ci.mem_usage == MemoryUsage::eCPUtoGPU || ci.mem_usage == MemoryUsage::eGPUtoCPU;
				// TODO: legacy buffer alloc can't signal errors
				dst[i] = legacy_gpu_allocator->allocate_buffer(ci.mem_usage, ci.buffer_usage, ci.size, ci.alignment, create_mapped);
			}
			return { expected_value };
		}

		void deallocate_buffers(std::span<const Buffer> src) override {
			for (auto& v : src) {
				if (v) {
					legacy_gpu_allocator->free_buffer(v);
				}
			}
		}

		Result<void, AllocateException> allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			assert(dst.size() == cis.size());
			for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
				VkResult res = vkCreateFramebuffer(device, &cis[i], nullptr, &dst[i]);
				if (res != VK_SUCCESS) {
					deallocate_framebuffers({ dst.data(), (uint64_t)i });
					return { expected_error, AllocateException{res} };
				}
			}
			return { expected_value };
		}

		void deallocate_framebuffers(std::span<const VkFramebuffer> src) override {
			for (auto& v : src) {
				if (v != VK_NULL_HANDLE) {
					vkDestroyFramebuffer(device, v, nullptr);
				}
			}
		}

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) override {
			assert(dst.size() == cis.size());
			for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
				// TODO: legacy image alloc can't signal errors

				dst[i] = legacy_gpu_allocator->create_image(cis[i]);
			}
			return { expected_value };
		}

		void deallocate_images(std::span<const Image> src) override {
			for (auto& v : src) {
				if (v != VK_NULL_HANDLE) {
					legacy_gpu_allocator->destroy_image(v);
				}
			}
		}

		Result<void, AllocateException> allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_image_views(std::span<const ImageView> src) override {
			for (auto& v : src) {
				if (v.payload != VK_NULL_HANDLE) {
					vkDestroyImageView(device, v.payload, nullptr);
				}
			}
		}

		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) override;

		Context& get_context() override {
			return *ctx;
		}

		Context* ctx;
		LegacyGPUAllocator* legacy_gpu_allocator;
		VkDevice device;
	};


	struct RingFrame;

	/*
	* allocates pass through to ring frame, deallocation is retained
	fence: linear
	semaphore: linear
	command buffers & pools: 1:1 buffers-to-pools for easy handout & threading - buffers are not freed individually
	*/
	struct FrameResource : VkResourceNested {
		FrameResource(VkDevice device, RingFrame& upstream);

		std::vector<VkFence> fences;
		std::vector<VkSemaphore> semaphores;

		void deallocate_semaphores(std::span<const VkSemaphore> src) override {
			auto& vec = semaphores;
			vec.insert(vec.end(), src.begin(), src.end());
		}

		void deallocate_fences(std::span<const VkFence> src) override {
			auto& vec = fences;
			vec.insert(vec.end(), src.begin(), src.end());
		}

		std::vector<HLCommandBuffer> cmdbuffers_to_free;
		std::vector<VkCommandPool> cmdpools_to_free;

		// TODO: error propagation
		Result<void, AllocateException> allocate_commandbuffers_hl(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			assert(dst.size() == cis.size());
			auto cmdpools_size = cmdpools_to_free.size();
			cmdpools_to_free.resize(cmdpools_to_free.size() + dst.size());

			for (uint64_t i = 0; i < dst.size(); i++) {
				auto& ci = cis[i];
				VkCommandPoolCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
				cpci.queueFamilyIndex = ci.queue_family_index;
				cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
				upstream->allocate_commandpools(std::span{ cmdpools_to_free.data() + cmdpools_size + i, 1 }, std::span{ &cpci, 1 }, loc);

				dst[i].command_pool = *(cmdpools_to_free.data() + cmdpools_size + i);
				VkCommandBufferAllocateInfo cbai{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
				cbai.commandBufferCount = 1;
				cbai.commandPool = dst[i].command_pool;
				cbai.level = ci.level;
				upstream->allocate_commandbuffers(std::span{ &dst[i].command_buffer, 1 }, std::span{ &cbai, 1 }, loc); // do not record cbuf, we deallocate it with the pool
			}

			return { expected_value };
		}

		void deallocate_commandbuffers_hl(std::span<const HLCommandBuffer> src) override {} // no-op, deallocated with pools

		void deallocate_commandbuffers(VkCommandPool pool, std::span<const VkCommandBuffer> src) override {
			cmdbuffers_to_free.reserve(cmdbuffers_to_free.size() + src.size());
			for (auto& s : src) {
				cmdbuffers_to_free.emplace_back(s, pool);
			}
		}

		void deallocate_commandpools(std::span<const VkCommandPool> src) override {
			auto& vec = cmdpools_to_free;
			vec.insert(vec.end(), src.begin(), src.end());
		}

		std::vector<Buffer> buffers;

		void deallocate_buffers(std::span<const Buffer> src) override {
			auto& vec = buffers;
			vec.insert(vec.end(), src.begin(), src.end());
		}

		std::vector<VkFramebuffer> framebuffers;

		void deallocate_framebuffers(std::span<const VkFramebuffer> src) override {
			auto& vec = framebuffers;
			vec.insert(vec.end(), src.begin(), src.end());
		}

		void wait() {
			if (fences.size() > 0) {
				vkWaitForFences(device, (uint32_t)fences.size(), fences.data(), true, UINT64_MAX);
			}
		}

		Context& get_context() override {
			return upstream->get_context();
		}

		VkDevice device;
	};

	/// @brief RingFrame is an allocator that gives out Frame allocators, and manages their resources
	struct RingFrame : VkResource {
		RingFrame(Context& ctx, uint64_t frames_in_flight);

		std::vector<FrameResource> frames;

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override {
			return direct.allocate_semaphores(dst, loc);
		}

		void deallocate_semaphores(std::span<const VkSemaphore> src) override {
			direct.deallocate_semaphores(src);
		}

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override {
			return direct.allocate_fences(dst, loc);
		}

		void deallocate_fences(std::span<const VkFence> src) override {
			direct.deallocate_fences(src);
		}

		Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<const VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) override {
			return direct.allocate_commandbuffers(dst, cis, loc);
		}

		void deallocate_commandbuffers(VkCommandPool pool, std::span<const VkCommandBuffer> dst) override {
			direct.deallocate_commandbuffers(pool, dst);
		}

		Result<void, AllocateException> allocate_commandbuffers_hl(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			assert(0 && "High level command buffers cannot be allocated from RingFrame.");
			return { expected_error, AllocateException{VK_ERROR_FEATURE_NOT_PRESENT} };
		}

		void deallocate_commandbuffers_hl(std::span<const HLCommandBuffer> dst) override {
			assert(0 && "High level command buffers cannot be deallocated from RingFrame.");
		}

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override {
			return direct.allocate_commandpools(dst, cis, loc);
		}

		void deallocate_commandpools(std::span<const VkCommandPool> src) override {
			direct.deallocate_commandpools(src);
		}

		Result<void, AllocateException> allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return direct.allocate_buffers(dst, cis, loc);
		}

		void deallocate_buffers(std::span<const Buffer> src) override {
			direct.deallocate_buffers(src);
		}

		Result<void, AllocateException> allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return direct.allocate_framebuffers(dst, cis, loc);
		}

		void deallocate_framebuffers(std::span<const VkFramebuffer> src) override {
			direct.deallocate_framebuffers(src);
		}

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) override {
			return direct.allocate_images(dst, cis, loc);
		}

		void deallocate_images(std::span<const Image> src) override {
			direct.deallocate_images(src);
		}

		Result<void, AllocateException> allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) override {
			return direct.allocate_image_views(dst, cis, loc);
		}

		void deallocate_image_views(std::span<const ImageView> src) override {
			return direct.deallocate_image_views(src);
		}

		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) override {
			return direct.allocate_persistent_descriptor_sets(dst, cis, loc);
		}

		void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) override {
			return direct.deallocate_persistent_descriptor_sets(src);
		}

		FrameResource& get_next_frame();

		void deallocate_frame(FrameResource& f) {
			direct.deallocate_fences(f.fences);
			direct.deallocate_semaphores(f.semaphores);
			for (auto& c : f.cmdbuffers_to_free) {
				direct.deallocate_commandbuffers(c.command_pool, std::span{ &c.command_buffer, 1 });
			}
			direct.deallocate_commandpools(f.cmdpools_to_free);
			direct.deallocate_buffers(f.buffers);
			direct.deallocate_framebuffers(f.framebuffers);

			f.fences.clear();
			f.semaphores.clear();
			f.cmdbuffers_to_free.clear();
			f.cmdpools_to_free.clear();
			f.buffers.clear();
			f.framebuffers.clear();
		}

		virtual ~RingFrame() {
			for (auto i = 0; i < frames_in_flight; i++) {
				auto lframe = (frame_counter + i) % frames_in_flight;
				auto& f = frames[lframe];
				f.wait();
				deallocate_frame(f);
			}
		}

		Context& get_context() override {
			return *direct.ctx;
		}

		Direct direct;
		std::atomic<uint64_t> frame_counter;
		std::atomic<uint64_t> local_frame;
		const uint64_t frames_in_flight;
	};

	inline FrameResource::FrameResource(VkDevice device, RingFrame& upstream) : device(device), VkResourceNested(&upstream) {}

	struct NLinear : VkResourceNested {
		enum class SyncScope { eInline, eScope };
		static constexpr SyncScope eInline = SyncScope::eInline;
		static constexpr SyncScope eScope = SyncScope::eScope;

		NLinear(VkResource& upstream, SyncScope scope);

		bool should_subsume = false;
		std::vector<VkSemaphore> semaphores;
		std::vector<VkFence> fences;

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override {
			auto result = upstream->allocate_semaphores(dst, loc);
			semaphores.insert(semaphores.end(), dst.begin(), dst.end());
			return result;
		}

		void deallocate_semaphores(std::span<const VkSemaphore>) override {} // linear allocator, noop

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override {
			auto result = upstream->allocate_fences(dst, loc);
			fences.insert(fences.end(), dst.begin(), dst.end());
			return result;
		}

		void deallocate_fences(std::span<const VkFence>) override {} // linear allocator, noop

		std::vector<VkCommandPool> command_pools;

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override {
			auto result = upstream->allocate_commandpools(dst, cis, loc);
			command_pools.insert(command_pools.end(), dst.begin(), dst.end());
			return result;
		}

		void deallocate_commandpools(std::span<const VkCommandPool>) override {} // linear allocator, noop

		// do not record the command buffers - they come from the pools
		Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<const VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_commandbuffers(dst, cis, loc);
		}

		void deallocate_commandbuffers(VkCommandPool, std::span<const VkCommandBuffer>) override {} // noop, the pools own the command buffers

		std::vector<VkCommandPool> direct_command_pools;

		// TODO: error propagation
		Result<void, AllocateException> allocate_commandbuffers_hl(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			for (uint64_t i = 0; i < dst.size(); i++) {
				auto& ci = cis[i];
				direct_command_pools.resize(direct_command_pools.size() < (ci.queue_family_index + 1) ? (ci.queue_family_index + 1) : direct_command_pools.size(), VK_NULL_HANDLE);
				auto& pool = direct_command_pools[ci.queue_family_index];
				if (pool == VK_NULL_HANDLE) {
					VkCommandPoolCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
					cpci.queueFamilyIndex = ci.queue_family_index;
					cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
					upstream->allocate_commandpools(std::span{ &pool, 1 }, std::span{ &cpci, 1 }, loc);
				}

				dst[i].command_pool = pool;
				VkCommandBufferAllocateInfo cbai{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
				cbai.commandBufferCount = 1;
				cbai.commandPool = pool;
				cbai.level = ci.level;
				upstream->allocate_commandbuffers(std::span{ &dst[i].command_buffer, 1 }, std::span{ &cbai, 1 }, loc);
			}
			return { expected_value };
		}

		std::vector<Buffer> buffers;

		Result<void, AllocateException> allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			auto result = upstream->allocate_buffers(dst, cis, loc);
			buffers.insert(buffers.end(), dst.begin(), dst.end());
			return result;
		}

		void deallocate_buffers(std::span<const Buffer>) override {} // linear allocator, noop

		std::vector<VkFramebuffer> framebuffers;

		Result<void, AllocateException> allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			auto result = upstream->allocate_framebuffers(dst, cis, loc);
			framebuffers.insert(framebuffers.end(), dst.begin(), dst.end());
			return result;
		}

		void deallocate_framebuffers(std::span<const VkFramebuffer>) override {} // linear allocator, noop

		void wait() {
			if (fences.size() > 0) {
				vkWaitForFences(device, (uint32_t)fences.size(), fences.data(), true, UINT64_MAX);
			}
		}

		Context& get_context() override {
			return *ctx;
		}

		~NLinear() {
			if (scope == SyncScope::eScope) {
				wait();
			}
			upstream->deallocate_fences(fences);
			upstream->deallocate_semaphores(semaphores);
			upstream->deallocate_commandpools(command_pools);
			upstream->deallocate_commandpools(direct_command_pools);
			upstream->deallocate_buffers(buffers);
			upstream->deallocate_framebuffers(framebuffers);
		}

		Context* ctx;
		VkDevice device;
		SyncScope scope;
	};

	template <class ContainerType>
	concept Container = requires(ContainerType a) {
		std::begin(a);
		std::end(a);
	};


	class NAllocator {
	public:
		explicit NAllocator(VkResource& mr) : ctx(&mr.get_context()), mr(&mr) {}

		Result<void, AllocateException> allocate(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
			return mr->allocate_semaphores(dst, loc);
		}

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) {
			return mr->allocate_semaphores(dst, loc);
		}

		void deallocate_impl(std::span<const VkSemaphore> src) {
			mr->deallocate_semaphores(src);
		}

		Result<void, AllocateException> allocate(std::span<VkFence> dst, SourceLocationAtFrame loc) {
			return mr->allocate_fences(dst, loc);
		}

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) {
			return mr->allocate_fences(dst, loc);
		}

		void deallocate_impl(std::span<const VkFence> src) {
			mr->deallocate_fences(src);
		}

		Result<void, AllocateException> allocate(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) {
			return mr->allocate_commandbuffers_hl(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_commandbuffers_hl(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) {
			return mr->allocate_commandbuffers_hl(dst, cis, loc);
		}

		void deallocate_impl(std::span<const HLCommandBuffer> src) {
			mr->deallocate_commandbuffers_hl(src);
		}

		Result<void, AllocateException> allocate(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
			return mr->allocate_buffers(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) {
			return mr->allocate_buffers(dst, cis, loc);
		}

		void deallocate_impl(std::span<const Buffer> src) {
			mr->deallocate_buffers(src);
		}

		Result<void, AllocateException> allocate(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
			return mr->allocate_framebuffers(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) {
			return mr->allocate_framebuffers(dst, cis, loc);
		}

		void deallocate_impl(std::span<const VkFramebuffer> src) {
			mr->deallocate_framebuffers(src);
		}

		Result<void, AllocateException> allocate(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
			return mr->allocate_images(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) {
			return mr->allocate_images(dst, cis, loc);
		}

		void deallocate_impl(std::span<const Image> src) {
			mr->deallocate_images(src);
		}

		Result<void, AllocateException> allocate(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
			return mr->allocate_image_views(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) {
			return mr->allocate_image_views(dst, cis, loc);
		}

		void deallocate_impl(std::span<const ImageView> src) {
			mr->deallocate_image_views(src);
		}

		Result<void, AllocateException> allocate(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) {
			return mr->allocate_persistent_descriptor_sets(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) {
			return mr->allocate_persistent_descriptor_sets(dst, cis, loc);
		}

		void deallocate_impl(std::span<const PersistentDescriptorSet> src) {
			mr->deallocate_persistent_descriptor_sets(src);
		}

		template<class T, size_t N>
		void deallocate(T(&src)[N]) {
			deallocate_impl(std::span<const T>{ src, N });
		}

		template<class T>
		void deallocate(const T& src) requires (!Container<T>) {
			deallocate_impl(std::span<const T>{ &src, 1 });
		}

		template<class T>
		void deallocate(const T& src) requires (Container<T>) {
			deallocate_impl(std::span(src));
		}

		VkResource& get_memory_resource() {
			return *mr;
		}

		Context& get_context() {
			return *ctx;
		}

	private:
		Context* ctx;
		VkResource* mr;
	};

	template<class T>
	Result<Unique<T>, AllocateException> allocate_semaphores(NAllocator& allocator, SourceLocationAtFrame loc) {
		Unique<T> semas(allocator);
		if (auto res = allocator.allocate_semaphores(*semas, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, semas };
	}

	inline Result<Unique<HLCommandBuffer>, AllocateException> allocate_hl_commandbuffer(NAllocator& allocator, const HLCommandBufferCreateInfo& cbci, SourceLocationAtFrame loc) {
		Unique<HLCommandBuffer> hlcb(allocator);
		if (auto res = allocator.allocate_commandbuffers_hl(std::span{ &hlcb.get(), 1 }, std::span{ &cbci, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(hlcb) };
	}

	inline Result<Unique<VkFence>, AllocateException> allocate_fence(NAllocator& allocator, SourceLocationAtFrame loc) {
		Unique<VkFence> fence(allocator);
		if (auto res = allocator.allocate_fences(std::span{ &fence.get(), 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(fence) };
	}

	inline Result<Unique<Image>, AllocateException> allocate_image(NAllocator& allocator, const ImageCreateInfo& ici, SourceLocationAtFrame loc) {
		Unique<Image> img(allocator);
		if (auto res = allocator.allocate_images(std::span{ &img.get(), 1 }, std::span{ &ici, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(img) };
	}

	inline Result<Unique<ImageView>, AllocateException> allocate_image_view(NAllocator& allocator, const ImageViewCreateInfo& ivci, SourceLocationAtFrame loc) {
		Unique<ImageView> iv(allocator);
		if (auto res = allocator.allocate_image_views(std::span{ &iv.get(), 1 }, std::span{ &ivci, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(iv) };
	}
}