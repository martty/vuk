#pragma once

#include <vuk/vuk_fwd.hpp>
#include <vuk/Config.hpp>
#include <vuk/Result.hpp>
#include <vuk/Exception.hpp>
#include <vuk/Types.hpp>
#include <vuk/Buffer.hpp>
#include <vuk/Descriptor.hpp>
#include <../src/RenderPass.hpp>
#include <../src/LegacyGPUAllocator.hpp> // TODO: remove
#include <span>
#include <vector>
#include <atomic>
#include <source_location>

namespace vuk {
#ifndef __cpp_consteval
	struct source_location {
		uint_least32_t _Line{};
		uint_least32_t _Column{};
		const char* _File = "";
		const char* _Function = "";

		[[nodiscard]] constexpr source_location() noexcept = default;

		[[nodiscard]] static source_location current(const uint_least32_t _Line_ = __builtin_LINE(),
			const uint_least32_t _Column_ = __builtin_COLUMN(), const char* const _File_ = __builtin_FILE(),
			const char* const _Function_ = __builtin_FUNCTION()) noexcept {
			source_location _Result;
			_Result._Line = _Line_;
			_Result._Column = _Column_;
			_Result._File = _File_;
			_Result._Function = _Function_;
			return _Result;
		}
	};

	struct SourceLocationAtFrame {
		source_location location;
		uint64_t absolute_frame;
	};
#else
	struct SourceLocationAtFrame {
		std::source_location location;
		uint64_t absolute_frame;
	};
#endif
#ifndef __cpp_consteval
#define VUK_HERE_AND_NOW() SourceLocationAtFrame{ vuk::source_location::current(), (uint64_t)-1LL }
#else
#define VUK_HERE_AND_NOW() SourceLocationAtFrame{ std::source_location::current(), (uint64_t)-1LL }
#endif
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

	struct HLCommandBufferCreateInfo {
		VkCommandBufferLevel level;
		uint32_t queue_family_index;
	};

	/*
	* HL cmdbuffers: 1:1 with pools
	*/
	struct HLCommandBuffer {
		HLCommandBuffer() = default;
		HLCommandBuffer(VkCommandBuffer command_buffer, VkCommandPool command_pool) : command_buffer(command_buffer), command_pool(command_pool) {}

		VkCommandBuffer command_buffer;
		VkCommandPool command_pool;

		operator VkCommandBuffer() {
			return command_buffer;
		}
	};

	struct BufferCreateInfo {
		MemoryUsage mem_usage;
		size_t size;
		size_t alignment;
	};

	struct PersistentDescriptorSetCreateInfo {
		DescriptorSetLayoutAllocInfo dslai;
		uint32_t num_descriptors;
	};


	struct CPUResource {
		virtual void* allocate(size_t bytes, size_t alignment, SourceLocationAtFrame loc) = 0;
		//virtual void allocate_at_least(size_t bytes, size_t alignment, SourceLocationAtFrame loc) = 0;
		virtual void deallocate(void* ptr, size_t bytes, size_t alignment) = 0;
	};

	struct CPUNestedResource : CPUResource {
		virtual void* allocate(size_t bytes, size_t alignment, SourceLocationAtFrame loc) {
			return upstream->allocate(bytes, alignment, loc);
		}
		virtual void deallocate(void* ptr, size_t bytes, size_t alignment) {
			return upstream->deallocate(ptr, bytes, alignment);
		}

		CPUResource* upstream = nullptr;
	};

	/// @brief A CrossDeviceResource represents objects that are used jointly by both CPU and GPU. 
	/// A CrossDeviceResource must prevent reuse of resources after deallocation until CPU-GPU timelines are synchronized.
	struct CrossDeviceResource {
		// missing here: Events (gpu only)

		// gpu only
		virtual Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_semaphores(std::span<const VkSemaphore> src) = 0;

		virtual Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_fences(std::span<const VkFence> dst) = 0;

		virtual Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<const VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_commandbuffers(VkCommandPool pool, std::span<const VkCommandBuffer> dst) = 0;

		virtual Result<void, AllocateException> allocate_hl_commandbuffers(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_hl_commandbuffers(std::span<const HLCommandBuffer> dst) = 0;

		virtual Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_commandpools(std::span<const VkCommandPool> dst) = 0;

		virtual Result<void, AllocateException> allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_buffers(std::span<const BufferCrossDevice> dst) = 0;

		// gpu only
		virtual Result<void, AllocateException> allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_buffers(std::span<const BufferGPU> dst) = 0;

		virtual Result<void, AllocateException> allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_framebuffers(std::span<const VkFramebuffer> dst) = 0;

		// gpu only
		virtual Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_images(std::span<const Image> dst) = 0;

		virtual Result<void, AllocateException> allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_image_views(std::span<const ImageView> src) = 0;

		virtual Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) = 0;

		virtual Result<void, AllocateException> allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_descriptor_sets(std::span<const DescriptorSet> src) = 0;

		/*
		virtual Result<void, AllocateException> allocate_timeline_semaphore(uint64_t initial_value, uint64_t frame, SourceLocation loc) { return upstream->allocate_timeline_semaphore(initial_value, frame, loc); }
		virtual void deallocate_timeline_semaphore(VkSemaphore sema) { upstream->deallocate_timeline_semaphore(sema); }
		*/

		virtual Context& get_context() = 0;
	};

	struct CrossDeviceNestedResource : CrossDeviceResource {
		CrossDeviceNestedResource(CrossDeviceResource* upstream) : upstream(upstream) {}

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override {
			return upstream->allocate_semaphores(dst, loc);
		}

		void deallocate_semaphores(std::span<const VkSemaphore> sema) override {
			upstream->deallocate_semaphores(sema);
		}

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override { return upstream->allocate_fences(dst, loc); }
		void deallocate_fences(std::span<const VkFence> dst) override { upstream->deallocate_fences(dst); }

		Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<const VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_commandbuffers(dst, cis, loc);
		}

		void deallocate_commandbuffers(VkCommandPool pool, std::span<const VkCommandBuffer> dst) override {
			upstream->deallocate_commandbuffers(pool, dst);
		}

		Result<void, AllocateException> allocate_hl_commandbuffers(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_hl_commandbuffers(dst, cis, loc);
		}

		void deallocate_hl_commandbuffers(std::span<const HLCommandBuffer> dst) override {
			upstream->deallocate_hl_commandbuffers(dst);
		}

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_commandpools(dst, cis, loc);
		}

		void deallocate_commandpools(std::span<const VkCommandPool> dst) override {
			upstream->deallocate_commandpools(dst);
		}

		Result<void, AllocateException> allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_buffers(dst, cis, loc);
		}

		void deallocate_buffers(std::span<const BufferCrossDevice> src) override {
			upstream->deallocate_buffers(src);
		}

		Result<void, AllocateException> allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_buffers(dst, cis, loc);
		}

		void deallocate_buffers(std::span<const BufferGPU> src) override {
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

		Result<void, AllocateException> allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_descriptor_sets(dst, cis, loc);
		}

		void deallocate_descriptor_sets(std::span<const DescriptorSet> src) override {
			upstream->deallocate_descriptor_sets(src);
		}

		CrossDeviceResource* upstream = nullptr;
	};

	struct CrossDeviceVkAllocator final : CrossDeviceResource {
		CrossDeviceVkAllocator(Context& ctx, LegacyGPUAllocator& alloc);

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

		Result<void, AllocateException> allocate_hl_commandbuffers(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
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

		void deallocate_hl_commandbuffers(std::span<const HLCommandBuffer> dst) override {
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

		Result<void, AllocateException> allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const BufferCrossDevice> src) override;

		Result<void, AllocateException> allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const BufferGPU> src) override;

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

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_images(std::span<const Image> src) override;

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

		Result<void, AllocateException> allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) override;

		void deallocate_descriptor_sets(std::span<const DescriptorSet> src);

		Context& get_context() override {
			return *ctx;
		}

		Context* ctx;
		LegacyGPUAllocator* legacy_gpu_allocator;
		VkDevice device;
	};


	struct CrossDeviceRingFrameResource;

	/*
	* allocates pass through to ring frame, deallocation is retained
	fence: linear
	semaphore: linear
	command buffers & pools: 1:1 buffers-to-pools for easy handout & threading - buffers are not freed individually
	*/
	struct CrossDeviceFrameResource : CrossDeviceNestedResource {
		CrossDeviceFrameResource(VkDevice device, CrossDeviceRingFrameResource& upstream);

		std::mutex sema_mutex;
		std::vector<VkSemaphore> semaphores;

		void deallocate_semaphores(std::span<const VkSemaphore> src) override {
			std::unique_lock _(sema_mutex);
			auto& vec = semaphores;
			vec.insert(vec.end(), src.begin(), src.end());
		}

		std::mutex fence_mutex;
		std::vector<VkFence> fences;

		void deallocate_fences(std::span<const VkFence> src) override {
			std::unique_lock _(fence_mutex);
			auto& vec = fences;
			vec.insert(vec.end(), src.begin(), src.end());
		}

		std::mutex cbuf_mutex;
		std::vector<HLCommandBuffer> cmdbuffers_to_free;
		std::vector<VkCommandPool> cmdpools_to_free;

		// TODO: error propagation
		Result<void, AllocateException> allocate_hl_commandbuffers(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			std::unique_lock _(cbuf_mutex);
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

		void deallocate_hl_commandbuffers(std::span<const HLCommandBuffer> src) override {} // no-op, deallocated with pools

		void deallocate_commandbuffers(VkCommandPool pool, std::span<const VkCommandBuffer> src) override {
			std::unique_lock _(cbuf_mutex);

			cmdbuffers_to_free.reserve(cmdbuffers_to_free.size() + src.size());
			for (auto& s : src) {
				cmdbuffers_to_free.emplace_back(s, pool);
			}
		}

		void deallocate_commandpools(std::span<const VkCommandPool> src) override {
			std::unique_lock _(cbuf_mutex);

			auto& vec = cmdpools_to_free;
			vec.insert(vec.end(), src.begin(), src.end());
		}

		// buffers are lockless
		Result<void, AllocateException> allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const BufferCrossDevice> src) override {} // no-op, linear

		Result<void, AllocateException> allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override;

		void deallocate_buffers(std::span<const BufferGPU> src) override {} // no-op, linear

		std::mutex framebuffer_mutex;
		std::vector<VkFramebuffer> framebuffers;

		void deallocate_framebuffers(std::span<const VkFramebuffer> src) override {
			std::unique_lock _(framebuffer_mutex);
			auto& vec = framebuffers;
			vec.insert(vec.end(), src.begin(), src.end());
		}

		std::mutex images_mutex;
		std::vector<Image> images;

		void deallocate_images(std::span<const Image> src) override {
			std::unique_lock _(images_mutex);

			auto& vec = images;
			vec.insert(vec.end(), src.begin(), src.end());
		}

		std::mutex image_views_mutex;
		std::vector<ImageView> image_views;
		void deallocate_image_views(std::span<const ImageView> src) override {
			std::unique_lock _(image_views_mutex);

			auto& vec = image_views;
			vec.insert(vec.end(), src.begin(), src.end());
		}


		template<class T>
		struct LRUEntry {
			T value;
			size_t last_use_frame;
		};

		template<class T>
		struct Cache {
			robin_hood::unordered_map<create_info_t<T>, LRUEntry<T>> lru_map;
			std::array<std::vector<T>, 32> per_thread_append_v;
			std::array<std::vector<create_info_t<T>>, 32> per_thread_append_k;

			std::mutex cache_mtx;

			T& acquire(uint64_t current_frame, const create_info_t<T>& ci);
			void collect(uint64_t current_frame, size_t threshold);
		};

		Cache<DescriptorSet> descriptor_sets;
		//Result<void, AllocateException> allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) override;

		void wait() {
			if (fences.size() > 0) {
				vkWaitForFences(device, (uint32_t)fences.size(), fences.data(), true, UINT64_MAX);
			}
		}

		Context& get_context() override {
			return upstream->get_context();
		}

		VkDevice device;
		uint64_t current_frame = 0;
		LegacyLinearAllocator linear_cpu_only;
		LegacyLinearAllocator linear_cpu_gpu;
		LegacyLinearAllocator linear_gpu_cpu;
		LegacyLinearAllocator linear_gpu_only;
	};

	/// @brief RingFrame is an allocator that gives out Frame allocators, and manages their resources
	struct CrossDeviceRingFrameResource : CrossDeviceResource {
		CrossDeviceRingFrameResource(Context& ctx, uint64_t frames_in_flight);

		std::unique_ptr<char[]> frames_storage;
		CrossDeviceFrameResource* frames;

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

		Result<void, AllocateException> allocate_hl_commandbuffers(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			assert(0 && "High level command buffers cannot be allocated from RingFrame.");
			return { expected_error, AllocateException{VK_ERROR_FEATURE_NOT_PRESENT} };
		}

		void deallocate_hl_commandbuffers(std::span<const HLCommandBuffer> dst) override {
			assert(0 && "High level command buffers cannot be deallocated from RingFrame.");
		}

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override {
			return direct.allocate_commandpools(dst, cis, loc);
		}

		void deallocate_commandpools(std::span<const VkCommandPool> src) override {
			direct.deallocate_commandpools(src);
		}

		Result<void, AllocateException> allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return direct.allocate_buffers(dst, cis, loc);
		}

		void deallocate_buffers(std::span<const BufferCrossDevice> src) override {
			direct.deallocate_buffers(src);
		}

		Result<void, AllocateException> allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return direct.allocate_buffers(dst, cis, loc);
		}

		void deallocate_buffers(std::span<const BufferGPU> src) override {
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

		Result<void, AllocateException> allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) override {
			return direct.allocate_descriptor_sets(dst, cis, loc);
		}

		void deallocate_descriptor_sets(std::span<const DescriptorSet> src) override {
			direct.deallocate_descriptor_sets(src);
		}

		CrossDeviceFrameResource& get_next_frame();

		void deallocate_frame(CrossDeviceFrameResource& f) {
			direct.deallocate_semaphores(f.semaphores);
			direct.deallocate_fences(f.fences);
			for (auto& c : f.cmdbuffers_to_free) {
				direct.deallocate_commandbuffers(c.command_pool, std::span{ &c.command_buffer, 1 });
			}
			direct.deallocate_commandpools(f.cmdpools_to_free);
			//direct.deallocate_buffers(f.buffers); // TODO: linear allocators don't suballocate :/
			direct.deallocate_framebuffers(f.framebuffers);
			direct.deallocate_images(f.images);
			direct.deallocate_image_views(f.image_views);

			f.semaphores.clear();
			f.fences.clear();
			f.cmdbuffers_to_free.clear();
			f.cmdpools_to_free.clear();
			auto& legacy = direct.legacy_gpu_allocator;
			legacy->reset_pool(f.linear_cpu_only);
			legacy->reset_pool(f.linear_cpu_gpu);
			legacy->reset_pool(f.linear_gpu_cpu);
			legacy->reset_pool(f.linear_gpu_only);
			f.framebuffers.clear();
			f.images.clear();
			f.image_views.clear();
		}

		virtual ~CrossDeviceRingFrameResource() {
			for (auto i = 0; i < frames_in_flight; i++) {
				auto lframe = (frame_counter + i) % frames_in_flight;
				auto& f = frames[lframe];
				f.wait();
				deallocate_frame(f);
				direct.legacy_gpu_allocator->destroy(f.linear_cpu_only);
				direct.legacy_gpu_allocator->destroy(f.linear_cpu_gpu);
				direct.legacy_gpu_allocator->destroy(f.linear_gpu_cpu);
				direct.legacy_gpu_allocator->destroy(f.linear_gpu_only);
				f.~CrossDeviceFrameResource();
			}
		}

		Context& get_context() override {
			return *direct.ctx;
		}

		CrossDeviceVkAllocator direct;
		std::mutex new_frame_mutex;
		std::atomic<uint64_t> frame_counter;
		std::atomic<uint64_t> local_frame;
		const uint64_t frames_in_flight;
	};

	struct CrossDeviceLinearResource : CrossDeviceNestedResource {
		enum class SyncScope { eInline, eScope };
		static constexpr SyncScope eInline = SyncScope::eInline;
		static constexpr SyncScope eScope = SyncScope::eScope;

		CrossDeviceLinearResource(CrossDeviceResource& upstream, SyncScope scope);

		bool should_subsume = false;
		std::vector<VkFence> fences;

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
		Result<void, AllocateException> allocate_hl_commandbuffers(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
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

		~CrossDeviceLinearResource() {
			if (scope == SyncScope::eScope) {
				wait();
			}
			upstream->deallocate_fences(fences);
			upstream->deallocate_commandpools(command_pools);
			upstream->deallocate_commandpools(direct_command_pools);
			upstream->deallocate_framebuffers(framebuffers);
		}

		Context* ctx;
		VkDevice device;
		SyncScope scope;
		LegacyLinearAllocator linear_cpu_only;
		LegacyLinearAllocator linear_cpu_gpu;
		LegacyLinearAllocator linear_gpu_cpu;
		LegacyLinearAllocator linear_gpu_only;
	};

	template <class ContainerType>
	concept Container = requires(ContainerType a) {
		std::begin(a);
		std::end(a);
	};


	class Allocator {
	public:
		explicit Allocator(CrossDeviceResource& cross_device) : ctx(&cross_device.get_context()), cross_device(&cross_device) {}

		Result<void, AllocateException> allocate(std::span<VkSemaphore> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_semaphores(dst, loc);
		}

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_semaphores(dst, loc);
		}

		void deallocate_impl(std::span<const VkSemaphore> src) {
			cross_device->deallocate_semaphores(src);
		}

		Result<void, AllocateException> allocate(std::span<VkFence> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_fences(dst, loc);
		}

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_fences(dst, loc);
		}

		void deallocate_impl(std::span<const VkFence> src) {
			cross_device->deallocate_fences(src);
		}

		Result<void, AllocateException> allocate(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_hl_commandbuffers(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_hl_commandbuffers(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_hl_commandbuffers(dst, cis, loc);
		}

		void deallocate_impl(std::span<const HLCommandBuffer> src) {
			cross_device->deallocate_hl_commandbuffers(src);
		}

		Result<void, AllocateException> allocate(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) = delete;
		Result<void, AllocateException> allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) = delete;

		Result<void, AllocateException> allocate(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_buffers(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_buffers(dst, cis, loc);
		}

		void deallocate_impl(std::span<const BufferCrossDevice> src) {
			cross_device->deallocate_buffers(src);
		}

		Result<void, AllocateException> allocate(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_buffers(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_buffers(dst, cis, loc);
		}

		void deallocate_impl(std::span<const BufferGPU> src) {
			cross_device->deallocate_buffers(src);
		}

		Result<void, AllocateException> allocate(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_framebuffers(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_framebuffers(dst, cis, loc);
		}

		void deallocate_impl(std::span<const VkFramebuffer> src) {
			cross_device->deallocate_framebuffers(src);
		}

		Result<void, AllocateException> allocate(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_images(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_images(dst, cis, loc);
		}

		void deallocate_impl(std::span<const Image> src) {
			cross_device->deallocate_images(src);
		}

		Result<void, AllocateException> allocate(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_image_views(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_image_views(dst, cis, loc);
		}

		void deallocate_impl(std::span<const ImageView> src) {
			cross_device->deallocate_image_views(src);
		}

		Result<void, AllocateException> allocate(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_persistent_descriptor_sets(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_persistent_descriptor_sets(dst, cis, loc);
		}

		void deallocate_impl(std::span<const PersistentDescriptorSet> src) {
			cross_device->deallocate_persistent_descriptor_sets(src);
		}

		Result<void, AllocateException> allocate(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_descriptor_sets(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
			return cross_device->allocate_descriptor_sets(dst, cis, loc);
		}

		void deallocate_impl(std::span<const DescriptorSet> src) {
			cross_device->deallocate_descriptor_sets(src);
		}



		// TODO: deallocation has to be a customization point
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

		CrossDeviceResource& get_cross_device_resource() {
			return *cross_device;
		}

		Context& get_context() {
			return *ctx;
		}

	private:
		Context* ctx;
		CrossDeviceResource* cross_device;
	};

	template<class T>
	Result<Unique<T>, AllocateException> allocate_semaphores(Allocator& allocator, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<T> semas(allocator);
		if (auto res = allocator.allocate_semaphores(*semas, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, semas };
	}

	inline Result<Unique<HLCommandBuffer>, AllocateException> allocate_hl_commandbuffer(Allocator& allocator, const HLCommandBufferCreateInfo& cbci, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<HLCommandBuffer> hlcb(allocator);
		if (auto res = allocator.allocate_hl_commandbuffers(std::span{ &hlcb.get(), 1 }, std::span{ &cbci, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(hlcb) };
	}

	inline Result<Unique<VkFence>, AllocateException> allocate_fence(Allocator& allocator, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<VkFence> fence(allocator);
		if (auto res = allocator.allocate_fences(std::span{ &fence.get(), 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(fence) };
	}

	inline Result<Unique<BufferCrossDevice>, AllocateException> allocate_buffer_cross_device(Allocator& allocator, const BufferCreateInfo& ici, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<BufferCrossDevice> buf(allocator);
		if (auto res = allocator.allocate_buffers(std::span{ &buf.get(), 1 }, std::span{ &ici, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(buf) };
	}

	inline Result<Unique<BufferGPU>, AllocateException> allocate_buffer_gpu(Allocator& allocator, const BufferCreateInfo& ici, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<BufferGPU> buf(allocator);
		if (auto res = allocator.allocate_buffers(std::span{ &buf.get(), 1 }, std::span{ &ici, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(buf) };
	}

	inline Result<Unique<Image>, AllocateException> allocate_image(Allocator& allocator, const ImageCreateInfo& ici, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<Image> img(allocator);
		if (auto res = allocator.allocate_images(std::span{ &img.get(), 1 }, std::span{ &ici, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(img) };
	}

	inline Result<Unique<ImageView>, AllocateException> allocate_image_view(Allocator& allocator, const ImageViewCreateInfo& ivci, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<ImageView> iv(allocator);
		if (auto res = allocator.allocate_image_views(std::span{ &iv.get(), 1 }, std::span{ &ivci, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(iv) };
	}

	template<typename Type>
	Unique<Type>::~Unique() noexcept {
		if (allocator && payload != Type{}) {
			allocator->deallocate(payload);
		}
	}

	template<typename Type>
	void Unique<Type>::reset(Type value) noexcept {
		if (payload != value) {
			if (allocator && payload != Type{}) {
				allocator->deallocate(std::move(payload));
			}
			payload = std::move(value);
		}
	}
}

#undef VUK_HERE_AND_NOW