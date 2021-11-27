#pragma once

#include <vuk/vuk_fwd.hpp>
#include <vuk/Config.hpp>
#include <vuk/Result.hpp>
#include <vuk/Exception.hpp>
#include <span>
#include <vector>
#include <atomic>

namespace vuk {
	struct SourceLocation {
		const char* file;
		unsigned line;
	};

	struct SourceLocationAtFrame {
		SourceLocation source_location;
		uint64_t absolute_frame;
	};

#define VUK_HERE_AND_NOW() vuk::SourceLocationAtFrame{vuk::SourceLocation{__FILE__, __LINE__}, (uint64_t)-1LL}
#define VUK_HERE_AT_FRAME(frame) vuk::SourceLocationAtFrame{vuk::SourceLocation{__FILE__, __LINE__}, frame}
#define VUK_DO_OR_RETURN(what) if(auto res = what; !res){ return { expected_error, res.error() }; }

	struct AllocateException : vuk::Exception {
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
	};

	struct VkResource {
		virtual Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_semaphores(std::span<const VkSemaphore> sema) = 0;

		virtual Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_fences(std::span<const VkFence> dst) = 0;

		virtual Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_commandbuffers(VkCommandPool pool, std::span<const VkCommandBuffer> dst) = 0;

		virtual Result<void, AllocateException> allocate_commandbuffers_hl(std::span<HLCommandBuffer> dst, std::span<HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_commandbuffers_hl(std::span<const HLCommandBuffer> dst) = 0;

		virtual Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_commandpools(std::span<const VkCommandPool> dst) = 0;
	};

	struct VkResourceNested : VkResource {
		VkResourceNested(VkResource* upstream) : upstream(upstream) {}

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) override { return upstream->allocate_semaphores(dst, loc); }
		void deallocate_semaphores(std::span<const VkSemaphore> sema) override { upstream->deallocate_semaphores(sema); }

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) override { return upstream->allocate_fences(dst, loc); }
		void deallocate_fences(std::span<const VkFence> dst) override { upstream->deallocate_fences(dst); }

		Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_commandbuffers(dst, cis, loc);
		}

		void deallocate_commandbuffers(VkCommandPool pool, std::span<const VkCommandBuffer> dst) override {
			upstream->deallocate_commandbuffers(pool, dst);
		}

		Result<void, AllocateException> allocate_commandbuffers_hl(std::span<HLCommandBuffer> dst, std::span<HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_commandbuffers_hl(dst, cis, loc);
		}

		void deallocate_commandbuffers_hl(std::span<const HLCommandBuffer> dst) override {
			upstream->deallocate_commandbuffers_hl(dst);
		}

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_commandpools(dst, cis, loc);
		}

		void deallocate_commandpools(std::span<const VkCommandPool> dst) override {
			upstream->deallocate_commandpools(dst);
		}


		/*virtual ImageView allocate_image_view(const ImageViewCreateInfo& info, uint64_t frame, SourceLocation loc) { return upstream->allocate_image_view(info, frame, loc); }
		virtual Sampler allocate_sampler(const SamplerCreateInfo& info, uint64_t frame, SourceLocation loc) { return upstream->allocate_sampler(info, frame, loc); }*/
		/*virtual DescriptorSet allocate_descriptorset(const SetBinding&, uint64_t frame, SourceLocation loc);
		virtual VkFramebuffer allocate_framebuffer(const struct FramebufferCreateInfo&, uint64_t frame, SourceLocation loc);
		virtual VkRenderPass allocate_renderpass(const struct RenderPassCreateInfo&, uint64_t frame, SourceLocation loc);*/

		/*
		virtual Result<void, AllocateException> allocate_timeline_semaphore(uint64_t initial_value, uint64_t frame, SourceLocation loc) { return upstream->allocate_timeline_semaphore(initial_value, frame, loc); }
		 }

		virtual void deallocate_image_view(ImageView iv) { upstream->deallocate_image_view(iv); }
		virtual void deallocate_sampler(Sampler samp) { upstream->deallocate_sampler(samp); }
		virtual void deallocate_timeline_semaphore(VkSemaphore sema) { upstream->deallocate_timeline_semaphore(sema); }
		*/

		VkResource* upstream = nullptr;
	};


	/*
	* HL cmdbuffers: 1:1 with pools
	*/
	struct Direct final : VkResource {
		Direct(VkDevice device) : device(device) {}

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

		Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) override {
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

		Result<void, AllocateException> allocate_commandbuffers_hl(std::span<HLCommandBuffer> dst, std::span<HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
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

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override {
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
		Result<void, AllocateException> allocate_commandbuffers_hl(std::span<HLCommandBuffer> dst, std::span<HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
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

		void wait() {
			if (fences.size() > 0) {
				vkWaitForFences(device, (uint32_t)fences.size(), fences.data(), true, UINT64_MAX);
			}
		}

		VkDevice device;
	};

	/// @brief RingFrame is an allocator that gives out Frame allocators, and manages their resources
	struct RingFrame : VkResource {
		RingFrame(VkDevice device, uint64_t frames_in_flight) : direct(device), frames_in_flight(frames_in_flight) {
			frames.resize(frames_in_flight, FrameResource{ device, *this });
		}

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

		Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) override {
			return direct.allocate_commandbuffers(dst, cis, loc);
		}

		void deallocate_commandbuffers(VkCommandPool pool, std::span<const VkCommandBuffer> dst) override {
			direct.deallocate_commandbuffers(pool, dst);
		}

		Result<void, AllocateException> allocate_commandbuffers_hl(std::span<HLCommandBuffer> dst, std::span<HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
			assert(0 && "High level command buffers cannot be allocated from RingFrame.");
			return { expected_error, AllocateException{VK_ERROR_FEATURE_NOT_PRESENT} };
		}

		void deallocate_commandbuffers_hl(std::span<const HLCommandBuffer> dst) override {
			assert(0 && "High level command buffers cannot be deallocated from RingFrame.");
		}

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override {
			return direct.allocate_commandpools(dst, cis, loc);
		}

		void deallocate_commandpools(std::span<const VkCommandPool> src) override {
			direct.deallocate_commandpools(src);
		}

		FrameResource& get_next_frame() {
			frame_counter++;
			local_frame = frame_counter % frames_in_flight;

			auto& f = frames[local_frame];
			f.wait();

			direct.deallocate_fences(f.fences);
			f.fences.clear();

			direct.deallocate_semaphores(f.semaphores);
			f.semaphores.clear();

			for (auto& c : f.cmdbuffers_to_free) {
				direct.deallocate_commandbuffers(c.command_pool, std::span{&c.command_buffer, 1});
			}
			f.cmdbuffers_to_free.clear();

			direct.deallocate_commandpools(f.cmdpools_to_free);
			f.cmdpools_to_free.clear();

			return f;
		}

		virtual ~RingFrame() {
			for (auto i = 0; i < frames_in_flight; i++) {
				auto lframe = (frame_counter + i) % frames_in_flight;
				auto& f = frames[lframe];
				f.wait();
				direct.deallocate_fences(f.fences);
				direct.deallocate_semaphores(f.semaphores);
				for (auto& c : f.cmdbuffers_to_free) {
					direct.deallocate_commandbuffers(c.command_pool, std::span{ &c.command_buffer, 1 });
				}
				direct.deallocate_commandpools(f.cmdpools_to_free);
			}
		}

		Direct direct;
		std::atomic<uint64_t> frame_counter;
		std::atomic<uint64_t> local_frame;
		const uint64_t frames_in_flight;
	};


	inline FrameResource::FrameResource(VkDevice device, RingFrame& upstream) : device(device), VkResourceNested(&upstream) {}

	struct NLinear : VkResourceNested {
		NLinear(VkDevice device, VkResource& upstream) : device(device), VkResourceNested(&upstream) {}

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

		Result<void, AllocateException> allocate_commandpools(std::span<VkCommandPool> dst, std::span<VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) override {
			auto result = upstream->allocate_commandpools(dst, cis, loc);
			command_pools.insert(command_pools.end(), dst.begin(), dst.end());
			return result;
		}

		void deallocate_commandpools(std::span<const VkCommandPool>) override {} // linear allocator, noop

		// do not record the command buffers - they come from the pools
		Result<void, AllocateException> allocate_commandbuffers(std::span<VkCommandBuffer> dst, std::span<VkCommandBufferAllocateInfo> cis, SourceLocationAtFrame loc) override {
			return upstream->allocate_commandbuffers(dst, cis, loc);
		}

		void deallocate_commandbuffers(VkCommandPool, std::span<const VkCommandBuffer>) override {} // noop, the pools own the command buffers

		std::vector<VkCommandPool> direct_command_pools;

		// TODO: error propagation
		Result<void, AllocateException> allocate_commandbuffers_hl(std::span<HLCommandBuffer> dst, std::span<HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) override {
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

		void subsume()&& {
			should_subsume = true;
		}

		void wait() {
			if (fences.size() > 0) {
				vkWaitForFences(device, (uint32_t)fences.size(), fences.data(), true, UINT64_MAX);
			}
		}

		~NLinear() {
			if (!should_subsume) {
				wait();
			}
			upstream->deallocate_fences(fences);
			upstream->deallocate_semaphores(semaphores);
			upstream->deallocate_commandpools(command_pools);
			upstream->deallocate_commandpools(direct_command_pools);
		}

		VkDevice device;
	};

	template <class ContainerType>
	concept Container = requires(ContainerType a) {
		std::begin(a);
		std::end(a);
	};


	struct NAllocator {
		explicit NAllocator(VkResource& mr) : mr(&mr) {}

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

		Result<void, AllocateException> allocate(std::span<HLCommandBuffer> dst, std::span<HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) {
			return mr->allocate_commandbuffers_hl(dst, cis, loc);
		}

		Result<void, AllocateException> allocate_commandbuffers_hl(std::span<HLCommandBuffer> dst, std::span<HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc) {
			return mr->allocate_commandbuffers_hl(dst, cis, loc);
		}

		void deallocate_impl(std::span<const HLCommandBuffer> src) {
			mr->deallocate_commandbuffers_hl(src);
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

		VkResource* mr;
	};

	template <typename Type>
	class NUnique {
		NAllocator* allocator;
		Type payload;
	public:
		using element_type = Type;

		explicit NUnique(NAllocator& allocator) : allocator(&allocator), payload{} {}
		explicit NUnique(NAllocator& allocator, Type payload) : allocator(&allocator), payload(std::move(payload)) {}
		NUnique(NUnique const&) = delete;

		NUnique(NUnique&& other) noexcept : allocator(other.allocator), payload(other.release()) {}

		~NUnique() noexcept {
			if (allocator) {
				allocator->deallocate(payload);
			}
		}

		NUnique& operator=(NUnique const&) = delete;

		NUnique& operator=(NUnique&& other) noexcept {
			auto tmp = other.allocator;
			reset(other.release());
			allocator = tmp;
			return *this;
		}

		explicit operator bool() const noexcept {
			return payload.operator bool();
		}

		Type const* operator->() const noexcept {
			return &payload;
		}

		Type* operator->() noexcept {
			return &payload;
		}

		Type const& operator*() const noexcept {
			return payload;
		}

		Type& operator*() noexcept {
			return payload;
		}

		const Type& get() const noexcept {
			return payload;
		}

		Type& get() noexcept {
			return payload;
		}

		void reset(Type value = Type()) noexcept {
			if (payload != value) {
				if (allocator && payload != Type{}) {
					allocator->deallocate(std::move(payload));
				}
				payload = std::move(value);
			}
		}

		Type release() noexcept {
			allocator = nullptr;
			return std::move(payload);
		}

		void swap(NUnique<Type>& rhs) noexcept {
			std::swap(payload, rhs.payload);
			std::swap(allocator, rhs.allocator);
		}
	};

	template <typename Type>
	inline void swap(NUnique<Type>& lhs, NUnique<Type>& rhs) noexcept {
		lhs.swap(rhs);
	}

	template<class T>
	vuk::Result<vuk::NUnique<T>, vuk::AllocateException> allocate_unique_semaphores(vuk::NAllocator allocator, vuk::SourceLocationAtFrame loc) {
		vuk::NUnique<T> semas(allocator);
		if (auto res = allocator.allocate_semaphores(*semas, loc); !res) {
			return { vuk::expected_error, res.error() };
		}
		return { vuk::expected_value, semas };
	}
}