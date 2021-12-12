#pragma once

#include <vuk/vuk_fwd.hpp>
#include <vuk/Config.hpp>
#include <vuk/Result.hpp>
#include <span>
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

	/// @brief A DeviceResource represents objects that are used jointly by both CPU and GPU. 
	/// A DeviceResource must prevent reuse of cross-device resources after deallocation until CPU-GPU timelines are synchronized. GPU-only resources may be reused immediately.
	struct DeviceResource {
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

	template <class ContainerType>
	concept Container = requires(ContainerType a) {
		std::begin(a);
		std::end(a);
	};

	struct DeviceVkResource;

	class Allocator {
	public:
		explicit Allocator(DeviceResource& device_resource) : ctx(&device_resource.get_context()), device_resource(&device_resource) {}
		explicit Allocator(DeviceVkResource& device_resource) = delete; // this resource is unsuitable for direct allocation

		Result<void, AllocateException> allocate(std::span<VkSemaphore> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		void deallocate(std::span<const VkSemaphore> src);

		Result<void, AllocateException> allocate(std::span<VkFence> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		void deallocate(std::span<const VkFence> src);

		Result<void, AllocateException> allocate(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		Result<void, AllocateException> allocate_hl_commandbuffers(std::span<HLCommandBuffer> dst, std::span<const HLCommandBufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		void deallocate(std::span<const HLCommandBuffer> src);

		Result<void, AllocateException> allocate(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) = delete;
		Result<void, AllocateException> allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) = delete;

		Result<void, AllocateException> allocate(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		Result<void, AllocateException> allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		void deallocate(std::span<const BufferCrossDevice> src);

		Result<void, AllocateException> allocate(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		Result<void, AllocateException> allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		void deallocate(std::span<const BufferGPU> src);

		Result<void, AllocateException> allocate(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		Result<void, AllocateException> allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		void deallocate(std::span<const VkFramebuffer> src);

		Result<void, AllocateException> allocate(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		void deallocate(std::span<const Image> src);

		Result<void, AllocateException> allocate(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		Result<void, AllocateException> allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		void deallocate(std::span<const ImageView> src);

		Result<void, AllocateException> allocate(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		void deallocate(std::span<const PersistentDescriptorSet> src);

		Result<void, AllocateException> allocate(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		Result<void, AllocateException> allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		void deallocate(std::span<const DescriptorSet> src);

		DeviceResource& get_cross_device_resource() {
			return *device_resource;
		}

		Context& get_context() {
			return *ctx;
		}

	private:
		Context* ctx;
		DeviceResource* device_resource;
	};

	/// @brief Customization point for deallocation of user types
	/// @tparam T 
	/// @param allocator 
	/// @param src 
	template<class T, size_t N>
	void deallocate(Allocator& allocator, T(&src)[N]) {
		allocator.deallocate(std::span<const T>{ src, N });
	}

	/// @brief Customization point for deallocation of user types
	/// @tparam T 
	/// @param allocator 
	/// @param src 
	template<class T>
	void deallocate(Allocator& allocator, const T& src) requires (!Container<T>) {
		allocator.deallocate(std::span<const T>{ &src, 1 });
	}

	/// @brief Customization point for deallocation of user types
	/// @tparam T 
	/// @param allocator 
	/// @param src 
	template<class T>
	void deallocate(Allocator& allocator, const T& src) requires (Container<T>) {
		allocator.deallocate(std::span(src));
	}

	template<typename Type>
	Unique<Type>::~Unique() noexcept {
		if (allocator && payload != Type{}) {
			deallocate(*allocator, payload);
		}
	}

	template<typename Type>
	void Unique<Type>::reset(Type value) noexcept {
		if (payload != value) {
			if (allocator && payload != Type{}) {
				deallocate(*allocator, std::move(payload));
			}
			payload = std::move(value);
		}
	}
}

#undef VUK_HERE_AND_NOW