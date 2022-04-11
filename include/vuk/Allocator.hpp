#pragma once

#include "vuk/Config.hpp"
#include "vuk/Image.hpp"
#include "vuk/Result.hpp"
#include "vuk/vuk_fwd.hpp"

#include <source_location>
#include <span>

namespace vuk {
	/// @cond INTERNAL
#ifndef __cpp_consteval
	struct source_location {
		uint_least32_t _Line{};
		uint_least32_t _Column{};
		const char* _File = "";
		const char* _Function = "";

		[[nodiscard]] constexpr source_location() noexcept = default;

		[[nodiscard]] static source_location current(const uint_least32_t _Line_ = __builtin_LINE(),
		                                             const uint_least32_t _Column_ = __builtin_COLUMN(),
		                                             const char* const _File_ = __builtin_FILE(),
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
#define VUK_HERE_AND_NOW()                                                                                                                                     \
	SourceLocationAtFrame {                                                                                                                                      \
		vuk::source_location::current(), (uint64_t)-1LL                                                                                                            \
	}
#else
#define VUK_HERE_AND_NOW()                                                                                                                                     \
	SourceLocationAtFrame {                                                                                                                                      \
		std::source_location::current(), (uint64_t)-1LL                                                                                                            \
	}
#endif
#define VUK_DO_OR_RETURN(what)                                                                                                                                 \
	if (auto res = what; !res) {                                                                                                                                 \
		return std::move(res);                                                                                                                                     \
	}
	/// @endcond

	struct TimelineSemaphore {
		VkSemaphore semaphore;
		uint64_t* value;

		bool operator==(const TimelineSemaphore& other) const noexcept {
			return semaphore == other.semaphore;
		}
	};

	/// @brief DeviceResource is a polymorphic interface over allocation of GPU resources.
	/// A DeviceResource must prevent reuse of cross-device resources after deallocation until CPU-GPU timelines are synchronized. GPU-only resources may be
	/// reused immediately.
	struct DeviceResource {
		// missing here: Events (gpu only)

		// gpu only
		virtual Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_semaphores(std::span<const VkSemaphore> src) = 0;

		virtual Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_fences(std::span<const VkFence> dst) = 0;

		virtual Result<void, AllocateException>
		allocate_command_buffers(std::span<CommandBufferAllocation> dst, std::span<const CommandBufferAllocationCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_command_buffers(std::span<const CommandBufferAllocation> dst) = 0;

		virtual Result<void, AllocateException>
		allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_command_pools(std::span<const CommandPool> dst) = 0;

		virtual Result<void, AllocateException>
		allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_buffers(std::span<const BufferCrossDevice> dst) = 0;

		// gpu only
		virtual Result<void, AllocateException> allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_buffers(std::span<const BufferGPU> dst) = 0;

		virtual Result<void, AllocateException>
		allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_framebuffers(std::span<const VkFramebuffer> dst) = 0;

		// gpu only
		virtual Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_images(std::span<const Image> dst) = 0;

		virtual Result<void, AllocateException>
		allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_image_views(std::span<const ImageView> src) = 0;

		virtual Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
		                                                                            std::span<const PersistentDescriptorSetCreateInfo> cis,
		                                                                            SourceLocationAtFrame loc) = 0;
		virtual void deallocate_persistent_descriptor_sets(std::span<const PersistentDescriptorSet> src) = 0;

		virtual Result<void, AllocateException>
		allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_descriptor_sets(std::span<const DescriptorSet> src) = 0;

		virtual Result<void, AllocateException>
		allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_timestamp_query_pools(std::span<const TimestampQueryPool> src) = 0;

		virtual Result<void, AllocateException>
		allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_timestamp_queries(std::span<const TimestampQuery> src) = 0;

		virtual Result<void, AllocateException> allocate_timeline_semaphores(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc) = 0;
		virtual void deallocate_timeline_semaphores(std::span<const TimelineSemaphore> src) = 0;

		virtual void deallocate_swapchains(std::span<const VkSwapchainKHR> src) = 0;

		virtual Context& get_context() = 0;
	};

	struct DeviceVkResource;

	/// @brief Interface for allocating device resources
	///
	/// The Allocator is a concrete value type wrapping over a polymorphic DeviceResource, forwarding allocations and deallocations to it.
	/// The allocation functions take spans of creation parameters and output values, reporting error through the return value of Result<void, AllocateException>.
	/// The deallocation functions can't fail.
	class Allocator {
	public:
		/// @brief Create new Allocator that wraps a DeviceResource
		/// @param device_resource The DeviceResource to allocate from
		explicit Allocator(DeviceResource& device_resource) : ctx(&device_resource.get_context()), device_resource(&device_resource) {}
		explicit Allocator(DeviceVkResource& device_resource) = delete; // this resource is unsuitable for direct allocation

		/// @brief Allocate semaphores from this Allocator
		/// @param dst Destination span to place allocated semaphores into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate(std::span<VkSemaphore> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate semaphores from this Allocator
		/// @param dst Destination span to place allocated semaphores into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate semaphores previously allocated from this Allocator
		/// @param src Span of semaphores to be deallocated
		void deallocate(std::span<const VkSemaphore> src);

		/// @brief Allocate fences from this Allocator
		/// @param dst Destination span to place allocated fences into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate(std::span<VkFence> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate fences from this Allocator
		/// @param dst Destination span to place allocated fences into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_fences(std::span<VkFence> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate fences previously allocated from this Allocator
		/// @param src Span of fences to be deallocated
		void deallocate(std::span<const VkFence> src);

		/// @brief Allocate command pools from this Allocator
		/// @param dst Destination span to place allocated command pools into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate command pools from this Allocator
		/// @param dst Destination span to place allocated command pools into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_command_pools(std::span<CommandPool> dst, std::span<const VkCommandPoolCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate command pools previously allocated from this Allocator
		/// @param src Span of command pools to be deallocated
		void deallocate(std::span<const CommandPool> src);

		/// @brief Allocate command buffers from this Allocator
		/// @param dst Destination span to place allocated command buffers into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<CommandBufferAllocation> dst, std::span<const CommandBufferAllocationCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate command buffers from this Allocator
		/// @param dst Destination span to place allocated command buffers into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_command_buffers(std::span<CommandBufferAllocation> dst,
		                                                         std::span<const CommandBufferAllocationCreateInfo> cis,
		                                                         SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate command buffers previously allocated from this Allocator
		/// @param src Span of command buffers to be deallocated
		void deallocate(std::span<const CommandBufferAllocation> src);

		Result<void, AllocateException>
		allocate(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) = delete;
		Result<void, AllocateException>
		allocate_buffers(std::span<Buffer> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) = delete;

		/// @brief Allocate cross-device buffers from this Allocator
		/// @param dst Destination span to place allocated cross-device buffers into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate cross-device buffers from this Allocator
		/// @param dst Destination span to place allocated cross-device buffers into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_buffers(std::span<BufferCrossDevice> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate cross-device buffers previously allocated from this Allocator
		/// @param src Span of cross-device buffers to be deallocated
		void deallocate(std::span<const BufferCrossDevice> src);

		/// @brief Allocate GPU-only buffers from this Allocator
		/// @param dst Destination span to place allocated GPU-only buffers into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate GPU-only buffers from this Allocator
		/// @param dst Destination span to place allocated GPU-only buffers into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_buffers(std::span<BufferGPU> dst, std::span<const BufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate GPU-only buffers previously allocated from this Allocator
		/// @param src Span of GPU-only buffers to be deallocated
		void deallocate(std::span<const BufferGPU> src);

		/// @brief Allocate framebuffers from this Allocator
		/// @param dst Destination span to place allocated framebuffers into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate framebuffers from this Allocator
		/// @param dst Destination span to place allocated framebuffers into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_framebuffers(std::span<VkFramebuffer> dst, std::span<const FramebufferCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate framebuffers previously allocated from this Allocator
		/// @param src Span of framebuffers to be deallocated
		void deallocate(std::span<const VkFramebuffer> src);

		/// @brief Allocate images from this Allocator
		/// @param dst Destination span to place allocated images into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate images from this Allocator
		/// @param dst Destination span to place allocated images into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_images(std::span<Image> dst, std::span<const ImageCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate images previously allocated from this Allocator
		/// @param src Span of images to be deallocated
		void deallocate(std::span<const Image> src);

		/// @brief Allocate image views from this Allocator
		/// @param dst Destination span to place allocated image views into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate image views from this Allocator
		/// @param dst Destination span to place allocated image views into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_image_views(std::span<ImageView> dst, std::span<const ImageViewCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate image views previously allocated from this Allocator
		/// @param src Span of image views to be deallocated
		void deallocate(std::span<const ImageView> src);

		/// @brief Allocate persistent descriptor sets from this Allocator
		/// @param dst Destination span to place allocated persistent descriptor sets into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<PersistentDescriptorSet> dst, std::span<const PersistentDescriptorSetCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate persistent descriptor sets from this Allocator
		/// @param dst Destination span to place allocated persistent descriptor sets into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_persistent_descriptor_sets(std::span<PersistentDescriptorSet> dst,
		                                                                    std::span<const PersistentDescriptorSetCreateInfo> cis,
		                                                                    SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate persistent descriptor sets previously allocated from this Allocator
		/// @param src Span of persistent descriptor sets to be deallocated
		void deallocate(std::span<const PersistentDescriptorSet> src);

		/// @brief Allocate descriptor sets from this Allocator
		/// @param dst Destination span to place allocated descriptor sets into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate descriptor sets from this Allocator
		/// @param dst Destination span to place allocated descriptor sets into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_descriptor_sets(std::span<DescriptorSet> dst, std::span<const SetBinding> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate descriptor sets previously allocated from this Allocator
		/// @param src Span of descriptor sets to be deallocated
		void deallocate(std::span<const DescriptorSet> src);

		/// @brief Allocate timestamp query pools from this Allocator
		/// @param dst Destination span to place allocated timestamp query pools into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<TimestampQueryPool> dst, std::span<const VkQueryPoolCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate timestamp query pools from this Allocator
		/// @param dst Destination span to place allocated timestamp query pools into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_timestamp_query_pools(std::span<TimestampQueryPool> dst,
		                                                               std::span<const VkQueryPoolCreateInfo> cis,
		                                                               SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate timestamp query pools previously allocated from this Allocator
		/// @param src Span of timestamp query pools to be deallocated
		void deallocate(std::span<const TimestampQueryPool> src);

		/// @brief Allocate timestamp queries from this Allocator
		/// @param dst Destination span to place allocated timestamp queries into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate timestamp queries from this Allocator
		/// @param dst Destination span to place allocated timestamp queries into
		/// @param cis Per-element construction info
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException>
		allocate_timestamp_queries(std::span<TimestampQuery> dst, std::span<const TimestampQueryCreateInfo> cis, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate timestamp queries previously allocated from this Allocator
		/// @param src Span of timestamp queries to be deallocated
		void deallocate(std::span<const TimestampQuery> src);

		/// @brief Allocate timeline semaphores from this Allocator
		/// @param dst Destination span to place allocated timeline semaphores into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Allocate timeline semaphores from this Allocator
		/// @param dst Destination span to place allocated timeline semaphores into
		/// @param loc Source location information
		/// @return Result<void, AllocateException> : void or AllocateException if the allocation could not be performed.
		Result<void, AllocateException> allocate_timeline_semaphores(std::span<TimelineSemaphore> dst, SourceLocationAtFrame loc = VUK_HERE_AND_NOW());

		/// @brief Deallocate timeline semaphores previously allocated from this Allocator
		/// @param src Span of timeline semaphores to be deallocated
		void deallocate(std::span<const TimelineSemaphore> src);

		/// @brief Deallocate swapchains previously allocated from this Allocator
		/// @param src Span of swapchains to be deallocated
		void deallocate(std::span<const VkSwapchainKHR> src);

		/// @brief Get the underlying DeviceResource
		/// @return the underlying DeviceResource
		DeviceResource& get_device_resource() {
			return *device_resource;
		}

		/// @brief Get the parent Context
		/// @return the parent Context
		Context& get_context() {
			return *ctx;
		}

	private:
		Context* ctx;
		DeviceResource* device_resource;
	};

	template<class ContainerType>
	concept Container = requires(ContainerType a) {
		std::begin(a);
		std::end(a);
	};

	/// @brief Customization point for deallocation of user types
	/// @tparam T
	/// @param allocator
	/// @param src
	template<class T, size_t N>
	void deallocate(Allocator& allocator, T (&src)[N]) {
		allocator.deallocate(std::span<const T>{ src, N });
	}

	/// @brief Customization point for deallocation of user types
	/// @tparam T
	/// @param allocator
	/// @param src
	template<class T>
	void deallocate(Allocator& allocator, const T& src) requires(!Container<T>) {
		allocator.deallocate(std::span<const T>{ &src, 1 });
	}

	/// @brief Customization point for deallocation of user types
	/// @tparam T
	/// @param allocator
	/// @param src
	template<class T>
	void deallocate(Allocator& allocator, const T& src) requires(Container<T>) {
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
} // namespace vuk

#undef VUK_HERE_AND_NOW