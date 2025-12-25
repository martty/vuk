#pragma once

#include "vuk/Exception.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/runtime/vk/Allocation.hpp"
#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/runtime/vk/Descriptor.hpp"
#include "vuk/runtime/vk/Query.hpp"
#include "vuk/SourceLocation.hpp"

namespace vuk {
	/// @brief Allocate a single semaphore from an Allocator
	/// @param allocator Allocator to use
	/// @param loc Source location information
	/// @return Semaphore in a RAII wrapper (Unique<T>) or AllocateException on error
	inline Result<Unique<VkSemaphore>, AllocateException> allocate_semaphore(Allocator& allocator, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<VkSemaphore> sema(allocator);
		if (auto res = allocator.allocate_semaphores(std::span{ &sema.get(), 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(sema) };
	}

	/// @brief Allocate a single command pool from an Allocator
	/// @param allocator Allocator to use
	/// @param cpci Command pool creation parameters
	/// @param loc Source location information
	/// @return Command pool in a RAII wrapper (Unique<T>) or AllocateException on error
	inline Result<Unique<CommandPool>, AllocateException>
	allocate_command_pool(Allocator& allocator, const VkCommandPoolCreateInfo& cpci, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<CommandPool> cp(allocator);
		if (auto res = allocator.allocate_command_pools(std::span{ &cp.get(), 1 }, std::span{ &cpci, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(cp) };
	}

	/// @brief Allocate a single command buffer from an Allocator
	/// @param allocator Allocator to use
	/// @param cbci Command buffer creation parameters
	/// @param loc Source location information
	/// @return Command buffer in a RAII wrapper (Unique<T>) or AllocateException on error
	inline Result<Unique<CommandBufferAllocation>, AllocateException>
	allocate_command_buffer(Allocator& allocator, const CommandBufferAllocationCreateInfo& cbci, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<CommandBufferAllocation> hlcb(allocator);
		if (auto res = allocator.allocate_command_buffers(std::span{ &hlcb.get(), 1 }, std::span{ &cbci, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(hlcb) };
	}

	/// @brief Allocate a single fence from an Allocator
	/// @param allocator Allocator to use
	/// @param loc Source location information
	/// @return Fence in a RAII wrapper (Unique<T>) or AllocateException on error
	inline Result<Unique<VkFence>, AllocateException> allocate_fence(Allocator& allocator, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<VkFence> fence(allocator);
		if (auto res = allocator.allocate_fences(std::span{ &fence.get(), 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(fence) };
	}

	template<class T = byte>
	inline Result<Unique<view<BufferLike<T>>>, AllocateException>
	allocate_buffer(Allocator& allocator, BufferCreateInfo bci, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<view<BufferLike<T>>> buf(allocator);
		if (auto res = allocator.allocate_memory(std::span{ static_cast<ptr_base*>(&buf->ptr), 1 }, std::span{ &bci, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		buf->sz_bytes = bci.size;
		return { expected_value, std::move(buf) };
	}

	/// @brief Allocate a single GPU-only buffer from an Allocator
	/// @param allocator Allocator to use
	/// @param bci Buffer creation parameters
	/// @param loc Source location information
	/// @return GPU-only buffer in a RAII wrapper (Unique<T>) or AllocateException on error
	template<class T>
	inline Result<Unique<ptr<BufferLike<T>>>, AllocateException>
	allocate_memory(Allocator& allocator, MemoryUsage memory_usage, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<ptr<BufferLike<T>>> buf(allocator);
		BufferCreateInfo bci{ .memory_usage = memory_usage, .size = sizeof(T), .alignment = alignof(T) };
		if (auto res = allocator.allocate_memory(std::span{ static_cast<ptr_base*>(&buf.get()), 1 }, std::span{ &bci, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(buf) };
	}

	template<class T>
	inline Result<Unique<view<BufferLike<T>>>, AllocateException>
	allocate_array(Allocator& allocator, size_t count, MemoryUsage memory_usage, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		BufferCreateInfo bci{ .memory_usage = memory_usage, .size = sizeof(T) * count, .alignment = alignof(T) };
		Unique<view<BufferLike<T>>> buf(allocator);
		if (auto res = allocator.allocate_memory(std::span{ static_cast<ptr_base*>(&buf->ptr), 1 }, std::span{ &bci, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		buf->sz_bytes = bci.size;
		return { expected_value, std::move(buf) };
	}

	/// @brief Allocate a single image from an Allocator
	/// @param allocator Allocator to use
	/// @param ici Image creation parameters
	/// @param loc Source location information
	/// @return Image in a RAII wrapper (Unique<T>) or AllocateException on error
	inline Result<Unique<Image<>>, AllocateException> allocate_image(Allocator& allocator, const ICI& ici, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<Image<>> img(allocator);
		if (auto res = allocator.allocate_images(std::span{ &img.get(), 1 }, std::span{ &ici, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(img) };
	}

	/// @brief Allocate a single image view from an Allocator
	/// @param allocator Allocator to use
	/// @param ivci Image view creation parameters
	/// @param loc Source location information
	/// @return ImageView in a RAII wrapper (Unique<T>) or AllocateException on error
	inline Result<Unique<ImageView<>>, AllocateException>
	allocate_image_view(Allocator& allocator, const IVCI& ivci, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<ImageView<>> iv(allocator);
		if (auto res = allocator.allocate_image_views(std::span{ &iv.get(), 1 }, std::span{ &ivci, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(iv) };
	}
} // namespace vuk