#pragma once

#include "vuk/Allocator.hpp"
#include "vuk/Exception.hpp"

#ifndef __cpp_consteval
#define VUK_HERE_AND_NOW() SourceLocationAtFrame{ vuk::source_location::current(), (uint64_t)-1LL }
#else
#define VUK_HERE_AND_NOW() SourceLocationAtFrame{ std::source_location::current(), (uint64_t)-1LL }
#endif

namespace vuk {
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
}

#undef VUK_HERE_AND_NOW