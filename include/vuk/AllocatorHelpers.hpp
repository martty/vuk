#pragma once

#include "vuk/Allocator.hpp"
#include "vuk/Buffer.hpp"
#include "vuk/Exception.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/Descriptor.hpp"
#include "vuk/Query.hpp"

/// @cond INTERNAL
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
/// @endcond

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

	/// @brief Allocate a single timeline semaphore from an Allocator
	/// @param allocator Allocator to use
	/// @param loc Source location information
	/// @return Timeline semaphore in a RAII wrapper (Unique<T>) or AllocateException on error
	inline Result<Unique<TimelineSemaphore>, AllocateException> allocate_timeline_semaphore(Allocator& allocator,
	                                                                                        SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<TimelineSemaphore> sema(allocator);
		if (auto res = allocator.allocate_timeline_semaphores(std::span{ &sema.get(), 1 }, loc); !res) {
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

	/// @brief Allocate a single GPU-only buffer from an Allocator
	/// @param allocator Allocator to use
	/// @param bci Buffer creation parameters
	/// @param loc Source location information
	/// @return GPU-only buffer in a RAII wrapper (Unique<T>) or AllocateException on error
	inline Result<Unique<Buffer>, AllocateException>
	allocate_buffer(Allocator& allocator, const BufferCreateInfo& bci, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<Buffer> buf(allocator);
		if (auto res = allocator.allocate_buffers(std::span{ &buf.get(), 1 }, std::span{ &bci, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(buf) };
	}

	/// @brief Allocate a single image from an Allocator
	/// @param allocator Allocator to use
	/// @param ici Image creation parameters
	/// @param loc Source location information
	/// @return Image in a RAII wrapper (Unique<T>) or AllocateException on error
	inline Result<Unique<Image>, AllocateException>
	allocate_image(Allocator& allocator, const ImageCreateInfo& ici, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<Image> img(allocator);
		if (auto res = allocator.allocate_images(std::span{ &img.get(), 1 }, std::span{ &ici, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(img) };
	}

	/// @brief Allocate a single image from an Allocator
	/// @param allocator Allocator to use
	/// @param attachment ImageAttachment to make the Image from
	/// @param loc Source location information
	/// @return Image in a RAII wrapper (Unique<T>) or AllocateException on error
	inline Result<Unique<Image>, AllocateException>
	allocate_image(Allocator& allocator, const ImageAttachment& attachment, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<Image> img(allocator);
		ImageCreateInfo ici;
		ici.format = vuk::Format(attachment.format);
		ici.imageType = attachment.image_type;
		ici.flags = attachment.image_flags;
		ici.arrayLayers = attachment.layer_count;
		ici.samples = attachment.sample_count.count;
		ici.tiling = attachment.tiling;
		ici.mipLevels = attachment.level_count;
		ici.usage = attachment.usage;
		assert(attachment.extent.sizing == Sizing::eAbsolute);
		ici.extent = static_cast<vuk::Extent3D>(attachment.extent.extent);

		VkImageFormatListCreateInfo listci = { VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO };
		if (attachment.allow_srgb_unorm_mutable) {
			auto unorm_fmt = srgb_to_unorm(attachment.format);
			auto srgb_fmt = unorm_to_srgb(attachment.format);
			VkFormat formats[2] = { (VkFormat)attachment.format, unorm_fmt == vuk::Format::eUndefined ? (VkFormat)srgb_fmt : (VkFormat)unorm_fmt };
			listci.pViewFormats = formats;
			listci.viewFormatCount = formats[1] == VK_FORMAT_UNDEFINED ? 1 : 2;
			if (listci.viewFormatCount > 1) {
				ici.flags |= vuk::ImageCreateFlagBits::eMutableFormat;
				ici.pNext = &listci;
			}
		}

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
	inline Result<Unique<ImageView>, AllocateException>
	allocate_image_view(Allocator& allocator, const ImageViewCreateInfo& ivci, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<ImageView> iv(allocator);
		if (auto res = allocator.allocate_image_views(std::span{ &iv.get(), 1 }, std::span{ &ivci, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(iv) };
	}

	/// @brief Allocate a single image view from an Allocator
	/// @param allocator Allocator to use
	/// @param attachment ImageAttachment to make the ImageView from
	/// @param loc Source location information
	/// @return ImageView in a RAII wrapper (Unique<T>) or AllocateException on error
	inline Result<Unique<ImageView>, AllocateException>
	allocate_image_view(Allocator& allocator, const ImageAttachment& attachment, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		Unique<ImageView> iv(allocator);
		ImageViewCreateInfo ivci;
		assert(attachment.image);
		ivci.flags = attachment.image_view_flags;
		ivci.image = attachment.image.image;
		ivci.viewType = attachment.view_type;
		ivci.format = vuk::Format(attachment.format);
		ivci.components = attachment.components;
		ivci.view_usage = attachment.usage;

		ImageSubresourceRange& isr = ivci.subresourceRange;
		isr.aspectMask = format_to_aspect(ivci.format);
		isr.baseArrayLayer = attachment.base_layer;
		isr.layerCount = attachment.layer_count;
		isr.baseMipLevel = attachment.base_level;
		isr.levelCount = attachment.level_count;

		if (auto res = allocator.allocate_image_views(std::span{ &iv.get(), 1 }, std::span{ &ivci, 1 }, loc); !res) {
			return { expected_error, res.error() };
		}
		return { expected_value, std::move(iv) };
	}
} // namespace vuk

#undef VUK_HERE_AND_NOW