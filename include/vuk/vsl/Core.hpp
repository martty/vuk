#pragma once

#include "vuk/RenderGraph.hpp"
#include "vuk/SourceLocation.hpp"
#include "vuk/Value.hpp"
#include "vuk/runtime/CommandBuffer.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include <math.h>
#include <span>

namespace vuk {
	/// @brief Fill a buffer with host data
	/// @param allocator Allocator to use for temporary allocations
	/// @param copy_domain The domain where the copy should happen (when dst is mapped, the copy happens on host)
	/// @param buffer Buffer to fill
	/// @param src_data pointer to source data
	/// @param size size of source data
	inline Value<Buffer> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer dst, const void* src_data, size_t size, VUK_CALLSTACK) {
		// host-mapped buffers just get memcpys
		if (dst.mapped_ptr) {
			memcpy(dst.mapped_ptr, src_data, size);
			return { vuk::acquire_buf("_dst", dst, Access::eNone, VUK_CALL) };
		}

		auto src = *allocate_buffer(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, 1 });
		::memcpy(src->mapped_ptr, src_data, size);

		auto src_buf = vuk::acquire_buf("_src", *src, Access::eNone, VUK_CALL);
		auto dst_buf = vuk::discard_buf("_dst", dst, VUK_CALL);
		auto pass = vuk::make_pass("upload buffer", [](vuk::CommandBuffer& command_buffer, VUK_BA(Access::eTransferRead) src, VUK_BA(Access::eTransferWrite) dst) {
			command_buffer.copy_buffer(src, dst);
			return dst;
		});
		return pass(std::move(src_buf), std::move(dst_buf), VUK_CALL);
	}

	/// @brief Fill a buffer with host data
	/// @param allocator Allocator to use for temporary allocations
	/// @param copy_domain The domain where the copy should happen (when dst is mapped, the copy happens on host)
	/// @param dst Buffer to fill
	/// @param data source data
	template<class T>
	Value<Buffer> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer dst, std::span<T> data, VUK_CALLSTACK) {
		return host_data_to_buffer(allocator, copy_domain, dst, data.data(), data.size_bytes(), VUK_CALL);
	}

	/// @brief Download a buffer to GPUtoCPU memory
	/// @param buffer_src Buffer to download
	inline Value<Buffer> download_buffer(Value<Buffer> buffer_src, VUK_CALLSTACK) {
		auto dst = declare_buf("dst", Buffer{ .memory_usage = MemoryUsage::eGPUtoCPU }, VUK_CALL);
		dst.same_size(buffer_src);
		auto download =
		    vuk::make_pass("download buffer", [](vuk::CommandBuffer& command_buffer, VUK_BA(Access::eTransferRead) src, VUK_BA(Access::eTransferWrite) dst) {
			    command_buffer.copy_buffer(src, dst);
			    return dst;
		    });
		return download(std::move(buffer_src), std::move(dst), VUK_CALL);
	}

	/// @brief Fill an image with host data
	/// @param allocator Allocator to use for temporary allocations
	/// @param copy_domain The domain where the copy should happen
	/// @param image ImageAttachment to fill
	/// @param src_data pointer to source data
	inline Value<ImageAttachment>
	host_data_to_image(Allocator& allocator, DomainFlagBits copy_domain, ImageAttachment image, const void* src_data, VUK_CALLSTACK) {
		size_t alignment = format_to_texel_block_size(image.format);
		size_t size = compute_image_size(image.format, image.extent);
		auto src = *allocate_buffer(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
		::memcpy(src->mapped_ptr, src_data, size);

		BufferImageCopy bc;
		bc.imageOffset = { 0, 0, 0 };
		bc.bufferRowLength = 0;
		bc.bufferImageHeight = 0;
		bc.imageExtent = image.extent;
		bc.imageSubresource.aspectMask = format_to_aspect(image.format);
		bc.imageSubresource.mipLevel = image.base_level;
		bc.imageSubresource.baseArrayLayer = image.base_layer;
		assert(image.layer_count == 1); // unsupported yet
		bc.imageSubresource.layerCount = image.layer_count;
		bc.bufferOffset = src->offset;

		auto srcbuf = acquire_buf("src", *src, Access::eNone, VUK_CALL);
		auto dst = declare_ia("dst", image, VUK_CALL);
		auto image_upload =
		    vuk::make_pass("image upload", [bc](vuk::CommandBuffer& command_buffer, VUK_BA(Access::eTransferRead) src, VUK_IA(Access::eTransferWrite) dst) {
			    command_buffer.copy_buffer_to_image(src, dst, bc);
			    return dst;
		    });

		return image_upload(std::move(srcbuf), std::move(dst), VUK_CALL);
	}

	/// @brief Allocates & fills a buffer with explicitly managed lifetime
	/// @param allocator Allocator to allocate this Buffer from
	/// @param mem_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
	template<class T>
	std::pair<Unique<Buffer>, Value<Buffer>>
	create_buffer(Allocator& allocator, vuk::MemoryUsage memory_usage, DomainFlagBits domain, std::span<T> data, size_t alignment = 1, VUK_CALLSTACK) {
		Unique<Buffer> buf(allocator);
		BufferCreateInfo bci{ memory_usage, sizeof(T) * data.size(), alignment };
		auto ret = allocator.allocate_buffers(std::span{ &*buf, 1 }, std::span{ &bci, 1 }); // TODO: dropping error
		Buffer b = buf.get();
		return { std::move(buf), host_data_to_buffer(allocator, domain, b, data, VUK_CALL) };
	}

	inline std::pair<Unique<Image>, Value<ImageAttachment>>
	create_image_with_data(Allocator& allocator, DomainFlagBits copy_domain, ImageAttachment& ia, const void* data, VUK_CALLSTACK) {
		auto image = allocate_image(allocator, ia, VUK_CALL);
		ia.image = **image;
		return { std::move(*image), host_data_to_image(allocator, copy_domain, ia, data, VUK_CALL) };
	}

	template<class T>
	std::pair<Unique<Image>, Value<ImageAttachment>>
	create_image_with_data(Allocator& allocator, DomainFlagBits copy_domain, ImageAttachment& ia, std::span<T> data, VUK_CALLSTACK) {
		return create_image_with_data(allocator, copy_domain, ia, data.data(), VUK_CALL);
	}

	inline std::tuple<Unique<Image>, Unique<ImageView>, Value<ImageAttachment>>
	create_image_and_view_with_data(Allocator& allocator, DomainFlagBits copy_domain, ImageAttachment& ia, const void* data, VUK_CALLSTACK) {
		auto image = allocate_image(allocator, ia, VUK_CALL);
		ia.image = **image;
		auto view = allocate_image_view(allocator, ia, VUK_CALL);
		ia.image_view = **view;
		return { std::move(*image), std::move(*view), host_data_to_image(allocator, copy_domain, ia, data, VUK_CALL) };
	}

	template<class T>
	std::tuple<Unique<Image>, Unique<ImageView>, Value<ImageAttachment>>
	create_image_and_view_with_data(Allocator& allocator, DomainFlagBits copy_domain, ImageAttachment ia, std::span<T> data, VUK_CALLSTACK) {
		return create_image_and_view_with_data(allocator, copy_domain, ia, data.data(), VUK_CALL);
	}

	inline Value<ImageAttachment> clear_image(Value<ImageAttachment> in, Clear clear_value, VUK_CALLSTACK) {
		auto clear = make_pass(
		    "clear image",
		    [=](CommandBuffer& cbuf, VUK_IA(Access::eClear) dst) {
			    cbuf.clear_image(dst, clear_value);
			    return dst;
		    },
		    DomainFlagBits::eGraphicsQueue);

		return clear(std::move(in), VUK_CALL);
	}

	inline Value<ImageAttachment> blit_image(Value<ImageAttachment> src, Value<ImageAttachment> dst, Filter filter, VUK_CALLSTACK) {
		auto blit = make_pass(
		    "blit image",
		    [=](CommandBuffer& cbuf, VUK_IA(Access::eTransferRead) src, VUK_IA(Access::eTransferWrite) dst) {
			    ImageBlit region = {};
			    region.srcOffsets[0] = Offset3D{};
			    region.srcOffsets[1] = Offset3D{ std::max(static_cast<int32_t>(src->extent.width) >> src->base_level, 1),
				                                   std::max(static_cast<int32_t>(src->extent.height) >> src->base_level, 1),
				                                   std::max(static_cast<int32_t>(src->extent.depth) >> src->base_level, 1) };
			    region.dstOffsets[0] = Offset3D{};
			    region.dstOffsets[1] = Offset3D{ std::max(static_cast<int32_t>(dst->extent.width) >> dst->base_level, 1),
				                                   std::max(static_cast<int32_t>(dst->extent.height) >> dst->base_level, 1),
				                                   std::max(static_cast<int32_t>(dst->extent.depth) >> dst->base_level, 1) };
			    region.srcSubresource.aspectMask = format_to_aspect(src->format);
			    region.srcSubresource.baseArrayLayer = src->base_layer;
			    region.srcSubresource.layerCount = src->layer_count;
			    region.srcSubresource.mipLevel = src->base_level;
			    assert(src->level_count == 1);
			    region.dstSubresource.baseArrayLayer = dst->base_layer;
			    region.dstSubresource.layerCount = dst->layer_count;
			    region.dstSubresource.mipLevel = dst->base_level;
			    assert(dst->level_count == 1);
			    region.dstSubresource.aspectMask = format_to_aspect(dst->format);

			    cbuf.blit_image(src, dst, region, filter);
			    return dst;
		    },
		    DomainFlagBits::eGraphicsQueue);

		return blit(std::move(src), std::move(dst), VUK_CALL);
	}

	inline Value<ImageAttachment> resolve_into(Value<ImageAttachment> src, Value<ImageAttachment> dst, VUK_CALLSTACK) {
		src.same_format_as(dst);
		src.same_shape_as(dst);
		dst->sample_count = Samples::e1;

		auto resolve = make_pass(
		    "resolve image",
		    [=](CommandBuffer& cbuf, VUK_IA(Access::eTransferRead) src, VUK_IA(Access::eTransferWrite) dst) {
			    cbuf.resolve_image(src, dst);
			    return dst;
		    },
		    DomainFlagBits::eGraphicsQueue);

		return resolve(std::move(src), std::move(dst), VUK_CALL);
	}

	/// @brief Generate mips for given ImageAttachment
	/// @param image input Future of ImageAttachment
	/// @param base_mip source mip level
	/// @param num_mips number of mip levels to generate
	inline vuk::Value<vuk::ImageAttachment> generate_mips(vuk::Value<vuk::ImageAttachment> image, uint32_t base_mip, uint32_t num_mips) {
		for (uint32_t mip_level = base_mip + 1; mip_level < (base_mip + num_mips + 1); mip_level++) {
			blit_image(image.mip(mip_level - 1), image.mip(mip_level), Filter::eLinear);
		}

		return image;
	}
} // namespace vuk
