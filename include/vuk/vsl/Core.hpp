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
	inline Value<Buffer<>> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer<> dst, const void* src_data, size_t size, VUK_CALLSTACK) {
		// host-mapped buffers just get memcpys
		if (&*dst.ptr) {
			memcpy(&*dst.ptr, src_data, size);
			return { acquire("_dst", dst, Access::eNone, VUK_CALL) };
		}

		auto src = *allocate_memory(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, 1 });
		::memcpy(&*src, src_data, size);

		auto src_buf = acquire("_src", Buffer<>{ src.get(), size }, Access::eNone, VUK_CALL);
		auto dst_buf = discard("_dst", dst, VUK_CALL);
		auto pass = make_pass("upload buffer", [](CommandBuffer& command_buffer, VUK_BA(Access::eTransferRead) src, VUK_BA(Access::eTransferWrite) dst) {
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
	template<class T = byte>
	Value<Buffer<T>> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer<T> dst, std::span<T> data, VUK_CALLSTACK) {
		return host_data_to_buffer(allocator, copy_domain, dst, data.data(), data.size_bytes(), VUK_CALL);
	}

	/// @brief Download a buffer to GPUtoCPU memory
	/// @param buffer_src Buffer to download
	template<class T = byte>
	inline Value<Buffer<T>> download_buffer(Value<Buffer<T>> buffer_src, VUK_CALLSTACK) {
		auto dst = declare_buf<T>("dst", { .memory_usage = MemoryUsage::eGPUtoCPU }, VUK_CALL);
		dst.same_size(buffer_src);
		auto download = make_pass("download buffer",
		                          [](CommandBuffer& command_buffer, VUK_ARG(Buffer<T>, Access::eTransferRead) src, VUK_ARG(Buffer<T>, Access::eTransferWrite) dst) {
			                          command_buffer.copy_buffer(src->to_byte_view(), dst->to_byte_view());
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
		auto src = *allocate_memory(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
		::memcpy(&*src, src_data, size);

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
		bc.bufferOffset = allocator.get_context().resolve_ptr(src.get()).buffer.offset;

		auto srcbuf = acquire("src", Buffer<>{ *src, size }, Access::eNone, VUK_CALL);
		auto dst = declare_ia("dst", image, VUK_CALL);
		auto image_upload = make_pass("image upload", [bc](CommandBuffer& command_buffer, VUK_BA(Access::eTransferRead) src, VUK_IA(Access::eTransferWrite) dst) {
			command_buffer.copy_buffer_to_image(src, dst, bc);
			return dst;
		});

		return image_upload(std::move(srcbuf), std::move(dst), VUK_CALL);
	}

	/// @brief Allocates & fills a buffer with explicitly managed lifetime
	/// @param allocator Allocator to allocate this Buffer from
	/// @param memory_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
	template<class T>
	std::pair<Unique<Buffer<T>>, Value<Buffer<T>>>
	create_buffer(Allocator& allocator, MemoryUsage memory_usage, DomainFlagBits domain, std::span<T> data, size_t alignment = 1, VUK_CALLSTACK) {
		Unique<Buffer> buf(allocator);
		BufferCreateInfo bci{ memory_usage, sizeof(T) * data.size(), alignment };
		auto ret = allocator.allocate_memory(std::span{ &*buf, 1 }, std::span{ &bci, 1 }); // TODO: dropping error
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
		    [=](CommandBuffer& cbuf, VUK_IA(Access::eBlitRead) src, VUK_IA(Access::eBlitWrite) dst) {
			    ImageBlit region = {};
			    region.srcOffsets[0] = Offset3D{};
			    auto src_extent = src->base_mip_extent();
			    region.srcOffsets[1] = Offset3D{ (int32_t)src_extent.width, (int32_t)src_extent.height, (int32_t)src_extent.depth };
			    region.dstOffsets[0] = Offset3D{};
			    auto dst_extent = dst->base_mip_extent();
			    region.dstOffsets[1] = Offset3D{ (int32_t)dst_extent.width, (int32_t)dst_extent.height, (int32_t)dst_extent.depth };
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

	template<class T>
	inline Value<Buffer<T>> copy(Value<ImageAttachment> src, Value<Buffer<T>> dst, VUK_CALLSTACK) {
		auto image2buf = make_pass("copy image to buffer", [](CommandBuffer& cbuf, VUK_IA(Access::eCopyRead) src, VUK_BA(Access::eTransferWrite) dst) {
			BufferImageCopy bc;
			bc.imageOffset = { 0, 0, 0 };
			bc.bufferRowLength = 0;
			bc.bufferImageHeight = 0;
			bc.imageExtent = src->base_mip_extent();
			bc.imageSubresource.aspectMask = format_to_aspect(src->format);
			bc.imageSubresource.mipLevel = src->base_level;
			bc.imageSubresource.baseArrayLayer = src->base_layer;
			assert(src->layer_count == 1); // unsupported yet
			bc.imageSubresource.layerCount = src->layer_count;
			auto& ae = cbuf.get_context().resolve_ptr(dst.ptr);
			bc.bufferOffset = ae.buffer.offset; // TODO: PAV: bad
			cbuf.copy_image_to_buffer(src, dst, bc);
			return dst;
		});

		return image2buf(src, dst, VUK_CALL);
	}

	template<class T>
	inline Value<Buffer<T>> copy(Value<Buffer<T>> src, Value<Buffer<T>> dst, VUK_CALLSTACK) {
		auto buf2buf =
		    vuk::make_pass("copy buffer to buffer", [](vuk::CommandBuffer& command_buffer, VUK_BA(vuk::eCopyRead) src, VUK_BA(vuk::eCopyWrite) dst) {
			    command_buffer.copy_buffer(src, dst);
			    return dst;
		    });
		return buf2buf(src, dst, VUK_CALL);
	}

	template<class T>
	inline void fill(Value<Buffer> dst, T value, VUK_CALLSTACK) {
		uint32_t value_as_uint;
		static_assert(sizeof(T) == sizeof(uint32_t), "T must be 4 bytes");
		memcpy(&value_as_uint, &value, sizeof(T));
		auto buf2buf = vuk::make_pass("fill buffer", [value_as_uint](vuk::CommandBuffer& command_buffer, VUK_BA(vuk::eClear) dst) {
			command_buffer.fill_buffer(dst, value_as_uint);
		    });
		buf2buf(dst, VUK_CALL);
	}

	inline Value<ImageAttachment> copy(Value<Buffer<T>> src, Value<ImageAttachment> dst, VUK_CALLSTACK) {
		auto buf2img = make_pass("copy buffer to image", [](CommandBuffer& cbuf, VUK_BA(Access::eCopyRead) src, VUK_IA(Access::eCopyWrite) dst) {
			BufferImageCopy bc;
			bc.imageOffset = { 0, 0, 0 };
			bc.bufferRowLength = 0;
			bc.bufferImageHeight = 0;
			bc.imageExtent = dst->base_mip_extent();
			bc.imageSubresource.aspectMask = format_to_aspect(dst->format);
			bc.imageSubresource.mipLevel = dst->base_level;
			bc.imageSubresource.baseArrayLayer = dst->base_layer;
			assert(dst->layer_count == 1); // unsupported yet
			bc.imageSubresource.layerCount = dst->layer_count;
			auto& ae = cbuf.get_context().resolve_ptr(dst.ptr);
			bc.bufferOffset = ae.buffer.offset; // TODO: PAV: bad
			cbuf.copy_buffer_to_image(src, dst, bc);
			return dst;
		});

		return buf2img(src, dst, VUK_CALL);
	}

	inline Value<ImageAttachment> copy(Value<ImageAttachment> src, Value<ImageAttachment> dst, VUK_CALLSTACK) {
		auto img2img = make_pass("copy image to image", [](CommandBuffer& cbuf, VUK_IA(Access::eCopyRead) src, VUK_IA(Access::eCopyWrite) dst) {
			assert(src->level_count == dst->level_count);

			ImageCopy bc;
			bc.imageExtent = dst->base_mip_extent();
			bc.srcOffsets = {};
			bc.srcSubresource.aspectMask = format_to_aspect(src->format);
			bc.srcSubresource.baseArrayLayer = src->base_layer;
			bc.srcSubresource.layerCount = src->layer_count;
			bc.dstOffsets = {};
			bc.dstSubresource.aspectMask = format_to_aspect(dst->format);
			bc.dstSubresource.baseArrayLayer = dst->base_layer;
			bc.dstSubresource.layerCount = dst->layer_count;

			for (uint32_t i = 0; i < src->level_count; i++) {
				bc.srcSubresource.mipLevel = src->base_level + i;
				bc.dstSubresource.mipLevel = dst->base_level + i;
				cbuf.copy_image(src, dst, bc);
			}

			return dst;
		});

		return img2img(src, dst, VUK_CALL);
	}

	inline Value<ImageAttachment> resolve_into(Value<ImageAttachment> src, Value<ImageAttachment> dst, VUK_CALLSTACK) {
		src.same_format_as(dst);
		src.same_shape_as(dst);
		dst->sample_count = Samples::e1;

		auto resolve = make_pass(
		    "resolve image",
		    [=](CommandBuffer& cbuf, VUK_IA(Access::eResolveRead) src, VUK_IA(Access::eResolveWrite) dst) {
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
	inline Value<ImageAttachment> generate_mips(Value<ImageAttachment> image, uint32_t base_mip, uint32_t num_mips) {
		for (uint32_t mip_level = base_mip + 1; mip_level < (base_mip + num_mips + 1); mip_level++) {
			blit_image(image.mip(mip_level - 1), image.mip(mip_level), Filter::eLinear);
		}

		return image;
	}
} // namespace vuk
