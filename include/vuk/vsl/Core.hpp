#pragma once

#include "vuk/RenderGraph.hpp"
#include "vuk/runtime/CommandBuffer.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/SourceLocation.hpp"
#include "vuk/Value.hpp"
#include <math.h>
#include <span>

namespace vuk {
	/// @brief Fill a buffer with host data
	/// @param allocator Allocator to use for temporary allocations
	/// @param copy_domain The domain where the copy should happen (when dst is mapped, the copy happens on host)
	/// @param buffer Buffer to fill
	/// @param src_data pointer to source data
	/// @param size size of source data
	template<class T>
	inline Value<Buffer<T>>
	host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer<T> dst, const void* src_data, size_t size, VUK_CALLSTACK) {
		// host-mapped buffers just get memcpys
		if (&*dst.ptr) {
			memcpy(&*dst.ptr, src_data, size);
			return { acquire("_dst", dst, Access::eNone, VUK_CALL) };
		}

		auto src = *allocate_buffer(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, 1 });
		::memcpy(&src[0], src_data, size);

		auto src_buf = acquire("_src", src.get(), Access::eNone, VUK_CALL);
		auto dst_buf = discard("_dst", dst, VUK_CALL);
		auto pass = make_pass("upload buffer",
		                      [](CommandBuffer& command_buffer, VUK_ARG(Buffer<>, Access::eTransferRead) src, VUK_ARG(Buffer<T>, Access::eTransferWrite) dst) {
			                      command_buffer.copy_buffer(src, dst->to_byte_view());
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
	Value<Buffer<T>> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer<T> dst, std::span<T> data, VUK_CALLSTACK) {
		return host_data_to_buffer(allocator, copy_domain, dst, data.data(), data.size_bytes(), VUK_CALL);
	}

	/// @brief Fill a buffer with host data
	/// @param allocator Allocator to use for temporary allocations
	/// @param copy_domain The domain where the copy should happen (when dst is mapped, the copy happens on host)
	/// @param dst Buffer to fill
	/// @param data source data
	template<class T>
	Value<Buffer<T>> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer<T> dst, std::span<const T> data, VUK_CALLSTACK) {
		return host_data_to_buffer(allocator, copy_domain, dst, data.data(), data.size_bytes(), VUK_CALL);
	}

	/// @brief Download a buffer to GPUtoCPU memory
	/// @param buffer_src Buffer to download
	template<class T>
	inline Value<Buffer<T>> download_buffer(Value<Buffer<T>> buffer_src, VUK_CALLSTACK) {
		auto dst = allocate<T>("dst", BufferCreateInfo{ .memory_usage = MemoryUsage::eGPUtoCPU }, VUK_CALL);
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
	/// @param image ImageView<> to fill
	/// @param src_data pointer to source data
	inline Value<ImageView<>> host_data_to_image(Allocator& allocator, DomainFlagBits copy_domain, ImageView<> image, const void* src_data, VUK_CALLSTACK) {
		auto& ve = image.get_meta();
		size_t alignment = format_to_texel_block_size(ve.format);
		size_t size = compute_image_size(ve.format, ve.extent);
		auto src = *allocate_buffer(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
		::memcpy(&src[0], src_data, size);
		BufferImageCopy bc;
		bc.imageOffset = { 0, 0, 0 };
		bc.bufferRowLength = 0;
		bc.bufferImageHeight = 0;
		bc.imageExtent = ve.extent;
		bc.imageSubresource.aspectMask = format_to_aspect(ve.format);
		bc.imageSubresource.mipLevel = ve.base_level;
		bc.imageSubresource.baseArrayLayer = ve.base_layer;
		assert(ve.layer_count == 1); // unsupported yet
		bc.imageSubresource.layerCount = ve.layer_count;
		bc.bufferOffset = allocator.get_context().ptr_to_buffer_offset(src->ptr).offset;

		auto srcbuf = acquire("src", *src, Access::eNone, VUK_CALL);
		auto dst = discard("dst", image, VUK_CALL);
		auto image_upload = make_pass("image upload", [bc](CommandBuffer& command_buffer, VUK_BA(Access::eTransferRead) src, VUK_IA(Access::eTransferWrite) dst) {
			command_buffer.copy_buffer_to_image(src, dst, bc);
			return dst;
		});

		return image_upload(std::move(srcbuf), std::move(dst), VUK_CALL);
	}

	/// @brief Allocates & fills a buffer with explicitly managed lifetime
	/// @param allocator Allocator to allocate this Buffer from
	/// @param memory_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
	template<class T, size_t Extent = dynamic_extent>
	std::pair<Unique<Buffer<std::remove_const_t<T>>>, Value<Buffer<std::remove_const_t<T>>>>
	create_buffer(Allocator& allocator, MemoryUsage memory_usage, DomainFlagBits domain, std::span<T, Extent> data, size_t alignment = 1, VUK_CALLSTACK) {
		Unique<Buffer<std::remove_const_t<T>>> buf(allocator);
		BufferCreateInfo bci{ memory_usage, sizeof(T) * data.size(), alignment };
		ptr_base ptr_;
		auto ret = allocator.allocate_memory(std::span{ &ptr_, 1 }, std::span{ &bci, 1 }); // TODO: dropping error
		buf->ptr = static_cast<ptr<BufferLike<std::remove_const_t<T>>>&>(ptr_);
		buf->sz_bytes = data.size_bytes();
		return { std::move(buf), host_data_to_buffer(allocator, domain, buf.get(), std::span<T, dynamic_extent>(data), VUK_CALL) };
	}

	inline std::pair<Unique<ImageView<>>, Value<ImageView<>>>
	create_image_with_data(Allocator& allocator, DomainFlagBits copy_domain, ICI ici, const void* data, VUK_CALLSTACK) {
		auto result = allocate_image(allocator, ici, VUK_CALL);
		if (!result) {
			return { Unique<ImageView<>>{ allocator }, Value<ImageView<>>{} };
		}
		auto& image = *result;
		auto view = Unique<ImageView<>>(allocator, image->default_view());
		return { std::move(view), host_data_to_image(allocator, copy_domain, image->default_view(), data, VUK_CALL) };
	}

	template<class T>
	std::pair<Unique<ImageView<>>, Value<ImageView<>>>
	create_image_with_data(Allocator& allocator, DomainFlagBits copy_domain, ICI ici, std::span<T> data, VUK_CALLSTACK) {
		return create_image_with_data(allocator, copy_domain, ici, data.data(), VUK_CALL);
	}

	inline Value<ImageView<>> clear_image(Value<ImageView<>> in, Clear clear_value, VUK_CALLSTACK) {
		auto clear = make_pass(
		    "clear image",
		    [=](CommandBuffer& cbuf, VUK_IA(Access::eClear) dst) {
			    cbuf.clear_image(dst, clear_value);
			    return dst;
		    },
		    DomainFlagBits::eGraphicsQueue);

		return clear(std::move(in), VUK_CALL);
	}

	inline Value<ImageView<>> blit_image(Value<ImageView<>> src, Value<ImageView<>> dst, Filter filter, VUK_CALLSTACK) {
		auto blit = make_pass(
		    "blit image",
		    [=](CommandBuffer& cbuf, VUK_IA(Access::eBlitRead) src, VUK_IA(Access::eBlitWrite) dst) {
			    auto& src_ve = src->get_meta();
			    auto& dst_ve = dst->get_meta();

			    ImageBlit region = {};
			    region.srcOffsets[0] = Offset3D{};
			    auto src_extent = src->base_mip_extent();
			    region.srcOffsets[1] = Offset3D{ (int32_t)src_extent.width, (int32_t)src_extent.height, (int32_t)src_extent.depth };
			    region.dstOffsets[0] = Offset3D{};
			    auto dst_extent = dst->base_mip_extent();
			    region.dstOffsets[1] = Offset3D{ (int32_t)dst_extent.width, (int32_t)dst_extent.height, (int32_t)dst_extent.depth };
			    region.srcSubresource.aspectMask = format_to_aspect(src_ve.format);
			    region.srcSubresource.baseArrayLayer = src_ve.base_layer;
			    region.srcSubresource.layerCount = src_ve.layer_count;
			    region.srcSubresource.mipLevel = src_ve.base_level;
			    assert(src_ve.level_count == 1);
			    region.dstSubresource.baseArrayLayer = dst_ve.base_layer;
			    region.dstSubresource.layerCount = dst_ve.layer_count;
			    region.dstSubresource.mipLevel = dst_ve.base_level;
			    assert(dst_ve.level_count == 1);
			    region.dstSubresource.aspectMask = format_to_aspect(dst_ve.format);

			    cbuf.blit_image(src, dst, region, filter);
			    return dst;
		    },
		    DomainFlagBits::eGraphicsQueue);

		return blit(std::move(src), std::move(dst), VUK_CALL);
	}

	template<class T>
	inline Value<Buffer<T>> copy(Value<ImageView<>> src, Value<Buffer<T>> dst, VUK_CALLSTACK) {
		auto image2buf = make_pass("copy image to buffer", [](CommandBuffer& cbuf, VUK_IA(Access::eCopyRead) src, VUK_ARG(Buffer<T>, Access::eCopyWrite) dst) {
			auto& src_ve = src->get_meta();

			BufferImageCopy bc;
			bc.imageOffset = { 0, 0, 0 };
			bc.bufferRowLength = 0;
			bc.bufferImageHeight = 0;
			bc.imageExtent = src->base_mip_extent();
			bc.imageSubresource.aspectMask = format_to_aspect(src_ve.format);
			bc.imageSubresource.mipLevel = src_ve.base_level;
			bc.imageSubresource.baseArrayLayer = src_ve.base_layer;
			assert(src_ve.layer_count == 1); // unsupported yet
			bc.imageSubresource.layerCount = src_ve.layer_count;
			bc.bufferOffset = cbuf.get_context().ptr_to_buffer_offset(dst->ptr).offset; // TODO: PAV: bad
			cbuf.copy_image_to_buffer(src, dst->to_byte_view(), bc);
			return dst;
		});

		return image2buf(src, dst, VUK_CALL);
	}

	template<class T>
	inline Value<Buffer<T>> copy(Value<Buffer<T>> src, Value<Buffer<T>> dst, VUK_CALLSTACK) {
		auto buf2buf = vuk::make_pass("copy buffer to buffer",
		                              [](vuk::CommandBuffer& command_buffer, VUK_ARG(Buffer<T>, vuk::eCopyRead) src, VUK_ARG(Buffer<T>, vuk::eCopyWrite) dst) {
			                              command_buffer.copy_buffer(src->to_byte_view(), dst->to_byte_view());
			                              return dst;
		                              });
		return buf2buf(src, dst, VUK_CALL);
	}

	template<class T>
	inline void fill(Value<Buffer<T>> dst, T value, VUK_CALLSTACK) {
		uint32_t value_as_uint;
		unsigned char* p = reinterpret_cast<unsigned char*>(&value_as_uint);
		static_assert(sizeof(T) <= sizeof(uint32_t), "T must be at most 4 bytes");
		for (size_t i = 0; i < (sizeof(uint32_t) / sizeof(T)); i++) {
			memcpy(p + i * sizeof(T), &value, sizeof(T));
		}
		auto buf2buf = vuk::make_pass("fill buffer", [value_as_uint](vuk::CommandBuffer& command_buffer, VUK_ARG(Buffer<T>, vuk::eTransferWrite) dst) {
			command_buffer.fill_buffer(dst->to_byte_view(), value_as_uint);
		});
		buf2buf(dst, VUK_CALL);
	}

	template<class T>
	inline Value<ImageView<>> copy(Value<Buffer<T>> src, Value<ImageView<>> dst, VUK_CALLSTACK) {
		auto buf2img = make_pass("copy buffer to image", [](CommandBuffer& cbuf, VUK_ARG(Buffer<T>, Access::eCopyRead) src, VUK_IA(Access::eCopyWrite) dst) {
			auto& dst_ve = dst->get_meta();

			BufferImageCopy bc;
			bc.imageOffset = { 0, 0, 0 };
			bc.bufferRowLength = 0;
			bc.bufferImageHeight = 0;
			bc.imageExtent = dst->base_mip_extent();
			bc.imageSubresource.aspectMask = format_to_aspect(dst_ve.format);
			bc.imageSubresource.mipLevel = dst_ve.base_level;
			bc.imageSubresource.baseArrayLayer = dst_ve.base_layer;
			assert(dst_ve.layer_count == 1); // unsupported yet
			bc.imageSubresource.layerCount = dst_ve.layer_count;
			bc.bufferOffset = cbuf.get_context().ptr_to_buffer_offset(src->ptr).offset; // TODO: PAV: bad
			cbuf.copy_buffer_to_image(src->to_byte_view(), dst, bc);
			return dst;
		});

		return buf2img(src, dst, VUK_CALL);
	}

	inline Value<ImageView<>> copy(Value<ImageView<>> src, Value<ImageView<>> dst, VUK_CALLSTACK) {
		auto img2img = make_pass("copy image to image", [](CommandBuffer& cbuf, VUK_IA(Access::eCopyRead) src, VUK_IA(Access::eCopyWrite) dst) {
			auto& src_ve = src->get_meta();
			auto& dst_ve = dst->get_meta();

			assert(src_ve.level_count == dst_ve.level_count);

			ImageCopy bc;
			bc.imageExtent = dst->base_mip_extent();
			bc.srcOffsets = {};
			bc.srcSubresource.aspectMask = format_to_aspect(src_ve.format);
			bc.srcSubresource.baseArrayLayer = src_ve.base_layer;
			bc.srcSubresource.layerCount = src_ve.layer_count;
			bc.dstOffsets = {};
			bc.dstSubresource.aspectMask = format_to_aspect(dst_ve.format);
			bc.dstSubresource.baseArrayLayer = dst_ve.base_layer;
			bc.dstSubresource.layerCount = dst_ve.layer_count;

			for (uint32_t i = 0; i < src_ve.level_count; i++) {
				bc.srcSubresource.mipLevel = src_ve.base_level + i;
				bc.dstSubresource.mipLevel = dst_ve.base_level + i;
				cbuf.copy_image(src, dst, bc);
			}

			return dst;
		});

		return img2img(src, dst, VUK_CALL);
	}

	inline Value<ImageView<>> resolve_into(Value<ImageView<>> src, Value<ImageView<>> dst, VUK_CALLSTACK) {
		src.same_format_as(dst);
		src.same_shape_as(dst);
		// TODO: set dst as single sampled

		auto resolve = make_pass(
		    "resolve image",
		    [=](CommandBuffer& cbuf, VUK_IA(Access::eResolveRead) src, VUK_IA(Access::eResolveWrite) dst) {
			    cbuf.resolve_image(src, dst);
			    return dst;
		    },
		    DomainFlagBits::eGraphicsQueue);

		return resolve(std::move(src), std::move(dst), VUK_CALL);
	}

	/// @brief Generate mips for given ImageView<>

	/// @param image input Future of ImageView<>
	/// @param base_mip source mip level
	/// @param num_mips number of mip levels to generate
	inline Value<ImageView<>> generate_mips(Value<ImageView<>> image, uint32_t base_mip, uint32_t num_mips) {
		for (uint32_t mip_level = base_mip + 1; mip_level < (base_mip + num_mips + 1); mip_level++) {
			blit_image(image.mip(mip_level - 1), image.mip(mip_level), Filter::eLinear);
		}

		return image;
	}
} // namespace vuk
