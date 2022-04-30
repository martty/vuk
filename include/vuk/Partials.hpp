#pragma once

#include "vuk/AllocatorHelpers.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/Future.hpp"
#include <math.h>
#include <span>

namespace vuk {
	/// @brief Fill a buffer with host data
	/// @param allocator Allocator to use for temporary allocations
	/// @param copy_domain The domain where the copy should happen (when dst is mapped, the copy happens on host)
	/// @param buffer Buffer to fill
	/// @param src_data pointer to source data
	/// @param size size of source data
	inline Future<Buffer> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer dst, const void* src_data, size_t size) {
		// host-mapped buffers just get memcpys
		if (dst.mapped_ptr) {
			memcpy(dst.mapped_ptr, src_data, size);
			return { allocator, std::move(dst) };
		}

		auto src = *allocate_buffer_cross_device(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, 1 });
		::memcpy(src->mapped_ptr, src_data, size);

		std::unique_ptr<RenderGraph> rgp = std::make_unique<RenderGraph>();
		rgp->add_pass({ .name = "BUFFER UPLOAD",
		                .execute_on = copy_domain,
		                .resources = { "_dst"_buffer >> vuk::Access::eTransferWrite, "_src"_buffer >> vuk::Access::eTransferRead },
		                .execute = [size](vuk::CommandBuffer& command_buffer) {
			                command_buffer.copy_buffer("_src", "_dst", size);
		                } });
		rgp->attach_buffer("_src", *src, vuk::Access::eNone, vuk::Access::eNone);
		rgp->attach_buffer("_dst", dst, vuk::Access::eNone, vuk::Access::eNone);
		return { allocator, std::move(rgp), "_dst+" };
	}

	/// @brief Fill a buffer with host data
	/// @param allocator Allocator to use for temporary allocations
	/// @param copy_domain The domain where the copy should happen (when dst is mapped, the copy happens on host)
	/// @param dst Buffer to fill
	/// @param data source data
	template<class T>
	Future<Buffer> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer dst, std::span<T> data) {
		return host_data_to_buffer(allocator, copy_domain, dst, data.data(), data.size_bytes());
	}

	/// @brief Fill an image with host data
	/// @param allocator Allocator to use for temporary allocations
	/// @param copy_domain The domain where the copy should happen (when dst is mapped, the copy happens on host)
	/// @param image ImageAttachment to fill
	/// @param src_data pointer to source data
	inline Future<ImageAttachment> host_data_to_image(Allocator& allocator, DomainFlagBits copy_domain, ImageAttachment image, const void* src_data) {
		size_t alignment = format_to_texel_block_size(image.format);
		assert(image.extent.sizing == Sizing::eAbsolute);
		size_t size = compute_image_size(image.format, static_cast<Extent3D>(image.extent.extent));
		auto src = *allocate_buffer_cross_device(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
		::memcpy(src->mapped_ptr, src_data, size);

		BufferImageCopy bc;
		bc.imageOffset = { 0, 0, 0 };
		bc.bufferRowLength = 0;
		bc.bufferImageHeight = 0;
		bc.imageExtent = static_cast<Extent3D>(image.extent.extent);
		bc.imageSubresource.aspectMask = format_to_aspect(image.format);
		bc.imageSubresource.mipLevel = image.base_level;
		bc.imageSubresource.baseArrayLayer = image.base_layer;
		assert(image.layer_count == 1); // unsupported yet
		bc.imageSubresource.layerCount = image.layer_count;

		std::unique_ptr<RenderGraph> rgp = std::make_unique<RenderGraph>();
		rgp->add_pass({ .name = "IMAGE UPLOAD",
		                .execute_on = copy_domain,
		                .resources = { "_dst"_image >> vuk::Access::eTransferWrite, "_src"_buffer >> vuk::Access::eTransferRead },
		                .execute = [bc](vuk::CommandBuffer& command_buffer) {
			                command_buffer.copy_buffer_to_image("_src", "_dst", bc);
		                } });
		rgp->attach_buffer("_src", *src, vuk::Access::eNone, vuk::Access::eNone);
		rgp->attach_image("_dst", image, vuk::Access::eNone, vuk::Access::eNone);
		return { allocator, std::move(rgp), "_dst+" };
	}

	/// @brief Transition image for given access - useful to force certain access across different RenderGraphs linked by Futures
	/// @param image input Future of ImageAttachment
	/// @param dst_access Access to have in the future
	inline Future<ImageAttachment> transition(Future<ImageAttachment> image, Access dst_access) {
		auto& allocator = image.get_allocator();
		std::unique_ptr<RenderGraph> rgp = std::make_unique<RenderGraph>();
		rgp->add_pass({ .name = "TRANSITION", .execute_on = DomainFlagBits::eDevice, .resources = { "_src"_image >> dst_access >> "_src+" } });
		rgp->attach_in("_src", std::move(image));
		return { allocator, std::move(rgp), "_src+" };
	}

	/// @brief Generate mips for given ImageAttachment
	/// @param image input Future of ImageAttachment
	/// @param base_mip source mip level
	/// @param num_mips number of mip levels to generate
	inline Future<ImageAttachment> generate_mips(Future<ImageAttachment> image, uint32_t base_mip, uint32_t num_mips) {
		auto& allocator = image.get_allocator();

		std::unique_ptr<RenderGraph> rgp = std::make_unique<RenderGraph>();
		for (uint32_t miplevel = base_mip + 1; miplevel < (base_mip + num_mips); miplevel++) {
			uint32_t dmiplevel = miplevel - base_mip;
			Name mip_dst = Name(std::to_string(miplevel));
			Name src = Name("_src");
			if (miplevel != base_mip + 1) {
				src = src.append("p");
			}
			Resource src_res(src, Resource::Type::eImage, Access::eTransferRead);
			src_res.subrange.image.base_level = miplevel - 1;
			src_res.subrange.image.level_count = 1;
			Resource dst_res("_src", Resource::Type::eImage, Access::eTransferWrite, "_srcp");
			dst_res.subrange.image.base_level = miplevel;
			dst_res.subrange.image.level_count = 1;
			rgp->add_pass({ .name = Name("MIP").append(mip_dst),
			                .execute_on = DomainFlagBits::eGraphicsOnGraphics,
			                .resources = { src_res, dst_res },
			                .execute = [src, dmiplevel, miplevel](CommandBuffer& command_buffer) {
				                ImageBlit blit;
				                auto src_ia = *command_buffer.get_resource_image_attachment(src);
				                auto dim = src_ia.extent;
				                assert(dim.sizing == Sizing::eAbsolute);
				                auto extent = dim.extent;
				                blit.srcSubresource.aspectMask = format_to_aspect(src_ia.format);
				                blit.srcSubresource.baseArrayLayer = src_ia.base_layer;
				                blit.srcSubresource.layerCount = src_ia.layer_count;
				                blit.srcSubresource.mipLevel = miplevel - 1;
				                blit.srcOffsets[0] = Offset3D{ 0 };
				                blit.srcOffsets[1] = Offset3D{ std::max((int32_t)extent.width >> (dmiplevel - 1), 1),
					                                             std::max((int32_t)extent.height >> (dmiplevel - 1), 1),
					                                             (int32_t)1 };
				                blit.dstSubresource = blit.srcSubresource;
				                blit.dstSubresource.mipLevel = miplevel;
				                blit.dstOffsets[0] = Offset3D{ 0 };
				                blit.dstOffsets[1] =
				                    Offset3D{ std::max((int32_t)extent.width >> (dmiplevel), 1), std::max((int32_t)extent.height >> (dmiplevel), 1), (int32_t)1 };
				                command_buffer.blit_image(src, "_srcp", blit, Filter::eLinear);
			                } });
		}

		rgp->converge_image("_src", "_src+");

		rgp->attach_in("_src", std::move(image));
		return { allocator, std::move(rgp), "_src+" };
	}

	/// @brief Allocates & fills a buffer with explicitly managed lifetime (cross-device scope)
	/// @param allocator Allocator to allocate this Buffer from
	/// @param mem_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
	template<class T>
	std::pair<Unique<BufferCrossDevice>, Future<Buffer>> create_buffer_cross_device(Allocator& allocator, MemoryUsage mem_usage, std::span<T> data) {
		Unique<BufferCrossDevice> buf(allocator);
		BufferCreateInfo bci{ mem_usage, sizeof(T) * data.size(), 1 };
		auto ret = allocator.allocate_buffers(std::span{ &*buf, 1 }, std::span{ &bci, 1 }); // TODO: dropping error
		memcpy(buf->mapped_ptr, data.data(), data.size_bytes());
		Buffer b = buf.get();
		return { std::move(buf), Future<Buffer>{ allocator, std::move(b) } };
	}

	/// @brief Allocates & fills a buffer with explicitly managed lifetime (device-only scope)
	/// @param allocator Allocator to allocate this Buffer from
	/// @param mem_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
	template<class T>
	std::pair<Unique<BufferGPU>, Future<Buffer>> create_buffer_gpu(Allocator& allocator, DomainFlagBits domain, std::span<T> data) {
		Unique<BufferGPU> buf(allocator);
		BufferCreateInfo bci{ MemoryUsage::eGPUonly, sizeof(T) * data.size(), 1 };
		auto ret = allocator.allocate_buffers(std::span{ &*buf, 1 }, std::span{ &bci, 1 }); // TODO: dropping error
		Buffer b = buf.get();
		return { std::move(buf), host_data_to_buffer(allocator, domain, b, data) };
	}

	/// @brief Allocates & fills an image, creates default ImageView for it (legacy)
	/// @param allocator Allocator to allocate this Texture from
	/// @param format Format of the image
	/// @param extent Extent3D of the image
	/// @param data pointer to data to fill the image with
	/// @param should_generate_mips if true, all mip levels are generated from the 0th level
	inline std::pair<Texture, Future<ImageAttachment>>
	create_texture(Allocator& allocator, Format format, Extent3D extent, void* data, bool should_generate_mips) {
		ImageCreateInfo ici;
		ici.format = format;
		ici.extent = extent;
		ici.samples = Samples::e1;
		ici.initialLayout = ImageLayout::eUndefined;
		ici.tiling = ImageTiling::eOptimal;
		ici.usage = ImageUsageFlagBits::eTransferRead | ImageUsageFlagBits::eTransferWrite | ImageUsageFlagBits::eSampled;
		ici.mipLevels = should_generate_mips ? (uint32_t)log2f((float)std::max(extent.width, extent.height)) + 1 : 1;
		ici.arrayLayers = 1;
		auto tex = allocator.get_context().allocate_texture(allocator, ici);

		auto upload_fut = host_data_to_image(allocator, DomainFlagBits::eTransferQueue, ImageAttachment::from_texture(tex), data);
		auto mipgen_fut = should_generate_mips ? generate_mips(std::move(upload_fut), 0, ici.mipLevels) : std::move(upload_fut);
		std::unique_ptr<RenderGraph> rgp = std::make_unique<RenderGraph>();
		rgp->add_pass({ .name = "TRANSITION", .execute_on = DomainFlagBits::eGraphicsQueue, .resources = { "_src"_image >> Access::eFragmentSampled >> "_src+" } });
		rgp->attach_in("_src", std::move(mipgen_fut));
		auto on_gfx = Future<ImageAttachment>{ allocator, std::move(rgp), "_src+" };

		return { std::move(tex), std::move(on_gfx) };
	}
} // namespace vuk
