#pragma once

#include "vuk/AllocatorHelpers.hpp"
#include "vuk/CommandBuffer.hpp"
#include "vuk/Future.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/SourceLocation.hpp"
#include <math.h>
#include <span>

namespace vuk {
	/// @brief Fill a buffer with host data
	/// @param allocator Allocator to use for temporary allocations
	/// @param copy_domain The domain where the copy should happen (when dst is mapped, the copy happens on host)
	/// @param buffer Buffer to fill
	/// @param src_data pointer to source data
	/// @param size size of source data
	inline TypedFuture<Buffer> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer dst, const void* src_data, size_t size) {
		// host-mapped buffers just get memcpys
		if (dst.mapped_ptr) {
			memcpy(dst.mapped_ptr, src_data, size);
			return { vuk::declare_buf("_dst", dst) };
		}

		auto src = *allocate_buffer(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, 1 });
		::memcpy(src->mapped_ptr, src_data, size);

		auto src_buf = vuk::declare_buf("_src", *src);
		auto dst_buf = vuk::declare_buf("_dst", dst);
		auto pass =
		    vuk::make_pass("upload buffer", [](vuk::CommandBuffer& command_buffer, VUK_BA(Access::eTransferRead) src, VUK_BA(Access::eTransferWrite) dst) {
			    command_buffer.copy_buffer(src, dst);
			    return dst;
		    });
		auto result = pass(src_buf, dst_buf);
		return result;
	}

	/// @brief Fill a buffer with host data
	/// @param allocator Allocator to use for temporary allocations
	/// @param copy_domain The domain where the copy should happen (when dst is mapped, the copy happens on host)
	/// @param dst Buffer to fill
	/// @param data source data
	template<class T>
	TypedFuture<Buffer> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer dst, std::span<T> data) {
		return host_data_to_buffer(allocator, copy_domain, dst, data.data(), data.size_bytes());
	}

	/// @brief Download a buffer to GPUtoCPU memory
	/// @param buffer_src Buffer to download
	inline TypedFuture<Buffer> download_buffer(TypedFuture<Buffer> buffer_src) {
		auto dst = declare_buf("dst", Buffer{ .memory_usage = MemoryUsage::eGPUtoCPU });
		dst.same_size(buffer_src);
		auto download =
		    vuk::make_pass("download buffer", [](vuk::CommandBuffer& command_buffer, VUK_BA(Access::eTransferRead) src, VUK_BA(Access::eTransferWrite) dst) {
			    command_buffer.copy_buffer(src, dst);
			    return dst;
		    });
		return download(buffer_src, dst);
	}

	/// @brief Fill an image with host data
	/// @param allocator Allocator to use for temporary allocations
	/// @param copy_domain The domain where the copy should happen
	/// @param image ImageAttachment to fill
	/// @param src_data pointer to source data
	inline TypedFuture<ImageAttachment> host_data_to_image(Allocator& allocator, DomainFlagBits copy_domain, ImageAttachment image, const void* src_data) {
		size_t alignment = format_to_texel_block_size(image.format);
		assert(image.extent.sizing == Sizing::eAbsolute);
		size_t size = compute_image_size(image.format, static_cast<Extent3D>(image.extent.extent));
		auto src = *allocate_buffer(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
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
		bc.bufferOffset = src->offset;

		auto srcbuf = declare_buf("src", *src);
		auto dst = declare_ia("dst", image);
		auto image_upload =
		    vuk::make_pass("image upload", [bc](vuk::CommandBuffer& command_buffer, VUK_BA(Access::eTransferRead) src, VUK_IA(Access::eTransferWrite) dst) {
			    command_buffer.copy_buffer_to_image(src, dst, bc);
			    return dst;
		    });

		return image_upload(srcbuf, dst);
	}

	/// @brief Transition image for given access - useful to force certain access across different RenderGraphs linked by Futures
	/// @param image input Future of ImageAttachment
	/// @param dst_access Access to have in the future
	/* inline Future transition(Future image, Access dst_access) {
		std::shared_ptr<RenderGraph> rgp = std::make_shared<RenderGraph>("transition");
		rgp->add_pass({ .name = "TRANSITION",
		                .execute_on = DomainFlagBits::eDevice,
		                .resources = { "_src"_image >> dst_access >> "_src+" },
		                .type = PassType::eForcedAccess });
		rgp->attach_in("_src", std::move(image));
		return { std::move(rgp), "_src+" };
	}

	/// @brief Generate mips for given ImageAttachment
	/// @param image input Future of ImageAttachment
	/// @param base_mip source mip level
	/// @param num_mips number of mip levels to generate
	inline Future generate_mips(Future image, uint32_t base_mip, uint32_t num_mips) {
		std::shared_ptr<RenderGraph> rgp = std::make_shared<RenderGraph>("generate_mips");
		rgp->attach_in("_src", std::move(image));
		Name mip = Name("_mip_");

		std::vector<Name> diverged_names;
		for (uint32_t miplevel = base_mip; miplevel < (base_mip + num_mips); miplevel++) {
			Name div_name = mip.append(std::to_string(miplevel));
			if (miplevel != base_mip) {
				diverged_names.push_back(div_name.append("+"));
			} else {
				diverged_names.push_back(div_name);
			}
			rgp->diverge_image("_src", { .base_level = miplevel, .level_count = 1 }, div_name);
		}

		for (uint32_t miplevel = base_mip + 1; miplevel < (base_mip + num_mips); miplevel++) {
			uint32_t dmiplevel = miplevel - base_mip;

			Name mip_src_name = mip.append(std::to_string(miplevel - 1));
			auto mip_dst = std::to_string(miplevel);
			Name mip_dst_name = mip.append(std::to_string(miplevel));
			if (miplevel != base_mip + 1) {
				mip_src_name = mip_src_name.append("+");
			}
			Resource src_res(mip_src_name, Resource::Type::eImage, Access::eTransferRead);
			Resource dst_res(mip_dst_name, Resource::Type::eImage, Access::eTransferWrite, mip_dst_name.append("+"));
			rgp->add_pass({ .name = Name("MIP").append(mip_dst),
			                .execute_on = DomainFlagBits::eGraphicsOnGraphics,
			                .resources = { src_res, dst_res },
			                .execute = [mip_src_name, mip_dst_name, dmiplevel, miplevel](CommandBuffer& command_buffer) {
				                ImageBlit blit;
				                auto src_ia = *command_buffer.get_resource_image_attachment(mip_src_name);
				                auto dst_ia = *command_buffer.get_resource_image_attachment(mip_dst_name);
				                auto dim = src_ia.extent;
				                assert(dim.sizing == Sizing::eAbsolute);
				                auto extent = dim.extent;
				                blit.srcSubresource.aspectMask = format_to_aspect(src_ia.format);
				                blit.srcSubresource.baseArrayLayer = src_ia.base_layer;
				                blit.srcSubresource.layerCount = src_ia.layer_count;
				                blit.srcSubresource.mipLevel = src_ia.base_level;
				                blit.srcOffsets[0] = Offset3D{ 0 };
				                blit.srcOffsets[1] = Offset3D{ std::max((int32_t)extent.width >> (dmiplevel - 1), 1),
					                                             std::max((int32_t)extent.height >> (dmiplevel - 1), 1),
					                                             std::max((int32_t)extent.depth >> (dmiplevel - 1), 1) };
				                blit.dstSubresource = blit.srcSubresource;
				                blit.dstSubresource.mipLevel = dst_ia.base_level;
				                blit.dstOffsets[0] = Offset3D{ 0 };
				                blit.dstOffsets[1] = Offset3D{ std::max((int32_t)extent.width >> (dmiplevel), 1),
					                                             std::max((int32_t)extent.height >> (dmiplevel), 1),
					                                             std::max((int32_t)extent.depth >> (dmiplevel), 1) };
				                command_buffer.blit_image(mip_src_name, mip_dst_name, blit, Filter::eLinear);
			                } });
		}

		rgp->converge_image_explicit(diverged_names, "_src+");
		return { std::move(rgp), "_src+" };
	}*/

	/// @brief Allocates & fills a buffer with explicitly managed lifetime
	/// @param allocator Allocator to allocate this Buffer from
	/// @param mem_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
	template<class T>
	std::pair<Unique<Buffer>, TypedFuture<Buffer>>
	create_buffer(Allocator& allocator, vuk::MemoryUsage memory_usage, DomainFlagBits domain, std::span<T> data, size_t alignment = 1) {
		Unique<Buffer> buf(allocator);
		BufferCreateInfo bci{ memory_usage, sizeof(T) * data.size(), alignment };
		auto ret = allocator.allocate_buffers(std::span{ &*buf, 1 }, std::span{ &bci, 1 }); // TODO: dropping error
		Buffer b = buf.get();
		return { std::move(buf), host_data_to_buffer(allocator, domain, b, data) };
	}

	std::pair<Unique<Image>, TypedFuture<ImageAttachment>> create_image_with_data(Allocator& allocator,
	                                                                              DomainFlagBits copy_domain,
	                                                                              ImageAttachment ia,
	                                                                              const void* data,
	                                                                              SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		auto image = allocate_image(allocator, ia, loc);
		ia.image = **image;
		return { std::move(*image), host_data_to_image(allocator, copy_domain, ia, data) };
	}

	template<class T>
	std::pair<Unique<Image>, TypedFuture<ImageAttachment>> create_image_with_data(Allocator& allocator,
	                                                                              DomainFlagBits copy_domain,
	                                                                              ImageAttachment ia,
	                                                                              std::span<T> data,
	                                                                              SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		return create_image_with_data(allocator, copy_domain, ia, data.data(), loc);
	}

	std::tuple<Unique<Image>, Unique<ImageView>, TypedFuture<ImageAttachment>> create_image_and_view_with_data(Allocator& allocator,
	                                                                                                           DomainFlagBits copy_domain,
	                                                                                                           ImageAttachment ia,
	                                                                                                           const void* data,
	                                                                                                           SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		auto image = allocate_image(allocator, ia, loc);
		ia.image = **image;
		auto view = allocate_image_view(allocator, ia, loc);
		ia.image_view = **view;
		return { std::move(*image), std::move(*view), host_data_to_image(allocator, copy_domain, ia, data) };
	}

	template<class T>
	std::tuple<Unique<Image>, Unique<ImageView>, TypedFuture<ImageAttachment>> create_image_and_view_with_data(Allocator& allocator,
	                                                                                                           DomainFlagBits copy_domain,
	                                                                                                           ImageAttachment ia,
	                                                                                                           std::span<T> data,
	                                                                                                           SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		return create_image_and_view_with_data(allocator, copy_domain, ia, data.data(), loc);
	}

	/// @brief Allocates & fills an image, creates default ImageView for it (legacy)
	/// @param allocator Allocator to allocate this Texture from
	/// @param format Format of the image
	/// @param extent Extent3D of the image
	/// @param data pointer to data to fill the image with
	/// @param should_generate_mips if true, all mip levels are generated from the 0th level
	/* inline std::pair<Texture, Future>
	create_texture(Allocator& allocator, Format format, Extent3D extent, void* data, bool should_generate_mips, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		ImageCreateInfo ici;
		ici.format = format;
		ici.extent = extent;
		ici.samples = Samples::e1;
		ici.initialLayout = ImageLayout::eUndefined;
		ici.tiling = ImageTiling::eOptimal;
		ici.usage = ImageUsageFlagBits::eTransferSrc | ImageUsageFlagBits::eTransferDst | ImageUsageFlagBits::eSampled;
		ici.mipLevels = should_generate_mips ? (uint32_t)log2f((float)std::max(std::max(extent.width, extent.height), extent.depth)) + 1 : 1;
		ici.arrayLayers = 1;
		auto tex = allocator.get_context().allocate_texture(allocator, ici, loc);

		auto upload_fut = host_data_to_image(allocator, DomainFlagBits::eTransferQueue, ImageAttachment::from_texture(tex), data);
		auto mipgen_fut = ici.mipLevels > 1 ? generate_mips(std::move(upload_fut), 0, ici.mipLevels) : std::move(upload_fut);
		std::shared_ptr<RenderGraph> rgp = std::make_shared<RenderGraph>("create_texture");
		rgp->add_pass({ .name = "TRANSITION",
		                .execute_on = DomainFlagBits::eGraphicsQueue,
		                .resources = { "_src"_image >> Access::eFragmentSampled >> "_src+" },
		                .type = PassType::eForcedAccess });
		rgp->attach_in("_src", std::move(mipgen_fut));
		auto on_gfx = Future{ std::move(rgp), "_src+" };

		return { std::move(tex), std::move(on_gfx) };
	}*/
} // namespace vuk
