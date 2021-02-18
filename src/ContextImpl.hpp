#include "vuk/Context.hpp"
#include "RGImage.hpp"

#include <math.h>
#include <mutex>
#include <queue>
#include <string_view>

#include "Allocator.hpp"
#include "Pool.hpp"
#include "Cache.hpp"
#include "RenderPass.hpp"

namespace vuk {
	struct TransientSubmitBundle {
		uint32_t queue_family_index;
		VkCommandPool cpool = VK_NULL_HANDLE;
		std::vector<VkCommandBuffer> command_buffers;
		vuk::Buffer buffer;
		VkFence fence = VK_NULL_HANDLE;
		VkSemaphore sema = VK_NULL_HANDLE;
		TransientSubmitBundle* next = nullptr;
	};

	struct ContextImpl {
		Allocator allocator;
		VkDevice device;

		std::mutex gfx_queue_lock;
		std::mutex xfer_queue_lock;
		Pool<VkCommandBuffer, Context::FC> cbuf_pools;
		Pool<VkSemaphore, Context::FC> semaphore_pools;
		Pool<VkFence, Context::FC> fence_pools;
		VkPipelineCache vk_pipeline_cache;
		Cache<PipelineBaseInfo> pipelinebase_cache;
		Cache<PipelineInfo> pipeline_cache;
		Cache<ComputePipelineInfo> compute_pipeline_cache;
		Cache<VkRenderPass> renderpass_cache;
		Cache<VkFramebuffer> framebuffer_cache;
		Cache<RGImage> transient_images;
		PerFrameCache<LinearAllocator, Context::FC> scratch_buffers;
		Cache<vuk::DescriptorPool> pool_cache;
		PerFrameCache<vuk::DescriptorSet, Context::FC> descriptor_sets;
		Cache<vuk::Sampler> sampler_cache;
		Pool<vuk::SampledImage, Context::FC> sampled_images;
		Cache<vuk::ShaderModule> shader_modules;
		Cache<vuk::DescriptorSetLayoutAllocInfo> descriptor_set_layouts;
		Cache<VkPipelineLayout> pipeline_layouts;

		std::mutex begin_frame_lock;

		std::array<std::mutex, Context::FC> recycle_locks;
		std::array<std::vector<vuk::Image>, Context::FC> image_recycle;
		std::array<std::vector<VkImageView>, Context::FC> image_view_recycle;
		std::array<std::vector<VkPipeline>, Context::FC> pipeline_recycle;
		std::array<std::vector<vuk::Buffer>, Context::FC> buffer_recycle;
		std::array<std::vector<vuk::PersistentDescriptorSet>, Context::FC> pds_recycle;

		std::mutex named_pipelines_lock;
		std::unordered_map<std::string_view, vuk::PipelineBaseInfo*> named_pipelines;
		std::unordered_map<std::string_view, vuk::ComputePipelineInfo*> named_compute_pipelines;

		std::mutex swapchains_lock;
		plf::colony<Swapchain> swapchains;

		std::mutex transient_submit_lock;
		/// @brief with stable addresses, so we can hand out opaque pointers
		plf::colony<TransientSubmitBundle> transient_submit_bundles;
		std::vector<plf::colony<TransientSubmitBundle>::iterator> transient_submit_freelist;

		TransientSubmitBundle* get_transient_bundle(uint32_t queue_family_index) {
			std::lock_guard _(transient_submit_lock);

			plf::colony<TransientSubmitBundle>::iterator it = transient_submit_bundles.end();
			for (auto fit = transient_submit_freelist.begin(); fit != transient_submit_freelist.end(); fit++) {
				if ((*fit)->queue_family_index == queue_family_index) {
					it = *fit;
					transient_submit_freelist.erase(fit);
					break;
				}
			}
			if (it == transient_submit_bundles.end()) { // didn't find suitable bundle
				it = transient_submit_bundles.emplace();
				auto& bundle = *it;

				VkCommandPoolCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
				cpci.queueFamilyIndex = queue_family_index;
				cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
				vkCreateCommandPool(device, &cpci, nullptr, &bundle.cpool);
			}
			return &*it;
		}

		void cleanup_transient_bundle_recursively(vuk::TransientSubmitBundle* ur) {
			if (ur->cpool) {
				vkResetCommandPool(device, ur->cpool, 0);
				vkFreeCommandBuffers(device, ur->cpool, (uint32_t)ur->command_buffers.size(), ur->command_buffers.data());
				ur->command_buffers.clear();
			}
			if (ur->buffer) {
				allocator.free_buffer(ur->buffer);
			}
			if (ur->fence) {
				vkDestroyFence(device, ur->fence, nullptr);
				ur->fence = VK_NULL_HANDLE;
			}
			if (ur->sema) {
				vkDestroySemaphore(device, ur->sema, nullptr);
				ur->sema = VK_NULL_HANDLE;
			}
			if (ur->next) {
				cleanup_transient_bundle_recursively(ur->next);
			}
		}

		VkCommandBuffer get_command_buffer(TransientSubmitBundle* bundle) {
			VkCommandBuffer cbuf;
			VkCommandBufferAllocateInfo cbai{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
			cbai.commandPool = bundle->cpool;
			cbai.commandBufferCount = 1;

			vkAllocateCommandBuffers(device, &cbai, &cbuf);
			bundle->command_buffers.push_back(cbuf);
			return cbuf;
		}

		VkFence get_unpooled_fence() {
			VkFenceCreateInfo fci{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
			VkFence fence;
			vkCreateFence(device, &fci, nullptr, &fence);
			return fence;
		}

		VkSemaphore get_unpooled_sema() {
			VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
			VkSemaphore sema;
			vkCreateSemaphore(device, &sci, nullptr, &sema);
			return sema;
		}

		ContextImpl(Context& ctx) : allocator(ctx.instance, ctx.device, ctx.physical_device, ctx.graphics_queue_family_index, ctx.transfer_queue_family_index),
			device(ctx.device),
			cbuf_pools(ctx),
			semaphore_pools(ctx),
			fence_pools(ctx),
			pipelinebase_cache(ctx),
			pipeline_cache(ctx),
			compute_pipeline_cache(ctx),
			renderpass_cache(ctx),
			framebuffer_cache(ctx),
			transient_images(ctx),
			scratch_buffers(ctx),
			pool_cache(ctx),
			descriptor_sets(ctx),
			sampler_cache(ctx),
			sampled_images(ctx),
			shader_modules(ctx),
			descriptor_set_layouts(ctx),
			pipeline_layouts(ctx) {

			VkPipelineCacheCreateInfo pcci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
			vkCreatePipelineCache(ctx.device, &pcci, nullptr, &vk_pipeline_cache);
		}
	};

	inline unsigned _prev(unsigned frame, unsigned amt, unsigned FC) {
		return ((frame - amt) % FC) + ((frame >= amt) ? 0 : FC - 1);
	}
	inline unsigned _next(unsigned frame, unsigned amt, unsigned FC) {
		return (frame + amt) % FC;
	}
	inline unsigned _next(unsigned frame, unsigned FC) {
		return (frame + 1) % FC;
	}
	inline size_t _next(size_t frame, unsigned FC) {
		return (frame + 1) % FC;
	}

	template<class T>
	Handle<T> Context::wrap(T payload) {
		return { { unique_handle_id_counter++ }, payload };
	}

	struct BufferCopyCommand {
		Buffer src;
		Buffer dst;
		TransferStub stub;
	};

	struct BufferImageCopyCommand {
		Buffer src;
		vuk::Image dst;
		vuk::Extent3D extent;
		uint32_t base_array_layer;
		uint32_t layer_count;
		uint32_t mip_level;
		bool generate_mips;
		TransferStub stub;
	};

	struct MipGenerateCommand {
		vuk::Image dst;
		vuk::Format format;
		vuk::Extent3D extent;
		uint32_t base_array_layer;
		uint32_t layer_count;
		uint32_t base_mip_level;
		TransferStub stub;
	};

	template<class T, size_t FC>
	typename Pool<T, FC>::PFView Pool<T, FC>::get_view(InflightContext& ctx) {
		return { ctx, *this, per_frame_storage[ctx.frame] };
	}

	template<class T, size_t FC>
	Pool<T, FC>::PFView::PFView(InflightContext& ifc, Pool<T, FC>& storage, plf::colony<PooledType<T>>& fv) : storage(storage), ifc(ifc), frame_values(fv) {
		storage.reset(ifc.frame);
	}
}

inline void record_mip_gen(VkCommandBuffer& cbuf, vuk::MipGenerateCommand& task, vuk::ImageLayout last_layout) {
	// transition top mip to transfersrc
	VkImageMemoryBarrier top_mip_to_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	top_mip_to_barrier.srcAccessMask = 0;
	top_mip_to_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	top_mip_to_barrier.oldLayout = (VkImageLayout)last_layout;
	top_mip_to_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eTransferSrcOptimal;
	top_mip_to_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	top_mip_to_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	top_mip_to_barrier.image = task.dst;
	top_mip_to_barrier.subresourceRange.aspectMask = (VkImageAspectFlags)vuk::format_to_aspect(task.format);
	top_mip_to_barrier.subresourceRange.baseMipLevel = task.base_mip_level;
	top_mip_to_barrier.subresourceRange.baseArrayLayer = task.base_array_layer;
	top_mip_to_barrier.subresourceRange.layerCount = task.layer_count;
	top_mip_to_barrier.subresourceRange.levelCount = 1;

	// transition other mips to transferdst
	VkImageMemoryBarrier rest_mip_to_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	rest_mip_to_barrier.srcAccessMask = 0;
	rest_mip_to_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	rest_mip_to_barrier.oldLayout = (VkImageLayout)last_layout;
	rest_mip_to_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
	rest_mip_to_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	rest_mip_to_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	rest_mip_to_barrier.image = task.dst;
	rest_mip_to_barrier.subresourceRange.aspectMask = (VkImageAspectFlags)vuk::format_to_aspect(task.format);
	rest_mip_to_barrier.subresourceRange.baseMipLevel = task.base_mip_level + 1;
	rest_mip_to_barrier.subresourceRange.baseArrayLayer = task.base_array_layer;
	rest_mip_to_barrier.subresourceRange.layerCount = task.layer_count;
	rest_mip_to_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;

	// transition top mip to SROO
	VkImageMemoryBarrier top_mip_use_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	top_mip_use_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	top_mip_use_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; // TODO: maybe memory read?
	top_mip_use_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferSrcOptimal;
	top_mip_use_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eShaderReadOnlyOptimal;
	top_mip_use_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	top_mip_use_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	top_mip_use_barrier.image = task.dst;
	top_mip_use_barrier.subresourceRange = top_mip_to_barrier.subresourceRange;

	// transition rest of the mips to SROO
	VkImageMemoryBarrier use_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };;
	use_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	use_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; // TODO: maybe memory read?
	use_barrier.oldLayout = (VkImageLayout)last_layout;
	use_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eShaderReadOnlyOptimal;
	use_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier.image = task.dst;
	use_barrier.subresourceRange = top_mip_to_barrier.subresourceRange;
	use_barrier.subresourceRange.baseMipLevel = task.base_mip_level + 1;
	use_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;


	VkImageMemoryBarrier to_bars[] = { top_mip_to_barrier, rest_mip_to_barrier };
	vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, to_bars);
	auto mips = (uint32_t)log2f((float)std::max(task.extent.width, task.extent.height));

	for (uint32_t miplevel = task.base_mip_level + 1; miplevel < mips; miplevel++) {
		VkImageBlit blit;
		blit.srcSubresource.aspectMask = top_mip_to_barrier.subresourceRange.aspectMask;
		blit.srcSubresource.baseArrayLayer = task.base_array_layer;
		blit.srcSubresource.layerCount = task.layer_count;
		blit.srcSubresource.mipLevel = 0; // blit from 0th mip
		blit.srcOffsets[0] = VkOffset3D{ 0 };
		blit.srcOffsets[1] = VkOffset3D{ (int32_t)task.extent.width, (int32_t)task.extent.height, (int32_t)task.extent.depth };
		blit.dstSubresource = blit.srcSubresource;
		blit.dstSubresource.mipLevel = miplevel;
		blit.dstOffsets[0] = VkOffset3D{ 0 };
		blit.dstOffsets[1] = VkOffset3D{ (int32_t)task.extent.width >> miplevel, (int32_t)task.extent.height >> miplevel, (int32_t)task.extent.depth };
		vkCmdBlitImage(cbuf, task.dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, task.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
	}

	VkImageMemoryBarrier bars[] = { use_barrier, top_mip_use_barrier };
	// wait for transfer, delay all (we don't know where this will be used, be safe)
	vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 0, nullptr, 0, nullptr, 2, bars);
}

// single queue copy + optional mip gen
inline void record_buffer_image_copy(VkCommandBuffer& cbuf, vuk::BufferImageCopyCommand& task) {
	VkBufferImageCopy bc;
	bc.bufferOffset = task.src.offset;
	bc.imageOffset = VkOffset3D{ 0, 0, 0 };
	bc.bufferRowLength = 0;
	bc.bufferImageHeight = 0;
	bc.imageExtent = task.extent;
	bc.imageSubresource.aspectMask = (VkImageAspectFlagBits)vuk::ImageAspectFlagBits::eColor;
	bc.imageSubresource.baseArrayLayer = task.base_array_layer;
	bc.imageSubresource.mipLevel = task.mip_level;
	bc.imageSubresource.layerCount = task.layer_count;

	VkImageMemoryBarrier copy_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	copy_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	copy_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eUndefined;
	copy_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
	copy_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	copy_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	copy_barrier.image = task.dst;
	copy_barrier.subresourceRange.aspectMask = bc.imageSubresource.aspectMask;
	copy_barrier.subresourceRange.layerCount = bc.imageSubresource.layerCount;
	copy_barrier.subresourceRange.baseArrayLayer = bc.imageSubresource.baseArrayLayer;
	copy_barrier.subresourceRange.baseMipLevel = bc.imageSubresource.mipLevel;
	copy_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;

	// transition top mip to transfersrc
	VkImageMemoryBarrier top_mip_to_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	top_mip_to_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	top_mip_to_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	top_mip_to_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
	top_mip_to_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eTransferSrcOptimal;
	top_mip_to_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	top_mip_to_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	top_mip_to_barrier.image = task.dst;
	top_mip_to_barrier.subresourceRange = copy_barrier.subresourceRange;
	top_mip_to_barrier.subresourceRange.levelCount = 1;

	// transition top mip to SROO
	VkImageMemoryBarrier top_mip_use_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	top_mip_use_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	top_mip_use_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	top_mip_use_barrier.oldLayout = task.generate_mips ? (VkImageLayout)vuk::ImageLayout::eTransferSrcOptimal : (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
	top_mip_use_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eShaderReadOnlyOptimal;
	top_mip_use_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	top_mip_use_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	top_mip_use_barrier.image = task.dst;
	top_mip_use_barrier.subresourceRange = copy_barrier.subresourceRange;
	top_mip_use_barrier.subresourceRange.levelCount = 1;

	// transition rest of the mips to SROO
	VkImageMemoryBarrier use_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };;
	use_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	use_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	use_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
	use_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eShaderReadOnlyOptimal;
	use_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier.image = task.dst;
	use_barrier.subresourceRange = copy_barrier.subresourceRange;
	use_barrier.subresourceRange.baseMipLevel = task.mip_level + 1;
	use_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;

	vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &copy_barrier);
	vkCmdCopyBufferToImage(cbuf, task.src.buffer, task.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);
	if (task.generate_mips) {
		vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &top_mip_to_barrier);

		auto mips = (uint32_t)log2f((float)std::max(task.extent.width, task.extent.height)) + 1;

		for (uint32_t miplevel = task.mip_level; miplevel < mips; miplevel++) {
			VkImageBlit blit;
			blit.srcSubresource.aspectMask = copy_barrier.subresourceRange.aspectMask;
			blit.srcSubresource.baseArrayLayer = task.base_array_layer;
			blit.srcSubresource.layerCount = task.layer_count;
			blit.srcSubresource.mipLevel = task.mip_level;
			blit.srcOffsets[0] = VkOffset3D{ 0 };
			blit.srcOffsets[1] = VkOffset3D{ (int32_t)task.extent.width, (int32_t)task.extent.height, (int32_t)task.extent.depth };
			blit.dstSubresource = blit.srcSubresource;
			blit.dstSubresource.mipLevel = miplevel;
			blit.dstOffsets[0] = VkOffset3D{ 0 };
			blit.dstOffsets[1] = VkOffset3D{ (int32_t)task.extent.width >> miplevel, (int32_t)task.extent.height >> miplevel, (int32_t)task.extent.depth };
			vkCmdBlitImage(cbuf, task.dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, task.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
		}

		vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &use_barrier);
	}

	vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &top_mip_use_barrier);
}

namespace vuk {
	struct PendingTransfer {
		size_t last_transfer_id;
		VkFence fence;
	};

	struct IFCImpl {
		Pool<VkFence, Context::FC>::PFView fence_pools; // must be first, so we wait for the fences
		Pool<VkCommandBuffer, Context::FC>::PFView commandbuffer_pools;
		Pool<VkSemaphore, Context::FC>::PFView semaphore_pools;
		Cache<PipelineInfo>::PFView pipeline_cache;
		Cache<ComputePipelineInfo>::PFView compute_pipeline_cache;
		Cache<PipelineBaseInfo>::PFView pipelinebase_cache;
		Cache<VkRenderPass>::PFView renderpass_cache;
		Cache<VkFramebuffer>::PFView framebuffer_cache;
		Cache<vuk::RGImage>::PFView transient_images;
		PerFrameCache<LinearAllocator, Context::FC>::PFView scratch_buffers;
		PerFrameCache<vuk::DescriptorSet, Context::FC>::PFView descriptor_sets;
		Cache<vuk::Sampler>::PFView sampler_cache;
		Pool<vuk::SampledImage, Context::FC>::PFView sampled_images;
		Cache<vuk::DescriptorPool>::PFView pool_cache;

		Cache<vuk::ShaderModule>::PFView shader_modules;
		Cache<vuk::DescriptorSetLayoutAllocInfo>::PFView descriptor_set_layouts;
		Cache<VkPipelineLayout>::PFView pipeline_layouts;

		// needs to be mpsc
		std::mutex transfer_mutex;
		std::queue<BufferCopyCommand> buffer_transfer_commands;
		std::queue<BufferImageCopyCommand> bufferimage_transfer_commands;
		// only accessed by DMAtask
		std::queue<PendingTransfer> pending_transfers;

		// recycle
		std::mutex recycle_lock;

		IFCImpl(Context& ctx, InflightContext& ifc) :
			fence_pools(ctx.impl->fence_pools.get_view(ifc)), // must be first, so we wait for the fences
			commandbuffer_pools(ctx.impl->cbuf_pools.get_view(ifc)),
			semaphore_pools(ctx.impl->semaphore_pools.get_view(ifc)),
			pipeline_cache(ifc, ctx.impl->pipeline_cache),
			compute_pipeline_cache(ifc, ctx.impl->compute_pipeline_cache),
			pipelinebase_cache(ifc, ctx.impl->pipelinebase_cache),
			renderpass_cache(ifc, ctx.impl->renderpass_cache),
			framebuffer_cache(ifc, ctx.impl->framebuffer_cache),
			transient_images(ifc, ctx.impl->transient_images),
			scratch_buffers(ifc, ctx.impl->scratch_buffers),
			descriptor_sets(ifc, ctx.impl->descriptor_sets),
			sampler_cache(ifc, ctx.impl->sampler_cache),
			sampled_images(ctx.impl->sampled_images.get_view(ifc)),
			pool_cache(ifc, ctx.impl->pool_cache),
			shader_modules(ifc, ctx.impl->shader_modules),
			descriptor_set_layouts(ifc, ctx.impl->descriptor_set_layouts),
			pipeline_layouts(ifc, ctx.impl->pipeline_layouts) {
		}
	};

	struct PTCImpl {
		Pool<VkCommandBuffer, Context::FC>::PFPTView commandbuffer_pool;
		Pool<VkSemaphore, Context::FC>::PFPTView semaphore_pool;
		Pool<VkFence, Context::FC>::PFPTView fence_pool;
		Cache<PipelineInfo>::PFPTView pipeline_cache;
		Cache<ComputePipelineInfo>::PFPTView compute_pipeline_cache;
		Cache<PipelineBaseInfo>::PFPTView pipelinebase_cache;
		Cache<VkRenderPass>::PFPTView renderpass_cache;
		Cache<VkFramebuffer>::PFPTView framebuffer_cache;
		Cache<vuk::RGImage>::PFPTView transient_images;
		PerFrameCache<LinearAllocator, Context::FC>::PFPTView scratch_buffers;
		PerFrameCache<vuk::DescriptorSet, Context::FC>::PFPTView descriptor_sets;
		Cache<vuk::Sampler>::PFPTView sampler_cache;
		Pool<vuk::SampledImage, Context::FC>::PFPTView sampled_images;
		Cache<vuk::DescriptorPool>::PFPTView pool_cache;
		Cache<vuk::ShaderModule>::PFPTView shader_modules;
		Cache<vuk::DescriptorSetLayoutAllocInfo>::PFPTView descriptor_set_layouts;
		Cache<VkPipelineLayout>::PFPTView pipeline_layouts;

		// recycling global objects
		std::vector<Buffer> buffer_recycle;
		std::vector<vuk::Image> image_recycle;
		std::vector<VkImageView> image_view_recycle;

		PTCImpl(InflightContext& ifc, PerThreadContext& ptc) :
			commandbuffer_pool(ifc.impl->commandbuffer_pools.get_view(ptc)),
			semaphore_pool(ifc.impl->semaphore_pools.get_view(ptc)),
			fence_pool(ifc.impl->fence_pools.get_view(ptc)),
			pipeline_cache(ptc, ifc.impl->pipeline_cache),
			compute_pipeline_cache(ptc, ifc.impl->compute_pipeline_cache),
			pipelinebase_cache(ptc, ifc.impl->pipelinebase_cache),
			renderpass_cache(ptc, ifc.impl->renderpass_cache),
			framebuffer_cache(ptc, ifc.impl->framebuffer_cache),
			transient_images(ptc, ifc.impl->transient_images),
			scratch_buffers(ptc, ifc.impl->scratch_buffers),
			descriptor_sets(ptc, ifc.impl->descriptor_sets),
			sampler_cache(ptc, ifc.impl->sampler_cache),
			sampled_images(ifc.impl->sampled_images.get_view(ptc)),
			pool_cache(ptc, ifc.impl->pool_cache),
			shader_modules(ptc, ifc.impl->shader_modules),
			descriptor_set_layouts(ptc, ifc.impl->descriptor_set_layouts),
			pipeline_layouts(ptc, ifc.impl->pipeline_layouts) {
		}
	};
}
