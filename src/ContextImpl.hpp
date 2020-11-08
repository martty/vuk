#include "vuk/Context.hpp"
#include "RGImage.hpp"

#include <mutex>

namespace vuk {
	struct ContextImpl {
		Allocator allocator;

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
		PerFrameCache<RGImage, Context::FC> transient_images;
		PerFrameCache<Allocator::Linear, Context::FC> scratch_buffers;
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

		// one pool per thread
		std::mutex one_time_pool_lock;
		std::vector<VkCommandPool> xfer_one_time_pools;
		std::vector<VkCommandPool> one_time_pools;

		ContextImpl(Context& ctx) : allocator(ctx.instance, ctx.device, ctx.physical_device),
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
}

inline void record_buffer_image_copy(VkCommandBuffer& cbuf, vuk::InflightContext::BufferImageCopyCommand& task) {
	VkBufferImageCopy bc;
	bc.bufferOffset = task.src.offset;
	bc.imageOffset = VkOffset3D{ 0, 0, 0 };
	bc.bufferRowLength = 0;
	bc.bufferImageHeight = 0;
	bc.imageExtent = task.extent;
	bc.imageSubresource.aspectMask = (VkImageAspectFlagBits)vuk::ImageAspectFlagBits::eColor;
	bc.imageSubresource.baseArrayLayer = 0;
	bc.imageSubresource.mipLevel = 0;
	bc.imageSubresource.layerCount = 1;

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
	use_barrier.subresourceRange.baseMipLevel = 1;

	vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &copy_barrier);
	vkCmdCopyBufferToImage(cbuf, task.src.buffer, task.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);
	if (task.generate_mips) {
		vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &top_mip_to_barrier);
		auto mips = (uint32_t)std::min(std::log2f((float)task.extent.width), std::log2f((float)task.extent.height));

		for (uint32_t miplevel = 1; miplevel < mips; miplevel++) {
			VkImageBlit blit;
			blit.srcSubresource.aspectMask = copy_barrier.subresourceRange.aspectMask;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.srcSubresource.mipLevel = 0;
			blit.srcOffsets[0] = VkOffset3D{ 0 };
			blit.srcOffsets[1] = VkOffset3D{ (int32_t)task.extent.width, (int32_t)task.extent.height, (int32_t)task.extent.depth };
			blit.dstSubresource = blit.srcSubresource;
			blit.dstSubresource.mipLevel = miplevel;
			blit.dstOffsets[0] = VkOffset3D{ 0 };
			blit.dstOffsets[1] = VkOffset3D{ (int32_t)task.extent.width >> miplevel, (int32_t)task.extent.height >> miplevel, (int32_t)task.extent.depth };
			vkCmdBlitImage(cbuf, task.dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, task.dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
		}

		vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &use_barrier);
	}

	vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &top_mip_use_barrier);
}


