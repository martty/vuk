#include "RGImage.hpp"
#include "vuk/Context.hpp"

#include <math.h>
#include <mutex>
#include <queue>
#include <string_view>

#include "Cache.hpp"
#include "LegacyGPUAllocator.hpp"
#include "RenderPass.hpp"
#include "vuk/Allocator.hpp"
#include "vuk/PipelineInstance.hpp"
#include "vuk/resources/DeviceVkResource.hpp"

namespace vuk {
	struct TransientSubmitBundle {
		uint32_t queue_family_index;
		VkCommandPool cpool = VK_NULL_HANDLE;
		std::vector<VkCommandBuffer> command_buffers;
		Buffer buffer;
		VkFence fence = VK_NULL_HANDLE;
		VkSemaphore sema = VK_NULL_HANDLE;
		TransientSubmitBundle* next = nullptr;
	};

	struct BufferCopyCommand {
		Buffer src;
		Buffer dst;
		TransferStub stub;
	};

	struct BufferImageCopyCommand {
		Buffer src;
		Image dst;
		Extent3D extent;
		uint32_t base_array_layer;
		uint32_t layer_count;
		uint32_t mip_level;
		bool generate_mips;
		TransferStub stub;
	};

	struct MipGenerateCommand {
		Image dst;
		Format format;
		Extent3D extent;
		uint32_t base_array_layer;
		uint32_t layer_count;
		uint32_t base_mip_level;
		TransferStub stub;
	};

	struct PendingTransfer {
		size_t last_transfer_id;
		VkFence fence;
	};

	struct ContextImpl {
		LegacyGPUAllocator legacy_gpu_allocator;
		VkDevice device;

		// TODO: gone
		std::mutex gfx_queue_lock;
		std::mutex xfer_queue_lock;

		VkPipelineCache vk_pipeline_cache = VK_NULL_HANDLE;
		Cache<PipelineBaseInfo> pipelinebase_cache;
		Cache<PipelineInfo> pipeline_cache;
		Cache<ComputePipelineBaseInfo> compute_pipelinebase_cache;
		Cache<ComputePipelineInfo> compute_pipeline_cache;
		Cache<VkRenderPass> renderpass_cache;
		Cache<RGImage> transient_images;
		Cache<DescriptorPool> pool_cache;
		Cache<Sampler> sampler_cache;
		Cache<ShaderModule> shader_modules;
		Cache<DescriptorSetLayoutAllocInfo> descriptor_set_layouts;
		Cache<VkPipelineLayout> pipeline_layouts;

		std::mutex begin_frame_lock;

		// needs to be mpsc
		std::mutex transfer_mutex;
		std::queue<BufferCopyCommand> buffer_transfer_commands;
		std::queue<BufferImageCopyCommand> bufferimage_transfer_commands;
		// only accessed by DMAtask
		std::queue<PendingTransfer> pending_transfers;

		std::mutex named_pipelines_lock;
		std::unordered_map<Name, PipelineBaseInfo*> named_pipelines;
		std::unordered_map<Name, ComputePipelineBaseInfo*> named_compute_pipelines;

		std::atomic<uint64_t> query_id_counter = 0;
		VkPhysicalDeviceProperties physical_device_properties;

		std::mutex swapchains_lock;
		plf::colony<Swapchain> swapchains;

		std::mutex transient_submit_lock;
		/// @brief with stable addresses, so we can hand out opaque pointers
		plf::colony<TransientSubmitBundle> transient_submit_bundles;
		std::vector<plf::colony<TransientSubmitBundle>::iterator> transient_submit_freelist;

		DeviceVkResource device_vk_resource;

		std::mutex query_lock;
		robin_hood::unordered_map<Query, uint64_t> timestamp_result_map;

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

		void cleanup_transient_bundle_recursively(TransientSubmitBundle* ur) {
			if (ur->cpool) {
				vkResetCommandPool(device, ur->cpool, 0);
				vkFreeCommandBuffers(device, ur->cpool, (uint32_t)ur->command_buffers.size(), ur->command_buffers.data());
				ur->command_buffers.clear();
			}
			if (ur->buffer) {
				legacy_gpu_allocator.free_buffer(ur->buffer);
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

		void collect(uint64_t absolute_frame) {
			transient_images.collect(absolute_frame, 6);
			// collect rarer resources
			static constexpr uint32_t cache_collection_frequency = 16;
			auto remainder = absolute_frame % cache_collection_frequency;
			switch (remainder) {
			case 0:
				pipeline_cache.collect(absolute_frame, cache_collection_frequency);
				break;
			case 1:
				compute_pipeline_cache.collect(absolute_frame, cache_collection_frequency);
				break;
			case 2:
				renderpass_cache.collect(absolute_frame, cache_collection_frequency);
				break;
				/*case 3:
				  ptc.impl->sampler_cache.collect(cache_collection_frequency); break;*/ // sampler cache can't be collected due to persistent descriptor sets
			case 4:
				pipeline_layouts.collect(absolute_frame, cache_collection_frequency);
				break;
			case 5:
				pipelinebase_cache.collect(absolute_frame, cache_collection_frequency);
				break;
			case 6:
				compute_pipelinebase_cache.collect(absolute_frame, cache_collection_frequency);
				break;
			case 7:
				pool_cache.collect(absolute_frame, cache_collection_frequency);
				break;
			}
		}

		ContextImpl(Context& ctx) :
		    legacy_gpu_allocator(ctx.instance, ctx.device, ctx.physical_device, ctx.graphics_queue_family_index, ctx.transfer_queue_family_index),
		    device(ctx.device),
		    pipelinebase_cache(ctx),
		    pipeline_cache(ctx),
		    compute_pipelinebase_cache(ctx),
		    compute_pipeline_cache(ctx),
		    renderpass_cache(ctx),
		    transient_images(ctx),
		    pool_cache(ctx),
		    sampler_cache(ctx),
		    shader_modules(ctx),
		    descriptor_set_layouts(ctx),
		    pipeline_layouts(ctx),
		    device_vk_resource(ctx, legacy_gpu_allocator) {
			vkGetPhysicalDeviceProperties(ctx.physical_device, &physical_device_properties);
		}
	};

	inline void record_mip_gen(VkCommandBuffer& cbuf, MipGenerateCommand& task, ImageLayout last_layout) {
		// transition top mip to transfersrc
		VkImageMemoryBarrier top_mip_to_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		top_mip_to_barrier.srcAccessMask = 0;
		top_mip_to_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		top_mip_to_barrier.oldLayout = (VkImageLayout)last_layout;
		top_mip_to_barrier.newLayout = (VkImageLayout)ImageLayout::eTransferSrcOptimal;
		top_mip_to_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		top_mip_to_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		top_mip_to_barrier.image = task.dst;
		top_mip_to_barrier.subresourceRange.aspectMask = (VkImageAspectFlags)format_to_aspect(task.format);
		top_mip_to_barrier.subresourceRange.baseMipLevel = task.base_mip_level;
		top_mip_to_barrier.subresourceRange.baseArrayLayer = task.base_array_layer;
		top_mip_to_barrier.subresourceRange.layerCount = task.layer_count;
		top_mip_to_barrier.subresourceRange.levelCount = 1;

		// transition other mips to transferdst
		VkImageMemoryBarrier rest_mip_to_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		rest_mip_to_barrier.srcAccessMask = 0;
		rest_mip_to_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		rest_mip_to_barrier.oldLayout = (VkImageLayout)last_layout;
		rest_mip_to_barrier.newLayout = (VkImageLayout)ImageLayout::eTransferDstOptimal;
		rest_mip_to_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		rest_mip_to_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		rest_mip_to_barrier.image = task.dst;
		rest_mip_to_barrier.subresourceRange.aspectMask = (VkImageAspectFlags)format_to_aspect(task.format);
		rest_mip_to_barrier.subresourceRange.baseMipLevel = task.base_mip_level + 1;
		rest_mip_to_barrier.subresourceRange.baseArrayLayer = task.base_array_layer;
		rest_mip_to_barrier.subresourceRange.layerCount = task.layer_count;
		rest_mip_to_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;

		// transition top mip to SROO
		VkImageMemoryBarrier top_mip_use_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		top_mip_use_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		top_mip_use_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; // TODO: maybe memory read?
		top_mip_use_barrier.oldLayout = (VkImageLayout)ImageLayout::eTransferSrcOptimal;
		top_mip_use_barrier.newLayout = (VkImageLayout)ImageLayout::eShaderReadOnlyOptimal;
		top_mip_use_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		top_mip_use_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		top_mip_use_barrier.image = task.dst;
		top_mip_use_barrier.subresourceRange = top_mip_to_barrier.subresourceRange;

		// transition rest of the mips to SROO
		VkImageMemoryBarrier use_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		;
		use_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		use_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; // TODO: maybe memory read?
		use_barrier.oldLayout = (VkImageLayout)last_layout;
		use_barrier.newLayout = (VkImageLayout)ImageLayout::eShaderReadOnlyOptimal;
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
	inline void record_buffer_image_copy(VkCommandBuffer& cbuf, BufferImageCopyCommand& task) {
		VkBufferImageCopy bc;
		bc.bufferOffset = task.src.offset;
		bc.imageOffset = VkOffset3D{ 0, 0, 0 };
		bc.bufferRowLength = 0;
		bc.bufferImageHeight = 0;
		bc.imageExtent = task.extent;
		bc.imageSubresource.aspectMask = (VkImageAspectFlagBits)ImageAspectFlagBits::eColor;
		bc.imageSubresource.baseArrayLayer = task.base_array_layer;
		bc.imageSubresource.mipLevel = task.mip_level;
		bc.imageSubresource.layerCount = task.layer_count;

		VkImageMemoryBarrier copy_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		copy_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		copy_barrier.oldLayout = (VkImageLayout)ImageLayout::eUndefined;
		copy_barrier.newLayout = (VkImageLayout)ImageLayout::eTransferDstOptimal;
		copy_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		copy_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		copy_barrier.image = task.dst;
		copy_barrier.subresourceRange.aspectMask = bc.imageSubresource.aspectMask;
		copy_barrier.subresourceRange.layerCount = bc.imageSubresource.layerCount;
		copy_barrier.subresourceRange.baseArrayLayer = bc.imageSubresource.baseArrayLayer;
		copy_barrier.subresourceRange.baseMipLevel = bc.imageSubresource.mipLevel;
		copy_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;

		// transition top mip to transfersrc
		VkImageMemoryBarrier mip_to_src_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		mip_to_src_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		mip_to_src_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		mip_to_src_barrier.oldLayout = (VkImageLayout)ImageLayout::eTransferDstOptimal;
		mip_to_src_barrier.newLayout = (VkImageLayout)ImageLayout::eTransferSrcOptimal;
		mip_to_src_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		mip_to_src_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		mip_to_src_barrier.image = task.dst;
		mip_to_src_barrier.subresourceRange = copy_barrier.subresourceRange;
		mip_to_src_barrier.subresourceRange.levelCount = 1;

		// transition top mip to SROO
		VkImageMemoryBarrier top_mip_use_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		top_mip_use_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		top_mip_use_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		top_mip_use_barrier.oldLayout = task.generate_mips ? (VkImageLayout)ImageLayout::eTransferSrcOptimal : (VkImageLayout)ImageLayout::eTransferDstOptimal;
		top_mip_use_barrier.newLayout = (VkImageLayout)ImageLayout::eShaderReadOnlyOptimal;
		top_mip_use_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		top_mip_use_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		top_mip_use_barrier.image = task.dst;
		top_mip_use_barrier.subresourceRange = copy_barrier.subresourceRange;
		top_mip_use_barrier.subresourceRange.levelCount = 1;

		// transition rest of the mips to SROO
		VkImageMemoryBarrier use_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		;
		use_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		use_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		use_barrier.oldLayout = (VkImageLayout)ImageLayout::eTransferSrcOptimal;
		use_barrier.newLayout = (VkImageLayout)ImageLayout::eShaderReadOnlyOptimal;
		use_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		use_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		use_barrier.image = task.dst;
		use_barrier.subresourceRange = copy_barrier.subresourceRange;
		use_barrier.subresourceRange.baseMipLevel = task.mip_level + 1;
		use_barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;

		vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &copy_barrier);
		vkCmdCopyBufferToImage(cbuf, task.src.buffer, task.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);
		if (task.generate_mips) {
			vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &mip_to_src_barrier);

			auto mips = (uint32_t)log2f((float)std::max(task.extent.width, task.extent.height)) + 1;

			for (uint32_t miplevel = task.mip_level + 1; miplevel < mips; miplevel++) {
				uint32_t dmiplevel = miplevel - task.mip_level;
				VkImageBlit blit;
				blit.srcSubresource.aspectMask = copy_barrier.subresourceRange.aspectMask;
				blit.srcSubresource.baseArrayLayer = task.base_array_layer;
				blit.srcSubresource.layerCount = task.layer_count;
				blit.srcSubresource.mipLevel = miplevel - 1;
				blit.srcOffsets[0] = VkOffset3D{ 0 };
				blit.srcOffsets[1] = VkOffset3D{ std::max((int32_t)task.extent.width >> (dmiplevel - 1), 1),
					                               std::max((int32_t)task.extent.height >> (dmiplevel - 1), 1),
					                               (int32_t)task.extent.depth };
				blit.dstSubresource = blit.srcSubresource;
				blit.dstSubresource.mipLevel = miplevel;
				blit.dstOffsets[0] = VkOffset3D{ 0 };
				blit.dstOffsets[1] = VkOffset3D{ std::max((int32_t)task.extent.width >> dmiplevel, 1),
					                               std::max((int32_t)task.extent.height >> dmiplevel, 1),
					                               (int32_t)task.extent.depth };
				vkCmdBlitImage(cbuf, task.dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, task.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

				mip_to_src_barrier.subresourceRange.baseMipLevel = miplevel;
				vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &mip_to_src_barrier);
			}

			vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &use_barrier);
		}

		vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &top_mip_use_barrier);
	}
} // namespace vuk
