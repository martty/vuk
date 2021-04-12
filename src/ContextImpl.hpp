#include "vuk/Context.hpp"
#include "RGImage.hpp"
#include <vuk/Types.hpp>

#include <mutex>
#include <queue>
#include <string_view>
#include <math.h>
#include <functional>

#include "Allocator.hpp"
#include "Pool.hpp"
#include "Cache.hpp"
#include "RenderPass.hpp"
#include "vuk/RenderGraph.hpp"
#include "ResourceBundle.hpp"
#include <vuk/GlobalAllocator.hpp>

namespace {
	inline static uint64_t token_generation[USHRT_MAX];
}

namespace vuk {
	struct ContextImpl {
		Context& ctx;
		VkDevice device;

		std::mutex gfx_queue_lock;
		std::mutex xfer_queue_lock;

		std::mutex named_pipelines_lock;
		std::unordered_map<Name, vuk::PipelineBaseInfo*> named_pipelines;
		std::unordered_map<Name, vuk::ComputePipelineInfo*> named_compute_pipelines;

		std::atomic<uint64_t> query_id_counter = 0;

		std::mutex swapchains_lock;
		plf::colony<Swapchain> swapchains;

		plf::colony<TokenData> token_data;

		std::mutex transient_submit_lock;
		/// @brief with stable addresses, so we can hand out opaque pointers
		plf::colony<LinearResourceAllocator<Allocator>> transient_submit_bundles;
		std::vector<plf::colony<LinearResourceAllocator<Allocator>>::iterator> transient_submit_freelist;

		// TODO: split queue family from allocator
		LinearResourceAllocator<Allocator>* get_linear_allocator(GlobalAllocator& ga, uint32_t queue_family_index) {
			std::lock_guard _(transient_submit_lock);

			auto it = transient_submit_bundles.end();
			for (auto fit = transient_submit_freelist.begin(); fit != transient_submit_freelist.end(); fit++) {
				if ((*fit)->queue_family_index == queue_family_index) {
					it = *fit;
					transient_submit_freelist.erase(fit);
					break;
				}
			}
			if (it == transient_submit_bundles.end()) { // didn't find suitable bundle
				it = transient_submit_bundles.emplace(static_cast<Allocator&>(ga));
				auto& bundle = *it;

				VkCommandPoolCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
				cpci.queueFamilyIndex = queue_family_index;
				cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
				assert(vkCreateCommandPool(device, &cpci, nullptr, &bundle.cpool) == VK_SUCCESS);
			}
			return &*it;
		}

		void cleanup_transient_bundle_recursively(vuk::LinearResourceAllocator<Allocator>* ur) {
			if (!ur) {
				return;
			}
			if (ur->next) {
				cleanup_transient_bundle_recursively(ur->next);
			}
			if (ur->sema) {
				vkDestroySemaphore(device, ur->sema, nullptr);
				ur->sema = VK_NULL_HANDLE;
			}
			if (ur->cpool) {
				vkResetCommandPool(device, ur->cpool, 0);
				if (ur->command_buffers.size() > 0) {
					vkFreeCommandBuffers(device, ur->cpool, (uint32_t)ur->command_buffers.size(), ur->command_buffers.data());
					ur->command_buffers.clear();
				}
			}
			if (ur->buffer) {
				//allocator.free_buffer(ur->buffer);
				ur->buffer = {};
			}
			if (ur->fence) {
				vkDestroyFence(device, ur->fence, nullptr);
				ur->fence = VK_NULL_HANDLE;
			}
			std::lock_guard _(transient_submit_lock);
			//TODO:
			//transient_submit_freelist.push_back(transient_submit_bundles.get_iterator_from_pointer(ur));
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

		Token create_token() {
			auto it = token_data.emplace();
			auto index = token_data.get_index_from_iterator(it);
			assert(index < USHRT_MAX);
			auto gen = token_generation[index];
			return { gen << 16 | index };
		}

		TokenData& get_token_data(Token tok) {
			auto index = tok.id & 0xFFFF;
			auto gen = tok.id >> 16;
			assert(token_generation[index] == gen && "Dead token!");
			return *token_data.get_iterator_from_index(index);
		}

		// token on kill :  token_generation[index]++

		ContextImpl(Context& ctx) : ctx(ctx), device(ctx.device) {
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
	VkImageMemoryBarrier mip_to_src_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	mip_to_src_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	mip_to_src_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	mip_to_src_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferDstOptimal;
	mip_to_src_barrier.newLayout = (VkImageLayout)vuk::ImageLayout::eTransferSrcOptimal;
	mip_to_src_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	mip_to_src_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	mip_to_src_barrier.image = task.dst;
	mip_to_src_barrier.subresourceRange = copy_barrier.subresourceRange;
	mip_to_src_barrier.subresourceRange.levelCount = 1;

	// transition top mip to SROO
	VkImageMemoryBarrier top_mip_use_barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	top_mip_use_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
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
	use_barrier.oldLayout = (VkImageLayout)vuk::ImageLayout::eTransferSrcOptimal;
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
			blit.srcOffsets[1] = VkOffset3D{ std::max((int32_t)task.extent.width >> (dmiplevel - 1), 1), std::max((int32_t)task.extent.height >> (dmiplevel - 1), 1), (int32_t)task.extent.depth };
			blit.dstSubresource = blit.srcSubresource;
			blit.dstSubresource.mipLevel = miplevel;
			blit.dstOffsets[0] = VkOffset3D{ 0 };
			blit.dstOffsets[1] = VkOffset3D{ std::max((int32_t)task.extent.width >> dmiplevel, 1), std::max((int32_t)task.extent.height >> dmiplevel, 1), (int32_t)task.extent.depth };
			vkCmdBlitImage(cbuf, task.dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, task.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

			mip_to_src_barrier.subresourceRange.baseMipLevel = miplevel;
			vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &mip_to_src_barrier);
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
}

