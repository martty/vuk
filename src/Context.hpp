#pragma once

#include <atomic>
#include <gsl/span>

#include "Pool.hpp"
#include "Cache.hpp"
#include "Allocator.hpp"
#include <string_view>

namespace vuk {
	struct RGImage {
		vk::Image image;
		vk::ImageView image_view;
	};
	struct RGCI {
		std::string_view name;
		vk::ImageCreateInfo ici;
		vk::ImageViewCreateInfo ivci;

		bool operator==(const RGCI& other) const {
			return std::tie(name, ici, ivci) == std::tie(other.name, other.ici, other.ivci);
		}
	};
	template<> struct create_info<RGImage> {
		using type = RGCI;
	};
}

namespace std {
	template <>
	struct hash<vuk::RGCI> {
		size_t operator()(vuk::RGCI const & x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.name, x.ici, x.ivci);
			return h;
		}
	};
};

namespace vuk {
	template<> struct create_info<Allocator::Pool> {
		using type = PoolSelect;
	};

}

#include <queue>
#include <algorithm>
namespace vuk {
	struct GPUHandleBase {
		size_t id;
	};

	template<class T>
	struct GPUHandle : public GPUHandleBase {};

	struct TransferStub {
		size_t id;
	};

	class Context {
	public:
		constexpr static size_t FC = 3;

		vk::Device device;
		vk::PhysicalDevice physical_device;
		Allocator allocator;
		Pool<vk::CommandBuffer, FC> cbuf_pools;
		Pool<vk::Semaphore, FC> semaphore_pools;
		Pool<vk::Fence, FC> fence_pools;
		Pool<vk::DescriptorSet, FC> descriptor_pools;
		vk::UniquePipelineCache vk_pipeline_cache;
		Cache<PipelineInfo> pipeline_cache;
		Cache<vk::RenderPass> renderpass_cache;
		PerFrameCache<RGImage, FC> transient_images;
		PerFrameCache<Allocator::Pool, FC> scratch_buffers;
		Cache<vk::DescriptorSet> descriptor_sets;

		std::array<std::vector<vk::Image>, Context::FC> image_recycle;
		std::array<std::vector<vk::ImageView>, Context::FC> image_view_recycle;

		std::unordered_map<std::string_view, create_info_t<vuk::PipelineInfo>> named_pipelines;

		vk::Queue graphics_queue;

		Context(vk::Device device, vk::PhysicalDevice physical_device) : device(device), physical_device(physical_device),
			allocator(device, physical_device),
			cbuf_pools(*this),
			semaphore_pools(*this),
			fence_pools(*this),
			descriptor_pools(*this),
			pipeline_cache(*this),
			renderpass_cache(*this),
			transient_images(*this),
			scratch_buffers(*this),
			descriptor_sets(*this)
		{
			vk_pipeline_cache = device.createPipelineCacheUnique({});
		}

		template<class T>
		void create_named(const char * name, create_info_t<T> ci) {
			if constexpr (std::is_same_v<T, vk::Pipeline>) {
				named_pipelines.emplace(name, ci);
			}
		}

		void destroy(const RGImage& image) {
			device.destroy(image.image_view);
			allocator.destroy_image(image.image);
		}

		void destroy(const Allocator::Pool& v) {
			allocator.destroy_scratch_pool(v);
		}


		std::atomic<size_t> frame_counter = 0;
		InflightContext begin();
	};

	inline unsigned prev_(unsigned frame, unsigned amt, unsigned FC) {
		return ((frame - amt) % FC) + ((frame >= amt) ? 0 : FC - 1);
	}

	class InflightContext {
	public:
		Context& ctx;
		unsigned frame;
		Pool<vk::CommandBuffer, Context::FC>::PFView commandbuffer_pools;
		Pool<vk::Semaphore, Context::FC>::PFView semaphore_pools;
		Pool<vk::Fence, Context::FC>::PFView fence_pools;
		Pool<vk::DescriptorSet, Context::FC>::PFView descriptor_pools;
		Cache<PipelineInfo>::PFView pipeline_cache;
		Cache<vk::RenderPass>::PFView renderpass_cache;
		PerFrameCache<vuk::RGImage, Context::FC>::PFView transient_images;
		PerFrameCache<Allocator::Pool, Context::FC>::PFView scratch_buffers;
		Cache<vk::DescriptorSet>::PFView descriptor_sets;

		InflightContext(Context& ctx, unsigned frame) : ctx(ctx), frame(frame),
			commandbuffer_pools(ctx.cbuf_pools.get_view(*this)),
			semaphore_pools(ctx.semaphore_pools.get_view(*this)),
			fence_pools(ctx.fence_pools.get_view(*this)),
			descriptor_pools(ctx.descriptor_pools.get_view(*this)),
			pipeline_cache(*this, ctx.pipeline_cache),
			renderpass_cache(*this, ctx.renderpass_cache),
			transient_images(*this, ctx.transient_images),
			scratch_buffers(*this, ctx.scratch_buffers),
			descriptor_sets(*this, ctx.descriptor_sets)
		{
			// image recycling
			for (auto& img : ctx.image_recycle[frame]) {
				ctx.allocator.destroy_image(img);
			}
			ctx.image_recycle[frame].clear();

			for (auto& iv : ctx.image_view_recycle[frame]) {
				ctx.device.destroy(iv);
			}
			ctx.image_view_recycle[frame].clear();

		}

		struct BufferCopyCommand {
			Allocator::Buffer src;
			Allocator::Buffer dst;
			TransferStub stub;
		};

		struct BufferImageCopyCommand {
			Allocator::Buffer src;
			vk::Image dst;
			vk::Extent3D extent;
			TransferStub stub;
		};

		
		std::atomic<size_t> transfer_id = 1;
		std::atomic<size_t> last_transfer_complete = 0;

		struct PendingTransfer {
			size_t last_transfer_id;
			vk::Fence fence;
		};
		// needs to be mpsc
		std::mutex transfer_mutex;
		std::queue<BufferCopyCommand> buffer_transfer_commands;
		std::queue<BufferImageCopyCommand> bufferimage_transfer_commands;
		// only accessed by DMAtask
		std::queue<PendingTransfer> pending_transfers;

		TransferStub enqueue_transfer(Allocator::Buffer src, Allocator::Buffer dst) {
			std::lock_guard _(transfer_mutex);
			TransferStub stub{ transfer_id++ };
			buffer_transfer_commands.push({ src, dst, stub });
			return stub;
		}

		TransferStub enqueue_transfer(Allocator::Buffer src, vk::Image dst, vk::Extent3D extent) {
			std::lock_guard _(transfer_mutex);
			TransferStub stub{ transfer_id++ };
			bufferimage_transfer_commands.push({ src, dst, extent, stub });
			return stub;
		}

		void wait_all_transfers() {
			std::lock_guard _(transfer_mutex);

			while (!pending_transfers.empty()) {
				ctx.device.waitForFences(pending_transfers.front().fence, true, UINT64_MAX);
				auto last = pending_transfers.front();
				last_transfer_complete = last.last_transfer_id;
				pending_transfers.pop();
			}
		}

		// recycle
		std::mutex recycle_mutex;

		void _add_images_to_recycle(std::vector<vk::Image>&& images) {
			std::lock_guard _(recycle_mutex);
			ctx.image_recycle[frame].insert(ctx.image_recycle[frame].end(), images.begin(), images.end());
		}
		void _add_image_views_to_recycle(std::vector<vk::ImageView>&& images) {
			std::lock_guard _(recycle_mutex);
			ctx.image_view_recycle[frame].insert(ctx.image_view_recycle[frame].end(), images.begin(), images.end());
		}


		PerThreadContext begin();
	};

	inline InflightContext Context::begin() {
		return InflightContext(*this, frame_counter++ % FC);
	}

	class PerThreadContext {
	public:
		Context& ctx;
		InflightContext& ifc;
		unsigned tid;
		Pool<vk::CommandBuffer, Context::FC>::PFPTView commandbuffer_pool;
		Pool<vk::Semaphore, Context::FC>::PFPTView semaphore_pool;
		Pool<vk::Fence, Context::FC>::PFPTView fence_pool;
		Pool<vk::DescriptorSet, Context::FC>::PFPTView descriptor_pool;
		Cache<PipelineInfo>::PFPTView pipeline_cache;
		Cache<vk::RenderPass>::PFPTView renderpass_cache;
		PerFrameCache<vuk::RGImage, Context::FC>::PFPTView transient_images;
		PerFrameCache<Allocator::Pool, Context::FC>::PFPTView scratch_buffers;
		Cache<vk::DescriptorSet>::PFPTView descriptor_sets;

		// recycling global objects
		std::vector<Allocator::Buffer> buffer_recycle;
		std::vector<vk::Image> image_recycle;
		std::vector<vk::ImageView> image_view_recycle;

		PerThreadContext(InflightContext& ifc, unsigned tid) : ctx(ifc.ctx), ifc(ifc), tid(tid),
			commandbuffer_pool(ifc.commandbuffer_pools.get_view(*this)),
			semaphore_pool(ifc.semaphore_pools.get_view(*this)),
			fence_pool(ifc.fence_pools.get_view(*this)),
			descriptor_pool(ifc.descriptor_pools.get_view(*this)),
			pipeline_cache(*this, ifc.pipeline_cache),
			renderpass_cache(*this, ifc.renderpass_cache),
			transient_images(*this, ifc.transient_images),
			scratch_buffers(*this, ifc.scratch_buffers),
			descriptor_sets(*this, ifc.descriptor_sets)
		{}

		~PerThreadContext() {
			ifc._add_images_to_recycle(std::move(image_recycle));
			ifc._add_image_views_to_recycle(std::move(image_view_recycle));
		}

		void destroy(vk::Image image) {
			image_recycle.push_back(image);
		}

		void destroy(vk::ImageView image) {
			image_view_recycle.push_back(image);
		}


		Allocator::Buffer _allocate_scratch_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, size_t size, bool create_mapped) {
			auto& pool = scratch_buffers.acquire({ mem_usage, buffer_usage });
			return ifc.ctx.allocator.allocate_buffer(pool, size, create_mapped);
		}

		bool is_ready(const TransferStub& stub) {
			return ifc.last_transfer_complete >= stub.id;
		}

		void wait_all_transfers() {
			// TODO: remove when we go MT
			dma_task(); // run one transfer so it is more easy to follow
			return ifc.wait_all_transfers();
		}

		template<class T>
		std::pair<Allocator::Buffer, TransferStub> create_scratch_buffer(gsl::span<T> data) {
			auto dst = _allocate_scratch_buffer(MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eUniformBuffer, sizeof(T) * data.size(), true);
			auto stub = upload(dst, data);
			return { dst, stub };
		}

		std::tuple<vk::Image, vk::ImageView, TransferStub> create_image(vk::Format format, vk::Extent3D extents, void* data) {
			vk::ImageCreateInfo ici;
			ici.format = format;
			ici.extent = extents;
			ici.arrayLayers = 1;
			ici.initialLayout = vk::ImageLayout::eUndefined;
			ici.mipLevels = 1;
			ici.imageType = vk::ImageType::e2D;
			ici.samples = vk::SampleCountFlagBits::e1;
			ici.tiling = vk::ImageTiling::eOptimal;
			ici.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
			auto dst = ifc.ctx.allocator.create_image(ici);
			auto stub = upload(dst, extents, gsl::span<std::byte>((std::byte*)data, extents.width * extents.height * extents.depth * 4));
			vk::ImageViewCreateInfo ivci;
			ivci.format = format;
			ivci.image = dst;
			ivci.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			ivci.subresourceRange.baseArrayLayer = 0;
			ivci.subresourceRange.baseMipLevel = 0;
			ivci.subresourceRange.layerCount = 1;
			ivci.subresourceRange.levelCount = 1;
			ivci.viewType = vk::ImageViewType::e2D;
			auto iv = ifc.ctx.device.createImageView(ivci);
			return { dst, std::move(iv), stub };
		}


		template<class T>
		TransferStub upload(Allocator::Buffer dst, gsl::span<T> data) {
			auto staging = _allocate_scratch_buffer(MemoryUsage::eCPUonly, vk::BufferUsageFlagBits::eTransferSrc, sizeof(T) * data.size(), true);
			::memcpy(staging.mapped_ptr, data.data(), sizeof(T) * data.size());
			
			return ifc.enqueue_transfer(staging, dst);
		}

		template<class T>
		TransferStub upload(vk::Image dst, vk::Extent3D extent, gsl::span<T> data) {
			auto staging = _allocate_scratch_buffer(MemoryUsage::eCPUonly, vk::BufferUsageFlagBits::eTransferSrc, sizeof(T) * data.size(), true);
			::memcpy(staging.mapped_ptr, data.data(), sizeof(T) * data.size());
			
			return ifc.enqueue_transfer(staging, dst, extent);
		}


		void dma_task() {
			std::lock_guard _(ifc.transfer_mutex);
			while(!ifc.pending_transfers.empty() && ctx.device.getFenceStatus(ifc.pending_transfers.front().fence) == vk::Result::eSuccess) {
				auto last = ifc.pending_transfers.front();
				ifc.last_transfer_complete = last.last_transfer_id;
				ifc.pending_transfers.pop();
			}

			if (ifc.buffer_transfer_commands.empty() && ifc.bufferimage_transfer_commands.empty()) return;
			auto cbuf = commandbuffer_pool.acquire(1)[0];
			cbuf.begin(vk::CommandBufferBeginInfo{});
			size_t last = 0;
			while (!ifc.buffer_transfer_commands.empty()) {
				auto task = ifc.buffer_transfer_commands.front();
				ifc.buffer_transfer_commands.pop();
				vk::BufferCopy bc;
				bc.dstOffset = task.dst.offset;
				bc.srcOffset = task.src.offset;
				bc.size = task.src.size;
				cbuf.copyBuffer(task.src.buffer, task.dst.buffer, bc);
				last = std::max(last, task.stub.id);
			}
			while (!ifc.bufferimage_transfer_commands.empty()) {
				auto task = ifc.bufferimage_transfer_commands.front();
				ifc.bufferimage_transfer_commands.pop();
				vk::BufferImageCopy bc;
				bc.bufferOffset = task.src.offset;
				bc.imageOffset = 0;
				bc.bufferRowLength = 0;
				bc.bufferImageHeight = 0;
				bc.imageExtent = task.extent;
				bc.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
				bc.imageSubresource.baseArrayLayer = 0;
				bc.imageSubresource.mipLevel = 0;
				bc.imageSubresource.layerCount = 1;

				vk::ImageMemoryBarrier copy_barrier;
				copy_barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
				copy_barrier.oldLayout = vk::ImageLayout::eUndefined;
				copy_barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
				copy_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				copy_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				copy_barrier.image = task.dst;
				copy_barrier.subresourceRange.aspectMask = bc.imageSubresource.aspectMask;
				copy_barrier.subresourceRange.layerCount = bc.imageSubresource.layerCount;
				copy_barrier.subresourceRange.baseArrayLayer = bc.imageSubresource.baseArrayLayer;
				copy_barrier.subresourceRange.baseMipLevel = bc.imageSubresource.mipLevel;
				copy_barrier.subresourceRange.levelCount = 1;

				vk::ImageMemoryBarrier use_barrier;
				use_barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
				use_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
				use_barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
				use_barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
				use_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				use_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				use_barrier.image = task.dst;
				use_barrier.subresourceRange = copy_barrier.subresourceRange;

				cbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits(0), {}, {}, copy_barrier);
				cbuf.copyBufferToImage(task.src.buffer, task.dst, vk::ImageLayout::eTransferDstOptimal, bc);
				cbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, vk::DependencyFlagBits(0), {}, {}, use_barrier);
				last = std::max(last, task.stub.id);
			}
			cbuf.end();
			auto fence = fence_pool.acquire(1)[0];
			vk::SubmitInfo si;
			si.commandBufferCount = 1;
			si.pCommandBuffers = &cbuf;
			ifc.ctx.graphics_queue.submit(si, fence);
			ifc.pending_transfers.emplace(InflightContext::PendingTransfer{ last, fence });
		}
	};

	inline PerThreadContext InflightContext::begin() {
		return PerThreadContext{ *this, 0 };
	}

	template<class T, size_t FC>
	typename Pool<T, FC>::PFView Pool<T, FC>::get_view(InflightContext& ctx) {
		return { ctx, *this, per_frame_storage[ctx.frame] };
	}

	template<class T, size_t FC>
	Pool<T, FC>::PFView::PFView(InflightContext& ifc, Pool<T, FC>& storage, plf::colony<PooledType<T>>& fv) : ifc(ifc), storage(storage), frame_values(fv) {
		storage.reset(ifc.frame);
	}

	template<class T>
	T create(PerThreadContext& ptc, const create_info_t<T>& cinfo) {
		auto& ctx = ptc.ifc.ctx;
		if constexpr (std::is_same_v<T, PipelineInfo>) {
			return { ctx.device.createGraphicsPipeline(*ctx.vk_pipeline_cache, cinfo.gpci), cinfo.pipeline_layout, cinfo.layout_info };
		} else if constexpr (std::is_same_v<T, vk::RenderPass>) {
			return ctx.device.createRenderPass(cinfo);
		} else if constexpr (std::is_same_v<T, vuk::RGImage>) {
			RGImage res;
			res.image = ctx.allocator.create_image_for_rendertarget(cinfo.ici);
			auto ivci = cinfo.ivci;
			ivci.image = res.image;
			res.image_view = ctx.device.createImageView(ivci);
			return res;
		} else if constexpr (std::is_same_v<T, Allocator::Pool>) {
			return ctx.allocator.allocate_pool(cinfo.mem_usage, cinfo.buffer_usage);
		} else if constexpr (std::is_same_v<T, vk::DescriptorSet>) {
			auto ds = ptc.descriptor_pool.acquire(cinfo.layout_info);
	
			unsigned long leading_zero = 0;
			auto mask = cinfo.used.to_ulong();
			auto is_null = _BitScanReverse(&leading_zero, mask);
			leading_zero++;
			std::array<vk::WriteDescriptorSet, VUK_MAX_BINDINGS> writes;
			for (unsigned i = 0; i < leading_zero; i++) {
				if (!cinfo.used.test(i)) continue;
				auto& write = writes[i];
				auto& binding = cinfo.bindings[i];
				write.descriptorType = binding.type;
				write.dstArrayElement = 0;
				write.descriptorCount = 1;
				write.dstBinding = i;
				write.dstSet = ds;
				switch (binding.type) {
					case vk::DescriptorType::eUniformBuffer:
					case vk::DescriptorType::eStorageBuffer:
						write.pBufferInfo = &binding.buffer;
						break;
					case vk::DescriptorType::eSampledImage:
					case vk::DescriptorType::eSampler:
					case vk::DescriptorType::eCombinedImageSampler:
						write.pImageInfo = &binding.image;
						break;
					default:
						assert(0);
				}
			}
			ctx.device.updateDescriptorSets(leading_zero, writes.data(), 0, nullptr);
			return ds;
		}
	}

}
