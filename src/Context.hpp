#pragma once

#include <atomic>
#include <gsl/span>

#include "Pool.hpp"
#include "Cache.hpp"
#include "Allocator.hpp"
#include "Program.hpp"
#include "Pipeline.hpp"
#include <string_view>

using Name = std::string_view;


namespace vuk {
	struct RGImage {
		vk::Image image;
		vuk::ImageView image_view;
	};
	struct RGCI {
		Name name;
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

	// high level type around binding a sampled image with a sampler
	struct SampledImage {
		struct Global {
			vuk::ImageView iv;
			vk::SamplerCreateInfo sci = {};
			vk::ImageLayout image_layout;
		};

		struct RenderGraphAttachment {
			Name attachment_name;
			vk::SamplerCreateInfo sci = {};
			vk::ImageLayout image_layout;
		};

		bool is_global;
		union {
			Global global = {};
			RenderGraphAttachment rg_attachment;
		};

		SampledImage(Global g) : global(g), is_global(true) {}
		SampledImage(RenderGraphAttachment g) : rg_attachment(g), is_global(false) {}

		SampledImage(const SampledImage& o) {
			*this = o;
		}

		SampledImage& operator=(const SampledImage& o) {
			if (o.is_global) {
				global = {};
				global = o.global;
			} else {
				rg_attachment = {};
				rg_attachment = o.rg_attachment;
			}
			is_global = o.is_global;
			return *this;
		}
	};

	// the returned values are pointer stable until the frame gets recycled
	template<>
	struct PooledType<vuk::SampledImage> {
		plf::colony<vuk::SampledImage> values;
		size_t needle = 0;

		PooledType(Context&) {}
		vuk::SampledImage& acquire(PerThreadContext& ptc, vuk::SampledImage si);
		void reset(Context&) { needle = 0; }
		void free(Context&) {} // nothing to free, this is non-owning
	};

	inline vuk::SampledImage& PooledType<vuk::SampledImage>::acquire(PerThreadContext& ptc, vuk::SampledImage si) {
		if (values.size() < (needle + 1)) {
			needle++;
			return *values.emplace(std::move(si));
		} else {
			auto it = values.begin();
			values.advance(it, needle++);
			*it = si;
			return *it;
		}
	}

}

#include <queue>
#include <algorithm>

namespace vuk {
	struct TransferStub {
		size_t id;
	};

	struct Swapchain {
		vk::SwapchainKHR swapchain;
		vk::SurfaceKHR surface;

		vk::Format format;
		vk::Extent2D extent = { 0, 0 };
		std::vector<vk::Image> images;
		std::vector<vk::ImageView> _ivs;
		std::vector<vuk::ImageView> image_views;
	};
	using SwapchainRef = Swapchain*;
	struct Program;

	class Context {
	public:
		constexpr static size_t FC = 3;

		vk::Device device;
		vk::PhysicalDevice physical_device;
		Allocator allocator;
		Pool<vk::CommandBuffer, FC> cbuf_pools;
		Pool<vk::Semaphore, FC> semaphore_pools;
		Pool<vk::Fence, FC> fence_pools;
		vk::UniquePipelineCache vk_pipeline_cache;
		Cache<PipelineInfo> pipeline_cache;
		Cache<vk::RenderPass> renderpass_cache;
		Cache<vk::Framebuffer> framebuffer_cache;
		PerFrameCache<RGImage, FC> transient_images;
		PerFrameCache<Allocator::Pool, FC> scratch_buffers;
		PerFrameCache<vuk::DescriptorPool, FC> pool_cache;
		Cache<vuk::DescriptorSet> descriptor_sets;
		Cache<vk::Sampler> sampler_cache;
		Pool<vuk::SampledImage, FC> sampled_images;
		Cache<vuk::ShaderModule> shader_modules;
		Cache<vuk::DescriptorSetLayoutAllocInfo> descriptor_set_layouts;
		Cache<vk::PipelineLayout> pipeline_layouts;

		std::array<std::vector<vk::Image>, Context::FC> image_recycle;
		std::array<std::vector<vk::ImageView>, Context::FC> image_view_recycle;

		std::unordered_map<std::string_view, create_info_t<vuk::PipelineInfo>> named_pipelines;

		plf::colony<Swapchain> swapchains;

		SwapchainRef add_swapchain(Swapchain sw) {
			sw.image_views.reserve(sw._ivs.size());
			for (auto& v : sw._ivs) {
				sw.image_views.push_back(wrap(vk::ImageView{ v }));
			}

			return &*swapchains.emplace(sw);
		}

		vk::Queue graphics_queue;

		Context(vk::Device device, vk::PhysicalDevice physical_device) : device(device), physical_device(physical_device),
			allocator(device, physical_device),
			cbuf_pools(*this),
			semaphore_pools(*this),
			fence_pools(*this),
			pipeline_cache(*this),
			renderpass_cache(*this),
			framebuffer_cache(*this),
			transient_images(*this),
			scratch_buffers(*this),
			descriptor_sets(*this),
			sampler_cache(*this),
			sampled_images(*this),
			pool_cache(*this),
			shader_modules(*this),
			descriptor_set_layouts(*this),
			pipeline_layouts(*this) {
			vk_pipeline_cache = device.createPipelineCacheUnique({});
		}

		template<class T>
		void create_named(const char* name, create_info_t<T> ci) {
			if constexpr (std::is_same_v<T, vk::Pipeline>) {
				named_pipelines.emplace(name, ci);
			}
		}

		Program compile(gsl::span<std::string> shaders);


		template<class T>
		Handle<T> wrap(T payload) {
			return { unique_handle_id_counter++, payload };
		}

		void destroy(const RGImage& image) {
			device.destroy(image.image_view.payload);
			allocator.destroy_image(image.image);
		}

		void destroy(const Allocator::Pool& v) {
			allocator.destroy_scratch_pool(v);
		}

		void destroy(const vuk::DescriptorPool& dp) {
			for (auto& p : dp.pools) {
				device.destroy(p);
			}
		}

		void destroy(vuk::PipelineInfo) {}

		void destroy(vuk::ShaderModule sm) {
			device.destroy(sm.shader_module);
		}

		void destroy(vuk::DescriptorSetLayoutAllocInfo ds) {
			device.destroy(ds.layout);
		}

		void destroy(vk::PipelineLayout pl) {
			device.destroy(pl);
		}

		void destroy(vk::RenderPass rp) {
			device.destroy(rp);
		}
		void destroy(vuk::DescriptorSet) {
			// no-op, we destroy the pools
		}
		void destroy(vk::Framebuffer fb) {
			device.destroy(fb);
		}
		void destroy(vk::Sampler sa) {
			device.destroy(sa);
		}

		~Context() {
			device.waitIdle();
			for (auto& s : swapchains) {
				for (auto& swiv : s.image_views) {
					device.destroy(swiv.payload);
				}
				device.destroy(s.swapchain);
			}
		}

		std::atomic<size_t> unique_handle_id_counter = 0;

		std::atomic<size_t> frame_counter = 0;
		InflightContext begin();
	};

	inline unsigned prev_(unsigned frame, unsigned amt, unsigned FC) {
		return ((frame - amt) % FC) + ((frame >= amt) ? 0 : FC - 1);
	}

	class InflightContext {
	public:
		Context& ctx;
		unsigned absolute_frame;
		unsigned frame;
		Pool<vk::CommandBuffer, Context::FC>::PFView commandbuffer_pools;
		Pool<vk::Semaphore, Context::FC>::PFView semaphore_pools;
		Pool<vk::Fence, Context::FC>::PFView fence_pools;
		Cache<PipelineInfo>::PFView pipeline_cache;
		Cache<vk::RenderPass>::PFView renderpass_cache;
		Cache<vk::Framebuffer>::PFView framebuffer_cache;
		PerFrameCache<vuk::RGImage, Context::FC>::PFView transient_images;
		PerFrameCache<Allocator::Pool, Context::FC>::PFView scratch_buffers;
		Cache<vuk::DescriptorSet>::PFView descriptor_sets;
		Cache<vk::Sampler>::PFView sampler_cache;
		Pool<vuk::SampledImage, Context::FC>::PFView sampled_images;
		PerFrameCache<vuk::DescriptorPool, Context::FC>::PFView pool_cache;

		Cache<vuk::ShaderModule>::PFView shader_modules;
		Cache<vuk::DescriptorSetLayoutAllocInfo>::PFView descriptor_set_layouts;
		Cache<vk::PipelineLayout>::PFView pipeline_layouts;


		InflightContext(Context& ctx, unsigned absolute_frame);

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

		void destroy(std::vector<vk::Image>&& images) {
			std::lock_guard _(recycle_mutex);
			ctx.image_recycle[frame].insert(ctx.image_recycle[frame].end(), images.begin(), images.end());
		}
		void destroy(std::vector<vk::ImageView>&& images) {
			std::lock_guard _(recycle_mutex);
			ctx.image_view_recycle[frame].insert(ctx.image_view_recycle[frame].end(), images.begin(), images.end());
		}

		PerThreadContext begin();
	};

	inline InflightContext Context::begin() {
		return InflightContext(*this, frame_counter++);
	}

	class PerThreadContext {
	public:
		Context& ctx;
		InflightContext& ifc;
		unsigned tid;
		Pool<vk::CommandBuffer, Context::FC>::PFPTView commandbuffer_pool;
		Pool<vk::Semaphore, Context::FC>::PFPTView semaphore_pool;
		Pool<vk::Fence, Context::FC>::PFPTView fence_pool;
		Cache<PipelineInfo>::PFPTView pipeline_cache;
		Cache<vk::RenderPass>::PFPTView renderpass_cache;
		Cache<vk::Framebuffer>::PFPTView framebuffer_cache;
		PerFrameCache<vuk::RGImage, Context::FC>::PFPTView transient_images;
		PerFrameCache<Allocator::Pool, Context::FC>::PFPTView scratch_buffers;
		Cache<vuk::DescriptorSet>::PFPTView descriptor_sets;
		Cache<vk::Sampler>::PFPTView sampler_cache;
		Pool<vuk::SampledImage, Context::FC>::PFPTView sampled_images;
		PerFrameCache<vuk::DescriptorPool, Context::FC>::PFPTView pool_cache;
		Cache<vuk::ShaderModule>::PFPTView shader_modules;
		Cache<vuk::DescriptorSetLayoutAllocInfo>::PFPTView descriptor_set_layouts;
		Cache<vk::PipelineLayout>::PFPTView pipeline_layouts;


		// recycling global objects
		std::vector<Allocator::Buffer> buffer_recycle;
		std::vector<vk::Image> image_recycle;
		std::vector<vk::ImageView> image_view_recycle;

		PerThreadContext(InflightContext& ifc, unsigned tid) : ctx(ifc.ctx), ifc(ifc), tid(tid),
			commandbuffer_pool(ifc.commandbuffer_pools.get_view(*this)),
			semaphore_pool(ifc.semaphore_pools.get_view(*this)),
			fence_pool(ifc.fence_pools.get_view(*this)),
			pipeline_cache(*this, ifc.pipeline_cache),
			renderpass_cache(*this, ifc.renderpass_cache),
			framebuffer_cache(*this, ifc.framebuffer_cache),
			transient_images(*this, ifc.transient_images),
			scratch_buffers(*this, ifc.scratch_buffers),
			descriptor_sets(*this, ifc.descriptor_sets),
			sampler_cache(*this, ifc.sampler_cache),
			sampled_images(ifc.sampled_images.get_view(*this)),
			pool_cache(*this, ifc.pool_cache),
			shader_modules(*this, ifc.shader_modules),
			descriptor_set_layouts(*this, ifc.descriptor_set_layouts),
			pipeline_layouts(*this, ifc.pipeline_layouts) {}

		~PerThreadContext() {
			ifc.destroy(std::move(image_recycle));
			ifc.destroy(std::move(image_view_recycle));
		}
		template<class T>
		void destroy(T t) {
			ctx.destroy(t);
		}

		void destroy(vk::Image image) {
			image_recycle.push_back(image);
		}

		void destroy(vuk::ImageView image) {
			image_view_recycle.push_back(image.payload);
		}

		void destroy(vuk::DescriptorSet ds) {
			// note that since we collect at integer times FC, we are releasing the DS back to the right pool
			pool_cache.acquire(ds.layout_info).free_sets.push_back(ds.descriptor_set);
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

		// since data is provided, we will add TransferDst to the flags automatically
		template<class T>
		std::pair<Allocator::Buffer, TransferStub> create_scratch_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, gsl::span<T> data) {
			auto dst = _allocate_scratch_buffer(mem_usage, vk::BufferUsageFlagBits::eTransferDst | buffer_usage, sizeof(T) * data.size(), false);
			auto stub = upload(dst, data);
			return { dst, stub };
		}

		std::tuple<vk::Image, vuk::ImageView, TransferStub> create_image(vk::Format format, vk::Extent3D extents, void* data) {
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
			return { dst, ifc.ctx.wrap(iv), stub };
		}


		template<class T>
		TransferStub upload(Allocator::Buffer dst, gsl::span<T> data) {
			if (data.empty()) return { 0 };
			auto staging = _allocate_scratch_buffer(MemoryUsage::eCPUonly, vk::BufferUsageFlagBits::eTransferSrc, sizeof(T) * data.size(), true);
			::memcpy(staging.mapped_ptr, data.data(), sizeof(T) * data.size());

			return ifc.enqueue_transfer(staging, dst);
		}

		template<class T>
		TransferStub upload(vk::Image dst, vk::Extent3D extent, gsl::span<T> data) {
			assert(!data.empty());
			auto staging = _allocate_scratch_buffer(MemoryUsage::eCPUonly, vk::BufferUsageFlagBits::eTransferSrc, sizeof(T) * data.size(), true);
			::memcpy(staging.mapped_ptr, data.data(), sizeof(T) * data.size());

			return ifc.enqueue_transfer(staging, dst, extent);
		}


		void dma_task() {
			std::lock_guard _(ifc.transfer_mutex);
			while (!ifc.pending_transfers.empty() && ctx.device.getFenceStatus(ifc.pending_transfers.front().fence) == vk::Result::eSuccess) {
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

		vuk::SampledImage& make_sampled_image(vuk::ImageView iv, vk::SamplerCreateInfo sci) {
			vuk::SampledImage si(vuk::SampledImage::Global{ iv, sci, vk::ImageLayout::eShaderReadOnlyOptimal });
			return sampled_images.acquire(si);
		}

		vuk::SampledImage& make_sampled_image(Name n, vk::SamplerCreateInfo sci) {
			vuk::SampledImage si(vuk::SampledImage::RenderGraphAttachment{ n, sci, vk::ImageLayout::eShaderReadOnlyOptimal });
			return sampled_images.acquire(si);
		}
	};

	inline InflightContext::InflightContext(Context& ctx, unsigned absolute_frame) : ctx(ctx),
		absolute_frame(absolute_frame),
		frame(absolute_frame% Context::FC),
		commandbuffer_pools(ctx.cbuf_pools.get_view(*this)),
		semaphore_pools(ctx.semaphore_pools.get_view(*this)),
		fence_pools(ctx.fence_pools.get_view(*this)),
		pipeline_cache(*this, ctx.pipeline_cache),
		renderpass_cache(*this, ctx.renderpass_cache),
		framebuffer_cache(*this, ctx.framebuffer_cache),
		transient_images(*this, ctx.transient_images),
		scratch_buffers(*this, ctx.scratch_buffers),
		descriptor_sets(*this, ctx.descriptor_sets),
		sampler_cache(*this, ctx.sampler_cache),
		sampled_images(ctx.sampled_images.get_view(*this)),
		pool_cache(*this, ctx.pool_cache),
		shader_modules(*this, ctx.shader_modules),
		descriptor_set_layouts(*this, ctx.descriptor_set_layouts),
		pipeline_layouts(*this, ctx.pipeline_layouts) {
		// image recycling
		for (auto& img : ctx.image_recycle[frame]) {
			ctx.allocator.destroy_image(img);
		}
		ctx.image_recycle[frame].clear();

		for (auto& iv : ctx.image_view_recycle[frame]) {
			ctx.device.destroy(iv);
		}
		ctx.image_view_recycle[frame].clear();

		for (auto& sb : scratch_buffers.cache.data[frame].pool) {
			ctx.allocator.reset_pool(sb);
		}

		auto ptc = begin();
		ptc.descriptor_sets.collect(Context::FC * 2);
	}

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
	/*
		vk::Framebuffer create(PerThreadContext& ptc, const create_info_t<vk::Framebuffer>& cinfo) {
			return ptc.ctx.device.createFramebuffer(cinfo);
		}
		*/
}
#include <spirv_cross.hpp>
#include <shaderc/shaderc.hpp>
#include <fstream>
#include <sstream>

inline std::string slurp(const std::string& path) {
	std::ostringstream buf;
	std::ifstream input(path.c_str());
	buf << input.rdbuf();
	return buf.str();
}

inline void burp(const std::string& in, const std::string& path) {
	std::ofstream output(path.c_str(), std::ios::trunc);
	if (!output.is_open()) {
	}
	output << in;
	output.close();
}

namespace vuk {
	template<class T>
	T create(PerThreadContext& ptc, const create_info_t<T>& cinfo) {
		auto& ctx = ptc.ifc.ctx;
		if constexpr (std::is_same_v<T, PipelineInfo>) {
			printf("Creating pipeline\n");
			std::vector<vk::PipelineShaderStageCreateInfo> psscis;

			// accumulate descriptors from all stages
			vuk::Program accumulated_reflection;
			for (auto& path : cinfo.shaders) {
				auto contents = slurp(path);
				auto& sm = ptc.shader_modules.acquire({ contents, path });
				vk::PipelineShaderStageCreateInfo shaderStage;
				shaderStage.pSpecializationInfo = nullptr;
				shaderStage.stage = sm.stage;
				shaderStage.module = sm.shader_module;
				shaderStage.pName = "main"; //TODO: make param
				psscis.push_back(shaderStage);
				accumulated_reflection.append(sm.reflection_info);
			}
			// acquire descriptor set layouts (1 per set)
			// acquire pipeline layout
			vuk::PipelineLayoutCreateInfo plci;
			plci.dslcis = vuk::PipelineCreateInfo::build_descriptor_layouts(accumulated_reflection);
			plci.pcrs = accumulated_reflection.push_constant_ranges;
			plci.plci.pushConstantRangeCount = accumulated_reflection.push_constant_ranges.size();
			plci.plci.pPushConstantRanges = accumulated_reflection.push_constant_ranges.data();
			std::array<vuk::DescriptorSetLayoutAllocInfo, VUK_MAX_SETS> dslai;
			std::vector<vk::DescriptorSetLayout> dsls;
			for (auto& dsl : plci.dslcis) {
				dsl.dslci.bindingCount = dsl.bindings.size();
				dsl.dslci.pBindings = dsl.bindings.data();
				auto l = ptc.descriptor_set_layouts.acquire(dsl);
				dslai[dsl.index] = l;
				dsls.push_back(dslai[dsl.index].layout);
			}
			plci.plci.pSetLayouts = dsls.data();
			plci.plci.setLayoutCount = dsls.size();
			// create gfx pipeline
			vk::GraphicsPipelineCreateInfo gpci = cinfo.to_vk();
			gpci.layout = ptc.pipeline_layouts.acquire(plci);
			gpci.pStages = psscis.data();
			gpci.stageCount = psscis.size();

			return { ctx.device.createGraphicsPipeline(*ctx.vk_pipeline_cache, gpci), gpci.layout, dslai };
		} else if constexpr (std::is_same_v<T, vuk::ShaderModule>) {
			shaderc::Compiler compiler;
			shaderc::CompileOptions options;

			shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(cinfo.source, shaderc_glsl_infer_from_source, cinfo.filename.c_str(), options);

			if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
				printf("%s", module.GetErrorMessage().c_str());
				//Platform::log->error("%s", module.GetErrorMessage().c_str());
				return {};
			} else {
				std::vector<uint32_t> spirv(module.cbegin(), module.cend());

				spirv_cross::Compiler refl(spirv.data(), spirv.size());
				vuk::Program p;
				auto stage = p.introspect(refl);

				vk::ShaderModuleCreateInfo moduleCreateInfo;
				moduleCreateInfo.codeSize = spirv.size() * sizeof(uint32_t);
				moduleCreateInfo.pCode = (uint32_t*)spirv.data();
				auto module = ctx.device.createShaderModule(moduleCreateInfo);
				return { module, p, stage };
			}
		} else if constexpr (std::is_same_v<T, vk::RenderPass>) {
			return ctx.device.createRenderPass(cinfo);
		} else if constexpr (std::is_same_v<T, vuk::RGImage>) {
			RGImage res;
			res.image = ctx.allocator.create_image_for_rendertarget(cinfo.ici);
			auto ivci = cinfo.ivci;
			ivci.image = res.image;
			res.image_view = ctx.wrap(ctx.device.createImageView(ivci));
			return res;
		} else if constexpr (std::is_same_v<T, Allocator::Pool>) {
			return ctx.allocator.allocate_pool(cinfo.mem_usage, cinfo.buffer_usage);
		} else if constexpr (std::is_same_v<T, vuk::DescriptorPool>) {
			return vuk::DescriptorPool{};
		} else if constexpr (std::is_same_v<T, vuk::DescriptorSet>) {
			auto& pool = ptc.pool_cache.acquire(cinfo.layout_info);
			auto ds = pool.acquire(ptc, cinfo.layout_info);
			unsigned long leading_zero = 0;
			auto mask = cinfo.used.to_ulong();
			auto is_null = _BitScanReverse(&leading_zero, mask);
			leading_zero++;
			std::array<vk::WriteDescriptorSet, VUK_MAX_BINDINGS> writes;
			std::array<vk::DescriptorImageInfo, VUK_MAX_BINDINGS> diis;
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
					diis[i] = binding.image;
					write.pImageInfo = &diis[i];
					break;
				default:
					assert(0);
				}
			}
			ctx.device.updateDescriptorSets(leading_zero, writes.data(), 0, nullptr);
			return { ds, cinfo.layout_info };
		} else if constexpr (std::is_same_v<T, vk::Framebuffer>) {
			return ptc.ctx.device.createFramebuffer(cinfo);
		} else if constexpr (std::is_same_v<T, vk::Sampler>) {
			return ptc.ctx.device.createSampler(cinfo);
		} else if constexpr (std::is_same_v<T, vuk::DescriptorSetLayoutAllocInfo>) {
			printf("Creating dslayout\n");
			vuk::DescriptorSetLayoutAllocInfo ret;
			ret.layout = ptc.ctx.device.createDescriptorSetLayout(cinfo.dslci);
			for (auto& b : cinfo.bindings) {
				ret.descriptor_counts[to_integral(b.descriptorType)] += b.descriptorCount;
			}
			return ret;
		} else if constexpr (std::is_same_v<T, vk::PipelineLayout>) {
			printf("Creating playout\n");
			return ptc.ctx.device.createPipelineLayout(cinfo.plci);
		}
	}
}

namespace vuk {
	struct RenderGraph;
	void execute_submit_and_present_to_one(PerThreadContext& ptc, RenderGraph& rg, SwapchainRef swapchain);
}
