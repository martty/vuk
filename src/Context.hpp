#pragma once

#include <atomic>
#include <gsl/span>

#include "Pool.hpp"
#include "Cache.hpp"
#include "Allocator.hpp"
#include "Program.hpp"
#include "Pipeline.hpp"
#include <queue>
#include <string_view>
#include "SampledImage.hpp"
#include "RenderPass.hpp"
#include "vuk_fwd.hpp"
#include <exception>

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

	struct ShaderCompilationException {
		std::string error_message;

		const char* what() const {
			return error_message.c_str();
		}
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

	class Context {
	public:
		constexpr static size_t FC = 3;
		
		vk::Instance instance;
		vk::Device device;
		vk::PhysicalDevice physical_device;
		vk::Queue graphics_queue;
        vk::Queue transfer_queue;
		Allocator allocator;

		std::mutex gfx_queue_lock;
        std::mutex xfer_queue_lock;
	private:
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

		std::mutex begin_frame_lock;

		std::array<std::mutex, FC> recycle_locks;
		std::array<std::vector<vk::Image>, FC> image_recycle;
		std::array<std::vector<vk::ImageView>, FC> image_view_recycle;
		std::array<std::vector<vk::Pipeline>, FC> pipeline_recycle;
		std::array<std::vector<vuk::Buffer>, FC> buffer_recycle;

		std::atomic<size_t> frame_counter = 0;
		std::atomic<size_t> unique_handle_id_counter = 0;

		std::mutex named_pipelines_lock;
		std::unordered_map<std::string_view, vuk::PipelineCreateInfo> named_pipelines;

		std::mutex swapchains_lock;
		plf::colony<Swapchain> swapchains;
	public:
		Context(vk::Instance instance, vk::Device device, vk::PhysicalDevice physical_device, vk::Queue graphics);
		~Context();

		struct DebugUtils {
			Context& ctx;
			PFN_vkSetDebugUtilsObjectNameEXT setDebugUtilsObjectNameEXT;
			PFN_vkCmdBeginDebugUtilsLabelEXT cmdBeginDebugUtilsLabelEXT;
			PFN_vkCmdEndDebugUtilsLabelEXT cmdEndDebugUtilsLabelEXT;

			bool enabled();

			DebugUtils(Context& ctx);
			void set_name(const vuk::Texture& iv, /*zstring_view*/Name name);
			template<class T>
			void set_name(const T& t, /*zstring_view*/Name name);

			void begin_region(const vk::CommandBuffer&, Name name, std::array<float, 4> color = { 1,1,1,1 });
			void end_region(const vk::CommandBuffer&);
		} debug;

		void create_named_pipeline(const char* name, vuk::PipelineCreateInfo ci);
		vuk::PipelineCreateInfo get_named_pipeline(const char* name);
		void invalidate_shadermodule_and_pipelines(Name);
        void compile_shader(Name path);
		vuk::ShaderModule create(const create_info_t<vuk::ShaderModule>& cinfo);

		// one pool per thread
        std::mutex one_time_pool_lock;
		std::vector<vk::CommandPool> one_time_pools;
        uint32_t (*get_thread_index)() = nullptr;

		struct Upload {
            vuk::Buffer dst;
            gsl::span<unsigned char> data;
		};
		vk::Fence fenced_upload(gsl::span<Upload>);

		Buffer allocate_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, size_t size);

		void enqueue_destroy(vk::Image);
		void enqueue_destroy(vuk::ImageView);
		void enqueue_destroy(vk::Pipeline);
		void enqueue_destroy(vuk::Buffer);

		template<class T>
		Handle<T> wrap(T payload);
		
		SwapchainRef add_swapchain(Swapchain sw);

		InflightContext begin();

		void wait_idle();
	private:
		void destroy(const RGImage& image);
		void destroy(const Allocator::Pool& v);
		void destroy(const vuk::DescriptorPool& dp);
		void destroy(vuk::PipelineInfo pi);
		void destroy(vuk::ShaderModule sm);
		void destroy(vuk::DescriptorSetLayoutAllocInfo ds);
		void destroy(vk::PipelineLayout pl);
		void destroy(vk::RenderPass rp);
		void destroy(vuk::DescriptorSet);
		void destroy(vk::Framebuffer fb);
		void destroy(vk::Sampler sa);

		friend class InflightContext;
		friend class PerThreadContext;
		template<class T> friend class Cache; // caches can directly destroy
		template<class T, size_t FC> friend class PerFrameCache;
	};

	class InflightContext {
	public:
		Context& ctx;
		const size_t absolute_frame;
		const unsigned frame;
	private:
		Pool<vk::Fence, Context::FC>::PFView fence_pools; // must be first, so we wait for the fences
		Pool<vk::CommandBuffer, Context::FC>::PFView commandbuffer_pools;
		Pool<vk::Semaphore, Context::FC>::PFView semaphore_pools;
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
	public:
		InflightContext(Context& ctx, size_t absolute_frame, std::lock_guard<std::mutex>&& recycle_guard);

		void wait_all_transfers();
		PerThreadContext begin();

	private:
		friend class PerThreadContext;

		struct BufferCopyCommand {
			Buffer src;
			Buffer dst;
			TransferStub stub;
		};

		struct BufferImageCopyCommand {
			Buffer src;
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

		TransferStub enqueue_transfer(Buffer src, Buffer dst);
		TransferStub enqueue_transfer(Buffer src, vk::Image dst, vk::Extent3D extent);

		// recycle
		std::mutex recycle_lock;
		void destroy(std::vector<vk::Image>&& images);
		void destroy(std::vector<vk::ImageView>&& images);
	};

	class PerThreadContext {
	public:
		Context& ctx;
		InflightContext& ifc;
		const unsigned tid = 0; // not yet implemented
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
	private:
		// recycling global objects
		std::vector<Buffer> buffer_recycle;
		std::vector<vk::Image> image_recycle;
		std::vector<vk::ImageView> image_view_recycle;
	public:
		PerThreadContext(InflightContext& ifc, unsigned tid);
		~PerThreadContext();

		PerThreadContext(const PerThreadContext& o) = delete;

		bool is_ready(const TransferStub& stub);
		void wait_all_transfers();

		Buffer _allocate_scratch_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, size_t size, bool create_mapped);
		Unique<Buffer> _allocate_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, size_t size, bool create_mapped);

		// since data is provided, we will add TransferDst to the flags automatically
		template<class T>
		std::pair<Buffer, TransferStub> create_scratch_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, gsl::span<T> data) {
			auto dst = _allocate_scratch_buffer(mem_usage, vk::BufferUsageFlagBits::eTransferDst | buffer_usage, sizeof(T) * data.size(), false);
			auto stub = upload(dst, data);
			return { dst, stub };
		}

		template<class T>
		std::pair<Unique<Buffer>, TransferStub> create_buffer(MemoryUsage mem_usage, vk::BufferUsageFlags buffer_usage, gsl::span<T> data) {
			auto dst = _allocate_buffer(mem_usage, vk::BufferUsageFlagBits::eTransferDst | buffer_usage, sizeof(T) * data.size(), false);
			auto stub = upload(*dst, data);
			return { std::move(dst), stub };
		}

		std::pair<vuk::Texture, TransferStub> create_texture(vk::Format format, vk::Extent3D extents, void* data);

		template<class T>
		TransferStub upload(Buffer dst, gsl::span<T> data) {
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

		void dma_task();

        vuk::SampledImage& make_sampled_image(vuk::ImageView iv, vk::SamplerCreateInfo sci);
        vuk::SampledImage& make_sampled_image(Name n, vk::SamplerCreateInfo sci);
        vuk::SampledImage& make_sampled_image(Name n, vk::ImageViewCreateInfo ivci, vk::SamplerCreateInfo sci);

        vuk::Program get_pipeline_reflection_info(vuk::PipelineCreateInfo pci);

        template<class T>
        void destroy(T t) {
            ctx.destroy(t);
        }

        void destroy(vk::Image image);
        void destroy(vuk::ImageView image);
        void destroy(vuk::DescriptorSet ds);

        PipelineInfo create(const create_info_t<PipelineInfo>& cinfo);
		vuk::ShaderModule create(const create_info_t<vuk::ShaderModule>& cinfo);
		vk::RenderPass create(const create_info_t<vk::RenderPass>& cinfo);
		vuk::RGImage create(const create_info_t<vuk::RGImage>& cinfo);
		vuk::Allocator::Pool create(const create_info_t<vuk::Allocator::Pool>& cinfo);
		vuk::DescriptorPool create(const create_info_t<vuk::DescriptorPool>& cinfo);
		vuk::DescriptorSet create(const create_info_t<vuk::DescriptorSet>& cinfo);
		vk::Framebuffer create(const create_info_t<vk::Framebuffer>& cinfo);
		vk::Sampler create(const create_info_t<vk::Sampler>& cinfo);
		vuk::DescriptorSetLayoutAllocInfo create(const create_info_t<vuk::DescriptorSetLayoutAllocInfo>& cinfo);
		vk::PipelineLayout create(const create_info_t<vk::PipelineLayout>& cinfo);
	};
	
	template<class T>
	void Context::DebugUtils::set_name(const T& t, Name name) {
		if (!enabled()) return;
		VkDebugUtilsObjectNameInfoEXT info;
		info.pNext = nullptr;
		info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
		info.pObjectName = name.data();
		info.objectType = (VkObjectType)t.objectType;
		info.objectHandle = reinterpret_cast<uint64_t>((typename T::CType)t);
		setDebugUtilsObjectNameEXT(ctx.device, &info);
	}
	
	template<class T>
	Handle<T> Context::wrap(T payload) {
		return { { unique_handle_id_counter++ }, payload };
	}
}

namespace vuk {
	template<class T, size_t FC>
	typename Pool<T, FC>::PFView Pool<T, FC>::get_view(InflightContext& ctx) {
		return { ctx, *this, per_frame_storage[ctx.frame] };
	}

	template<class T, size_t FC>
	Pool<T, FC>::PFView::PFView(InflightContext& ifc, Pool<T, FC>& storage, plf::colony<PooledType<T>>& fv) : storage(storage), ifc(ifc), frame_values(fv) {
		storage.reset(ifc.frame);
	}	

    template<typename Type>
    inline Unique<Type>::~Unique() noexcept {
        if (context) context->enqueue_destroy(payload);
    }
    template<typename Type>
    inline void Unique<Type>::reset(Type const& value) noexcept {
        if (payload != value) {
            if (context) context->enqueue_destroy(payload);
            payload = value;
        }
    }

	struct RenderGraph;
	void execute_submit_and_present_to_one(PerThreadContext& ptc, RenderGraph& rg, SwapchainRef swapchain);
}
