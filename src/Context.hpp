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
	class Context {
	public:
		constexpr static size_t FC = 3;

		vk::Device device;
		vk::PhysicalDevice physical_device;
		Allocator allocator;
		Pool<vk::CommandBuffer, FC> cbuf_pools;
		Pool<vk::Semaphore, FC> semaphore_pools;
		vk::UniquePipelineCache vk_pipeline_cache;
		Cache<vk::Pipeline> pipeline_cache;
		Cache<vk::RenderPass> renderpass_cache;
		PerFrameCache<RGImage, FC> transient_images;

		std::unordered_map<std::string_view, create_info_t<vk::Pipeline>> named_pipelines;

		Context(vk::Device device, vk::PhysicalDevice physical_device) : device(device), physical_device(physical_device),
			allocator(device, physical_device),
			cbuf_pools(*this),
			semaphore_pools(*this),
			pipeline_cache(*this),
			renderpass_cache(*this),
			transient_images(*this)
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
		Cache<vk::Pipeline>::View pipeline_cache;
		Cache<vk::RenderPass>::View renderpass_cache;
		PerFrameCache<vuk::RGImage, Context::FC>::View transient_images;

		InflightContext(Context& ctx, unsigned frame) : ctx(ctx), frame(frame),
			commandbuffer_pools(ctx.cbuf_pools.get_view(*this)),
			semaphore_pools(ctx.semaphore_pools.get_view(*this)),
			pipeline_cache(*this, ctx.pipeline_cache),
			renderpass_cache(*this, ctx.renderpass_cache),
			transient_images(*this, ctx.transient_images)
		{
			auto prev_frame = prev_(frame, 1, Context::FC);
			ctx.cbuf_pools.reset(prev_frame);
			ctx.semaphore_pools.reset(prev_frame);
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

		PerThreadContext(InflightContext& ifc, unsigned tid) : ctx(ifc.ctx), ifc(ifc), tid(tid),
			commandbuffer_pool(ifc.commandbuffer_pools.get_view(*this)),
			semaphore_pool(ifc.semaphore_pools.get_view(*this))
		{}

	};

	inline PerThreadContext InflightContext::begin() {
		return PerThreadContext{ *this, 0 };
	}

	template<class T, size_t FC>
	typename Pool<T, FC>::PFView Pool<T, FC>::get_view(InflightContext& ctx) {
		return { ctx, *this, per_frame_storage[ctx.frame] };
	}

	template<class T>
	T create(Context& ctx, create_info_t<T> cinfo) {
		if constexpr (std::is_same_v<T, vk::Pipeline>) {
			return ctx.device.createGraphicsPipeline(*ctx.vk_pipeline_cache, cinfo);
		} else if constexpr (std::is_same_v<T, vk::RenderPass>) {
			return ctx.device.createRenderPass(cinfo);
		} else if constexpr (std::is_same_v<T, vuk::RGImage>) {
			RGImage res;
			res.image = ctx.allocator.create_image_for_rendertarget(cinfo.ici);
			cinfo.ivci.image = res.image;
			res.image_view = ctx.device.createImageView(cinfo.ivci);
			return res;
		}
	}

}
