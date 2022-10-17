#include "Cache.hpp"
#include "LegacyGPUAllocator.hpp"
#include "RGImage.hpp"
#include "RenderPass.hpp"
#include "vuk/Allocator.hpp"
#include "vuk/Context.hpp"
#include "vuk/PipelineInstance.hpp"
#include "vuk/Query.hpp"
#include "vuk/resources/DeviceVkResource.hpp"

#include <atomic>
#include <math.h>
#include <mutex>
#include <plf_colony.h>
#include <queue>
#include <robin_hood.h>
#include <string_view>

namespace vuk {
	template<class T>
	struct FN {
		static T create_fn(void* ctx, const create_info_t<T>& ci) {
			return reinterpret_cast<Context*>(ctx)->create(ci);
		}

		static void destroy_fn(void* ctx, const T& v) {
			return reinterpret_cast<Context*>(ctx)->destroy(v);
		}
	};

	struct ContextImpl {
		LegacyGPUAllocator legacy_gpu_allocator;
		VkDevice device;

		VkPipelineCache vk_pipeline_cache = VK_NULL_HANDLE;
		Cache<PipelineBaseInfo> pipelinebase_cache;
		Cache<PipelineInfo> pipeline_cache;
		Cache<ComputePipelineInfo> compute_pipeline_cache;
		Cache<RayTracingPipelineInfo> ray_tracing_pipeline_cache;
		Cache<VkRenderPass> renderpass_cache;
		Cache<RGImage> transient_images;
		Cache<DescriptorPool> pool_cache;
		Cache<Sampler> sampler_cache;
		Cache<ShaderModule> shader_modules;
		Cache<DescriptorSetLayoutAllocInfo> descriptor_set_layouts;
		Cache<VkPipelineLayout> pipeline_layouts;

		std::mutex begin_frame_lock;

		std::atomic<size_t> frame_counter = 0;
		std::atomic<size_t> unique_handle_id_counter = 0;

		std::mutex named_pipelines_lock;
		std::unordered_map<Name, PipelineBaseInfo*> named_pipelines;

		std::atomic<uint64_t> query_id_counter = 0;
		VkPhysicalDeviceProperties physical_device_properties;

		std::mutex swapchains_lock;
		plf::colony<Swapchain> swapchains;

		DeviceVkResource device_vk_resource;

		std::mutex query_lock;
		robin_hood::unordered_map<Query, uint64_t> timestamp_result_map;

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
			/* case 5:
				pipelinebase_cache.collect(absolute_frame, cache_collection_frequency);
				break;*/ // can't be collected since we keep the pointer around in PipelineInfos
			case 6:
				pool_cache.collect(absolute_frame, cache_collection_frequency);
				break;
			}
		}

		ContextImpl(Context& ctx) :
		    legacy_gpu_allocator(ctx.instance,
		                         ctx.device,
		                         ctx.physical_device,
		                         ctx.graphics_queue_family_index,
		                         ctx.compute_queue_family_index,
		                         ctx.transfer_queue_family_index,
		                         ctx.debug.setDebugUtilsObjectNameEXT),
		    device(ctx.device),
		    pipelinebase_cache(&ctx, &FN<struct PipelineBaseInfo>::create_fn, &FN<struct PipelineBaseInfo>::destroy_fn),
		    pipeline_cache(&ctx, &FN<struct PipelineInfo>::create_fn, &FN<struct PipelineInfo>::destroy_fn),
		    compute_pipeline_cache(&ctx, &FN<struct ComputePipelineInfo>::create_fn, &FN<struct ComputePipelineInfo>::destroy_fn),
		    ray_tracing_pipeline_cache(&ctx, &FN<struct RayTracingPipelineInfo>::create_fn, &FN<struct RayTracingPipelineInfo>::destroy_fn),
		    renderpass_cache(&ctx, &FN<VkRenderPass>::create_fn, &FN<VkRenderPass>::destroy_fn),
		    transient_images(&ctx, &FN<struct RGImage>::create_fn, &FN<struct RGImage>::destroy_fn),
		    pool_cache(&ctx, &FN<struct DescriptorPool>::create_fn, &FN<struct DescriptorPool>::destroy_fn),
		    sampler_cache(&ctx, &FN<Sampler>::create_fn, &FN<Sampler>::destroy_fn),
		    shader_modules(&ctx, &FN<struct ShaderModule>::create_fn, &FN<struct ShaderModule>::destroy_fn),
		    descriptor_set_layouts(&ctx, &FN<struct DescriptorSetLayoutAllocInfo>::create_fn, &FN<struct DescriptorSetLayoutAllocInfo>::destroy_fn),
		    pipeline_layouts(&ctx, &FN<VkPipelineLayout>::create_fn, &FN<VkPipelineLayout>::destroy_fn),
		    device_vk_resource(ctx, legacy_gpu_allocator) {
			vkGetPhysicalDeviceProperties(ctx.physical_device, &physical_device_properties);
		}
	};
} // namespace vuk
