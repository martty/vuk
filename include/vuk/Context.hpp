#pragma once

#include <atomic>
#include <span>
#include <string_view>

#include "vuk_fwd.hpp"
#include "vuk/Pipeline.hpp"
#include "vuk/SampledImage.hpp"
#include "vuk/Image.hpp"
#include "vuk/Buffer.hpp"
#include "vuk/Swapchain.hpp"
#include "vuk/Query.hpp"

#include <vuk/Allocator.hpp>

namespace vuk {
	struct TransferStub {
		size_t id;
	};

	enum class Domain {
		eNone = 0,
		eHost = 1 << 0,
		eGraphicsQueue = 1 << 1,
		eGraphicsOperation = 1 << 2,
		eComputeQueue = 1 << 3,
		eComputeOperation = 1 << 4,
		eTransferQueue = 1 << 5,
		eTransferOperation = 1 << 6,
		eGraphicsOnGraphics = eGraphicsQueue | eGraphicsOperation,
		eComputeOnGraphics = eGraphicsQueue | eComputeOperation,
		eTransferOnGraphics = eGraphicsQueue | eTransferOperation,
		eComputeOnCompute = eComputeQueue | eComputeOperation,
		eTransferOnCompute = eComputeQueue | eComputeOperation,
		eTransferOnTransfer = eTransferQueue | eTransferOperation,
		eDevice = eGraphicsQueue | eComputeQueue | eTransferQueue,
		eAny = eDevice | eHost
	};

	struct ContextCreateParameters {
		VkInstance instance;
		VkDevice device;
		VkPhysicalDevice physical_device;
		VkQueue graphics_queue;
		uint32_t graphics_queue_family_index;
		/// @brief Optional transfer queue
		VkQueue transfer_queue = VK_NULL_HANDLE;
		/// @brief Optional transfer queue index
		uint32_t transfer_queue_family_index = VK_QUEUE_FAMILY_IGNORED;
	};

	class Context {
	public:
		VkInstance instance;
		VkDevice device;
		VkPhysicalDevice physical_device;
		VkQueue graphics_queue;
		uint32_t graphics_queue_family_index;
		VkQueue transfer_queue;
		uint32_t transfer_queue_family_index;

		PFN_vkQueueSubmit2KHR queueSubmit2KHR;

		std::atomic<size_t> frame_counter = 0;

		/// @brief Create a new Context
		/// @param params Vulkan parameters initialized beforehand
		Context(ContextCreateParameters params);
		~Context();

		struct DebugUtils {
			Context& ctx;
			PFN_vkSetDebugUtilsObjectNameEXT setDebugUtilsObjectNameEXT;
			PFN_vkCmdBeginDebugUtilsLabelEXT cmdBeginDebugUtilsLabelEXT;
			PFN_vkCmdEndDebugUtilsLabelEXT cmdEndDebugUtilsLabelEXT;

			bool enabled();

			DebugUtils(Context& ctx);
			void set_name(const Texture& iv, Name name);
			template<class T>
			void set_name(const T& t, Name name);

			void begin_region(const VkCommandBuffer&, Name name, std::array<float, 4> color = { 1,1,1,1 });
			void end_region(const VkCommandBuffer&);
		} debug;

		void create_named_pipeline(Name name, PipelineBaseCreateInfo pbci);
		void create_named_pipeline(Name name, ComputePipelineBaseCreateInfo pbci);

		PipelineBaseInfo* get_named_pipeline(Name name);
		ComputePipelineBaseInfo* get_named_compute_pipeline(Name name);

		PipelineBaseInfo* get_pipeline(const PipelineBaseCreateInfo& pbci);
		ComputePipelineBaseInfo* get_pipeline(const ComputePipelineBaseCreateInfo& pbci);
		Program get_pipeline_reflection_info(const PipelineBaseCreateInfo& pbci);
		Program get_pipeline_reflection_info(const ComputePipelineBaseCreateInfo& pbci);
		ShaderModule compile_shader(ShaderSource source, std::string path);

		bool load_pipeline_cache(std::span<std::byte> data);
		std::vector<std::byte> save_pipeline_cache();

		Query create_timestamp_query();

		// Allocator support

		/// @brief Return an allocator over the direct resource - resources will be allocated from the Vulkan runtime
		/// @return The resource
		DeviceVkResource& get_vk_resource();

		uint32_t(*get_thread_index)() = nullptr;

		struct UploadItem {
			/// @brief Describes a single upload to a Buffer
			struct BufferUpload {
				/// @brief Buffer to upload to
				Buffer dst;
				/// @brief Data to upload
				std::span<unsigned char> data;
			};

			/// @brief Describes a single upload to an Image
			struct ImageUpload {
				/// @brief Image to upload to
				Image dst;
				/// @brief Format of the image data
				Format format;
				/// @brief Extent of the image data
				Extent3D extent;
				/// @brief Mip level
				uint32_t mip_level;
				/// @brief Base array layer
				uint32_t base_array_layer;
				/// @brief Should mips be automatically generated for levels higher than mip_level
				bool generate_mips;
				/// @brief Image data
				std::span<unsigned char> data;
			};

			UploadItem(BufferUpload bu) : buffer(std::move(bu)), is_buffer(true) {}
			UploadItem(ImageUpload bu) : image(std::move(bu)), is_buffer(false) {}

			union {
				BufferUpload buffer = {};
				ImageUpload image;
			};
			bool is_buffer;
		};

		using TransientSubmitStub = struct TransientSubmitBundle*;

		/// @brief Enqueue buffer or image data for upload
		/// @param uploads UploadItem structures describing the upload parameters
		/// @param dst_queue_family The queue family where the uploads will be used (ignored for buffers)
		TransientSubmitStub fenced_upload(std::span<UploadItem> uploads, uint32_t dst_queue_family);

		/// @brief Check if the upload has finished. If the upload has finished, resources will be reclaimed automatically. If this function returns true you must not poll again.
		/// @param pending TransientSubmitStub object to check. 
		bool poll_upload(TransientSubmitStub pending);

		Texture allocate_texture(Allocator& allocator, ImageCreateInfo ici);
		std::pair<Texture, TransferStub> create_texture(Allocator& allocator, Format format, Extent3D extents, void* data, bool generate_mips = false);

		size_t get_allocation_size(Buffer);

		/// @brief Allocates & fills a buffer with explicitly managed lifetime
		/// @param mem_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
		/// @param buffer_usage How this buffer will be used (since data is provided, TransferDst is added to the flags)
		/// @return The allocated Buffer
		template<class T>
		std::pair<Unique<BufferCrossDevice>, TransferStub> create_buffer_cross_device(Allocator& allocator, MemoryUsage mem_usage, std::span<T> data) {
			Unique<BufferCrossDevice> buf(allocator);
			BufferCreateInfo bci{ mem_usage, sizeof(T) * data.size(), 1 };
			auto ret = allocator.allocate_buffers(std::span{ &*buf, 1 }, std::span{ &bci, 1 }); // TODO: dropping error
			memcpy(buf->mapped_ptr, data.data(), data.size_bytes());
			return { std::move(buf), TransferStub{0} };
		}

		template<class T>
		std::pair<Unique<BufferGPU>, TransferStub> create_buffer_gpu(Allocator& allocator, std::span<T> data) {
			Unique<BufferGPU> buf(allocator);
			BufferCreateInfo bci{ MemoryUsage::eGPUonly, sizeof(T) * data.size(), 1 };
			auto ret = allocator.allocate_buffers(std::span{ &*buf, 1 }, std::span{ &bci, 1 }); // TODO: dropping error
			auto stub = upload(allocator, *buf, data);
			return { std::move(buf), stub };
		}

		/*
		*  These uploads should also put the fence/sync onto the allocator, otherwise what happens here is not safe
		*  For now we just immediately wait, which is also safe but bad
		*/

		template<class T>
		TransferStub upload(Allocator& allocator, Buffer dst, std::span<T> data) {
			if (data.empty()) return { 0 };
			Unique<BufferCrossDevice> staging(allocator);
			BufferCreateInfo bci{ MemoryUsage::eCPUonly, sizeof(T) * data.size(), 1 };
			auto ret = allocator.allocate_buffers(std::span{ &*staging, 1 }, std::span{ &bci, 1 }); // TODO: dropping error
			::memcpy(staging->mapped_ptr, data.data(), sizeof(T) * data.size());

			auto stub = enqueue_transfer(*staging, dst);
			return stub;
		}

		template<class T>
		TransferStub upload(Allocator& allocator, Image dst, Format format, Extent3D extent, uint32_t base_layer, std::span<T> data, bool generate_mips) {
			assert(!data.empty());
			// compute staging buffer alignment as texel block size
			size_t alignment = format_to_texel_block_size(format);
			Unique<BufferCrossDevice> staging(allocator);
			BufferCreateInfo bci{ MemoryUsage::eCPUonly, sizeof(T) * data.size(), alignment };
			auto ret = allocator.allocate_buffers(std::span{ &*staging, 1 }, std::span{ &bci, 1 }); // TODO: dropping error
			::memcpy(staging->mapped_ptr, data.data(), sizeof(T) * data.size());

			auto stub = enqueue_transfer(*staging, dst, extent, base_layer, generate_mips);
			return stub;
		}

		// TODO: temporary stuff
		std::atomic<size_t> transfer_id = 1;
		std::atomic<size_t> last_transfer_complete = 0;

		TransferStub enqueue_transfer(Buffer src, Buffer dst);
		TransferStub enqueue_transfer(Buffer src, Image dst, Extent3D extent, uint32_t base_layer, bool generate_mips);
		void dma_task(Allocator& allocator);
		void wait_all_transfers(Allocator& allocator);

		/// @brief Add a swapchain to be managed by the Context
		/// @return Reference to the new swapchain that can be used during presentation
		SwapchainRef add_swapchain(Swapchain);

		/// @brief Remove a swapchain that is managed by the Context
		/// the swapchain is not destroyed
		void remove_swapchain(SwapchainRef);

		/// @brief Advance internal counter used for caching and garbage collect caches
		void next_frame();

		/// @brief Wait for the device to become idle. Useful for only a few synchronisation events, like resizing or shutting down.
		void wait_idle();

		/// @brief Create a wrapped handle type (eg. a ImageView) from an externally sourced Vulkan handle
		/// @tparam T Vulkan handle type to wrap
		/// @param payload Vulkan handle to wrap
		/// @return The wrapped handle.
		template<class T>
		Handle<T> wrap(T payload);
		ImageView wrap(VkImageView payload, ImageViewCreateInfo);


		Result<void> submit_graphics(std::span<VkSubmitInfo>, VkFence);
		Result<void> submit_transfer(std::span<VkSubmitInfo>, VkFence);
		Result<void> submit_graphics(std::span<VkSubmitInfo2KHR>, VkFence);
		Result<void> submit_transfer(std::span<VkSubmitInfo2KHR>, VkFence);

		LegacyGPUAllocator& get_legacy_gpu_allocator();

		// Query functionality
		bool is_timestamp_available(Query q);

		std::optional<uint64_t> retrieve_timestamp(Query q);

		std::optional<double> retrieve_duration(Query q1, Query q2);

		Result<void> make_timestamp_results_available(std::span<const TimestampQueryPool> pool);

		RGImage acquire_rendertarget(const struct RGCI&, uint64_t absolute_frame);
		Sampler acquire_sampler(const SamplerCreateInfo&, uint64_t absolute_frame);
		VkRenderPass acquire_renderpass(const struct RenderPassCreateInfo&, uint64_t absolute_frame);
		struct PipelineInfo acquire_pipeline(const struct PipelineInstanceCreateInfo&, uint64_t absolute_frame);
		struct ComputePipelineInfo acquire_pipeline(const struct ComputePipelineInstanceCreateInfo&, uint64_t absolute_frame);
		struct DescriptorPool& acquire_descriptor_pool(const DescriptorSetLayoutAllocInfo& dslai, uint64_t absolute_frame);

		Unique<PersistentDescriptorSet> create_persistent_descriptorset(Allocator& allocator, DescriptorSetLayoutCreateInfo dslci, unsigned num_descriptors);
		Unique<PersistentDescriptorSet> create_persistent_descriptorset(Allocator& allocator, const PipelineBaseInfo& base, unsigned set, unsigned num_descriptors);
		Unique<PersistentDescriptorSet> create_persistent_descriptorset(Allocator& allocator, const ComputePipelineBaseInfo& base, unsigned set, unsigned num_descriptors);
		Unique<PersistentDescriptorSet> create_persistent_descriptorset(Allocator& allocator, const PersistentDescriptorSetCreateInfo&);
		void commit_persistent_descriptorset(PersistentDescriptorSet& array);

		void collect(uint64_t frame);

	private:
		struct ContextImpl* impl;
		std::atomic<size_t> unique_handle_id_counter = 0;

		void destroy(const struct RGImage& image);
		void destroy(const struct LegacyPoolAllocator& v);
		void destroy(const struct LegacyLinearAllocator& v);
		void destroy(const DescriptorPool& dp);
		void destroy(const struct PipelineInfo& pi);
		void destroy(const struct ComputePipelineInfo& pi);
		void destroy(const ShaderModule& sm);
		void destroy(const DescriptorSetLayoutAllocInfo& ds);
		void destroy(const VkPipelineLayout& pl);
		void destroy(const VkRenderPass& rp);
		void destroy(const DescriptorSet&);
		void destroy(const VkFramebuffer& fb);
		void destroy(const Sampler& sa);
		void destroy(const PipelineBaseInfo& pbi);
		void destroy(const ComputePipelineBaseInfo& pbi);

		ShaderModule create(const create_info_t<ShaderModule>& cinfo);
		PipelineBaseInfo create(const create_info_t<PipelineBaseInfo>& cinfo);
		ComputePipelineBaseInfo create(const create_info_t<ComputePipelineBaseInfo>& cinfo);
		VkPipelineLayout create(const create_info_t<VkPipelineLayout>& cinfo);
		DescriptorSetLayoutAllocInfo create(const create_info_t<DescriptorSetLayoutAllocInfo>& cinfo);
		DescriptorPool create(const struct DescriptorSetLayoutAllocInfo& cinfo);
		PipelineInfo create(const struct PipelineInstanceCreateInfo& cinfo);
		ComputePipelineInfo create(const struct ComputePipelineInstanceCreateInfo& cinfo);
		VkRenderPass create(const struct RenderPassCreateInfo& cinfo);
		RGImage create(const struct RGCI& cinfo);
		Sampler create(const struct SamplerCreateInfo& cinfo);

		template<class T> friend class Cache; // caches can directly destroy
	};

	template<class T>
	Handle<T> Context::wrap(T payload) {
		return { { unique_handle_id_counter++ }, payload };
	}

	template<class T>
	void Context::DebugUtils::set_name(const T& t, Name name) {
		if (!enabled()) return;
		VkDebugUtilsObjectNameInfoEXT info = { .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
		info.pObjectName = name.c_str();
		if constexpr (std::is_same_v<T, VkImage>) {
			info.objectType = VK_OBJECT_TYPE_IMAGE;
		} else if constexpr (std::is_same_v<T, VkImageView>) {
			info.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
		} else if constexpr (std::is_same_v<T, VkShaderModule>) {
			info.objectType = VK_OBJECT_TYPE_SHADER_MODULE;
		} else if constexpr (std::is_same_v<T, VkPipeline>) {
			info.objectType = VK_OBJECT_TYPE_PIPELINE;
		}
		info.objectHandle = reinterpret_cast<uint64_t>(t);
		setDebugUtilsObjectNameEXT(ctx.device, &info);
	}
}

// utility functions
namespace vuk {
	struct ExecutableRenderGraph;
	Result<void> execute_submit_and_present_to_one(Allocator& nalloc, ExecutableRenderGraph&& rg, SwapchainRef swapchain);
	Result<void> execute_submit_and_wait(Allocator& nalloc, ExecutableRenderGraph&& rg);

	SampledImage make_sampled_image(ImageView iv, SamplerCreateInfo sci);

	SampledImage make_sampled_image(Name n, SamplerCreateInfo sci);

	SampledImage make_sampled_image(Name n, ImageViewCreateInfo ivci, SamplerCreateInfo sci);
}

// futures

namespace vuk {
	template<class T>
	struct Future {
		Future() = default;
		Future(Allocator& alloc, struct RenderGraph& rg, Name output_binding);
		Future(Allocator& alloc, T&& value);

		Allocator* alloc;
		T result;
		Name output_binding;

		RenderGraph* rg;

		enum class Status { initial, value_bound, rg_bound, compiled, submitted, done } status = Status::initial;
		Domain available = Domain::eNone;
		
		Allocator& get_allocator();

		Result<void> compile(); // turn RG into ERG
		Result<void> execute(); // turn ERG into cmdbufs
		Result<void> submit(); // turn cmdbufs into possibly a TS
		Result<T> get(); // wait on host for T to be produced by the computation
	};
}
