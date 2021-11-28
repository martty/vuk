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

	struct TimestampQuery {
		VkQueryPool pool;
		uint32_t id;
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
		constexpr static size_t FC = 3;
		VkInstance instance;
		VkDevice device;
		VkPhysicalDevice physical_device;
		VkQueue graphics_queue;
		uint32_t graphics_queue_family_index;
		VkQueue transfer_queue;
		uint32_t transfer_queue_family_index;

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
			void set_name(const vuk::Texture& iv, Name name);
			template<class T>
			void set_name(const T& t, Name name);

			void begin_region(const VkCommandBuffer&, Name name, std::array<float, 4> color = { 1,1,1,1 });
			void end_region(const VkCommandBuffer&);
		} debug;

		void create_named_pipeline(Name name, vuk::PipelineBaseCreateInfo pbci);
		void create_named_pipeline(Name name, vuk::ComputePipelineBaseCreateInfo pbci);

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
		/// @return 
		NAllocator& get_direct_allocator();

		uint32_t(*get_thread_index)() = nullptr;

		struct UploadItem {
			/// @brief Describes a single upload to a Buffer
			struct BufferUpload {
				/// @brief Buffer to upload to
				vuk::Buffer dst;
				/// @brief Data to upload
				std::span<unsigned char> data;
			};

			/// @brief Describes a single upload to an Image
			struct ImageUpload {
				/// @brief Image to upload to
				vuk::Image dst;
				/// @brief Format of the image data
				vuk::Format format;
				/// @brief Extent of the image data
				vuk::Extent3D extent;
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

		Texture allocate_texture(vuk::ImageCreateInfo ici);
		Unique<ImageView> create_image_view(vuk::ImageViewCreateInfo);
		std::pair<vuk::Texture, TransferStub> create_texture(NAllocator&, vuk::Format format, vuk::Extent3D extents, void* data, bool generate_mips = false);


		size_t get_allocation_size(Buffer);

		/// @brief Allocates a buffer with explicitly managed lifetime
		/// @param mem_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
		/// @param buffer_usage How this buffer will be used
		/// @param size Size of the buffer
		/// @param alignment Alignment of the buffer
		/// @return The allocated Buffer
		NUnique<Buffer> allocate_buffer(NAllocator&, MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment);

		// temporary stuff
		std::atomic<size_t> transfer_id = 1;
		std::atomic<size_t> last_transfer_complete = 0;

		TransferStub enqueue_transfer(Buffer src, Buffer dst);
		TransferStub enqueue_transfer(Buffer src, vuk::Image dst, vuk::Extent3D extent, uint32_t base_layer, bool generate_mips);
		void dma_task();

		/// @brief Allocates & fills a buffer with explicitly managed lifetime
		/// @param mem_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
		/// @param buffer_usage How this buffer will be used (since data is provided, TransferDst is added to the flags)
		/// @return The allocated Buffer
		template<class T>
		std::pair<NUnique<Buffer>, TransferStub> create_buffer(NAllocator& allocator, MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, std::span<T> data) {
			NUnique<Buffer> buf(allocator);
			BufferCreateInfo bci{ mem_usage, vuk::BufferUsageFlagBits::eTransferDst | buffer_usage, sizeof(T) * data.size(), 1 };
			auto ret = allocator.allocate_buffers(std::span{ &*buf, 1 }, std::span{ &bci, 1 }, VUK_HERE_AND_NOW()); // TODO: dropping error
			auto stub = upload(allocator, *buf, data);
			return { std::move(buf), stub };
		}

		/*
		*  These uploads should also put the fence/sync onto the allocator, otherwise what happens here is not safe
		*  For now we just immediately wait, which is also safe but bad
		*/

		template<class T>
		TransferStub upload(NAllocator& allocator, Buffer dst, std::span<T> data) {
			if (data.empty()) return { 0 };
			NUnique<Buffer> staging(allocator);
			BufferCreateInfo bci{ MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, sizeof(T) * data.size(), 1 };
			auto ret = allocator.allocate_buffers(std::span{ &*staging, 1 }, std::span{ &bci, 1 }, VUK_HERE_AND_NOW()); // TODO: dropping error
			::memcpy(staging->mapped_ptr, data.data(), sizeof(T) * data.size());

			auto stub = enqueue_transfer(*staging, dst);
			wait_all_transfers();
			return stub;
		}

		template<class T>
		TransferStub upload(NAllocator& allocator, vuk::Image dst, vuk::Format format, vuk::Extent3D extent, uint32_t base_layer, std::span<T> data, bool generate_mips) {
			assert(!data.empty());
			// compute staging buffer alignment as texel block size
			size_t alignment = format_to_texel_block_size(format);
			NUnique<Buffer> staging(allocator);
			BufferCreateInfo bci{ MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, sizeof(T) * data.size(), alignment };
			auto ret = allocator.allocate_buffers(std::span{ &*staging, 1 }, std::span{ &bci, 1 }, VUK_HERE_AND_NOW()); // TODO: dropping error
			::memcpy(staging->mapped_ptr, data.data(), sizeof(T) * data.size());

			auto stub = enqueue_transfer(*staging, dst, extent, base_layer, generate_mips);
			wait_all_transfers();
			return stub;
		}

		void wait_all_transfers() {
			dma_task();
		}

		/// @brief Manually request destruction of vuk::Image
		void enqueue_destroy(vuk::Image);
		/// @brief Manually request destruction of vuk::ImageView
		void enqueue_destroy(vuk::ImageView);
		/// @brief Manually request destruction of vuk::Buffer
		void enqueue_destroy(vuk::Buffer);
		/// @brief Manually request destruction of vuk::PersistentDescriptorSet
		void enqueue_destroy(vuk::PersistentDescriptorSet);
		/// @brief Manually request destruction of VkFramebuffer
		void enqueue_destroy(VkFramebuffer fb);

		/// @brief Add a swapchain to be managed by the Context
		/// @return Reference to the new swapchain that can be used during presentation
		SwapchainRef add_swapchain(Swapchain);

		/// @brief Remove a swapchain that is managed by the Context
		/// the swapchain is not destroyed
		void remove_swapchain(SwapchainRef);

		/// @brief Begin new frame, with a new InflightContext
		/// @return the new InflightContext
		InflightContext begin();

		/// @brief Wait for the device to become idle. Useful for only a few synchronisation events, like resizing or shutting down.
		void wait_idle();

		/// @brief Create a wrapped handle type (eg. a vuk::ImageView) from an externally sourced Vulkan handle
		/// @tparam T Vulkan handle type to wrap
		/// @param payload Vulkan handle to wrap
		/// @return The wrapped handle.
		template<class T>
		Handle<T> wrap(T payload);
		vuk::ImageView wrap(VkImageView payload, vuk::ImageViewCreateInfo);


		void submit_graphics(VkSubmitInfo, VkFence);
		void submit_transfer(VkSubmitInfo, VkFence);

		Allocator& get_gpumem();

		Unique<VkFramebuffer> create(const struct FramebufferCreateInfo& cinfo);
		struct LinearAllocator create(const struct PoolSelect& cinfo);
		struct DescriptorSet create(const struct SetBinding& cinfo);
		RGImage acquire_rendertarget(const struct RGCI&, uint64_t absolute_frame);
		Sampler acquire_sampler(const SamplerCreateInfo&, uint64_t absolute_frame);
		DescriptorSet acquire_descriptorset(const SetBinding&, uint64_t absolute_frame);
		VkRenderPass acquire_renderpass(const struct RenderPassCreateInfo&, uint64_t absolute_frame);
		struct PipelineInfo acquire_pipeline(const struct PipelineInstanceCreateInfo&, uint64_t absolute_frame);
		struct ComputePipelineInfo acquire_pipeline(const struct ComputePipelineInstanceCreateInfo&, uint64_t absolute_frame);

		Unique<PersistentDescriptorSet> create_persistent_descriptorset(const PipelineBaseInfo& base, unsigned set, unsigned num_descriptors);
		Unique<PersistentDescriptorSet> create_persistent_descriptorset(const ComputePipelineBaseInfo& base, unsigned set, unsigned num_descriptors);
		Unique<PersistentDescriptorSet> create_persistent_descriptorset(const DescriptorSetLayoutAllocInfo& dslai, unsigned num_descriptors);
		Unique<PersistentDescriptorSet> create_persistent_descriptorset(DescriptorSetLayoutCreateInfo dslci, unsigned num_descriptors);
		void commit_persistent_descriptorset(PersistentDescriptorSet& array);

		void collect(uint64_t frame);

	private:
		struct ContextImpl* impl;
		std::atomic<size_t> unique_handle_id_counter = 0;

		void enqueue_destroy(VkPipeline);

		void destroy(const struct RGImage& image);
		void destroy(const struct PoolAllocator& v);
		void destroy(const struct LinearAllocator& v);
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

		friend class InflightContext;
		friend class PerThreadContext;
		friend struct IFCImpl;
		friend struct PTCImpl;
		template<class T> friend class Cache; // caches can directly destroy
		template<class T, size_t FC> friend class PerFrameCache;
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

	template<typename Type>
	inline Unique<Type>::~Unique() noexcept {
		if (context && payload != Type{})
			context->enqueue_destroy(std::move(payload));
	}
	template<typename Type>
	inline void Unique<Type>::reset(Type value) noexcept {
		if (payload != value) {
			if (context && payload != Type{}) {
				context->enqueue_destroy(std::move(payload));
			}
			payload = std::move(value);
		}
	}
}

// utility functions
namespace vuk {
	struct ExecutableRenderGraph;
	Result<void> execute_submit_and_present_to_one(Context& ctx, NAllocator& nalloc, ExecutableRenderGraph&& rg, SwapchainRef swapchain);
	Result<void> execute_submit_and_wait(Context& ctx, NAllocator& nalloc, ExecutableRenderGraph&& rg);
}
