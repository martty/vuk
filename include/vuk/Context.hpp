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

namespace vuk {
	struct TransferStub {
		size_t id;
	};

	struct Token {
		size_t id;
	};

	struct TokenWithContext {
		Context& ctx;
		Token token;

		void operator+=(Token other);

		operator Token() {
			return token;
		}
	};

	struct TimestampQuery {
		VkQueryPool pool;
		uint32_t id;
	};

	enum class Domain {
		eNone = 0,
		eHost = 1 << 0,
		eGeneralQueue = 1 << 1,
		eComputeQueue = 1 << 2,
		eTransferQueue = 1 << 3
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
			void set_name(const vuk::Texture& iv, /*zstring_view*/Name name);
			template<class T>
			void set_name(const T& t, /*zstring_view*/Name name);

			void begin_region(const VkCommandBuffer&, Name name, std::array<float, 4> color = { 1,1,1,1 });
			void end_region(const VkCommandBuffer&);
		} debug;

		void create_named_pipeline(Name name, vuk::PipelineBaseCreateInfo pbci);
		void create_named_pipeline(Name name, vuk::ComputePipelineCreateInfo pbci);

		PipelineBaseInfo* get_named_pipeline(Name name);
		ComputePipelineInfo* get_named_compute_pipeline(Name name);

		PipelineBaseInfo* get_pipeline(const PipelineBaseCreateInfo& pbci);
		ComputePipelineInfo* get_pipeline(const ComputePipelineCreateInfo& pbci);
		Program get_pipeline_reflection_info(PipelineBaseCreateInfo pbci);
		ShaderModule compile_shader(ShaderSource source, std::string path);

		bool load_pipeline_cache(std::span<uint8_t> data);
		std::vector<uint8_t> save_pipeline_cache();

		Query create_timestamp_query();

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

		using TransientSubmitStub = struct LinearResourceAllocator*;

		/// @brief Enqueue buffer or image data for upload
		/// @param uploads UploadItem structures describing the upload parameters
		/// @param dst_queue_family The queue family where the uploads will be used (ignored for buffers)
		TransientSubmitStub fenced_upload(std::span<UploadItem> uploads, uint32_t dst_queue_family);

		/// @brief Check if the upload has finished. If the upload has finished, resources will be reclaimed automatically. If this function returns true you must not poll again.
		/// @param pending TransientSubmitStub object to check. 
		bool poll_upload(TransientSubmitStub pending);

		/// @brief Allocate a Buffer in device-visible memory (GPU or CPU).
		/// @param mem_usage Determines which memory will be used.
		/// @param buffer_usage Set to the usage of the buffer.
		/// @param size Size of the allocation.
		/// @param alignment Minimum alignment of the allocation.
		/// @param create_mapped Should the memory be mapped. Should only be true for CPU-visible memory.
		/// @return The allocated buffer in a RAII handle.
		Unique<Buffer> allocate_buffer(MemoryUsage mem_usage, BufferUsageFlags buffer_usage, size_t size, size_t alignment, bool create_mapped);
		Texture allocate_texture(ImageCreateInfo ici);
		Unique<ImageView> create_image_view(ImageViewCreateInfo);
		VkRenderPass acquire_renderpass(const struct RenderPassCreateInfo&);

		/// @brief Manually request destruction of vuk::Image
		void enqueue_destroy(vuk::Image);
		/// @brief Manually request destruction of vuk::ImageView
		void enqueue_destroy(vuk::ImageView);
		/// @brief Manually request destruction of vuk::Buffer
		void enqueue_destroy(vuk::Buffer);
		/// @brief Manually request destruction of vuk::PersistentDescriptorSet
		void enqueue_destroy(vuk::PersistentDescriptorSet);

		Token create_token();
		TokenWithContext transition_image(vuk::Texture&, vuk::Access src_access, vuk::Access dst_access);
		TokenWithContext copy_to_buffer(vuk::Buffer buffer, void* data, size_t size);

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

		void submit_graphics(VkSubmitInfo, VkFence);
		void submit_transfer(VkSubmitInfo, VkFence);
	private:
		struct ContextImpl* impl;
		std::atomic<size_t> unique_handle_id_counter = 0;

		void enqueue_destroy(VkPipeline);

		void destroy(const struct RGImage& image);
		void destroy(const struct PoolAllocator& v);
		void destroy(const struct LinearAllocator& v);
		void destroy(const DescriptorPool& dp);
		void destroy(const PipelineInfo& pi);
		void destroy(const ShaderModule& sm);
		void destroy(const DescriptorSetLayoutAllocInfo& ds);
		void destroy(const VkPipelineLayout& pl);
		void destroy(const VkRenderPass& rp);
		void destroy(const DescriptorSet&);
		void destroy(const VkFramebuffer& fb);
		void destroy(const Sampler& sa);
		void destroy(const PipelineBaseInfo& pbi);

		ShaderModule create(const create_info_t<ShaderModule>& cinfo);
		PipelineBaseInfo create(const create_info_t<PipelineBaseInfo>& cinfo);
		VkPipelineLayout create(const create_info_t<VkPipelineLayout>& cinfo);
		DescriptorSetLayoutAllocInfo create(const create_info_t<DescriptorSetLayoutAllocInfo>& cinfo);
		ComputePipelineInfo create(const create_info_t<ComputePipelineInfo>& cinfo);
		PipelineInfo create(const struct PipelineInstanceCreateInfo& cinfo);
		VkRenderPass create(const struct RenderPassCreateInfo& cinfo);
		VkFramebuffer create(const struct FramebufferCreateInfo& cinfo);
		Sampler create(const struct SamplerCreateInfo& cinfo);
		DescriptorPool create(const struct DescriptorSetLayoutAllocInfo& cinfo);
		RGImage create(const struct RGCI& cinfo);

		struct TokenData& get_token_data(Token);

		friend class InflightContext;
		friend class PerThreadContext;
		friend struct LinearResourceAllocator;
		friend struct TokenWithContext;
		friend struct IFCImpl;
		friend struct PTCImpl;
		friend struct ExecutableRenderGraph;
		template<class T> friend class Cache; // caches can directly destroy
		template<class T, size_t FC> friend class PerFrameCache;
	};

	class InflightContext {
	public:
		Context& ctx;
		const size_t absolute_frame;
		const unsigned frame;
		InflightContext(Context& ctx, size_t absolute_frame, std::lock_guard<std::mutex>&& recycle_guard);
		~InflightContext();

		void wait_all_transfers();
		PerThreadContext begin();

		std::optional<uint64_t> get_timestamp_query_result(Query);
		std::optional<double> get_duration_query_result(Query start, Query end);
		//std::optional<double> get_named_timestamp_query_results(Name);

		std::vector<SampledImage> get_sampled_images();
	private:
		struct IFCImpl* impl;
		friend class PerThreadContext;
		friend struct PTCImpl;

		std::atomic<size_t> transfer_id = 1;
		std::atomic<size_t> last_transfer_complete = 0;

		TransferStub enqueue_transfer(Buffer src, Buffer dst);
		TransferStub enqueue_transfer(Buffer src, vuk::Image dst, vuk::Extent3D extent, uint32_t base_layer, bool generate_mips);

		void destroy(std::vector<vuk::Image>&& images);
		void destroy(std::vector<VkImageView>&& images);
		void destroy(std::vector<struct LinearResourceAllocator*>&&);
	};

	class PerThreadContext {
	public:
		Context& ctx;
		InflightContext* ifc;
		unsigned tid = 0;

		PerThreadContext(InflightContext& ifc, unsigned tid);
		~PerThreadContext();

		PerThreadContext(const PerThreadContext& o) = delete;
		PerThreadContext& operator=(const PerThreadContext& o) = delete;

		PerThreadContext(PerThreadContext&& o) : ctx(o.ctx) {
			*this = std::move(o);
		}
		PerThreadContext& operator=(PerThreadContext&& o) {
			ifc = o.ifc;
			tid = o.tid;
			impl = std::exchange(o.impl, nullptr);
			return *this;
		}

		PerThreadContext clone();
		Context& get_context() {
			return ctx;
		}

		/// @brief Checks if the given transfer is complete (ready)
		/// @param stub The transfer to check
		/// @return True if the transfer has completed
		bool is_ready(const TransferStub& stub);
		void wait_all_transfers();

		Unique<PersistentDescriptorSet> create_persistent_descriptorset(const PipelineBaseInfo& base, unsigned set, unsigned num_descriptors);
		Unique<PersistentDescriptorSet> create_persistent_descriptorset(const ComputePipelineInfo& base, unsigned set, unsigned num_descriptors);
		Unique<PersistentDescriptorSet> create_persistent_descriptorset(const DescriptorSetLayoutAllocInfo& dslai, unsigned num_descriptors);
		void commit_persistent_descriptorset(PersistentDescriptorSet& array);

		size_t get_allocation_size(Buffer);
		/// @brief Allocates a scratch buffer, i.e. a buffer that has its lifetime bound to the current inflight frame
		/// @param mem_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
		/// @param buffer_usage How this buffer will be used
		/// @param size Size of the buffer
		/// @param alignment Alignment of the buffer
		/// @return The allocated Buffer
		Buffer allocate_scratch_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment);

		/// @brief Allocates a buffer with explicitly managed lifetime
		/// @param mem_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
		/// @param buffer_usage How this buffer will be used
		/// @param size Size of the buffer
		/// @param alignment Alignment of the buffer
		/// @return The allocated Buffer
		Unique<Buffer> allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment);

		/// @brief Allocates & fills a scratch buffer, i.e. a buffer that has its lifetime bound to the current inflight frame
		/// @param mem_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
		/// @param buffer_usage How this buffer will be used (since data is provided, TransferDst is added to the flags)
		/// @return The allocated Buffer
		template<class T>
		std::pair<Buffer, TransferStub> create_scratch_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, std::span<T> data) {
			auto dst = allocate_scratch_buffer(mem_usage, vuk::BufferUsageFlagBits::eTransferDst | buffer_usage, data.size_bytes(), 1);
			if (dst.mapped_ptr) {
				memcpy(dst.mapped_ptr, data.data(), data.size_bytes());
				return { dst, TransferStub{} };
			}
			auto stub = upload(dst, data);
			return { dst, stub };
		}

		/// @brief Allocates & fills a buffer with explicitly managed lifetime
		/// @param mem_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
		/// @param buffer_usage How this buffer will be used (since data is provided, TransferDst is added to the flags)
		/// @return The allocated Buffer
		template<class T>
		std::pair<Unique<Buffer>, TransferStub> create_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, std::span<T> data) {
			auto dst = allocate_buffer(mem_usage, vuk::BufferUsageFlagBits::eTransferDst | buffer_usage, data.size_bytes(), 1);
			if (dst->mapped_ptr) {
				memcpy(dst->mapped_ptr, data.data(), data.size_bytes());
				return { std::move(dst), TransferStub{} };
			}
			auto stub = upload(*dst, data);
			return { std::move(dst), stub };
		}

		vuk::Texture allocate_texture(vuk::ImageCreateInfo);
		std::pair<vuk::Texture, TransferStub> create_texture(vuk::Format format, vuk::Extent3D extents, void* data, bool generate_mips = false);
		Unique<ImageView> create_image_view(vuk::ImageViewCreateInfo);

		template<class T>
		TransferStub upload(Buffer dst, std::span<T> data) {
			if (data.empty()) return { 0 };
			auto staging = allocate_scratch_buffer(MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, sizeof(T) * data.size(), 1);
			::memcpy(staging.mapped_ptr, data.data(), sizeof(T) * data.size());

			return ifc->enqueue_transfer(staging, dst);
		}

		template<class T>
		TransferStub upload(vuk::Image dst, vuk::Format format, vuk::Extent3D extent, uint32_t base_layer, std::span<T> data, bool generate_mips) {
			assert(!data.empty());
			// compute staging buffer alignment as texel block size
			size_t alignment = format_to_texel_block_size(format);
			auto staging = allocate_scratch_buffer(MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, sizeof(T) * data.size(), alignment);
			::memcpy(staging.mapped_ptr, data.data(), sizeof(T) * data.size());

			return ifc->enqueue_transfer(staging, dst, extent, base_layer, generate_mips);
		}

		void dma_task();

		vuk::SampledImage& make_sampled_image(vuk::ImageView iv, vuk::SamplerCreateInfo sci);
		vuk::SampledImage& make_sampled_image(Name n, vuk::SamplerCreateInfo sci);
		vuk::SampledImage& make_sampled_image(Name n, vuk::ImageViewCreateInfo ivci, vuk::SamplerCreateInfo sci);

		vuk::Program get_pipeline_reflection_info(vuk::PipelineBaseCreateInfo pci);

		TimestampQuery register_timestamp_query(Query);

		Token submit(Token, Domain wait_domain);
		void wait(Token);
		void free(Token);

		template<class T>
		void destroy(const T& t) {
			ctx.destroy(t);
		}

		void destroy(vuk::Image image);
		void destroy(vuk::ImageView image);
		void destroy(vuk::DescriptorSet ds);

		VkFence acquire_fence();
		VkCommandBuffer acquire_command_buffer(VkCommandBufferLevel);
		VkSemaphore acquire_semaphore();
		VkFramebuffer acquire_framebuffer(const struct FramebufferCreateInfo&);
		VkRenderPass acquire_renderpass(const struct RenderPassCreateInfo&);
		RGImage acquire_rendertarget(const struct RGCI&);
		Sampler acquire_sampler(const SamplerCreateInfo&);
		DescriptorSet acquire_descriptorset(const SetBinding&);
		PipelineInfo acquire_pipeline(const PipelineInstanceCreateInfo&);

		const plf::colony<SampledImage>& get_sampled_images();

		PipelineBaseInfo create(const struct PipelineBaseCreateInfo& cinfo);
		ShaderModule create(const struct ShaderModuleCreateInfo& cinfo);
		VkRenderPass create(const struct RenderPassCreateInfo& cinfo);
		RGImage create(const struct RGCI& cinfo);
		LinearAllocator create(const struct PoolSelect& cinfo);
		DescriptorPool create(const struct DescriptorSetLayoutAllocInfo& cinfo);
		DescriptorSet create(const struct SetBinding& cinfo);
		VkFramebuffer create(const struct FramebufferCreateInfo& cinfo);
		Sampler create(const struct SamplerCreateInfo& cinfo);
		DescriptorSetLayoutAllocInfo create(const struct DescriptorSetLayoutCreateInfo& cinfo);
		VkPipelineLayout create(const struct PipelineLayoutCreateInfo& cinfo);
		ComputePipelineInfo create(const struct ComputePipelineCreateInfo& cinfo);
		PipelineInfo create(const struct PipelineInstanceCreateInfo& cinfo);
	private:
		friend class InflightContext;

		struct PTCImpl* impl;
	};

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
	bool execute_submit_and_present_to_one(PerThreadContext& ptc, ExecutableRenderGraph&& rg, SwapchainRef swapchain);
	void execute_submit_and_wait(PerThreadContext& ptc, ExecutableRenderGraph&& rg);
}
