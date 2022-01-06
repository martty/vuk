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

	template<class T>
	struct Future;

	enum class DomainFlagBits {
		eNone = 0,
		eHost = 1 << 0,
		eGraphicsQueue = 1 << 1,
		eComputeQueue = 1 << 2,
		eTransferQueue = 1 << 3,
		eGraphicsOperation = 1 << 4,
		eComputeOperation = 1 << 5,
		eTransferOperation = 1 << 6,
		eQueueMask = 0b1110,
		eOpMask = 0b1110000,
		eGraphicsOnGraphics = eGraphicsQueue | eGraphicsOperation,
		eComputeOnGraphics = eGraphicsQueue | eComputeOperation,
		eTransferOnGraphics = eGraphicsQueue | eTransferOperation,
		eComputeOnCompute = eComputeQueue | eComputeOperation,
		eTransferOnCompute = eComputeQueue | eComputeOperation,
		eTransferOnTransfer = eTransferQueue | eTransferOperation,
		eDevice = eGraphicsQueue | eComputeQueue | eTransferQueue,
		eAny = eDevice | eHost
	};

	using DomainFlags = Flags<DomainFlagBits>;
	inline constexpr DomainFlags operator|(DomainFlagBits bit0, DomainFlagBits bit1) noexcept {
		return DomainFlags(bit0) | bit1;
	}

	inline constexpr DomainFlags operator&(DomainFlagBits bit0, DomainFlagBits bit1) noexcept {
		return DomainFlags(bit0) & bit1;
	}

	inline constexpr DomainFlags operator^(DomainFlagBits bit0, DomainFlagBits bit1) noexcept {
		return DomainFlags(bit0) ^ bit1;
	}

	struct ContextCreateParameters {
		VkInstance instance;
		VkDevice device;
		VkPhysicalDevice physical_device;
		/// @brief Optional graphics queue
		VkQueue graphics_queue = VK_NULL_HANDLE;
		/// @brief Optional graphics queue family index
		uint32_t graphics_queue_family_index = VK_QUEUE_FAMILY_IGNORED;
		/// @brief Optional compute queue
		VkQueue compute_queue = VK_NULL_HANDLE;
		/// @brief Optional compute queue family index
		uint32_t compute_queue_family_index = VK_QUEUE_FAMILY_IGNORED;
		/// @brief Optional transfer queue
		VkQueue transfer_queue = VK_NULL_HANDLE;
		/// @brief Optional transfer queue family index
		uint32_t transfer_queue_family_index = VK_QUEUE_FAMILY_IGNORED;
	};

	struct Queue {
		Queue(PFN_vkQueueSubmit2KHR fn, VkQueue queue, uint32_t queue_family_index, TimelineSemaphore ts);

		std::mutex queue_lock;
		PFN_vkQueueSubmit2KHR queueSubmit2KHR;
		TimelineSemaphore submit_sync;
		VkQueue queue;
		std::array<std::atomic<uint64_t>, 3> last_device_waits;
		std::atomic<uint64_t> last_host_wait;
		uint32_t family_index;

		Result<void> submit(std::span<VkSubmitInfo> submit_infos, VkFence fence);
		Result<void> submit(std::span<VkSubmitInfo2KHR> submit_infos, VkFence fence);
	};

	class Context {
	public:
		VkInstance instance;
		VkDevice device;
		VkPhysicalDevice physical_device;
		uint32_t graphics_queue_family_index;
		uint32_t compute_queue_family_index;
		uint32_t transfer_queue_family_index;

		std::optional<Queue> dedicated_graphics_queue;
		std::optional<Queue> dedicated_compute_queue;
		std::optional<Queue> dedicated_transfer_queue;

		Queue* graphics_queue = nullptr;
		Queue* compute_queue = nullptr;
		Queue* transfer_queue = nullptr;

		Result<void> wait_for_domains(std::span<std::pair<DomainFlags, uint64_t>> queue_waits);

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

		Queue& domain_to_queue(DomainFlags);
		uint32_t domain_to_queue_index(DomainFlags);
		uint32_t domain_to_queue_family_index(DomainFlags);

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

		size_t get_allocation_size(Buffer);

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
		Result<void> submit_graphics(std::span<VkSubmitInfo2KHR>);
		Result<void> submit_transfer(std::span<VkSubmitInfo2KHR>);

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

#include "vuk/Exception.hpp"
// utility functions
namespace vuk {
	struct ExecutableRenderGraph;
	Result<void> link_execute_submit(Allocator& allocator, std::span<std::pair<Allocator*, struct RenderGraph*>> rgs);
	Result<void> execute_submit(Allocator& allocator, std::span<std::pair<Allocator*, ExecutableRenderGraph*>> rgs, std::vector<std::pair<SwapchainRef, size_t>> swapchains_with_indexes, VkSemaphore present_rdy, VkSemaphore render_complete);
	Result<void> execute_submit_and_present_to_one(Allocator& nalloc, ExecutableRenderGraph&& rg, SwapchainRef swapchain);
	Result<void> execute_submit_and_wait(Allocator& nalloc, ExecutableRenderGraph&& rg);

	struct FutureBase;

	template<class T>
	int get_allocator(T&) {
		return 0;
	}

	template<class... Args>
	Result<void> wait_for_futures(Allocator& alloc, Args&... futs) {
		std::array controls = { futs.control.get()...};
		std::array rgs = { futs.rg... };
		std::vector<std::pair<Allocator*, RenderGraph*>> rgs_to_run;
		for (uint64_t i = 0; i < controls.size(); i++) {
			auto& control = controls[i];
			if (control->status == FutureBase::Status::eInputAttached || control->status == FutureBase::Status::eInitial) {
				return { expected_error };
			} else if (control->status == FutureBase::Status::eHostAvailable || control->status == FutureBase::Status::eSubmitted) {
				continue;
			} else {
				rgs_to_run.emplace_back(&control->get_allocator(), rgs[i]);
			}
		}

		VUK_DO_OR_RETURN(link_execute_submit(alloc, std::span(rgs_to_run)));

		std::vector<std::pair<DomainFlags, uint64_t>> waits;
		for (uint64_t i = 0; i < controls.size(); i++) {
			auto& control = controls[i];
			if (control->status != FutureBase::Status::eSubmitted) {
				continue;
			}
			waits.emplace_back(control->initial_domain, control->initial_visibility);
		}
		alloc.get_context().wait_for_domains(std::span(waits));

		return { expected_value };
	}

	SampledImage make_sampled_image(ImageView iv, SamplerCreateInfo sci);

	SampledImage make_sampled_image(Name n, SamplerCreateInfo sci);

	SampledImage make_sampled_image(Name n, ImageViewCreateInfo ivci, SamplerCreateInfo sci);
}

// futures
namespace vuk {

	struct QueueResourceUse {
		vuk::Access original;
		vuk::PipelineStageFlags stages;
		vuk::AccessFlags access;
		vuk::ImageLayout layout; // ignored for buffers
		vuk::DomainFlagBits domain;
	};

	struct ImageAttachment {
		vuk::Image image;
		vuk::ImageView image_view;

		vuk::Extent2D extent;
		vuk::Format format;
		vuk::Samples sample_count = vuk::Samples::e1;
		Clear clear_value;

		uint32_t base_level = 0;
		uint32_t level_count = VK_REMAINING_MIP_LEVELS;

		uint32_t base_layer = 0;
		uint32_t layer_count = VK_REMAINING_ARRAY_LAYERS;

		static ImageAttachment from_texture(const vuk::Texture& t, Clear clear_value) {
			return ImageAttachment{
				.image = t.image.get(), .image_view = t.view.get(), .extent = {t.extent.width, t.extent.height}, .format = t.format, .sample_count = {t.sample_count}, .clear_value = clear_value };
		}
		static ImageAttachment from_texture(const vuk::Texture& t) {
			return ImageAttachment{
				.image = t.image.get(), .image_view = t.view.get(), .extent = {t.extent.width, t.extent.height}, .format = t.format, .sample_count = {t.sample_count}, .layer_count = t.view->layer_count };
		}
	};

	struct FutureBase {
		FutureBase() = default;
		FutureBase(Allocator&);

		Allocator* allocator;

		enum class Status {
			eInitial, // default-constructed future
			eRenderGraphBound, // a rendergraph was bound to this future
			eInputAttached, // this future was attached to a rendergraph as input
			eOutputAttached, // this future was attached to a rendergraph as output
			eSubmitted, // the rendergraph referenced by this future was submitted (result is available on device with appropriate sync)
			eHostAvailable // the result is available on host, available on device without sync
		} status = Status::eInitial;

		Allocator& get_allocator() { return *allocator; }

		DomainFlagBits initial_domain = DomainFlagBits::eNone; // the domain where we submitted this Future to
		QueueResourceUse last_use; // the results of the future are available if waited for on the initial_domain
		uint64_t initial_visibility; // the results of the future are available if waited for {initial_domain, initial_visibility}

		ImageAttachment result_image;
		Buffer result_buffer;

		template<class T>
		T& get_result();

		template<>
		ImageAttachment& get_result() {
			return result_image;
		}

		template<>
		Buffer& get_result() {
			return result_buffer;
		}
	};

	template<class T>
	struct Future {
		Future() = default;
		/// @brief Create a Future with ownership of a RenderGraph and bind to an output
		/// @param allocator 
		/// @param rg 
		/// @param output_binding 
		Future(Allocator& allocator, std::unique_ptr<struct RenderGraph> rg, Name output_binding);
		/// @brief Create a Future without ownership of a RenderGraph and bind to an output
		/// @param allocator 
		/// @param rg 
		/// @param output_binding 
		Future(Allocator& allocator, struct RenderGraph& rg, Name output_binding);
		/// @brief Create a Future from a value, automatically making it host available
		/// @param allocator 
		/// @param value 
		Future(Allocator& allocator, T&& value);

		Name output_binding;

		std::unique_ptr<RenderGraph> owned_rg;
		RenderGraph* rg = nullptr;

		std::unique_ptr<FutureBase> control;

		FutureBase::Status& get_status() {
			return control->status;
		}

		Allocator& get_allocator() { return *control->allocator; }

		Result<void> submit(); // turn cmdbufs into possibly a TS
		Result<T> get(); // wait on host for T to be produced by the computation
	};
}
