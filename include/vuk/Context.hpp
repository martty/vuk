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

namespace vuk {
	struct TransferStub {
		size_t id;
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

		Context(VkInstance instance, VkDevice device, VkPhysicalDevice physical_device, VkQueue graphics);
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

		void create_named_pipeline(const char* name, vuk::PipelineBaseCreateInfo pbci);
		void create_named_pipeline(const char* name, vuk::ComputePipelineCreateInfo pbci);

		PipelineBaseInfo* get_named_pipeline(const char* name);
		ComputePipelineInfo* get_named_compute_pipeline(const char* name);

		PipelineBaseInfo* get_pipeline(const PipelineBaseCreateInfo& pbci);
		ComputePipelineInfo* get_pipeline(const ComputePipelineCreateInfo& pbci);
		Program get_pipeline_reflection_info(PipelineBaseCreateInfo pbci);
		ShaderModule compile_shader(std::string source, Name path);

		ShaderModule create(const create_info_t<ShaderModule>& cinfo);
		PipelineBaseInfo create(const create_info_t<PipelineBaseInfo>& cinfo);
		VkPipelineLayout create(const create_info_t<VkPipelineLayout>& cinfo);
		DescriptorSetLayoutAllocInfo create(const create_info_t<DescriptorSetLayoutAllocInfo>& cinfo);
		ComputePipelineInfo create(const create_info_t<ComputePipelineInfo>& cinfo);

		bool load_pipeline_cache(std::span<uint8_t> data);
		std::vector<uint8_t> save_pipeline_cache();

		uint32_t(*get_thread_index)() = nullptr;

		// when the fence is signaled, caller should clean up the resources
		struct UploadResult {
			VkFence fence;
			VkCommandBuffer command_buffer;
			vuk::Buffer staging;
			bool is_buffer;
			unsigned thread_index;
		};

		struct BufferUpload {
			vuk::Buffer dst;
			std::span<unsigned char> data;
		};
		UploadResult fenced_upload(std::span<BufferUpload>);

		struct ImageUpload {
			vuk::Image dst;
			vuk::Extent3D extent;
			std::span<unsigned char> data;
		};
		UploadResult fenced_upload(std::span<ImageUpload>);
		void free_upload_resources(const UploadResult&);

		Buffer allocate_buffer(MemoryUsage mem_usage, BufferUsageFlags buffer_usage, size_t size, size_t alignment);
		Texture allocate_texture(vuk::ImageCreateInfo ici);

		void enqueue_destroy(vuk::Image);
		void enqueue_destroy(vuk::ImageView);
		void enqueue_destroy(VkPipeline);
		void enqueue_destroy(vuk::Buffer);
		void enqueue_destroy(vuk::PersistentDescriptorSet);

		template<class T>
		Handle<T> wrap(T payload);

		SwapchainRef add_swapchain(Swapchain sw);

		InflightContext begin();

		void wait_idle();

		void submit_graphics(VkSubmitInfo, VkFence);
		void submit_transfer(VkSubmitInfo, VkFence);
	private:
		struct ContextImpl* impl;
		std::atomic<size_t> unique_handle_id_counter = 0;

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

		friend class InflightContext;
		friend class PerThreadContext;
		friend struct IFCImpl;
		friend struct PTCImpl;
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

		std::vector<SampledImage> get_sampled_images();
	private:
		struct IFCImpl* impl;
		friend class PerThreadContext;
		friend struct PTCImpl;

		std::atomic<size_t> transfer_id = 1;
		std::atomic<size_t> last_transfer_complete = 0;

		TransferStub enqueue_transfer(Buffer src, Buffer dst);
		TransferStub enqueue_transfer(Buffer src, vuk::Image dst, vuk::Extent3D extent, bool generate_mips);
		void destroy(std::vector<vuk::Image>&& images);
		void destroy(std::vector<VkImageView>&& images);
	};

	class PerThreadContext {
	public:
		Context& ctx;
		InflightContext& ifc;
		const unsigned tid = 0;

		PerThreadContext(InflightContext& ifc, unsigned tid);
		~PerThreadContext();

		PerThreadContext(const PerThreadContext& o) = delete;
		PerThreadContext& operator=(const PerThreadContext& o) = delete;

		bool is_ready(const TransferStub& stub);
		void wait_all_transfers();

		Unique<PersistentDescriptorSet> create_persistent_descriptorset(const PipelineBaseInfo& base, unsigned set, unsigned num_descriptors);
		Unique<PersistentDescriptorSet> create_persistent_descriptorset(const ComputePipelineInfo& base, unsigned set, unsigned num_descriptors);
		Unique<PersistentDescriptorSet> create_persistent_descriptorset(const DescriptorSetLayoutAllocInfo& dslai, unsigned num_descriptors);
		void commit_persistent_descriptorset(PersistentDescriptorSet& array);

		size_t get_allocation_size(Buffer);
		Buffer _allocate_scratch_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment, bool create_mapped);
		Unique<Buffer> _allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment, bool create_mapped);

		// since data is provided, we will add TransferDst to the flags automatically
		template<class T>
		std::pair<Buffer, TransferStub> create_scratch_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, std::span<T> data) {
			auto dst = _allocate_scratch_buffer(mem_usage, vuk::BufferUsageFlagBits::eTransferDst | buffer_usage, sizeof(T) * data.size(), 1, false);
			auto stub = upload(dst, data);
			return { dst, stub };
		}

		template<class T>
		std::pair<Unique<Buffer>, TransferStub> create_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, std::span<T> data) {
			auto dst = _allocate_buffer(mem_usage, vuk::BufferUsageFlagBits::eTransferDst | buffer_usage, sizeof(T) * data.size(), 1, false);
			auto stub = upload(*dst, data);
			return { std::move(dst), stub };
		}


		vuk::Texture allocate_texture(vuk::ImageCreateInfo);
		std::pair<vuk::Texture, TransferStub> create_texture(vuk::Format format, vuk::Extent3D extents, void* data);

		template<class T>
		TransferStub upload(Buffer dst, std::span<T> data) {
			if (data.empty()) return { 0 };
			auto staging = _allocate_scratch_buffer(MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, sizeof(T) * data.size(), 1, true);
			::memcpy(staging.mapped_ptr, data.data(), sizeof(T) * data.size());

			return ifc.enqueue_transfer(staging, dst);
		}

		template<class T>
		TransferStub upload(vuk::Image dst, vuk::Extent3D extent, std::span<T> data, bool generate_mips) {
			assert(!data.empty());
			auto staging = _allocate_scratch_buffer(MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, sizeof(T) * data.size(), 1, true);
			::memcpy(staging.mapped_ptr, data.data(), sizeof(T) * data.size());

			return ifc.enqueue_transfer(staging, dst, extent, generate_mips);
		}

		void dma_task();

		vuk::SampledImage& make_sampled_image(vuk::ImageView iv, vuk::SamplerCreateInfo sci);
		vuk::SampledImage& make_sampled_image(Name n, vuk::SamplerCreateInfo sci);
		vuk::SampledImage& make_sampled_image(Name n, vuk::ImageViewCreateInfo ivci, vuk::SamplerCreateInfo sci);

		vuk::Program get_pipeline_reflection_info(vuk::PipelineBaseCreateInfo pci);

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
		PipelineInfo create(const struct PipelineInstanceCreateInfo& cinfo);
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

	private:
		friend class InflightContext;

		struct PTCImpl* impl;
	};

	template<class T>
	void Context::DebugUtils::set_name(const T& t, Name name) {
		if (!enabled()) return;
		VkDebugUtilsObjectNameInfoEXT info = { .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
		info.pObjectName = name.data();
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
	struct RenderGraph;
	bool execute_submit_and_present_to_one(PerThreadContext& ptc, RenderGraph& rg, SwapchainRef swapchain);
	void execute_submit_and_wait(PerThreadContext& ptc, RenderGraph& rg);
}
