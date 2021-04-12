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

	template<class Parent> struct LinearResourceAllocator;
	struct TokenData {
		enum class State { eInitial, eArmed, ePending, eComplete } state = State::eInitial;  // token state : nothing -> armed (ready for submit) -> pending (submitted) -> complete (observed on CPU)
		enum class TokenType { eUndecided, eTimeline, eAnyDevice, eEvent, eBarrier, eOrder } token_type = TokenType::eUndecided;
		LinearResourceAllocator<Allocator>* resources = nullptr; // internally may have resources bound to the token, which get freed or enqueued for deletion
		struct RenderGraph* rg;
		TokenData* next = nullptr;
	};

	struct TimestampQuery {
		VkQueryPool pool;
		uint32_t id;
	};

	enum class Domain {
		eNone = 0,
		eHost = 1 << 0,
		eGraphics = 1 << 1,
		eCompute = 1 << 2,
		eTransfer = 1 << 3,
		eDevice = eGraphics | eCompute | eTransfer,
		eAny = eDevice | eHost
	};

	struct DebugUtils {
		VkDevice device;
		PFN_vkSetDebugUtilsObjectNameEXT setDebugUtilsObjectNameEXT;
		PFN_vkCmdBeginDebugUtilsLabelEXT cmdBeginDebugUtilsLabelEXT;
		PFN_vkCmdEndDebugUtilsLabelEXT cmdEndDebugUtilsLabelEXT;

		DebugUtils(VkDevice device);

		bool enabled();
		void set_name(const vuk::Texture& iv, /*zstring_view*/Name name);
		template<class T>
		void set_name(const T& t, /*zstring_view*/Name name);

		void begin_region(const VkCommandBuffer&, Name name, std::array<float, 4> color = { 1,1,1,1 });
		void end_region(const VkCommandBuffer&);
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
		VkPhysicalDeviceProperties physical_device_properties;

		std::atomic<size_t> frame_counter = 0;

		/// @brief Create a new Context
		/// @param params Vulkan parameters initialized beforehand
		Context(ContextCreateParameters params);
		~Context();

		void create_named_pipeline(Name name, vuk::PipelineBaseInfo&);
		void create_named_pipeline(Name name, vuk::ComputePipelineCreateInfo pbci);

		PipelineBaseInfo* get_named_pipeline(Name name);
		ComputePipelineInfo* get_named_compute_pipeline(Name name);

		PipelineBaseInfo* get_pipeline(const PipelineBaseCreateInfo& pbci);
		ComputePipelineInfo* get_pipeline(const ComputePipelineCreateInfo& pbci);
		Program get_pipeline_reflection_info(PipelineBaseCreateInfo pbci);
		ShaderModule compile_shader(ShaderSource source, std::string path);

		bool load_pipeline_cache(std::span<uint8_t> data);
		std::vector<uint8_t> save_pipeline_cache();

		Token create_token();
		TokenData& get_token_data(Token t);
		Token submit(Allocator&, Token, Domain wait_domain);
		void wait(Allocator&, Token);
		void destroy_token(Token);

		TokenWithContext transition_image(vuk::Texture&, vuk::Access src_access, vuk::Access dst_access);
		TokenWithContext copy_to_buffer(vuk::Domain copy_domain, vuk::Buffer buffer, void* data, size_t size);
		TokenWithContext copy_to_image(vuk::Image dst, vuk::Format format, vuk::Extent3D extent, uint32_t base_layer, void* data, size_t size);

		/// @brief Wait for the device to become idle. Useful for only a few synchronisation events, like resizing or shutting down.
		void wait_idle();

		void submit_graphics(VkSubmitInfo, VkFence);
		void submit_transfer(VkSubmitInfo, VkFence);
	private:
		struct ContextImpl* impl;

		template<class Parent> friend struct LinearResourceAllocator;
		friend struct TokenWithContext;
		friend struct IFCImpl;
		friend struct PTCImpl;
		friend struct ExecutableRenderGraph;
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
		Unique<ImageView> create_image_view(vuk::ImageViewCreateInfo);

		template<class T>
		TransferStub upload(vuk::Image dst, vuk::Format format, vuk::Extent3D extent, uint32_t base_layer, std::span<T> data, bool generate_mips) {
			assert(!data.empty());
			// compute staging buffer alignment as texel block size
			size_t alignment = format_to_texel_block_size(format);
			auto staging = allocate_scratch_buffer(MemoryUsage::eCPUonly, vuk::BufferUsageFlagBits::eTransferSrc, sizeof(T) * data.size(), alignment);
			::memcpy(staging.mapped_ptr, data.data(), sizeof(T) * data.size());

			return {};// ifc->enqueue_transfer(staging, dst, extent, base_layer, generate_mips);
		}

		void dma_task();

		vuk::SampledImage& make_sampled_image(vuk::ImageView iv, vuk::SamplerCreateInfo sci);
		vuk::SampledImage& make_sampled_image(Name n, vuk::SamplerCreateInfo sci);
		vuk::SampledImage& make_sampled_image(Name n, vuk::ImageViewCreateInfo ivci, vuk::SamplerCreateInfo sci);

		vuk::Program get_pipeline_reflection_info(vuk::PipelineBaseCreateInfo pci);

		TimestampQuery register_timestamp_query(Query);

		const plf::colony<SampledImage>& get_sampled_images();
	private:
		struct PTCImpl* impl;
	};

	template<class T>
	void DebugUtils::set_name(const T& t, Name name) {
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
		setDebugUtilsObjectNameEXT(device, &info);
	}

	template<typename Type, class Allocator>
	inline Unique<Type, Allocator>::~Unique() noexcept {
		if (context && payload != Type{})
			context->destroy(std::move(payload));
	}
	template<typename Type, class Allocator>
	inline void Unique<Type, Allocator>::reset(Type value) noexcept {
		if (payload != value) {
			if (context && payload != Type{}) {
				context->destroy(std::move(payload));
			}
			payload = std::move(value);
		}
	}
}

// utility functions
namespace vuk {
	struct ExecutableRenderGraph;
	// TODO: template these?
	bool execute_submit_and_present_to_one(Context&, struct ThreadLocalFrameAllocator&, ExecutableRenderGraph&& rg, vuk::Swapchain& swapchain);
	void execute_submit_and_wait(Context&, struct ThreadLocalFrameAllocator&, ExecutableRenderGraph&& rg);
}
