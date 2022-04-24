#pragma once

#include <array>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "vuk/Allocator.hpp"
#include "vuk/Buffer.hpp"
#include "vuk/Image.hpp"
#include "vuk/Swapchain.hpp"
#include "vuk_fwd.hpp"

namespace std {
	class mutex;
	class recursive_mutex;
} // namespace std

namespace vuk {
	/// @brief Parameters used for creating a Context
	struct ContextCreateParameters {
		/// @brief Vulkan instance
		VkInstance instance;
		/// @brief Vulkan device
		VkDevice device;
		/// @brief Vulkan physical device
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

	/// @brief Abstraction of a device queue in Vulkan
	struct Queue {
		Queue(PFN_vkQueueSubmit2KHR fn, VkQueue queue, uint32_t queue_family_index, TimelineSemaphore ts);
		~Queue();

		TimelineSemaphore& get_submit_sync();
		std::recursive_mutex& get_queue_lock();

		Result<void> submit(std::span<VkSubmitInfo> submit_infos, VkFence fence);
		Result<void> submit(std::span<VkSubmitInfo2KHR> submit_infos, VkFence fence);

		struct QueueImpl* impl;
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

		uint64_t get_frame_count() const;

		/// @brief Create a new Context
		/// @param params Vulkan parameters initialized beforehand
		Context(ContextCreateParameters params);
		~Context();

		struct DebugUtils {
			Context& ctx;
			PFN_vkSetDebugUtilsObjectNameEXT setDebugUtilsObjectNameEXT;
			PFN_vkCmdBeginDebugUtilsLabelEXT cmdBeginDebugUtilsLabelEXT;
			PFN_vkCmdEndDebugUtilsLabelEXT cmdEndDebugUtilsLabelEXT;

			bool enabled() const;

			DebugUtils(Context& ctx);
			void set_name(const Texture& iv, Name name);
			template<class T>
			void set_name(const T& t, Name name);

			void begin_region(const VkCommandBuffer&, Name name, std::array<float, 4> color = { 1, 1, 1, 1 });
			void end_region(const VkCommandBuffer&);
		} debug;

		void create_named_pipeline(Name name, PipelineBaseCreateInfo pbci);

		PipelineBaseInfo* get_named_pipeline(Name name);

		PipelineBaseInfo* get_pipeline(const PipelineBaseCreateInfo& pbci);
		Program get_pipeline_reflection_info(const PipelineBaseCreateInfo& pbci);
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

		Texture allocate_texture(Allocator& allocator, ImageCreateInfo ici);

		size_t get_allocation_size(Buffer);

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

		/// @brief Checks if a timestamp query is available
		/// @param q the Query to check
		/// @return true if the timestamp is available
		bool is_timestamp_available(Query q);

		/// @brief Retrieve a timestamp if available
		/// @param q the Query to check
		/// @return the timestamp value if it was available, null optional otherwise
		std::optional<uint64_t> retrieve_timestamp(Query q);

		/// @brief Retrive a duration if available
		/// @param q1 the start timestamp Query
		/// @param q2 the end timestamp Query
		/// @return the duration in seconds if both timestamps were available, null optional otherwise
		std::optional<double> retrieve_duration(Query q1, Query q2);

		/// @brief Retrieve results from `TimestampQueryPool`s and make them available to retrieve_timestamp and retrieve_duration
		Result<void> make_timestamp_results_available(std::span<const TimestampQueryPool> pools);

		// Caches

		/// @brief Acquire a cached rendertarget
		RGImage acquire_rendertarget(const struct RGCI& ci, uint64_t absolute_frame);
		/// @brief Acquire a cached sampler
		Sampler acquire_sampler(const SamplerCreateInfo& cu, uint64_t absolute_frame);
		/// @brief Acquire a cached VkRenderPass
		VkRenderPass acquire_renderpass(const struct RenderPassCreateInfo& ci, uint64_t absolute_frame);
		/// @brief Acquire a cached pipeline
		struct PipelineInfo acquire_pipeline(const struct PipelineInstanceCreateInfo& ci, uint64_t absolute_frame);
		/// @brief Acquire a cached compute pipeline
		struct ComputePipelineInfo acquire_pipeline(const struct ComputePipelineInstanceCreateInfo& ci, uint64_t absolute_frame);
		/// @brief Acquire a cached descriptor pool
		struct DescriptorPool& acquire_descriptor_pool(const struct DescriptorSetLayoutAllocInfo& dslai, uint64_t absolute_frame);

		// Persistent descriptor sets

		Unique<PersistentDescriptorSet> create_persistent_descriptorset(Allocator& allocator, struct DescriptorSetLayoutCreateInfo dslci, unsigned num_descriptors);
		Unique<PersistentDescriptorSet> create_persistent_descriptorset(Allocator& allocator, const PipelineBaseInfo& base, unsigned set, unsigned num_descriptors);
		Unique<PersistentDescriptorSet> create_persistent_descriptorset(Allocator& allocator, const PersistentDescriptorSetCreateInfo&);
		void commit_persistent_descriptorset(PersistentDescriptorSet& array);

		void collect(uint64_t frame);

		uint64_t get_unique_handle_id();

	private:
		struct ContextImpl* impl;

		void destroy(const struct RGImage& image);
		void destroy(const struct LegacyPoolAllocator& v);
		void destroy(const struct LegacyLinearAllocator& v);
		void destroy(const struct DescriptorPool& dp);
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

		ShaderModule create(const struct ShaderModuleCreateInfo& cinfo);
		PipelineBaseInfo create(const struct PipelineBaseCreateInfo& cinfo);
		VkPipelineLayout create(const struct PipelineLayoutCreateInfo& cinfo);
		DescriptorSetLayoutAllocInfo create(const struct DescriptorSetLayoutCreateInfo& cinfo);
		DescriptorPool create(const struct DescriptorSetLayoutAllocInfo& cinfo);
		PipelineInfo create(const struct PipelineInstanceCreateInfo& cinfo);
		ComputePipelineInfo create(const struct ComputePipelineInstanceCreateInfo& cinfo);
		VkRenderPass create(const struct RenderPassCreateInfo& cinfo);
		RGImage create(const struct RGCI& cinfo);
		Sampler create(const struct SamplerCreateInfo& cinfo);

		template<class T>
		friend class Cache; // caches can directly destroy
	};

	template<class T>
	Handle<T> Context::wrap(T payload) {
		return { { get_unique_handle_id() }, payload };
	}

	template<class T>
	void Context::DebugUtils::set_name(const T& t, Name name) {
		if (!enabled())
			return;
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
} // namespace vuk

#include "vuk/Exception.hpp"
// utility functions
namespace vuk {
	struct ExecutableRenderGraph;

	/// @brief Compile & link given `RenderGraph`s, then execute them into API VkCommandBuffers, then submit them to queues
	/// @param allocator Allocator to use for submission resources
	/// @param rendergraphs `RenderGraph`s for compilation
	Result<void> link_execute_submit(Allocator& allocator, std::span<std::pair<Allocator*, struct RenderGraph*>> rendergraphs);
	/// @brief Execute given `ExecutableRenderGraph`s into API VkCommandBuffers, then submit them to queues
	/// @param allocator Allocator to use for submission resources
	/// @param executable_rendergraphs `ExecutableRenderGraph`s for execution
	/// @param swapchains_with_indexes Swapchains references by the rendergraphs
	/// @param present_rdy Semaphore used to gate device-side execution
	/// @param render_complete Semaphore used to gate presentation
	Result<void> execute_submit(Allocator& allocator,
	                            std::span<std::pair<Allocator*, ExecutableRenderGraph*>> executable_rendergraphs,
	                            std::vector<std::pair<SwapchainRef, size_t>> swapchains_with_indexes,
	                            VkSemaphore present_rdy,
	                            VkSemaphore render_complete);
	/// @brief Execute given `ExecutableRenderGraph` into API VkCommandBuffers, then submit them to queues, presenting to a single swapchain
	/// @param allocator Allocator to use for submission resources
	/// @param executable_rendergraph `ExecutableRenderGraph`s for execution
	/// @param swapchain Swapchain referenced by the rendergraph
	Result<void> execute_submit_and_present_to_one(Allocator& allocator, ExecutableRenderGraph&& executable_rendergraph, SwapchainRef swapchain);
	/// @brief Execute given `ExecutableRenderGraph` into API VkCommandBuffers, then submit them to queues, then blocking-wait for the submission to complete
	/// @param allocator Allocator to use for submission resources
	/// @param executable_rendergraph `ExecutableRenderGraph`s for execution
	Result<void> execute_submit_and_wait(Allocator& allocator, ExecutableRenderGraph&& executable_rendergraph);

	struct SampledImage make_sampled_image(ImageView iv, SamplerCreateInfo sci);

	struct SampledImage make_sampled_image(Name n, SamplerCreateInfo sci);

	struct SampledImage make_sampled_image(Name n, ImageViewCreateInfo ivci, SamplerCreateInfo sci);
} // namespace vuk