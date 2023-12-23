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
#include "vuk/runtime/vk/VulkanQueueExecutor.hpp"
#include "vuk_fwd.hpp"

#include "vuk/SourceLocation.hpp"

namespace std {
	class mutex;
	class recursive_mutex;
} // namespace std

namespace vuk {
	namespace rtvk {
#define VUK_X(name) PFN_##name name = nullptr;
#define VUK_Y(name) PFN_##name name = nullptr;

		struct FunctionPointers {
			PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
			PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;
#include "vuk/VulkanPFNOptional.hpp"
#include "vuk/VulkanPFNRequired.hpp"

			/// @brief Check if all required function pointers are available (if providing them externally)
			bool check_pfns();
			/// @brief Load function pointers that the runtime needs
			/// @param instance Vulkan instance
			/// @param device Vulkan device
			/// @param allow_dynamic_loading_of_vk_function_pointers If true, then this function will attempt dynamic loading of the fn pointers
			/// If this is false, then you must fill in all required function pointers
			vuk::Result<void> load_pfns(VkInstance instance, VkDevice device, bool allow_dynamic_loading_of_vk_function_pointers);
		};
#undef VUK_X
#undef VUK_Y

		std::unique_ptr<Executor> create_vkqueue_executor(const FunctionPointers& fps, VkDevice device, VkQueue queue, uint32_t queue_family_index, DomainFlagBits domain);
	} // namespace rtvk

	/// @brief Parameters used for creating a Context
	struct ContextCreateParameters {
		/// @brief Vulkan instance
		VkInstance instance;
		/// @brief Vulkan device
		VkDevice device;
		/// @brief Vulkan physical device
		VkPhysicalDevice physical_device;
		/// @brief Executors available to the runtime for scheduling
		std::vector<std::unique_ptr<Executor>> executors;
		/// @brief User provided function pointers. If you want dynamic loading, you must set vkGetInstanceProcAddr & vkGetDeviceProcAddr
		rtvk::FunctionPointers pointers;
	};

	class Context : public rtvk::FunctionPointers {
	public:
		/// @brief Create a new Context
		/// @param params Vulkan parameters initialized beforehand
		Context(ContextCreateParameters params);
		~Context();

		Context(const Context&) = delete;
		Context& operator=(const Context&) = delete;

		Context(Context&&) noexcept;
		Context& operator=(Context&&) noexcept;

		// Vulkan instance and device

		VkInstance instance;
		VkDevice device;
		VkPhysicalDevice physical_device;

		// Vulkan properties

		VkPhysicalDeviceProperties physical_device_properties;
		VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_properties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };
		VkPhysicalDeviceAccelerationStructurePropertiesKHR as_properties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR };
		size_t min_buffer_alignment;

		// Executors
		std::vector<uint32_t> all_queue_families;
		// retrieve a specific executor from the runtime
		Executor* get_executor(Executor::Tag tag);
		// retrieve an executor for the given domain from the runtime
		Executor* get_executor(DomainFlagBits domain);

		// Debug functions

		/// @brief If debug utils is available and debug names & markers are supported
		bool debug_enabled() const;

		/// @brief Set debug name for object
		template<class T>
		void set_name(const T& t, Name name);

		/// @brief Add debug region to command buffer
		/// @param name Name of the region
		/// @param color Display color of the region
		void begin_region(const VkCommandBuffer&, Name name, std::array<float, 4> color = { 1, 1, 1, 1 });
		/// @brief End debug region in command buffer
		void end_region(const VkCommandBuffer&);

		// Pipeline management

		/// Internal pipeline cache to use
		VkPipelineCache vk_pipeline_cache = VK_NULL_HANDLE;

		/// @brief Create a pipeline base that can be recalled by name
		void create_named_pipeline(Name name, PipelineBaseCreateInfo pbci);

		/// @brief Recall name pipeline base
		PipelineBaseInfo* get_named_pipeline(Name name);

		/// @brief Checks if a pipeline is available
		/// @param name the Name of the pipeline to check
		/// @return true if the pipeline is available
		bool is_pipeline_available(Name name) const;

		PipelineBaseInfo* get_pipeline(const PipelineBaseCreateInfo& pbci);
		/// @brief Reflect given pipeline base
		Program get_pipeline_reflection_info(const PipelineBaseCreateInfo& pbci);
		/// @brief Explicitly compile give ShaderSource into a ShaderModule
		ShaderModule compile_shader(ShaderSource source, std::string path);

		/// @brief Load a Vulkan pipeline cache
		bool load_pipeline_cache(std::span<std::byte> data);
		/// @brief Retrieve the current Vulkan pipeline cache
		std::vector<std::byte> save_pipeline_cache();

		// Allocator support

		/// @brief Return an allocator over the direct resource - resources will be allocated from the Vulkan runtime
		/// @return The resource
		DeviceVkResource& get_vk_resource();

		// Frame management

		/// @brief Retrieve the current frame count
		uint64_t get_frame_count() const;

		/// @brief Advance internal counter used for caching and garbage collect caches
		void next_frame();

		/// @brief Wait for the device to become idle. Useful for only a few synchronisation events, like resizing or shutting down.
		Result<void> wait_idle();

		Result<void> wait_for_domains(std::span<struct SyncPoint> sync_points);

		// Query functionality

		/// @brief Create a timestamp query to record timing information
		Query create_timestamp_query();

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

		/// @brief Acquire a cached sampler
		Sampler acquire_sampler(const SamplerCreateInfo& cu, uint64_t absolute_frame);
		/// @brief Acquire a cached descriptor pool
		struct DescriptorPool& acquire_descriptor_pool(const struct DescriptorSetLayoutAllocInfo& dslai, uint64_t absolute_frame);
		/// @brief Force collection of caches
		void collect(uint64_t frame);

		// Persistent descriptor sets

		Unique<PersistentDescriptorSet> create_persistent_descriptorset(Allocator& allocator, struct DescriptorSetLayoutCreateInfo dslci, unsigned num_descriptors);
		Unique<PersistentDescriptorSet> create_persistent_descriptorset(Allocator& allocator, const PipelineBaseInfo& base, unsigned set, unsigned num_descriptors);
		Unique<PersistentDescriptorSet> create_persistent_descriptorset(Allocator& allocator, const PersistentDescriptorSetCreateInfo&);

		// Misc.

		/// @brief Descriptor set strategy to use by default, can be overridden on the CommandBuffer
		DescriptorSetStrategyFlags default_descriptor_set_strategy = {};
		/// @brief Retrieve a unique uint64_t value
		uint64_t get_unique_handle_id();

		/// @brief Create a wrapped handle type (eg. a ImageView) from an externally sourced Vulkan handle
		/// @tparam T Vulkan handle type to wrap
		/// @param payload Vulkan handle to wrap
		/// @return The wrapped handle.
		template<class T>
		Handle<T> wrap(T payload);

	private:
		struct ContextImpl* impl;
		friend struct ContextImpl;

		// internal functions
		void destroy(const struct DescriptorPool& dp);
		void destroy(const ShaderModule& sm);
		void destroy(const DescriptorSetLayoutAllocInfo& ds);
		void destroy(const VkPipelineLayout& pl);
		void destroy(const DescriptorSet&);
		void destroy(const Sampler& sa);
		void destroy(const PipelineBaseInfo& pbi);

		ShaderModule create(const struct ShaderModuleCreateInfo& cinfo);
		PipelineBaseInfo create(const struct PipelineBaseCreateInfo& cinfo);
		VkPipelineLayout create(const struct PipelineLayoutCreateInfo& cinfo);
		DescriptorSetLayoutAllocInfo create(const struct DescriptorSetLayoutCreateInfo& cinfo);
		DescriptorPool create(const struct DescriptorSetLayoutAllocInfo& cinfo);
		Sampler create(const struct SamplerCreateInfo& cinfo);
	};

	template<class T>
	Handle<T> Context::wrap(T payload) {
		return { { get_unique_handle_id() }, payload };
	}

	template<class T>
	void Context::set_name(const T& t, Name name) {
		if (!debug_enabled())
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
		} else if constexpr (std::is_same_v<T, VkBuffer>) {
			info.objectType = VK_OBJECT_TYPE_BUFFER;
		} else if constexpr (std::is_same_v<T, VkQueue>) {
			info.objectType = VK_OBJECT_TYPE_QUEUE;
		}
		info.objectHandle = reinterpret_cast<uint64_t>(t);
		this->vkSetDebugUtilsObjectNameEXT(device, &info);
	}
} // namespace vuk

#include "vuk/Exception.hpp"
// utility functions
namespace vuk {
	struct ExecutableRenderGraph;

	/// @brief Compile & link given `RenderGraph`s, then execute them into API VkCommandBuffers, then submit them to queues
	/// @param allocator Allocator to use for submission resources
	/// @param rendergraphs `RenderGraph`s for compilation
	/// @param option Compilation options
	Result<void>
	link_execute_submit(Allocator& allocator, Compiler& compiler, std::span<std::shared_ptr<struct RG>> rendergraphs, RenderGraphCompileOptions options = {});
	/// @brief Execute given `ExecutableRenderGraph`s into API VkCommandBuffers, then submit them to queues
	/// @param allocator Allocator to use for submission resources
	/// @param executable_rendergraphs `ExecutableRenderGraph`s for execution
	/// @param swapchains_with_indexes Swapchains references by the rendergraphs
	/// @param present_rdy Semaphore used to gate device-side execution
	/// @param render_complete Semaphore used to gate presentation
	Result<void> execute_submit(Allocator& allocator, std::span<std::pair<Allocator*, ExecutableRenderGraph*>> executable_rendergraphs);

	/// @brief Execute given `ExecutableRenderGraph` into API VkCommandBuffers, then submit them to queues, then blocking-wait for the submission to complete
	/// @param allocator Allocator to use for submission resources
	/// @param executable_rendergraph `ExecutableRenderGraph`s for execution
	Result<void> execute_submit_and_wait(Allocator& allocator, ExecutableRenderGraph&& executable_rendergraph);

	struct RenderGraphCompileOptions;

	struct SampledImage make_sampled_image(ImageView iv, SamplerCreateInfo sci);

	struct SampledImage make_sampled_image(struct NameReference n, SamplerCreateInfo sci);

	struct SampledImage make_sampled_image(struct NameReference n, ImageViewCreateInfo ivci, SamplerCreateInfo sci);
} // namespace vuk