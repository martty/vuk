#pragma once

#include <utility>
#include <optional>
#include <vuk/Config.hpp>
#include <vuk/vuk_fwd.hpp>
#include <vuk/FixedVector.hpp>
#include <vuk/Types.hpp>
#include <vuk/Image.hpp>
#include <vuk/Query.hpp>
#include <vuk/PipelineInstance.hpp>
#include "vuk/Exception.hpp"

namespace vuk {
	class Context;

	struct Ignore {
		Ignore(size_t bytes) : format(Format::eUndefined), bytes((uint32_t)bytes) {}
		Ignore(Format format) : format(format) {}
		Format format;
		uint32_t bytes = 0;

		uint32_t to_size();
	};

	struct FormatOrIgnore {
		FormatOrIgnore(Format format);
		FormatOrIgnore(Ignore ign);

		bool ignore;
		Format format;
		uint32_t size;
	};

	struct Packed {
		Packed() {}
		Packed(std::initializer_list<FormatOrIgnore> ilist) : list(ilist) {}
		fixed_vector<FormatOrIgnore, VUK_MAX_ATTRIBUTES> list;
	};

	struct DrawIndexedIndirectCommand {
		uint32_t indexCount = {};
		uint32_t instanceCount = {};
		uint32_t firstIndex = {};
		int32_t vertexOffset = {};
		uint32_t firstInstance = {};

		operator VkDrawIndexedIndirectCommand const& () const noexcept {
			return *reinterpret_cast<const VkDrawIndexedIndirectCommand*>(this);
		}

		operator VkDrawIndexedIndirectCommand& () noexcept {
			return *reinterpret_cast<VkDrawIndexedIndirectCommand*>(this);
		}

		bool operator==(DrawIndexedIndirectCommand const& rhs) const noexcept {
			return (indexCount == rhs.indexCount)
				&& (instanceCount == rhs.instanceCount)
				&& (firstIndex == rhs.firstIndex)
				&& (vertexOffset == rhs.vertexOffset)
				&& (firstInstance == rhs.firstInstance);
		}

		bool operator!=(DrawIndexedIndirectCommand const& rhs) const noexcept {
			return !operator==(rhs);
		}
	};
	static_assert(sizeof(DrawIndexedIndirectCommand) == sizeof(VkDrawIndexedIndirectCommand), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<DrawIndexedIndirectCommand>::value, "struct wrapper is not a standard layout!");

	struct ImageSubresourceLayers {
		ImageAspectFlags aspectMask = {};
		uint32_t mipLevel = 0;
		uint32_t baseArrayLayer = 0;
		uint32_t layerCount = 1;

		operator VkImageSubresourceLayers const& () const noexcept {
			return *reinterpret_cast<const VkImageSubresourceLayers*>(this);
		}

		operator VkImageSubresourceLayers& () noexcept {
			return *reinterpret_cast<VkImageSubresourceLayers*>(this);
		}

		bool operator==(ImageSubresourceLayers const& rhs) const noexcept {
			return (aspectMask == rhs.aspectMask)
				&& (mipLevel == rhs.mipLevel)
				&& (baseArrayLayer == rhs.baseArrayLayer)
				&& (layerCount == rhs.layerCount);
		}

		bool operator!=(ImageSubresourceLayers const& rhs) const noexcept {
			return !operator==(rhs);
		}

	};
	static_assert(sizeof(ImageSubresourceLayers) == sizeof(VkImageSubresourceLayers), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<ImageSubresourceLayers>::value, "struct wrapper is not a standard layout!");


	struct ImageBlit {
		ImageSubresourceLayers srcSubresource = {};
		std::array<Offset3D, 2> srcOffsets = {};
		ImageSubresourceLayers dstSubresource = {};
		std::array<Offset3D, 2> dstOffsets = {};

		operator VkImageBlit const& () const noexcept {
			return *reinterpret_cast<const VkImageBlit*>(this);
		}

		operator VkImageBlit& () noexcept {
			return *reinterpret_cast<VkImageBlit*>(this);
		}

		bool operator==(ImageBlit const& rhs) const noexcept {
			return (srcSubresource == rhs.srcSubresource)
				&& (srcOffsets == rhs.srcOffsets)
				&& (dstSubresource == rhs.dstSubresource)
				&& (dstOffsets == rhs.dstOffsets);
		}

		bool operator!=(ImageBlit const& rhs) const noexcept {
			return !operator==(rhs);
		}
	};
	static_assert(sizeof(ImageBlit) == sizeof(VkImageBlit), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<ImageBlit>::value, "struct wrapper is not a standard layout!");

	struct BufferImageCopy {
		VkDeviceSize bufferOffset = {};
		uint32_t bufferRowLength = {};
		uint32_t bufferImageHeight = {};
		ImageSubresourceLayers imageSubresource = {};
		Offset3D imageOffset = {};
		Extent3D imageExtent = {};


		operator VkBufferImageCopy const& () const noexcept {
			return *reinterpret_cast<const VkBufferImageCopy*>(this);
		}

		operator VkBufferImageCopy& () noexcept {
			return *reinterpret_cast<VkBufferImageCopy*>(this);
		}

		bool operator==(BufferImageCopy const& rhs) const noexcept {
			return (bufferOffset == rhs.bufferOffset)
				&& (bufferRowLength == rhs.bufferRowLength)
				&& (bufferImageHeight == rhs.bufferImageHeight)
				&& (imageSubresource == rhs.imageSubresource)
				&& (imageOffset == rhs.imageOffset)
				&& (imageExtent == rhs.imageExtent);
		}

		bool operator!=(BufferImageCopy const& rhs) const noexcept {
			return !operator==(rhs);
		}
	};
	static_assert(sizeof(BufferImageCopy) == sizeof(VkBufferImageCopy), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<BufferImageCopy>::value, "struct wrapper is not a standard layout!");


	struct ExecutableRenderGraph;
	struct PassInfo;
	struct Query;
	class Allocator;

	class CommandBuffer {
	protected:
		friend struct ExecutableRenderGraph;
		ExecutableRenderGraph* rg = nullptr;
		Context& ctx;
		Allocator* allocator;
		CommandBufferAllocation command_buffer_allocation;
		VkCommandBuffer command_buffer;

		struct RenderPassInfo {
			VkRenderPass renderpass;
			uint32_t subpass;
			Extent2D extent;
			SampleCountFlagBits samples;
			VkAttachmentReference const* depth_stencil_attachment;
			std::array<Name, VUK_MAX_COLOR_ATTACHMENTS> color_attachment_names;
			std::span<const VkAttachmentReference> color_attachments;
		};
		std::optional<RenderPassInfo> ongoing_renderpass;
		PassInfo* current_pass = nullptr;

		mutable bool extracted = false;
		Exception* current_exception = nullptr;
		std::optional<AllocateException> allocate_except;
		std::optional<RenderGraphException> rg_except;

		// Pipeline state
		// Enabled dynamic state
		DynamicStateFlags dynamic_state_flags = {};

		// Current & next graphics & compute pipelines
		PipelineBaseInfo* next_pipeline = nullptr;
		ComputePipelineBaseInfo* next_compute_pipeline = nullptr;
		std::optional<PipelineInfo> current_pipeline;
		std::optional<ComputePipelineInfo> current_compute_pipeline;

		// Input assembly & fixed-function attributes
		PrimitiveTopology topology = PrimitiveTopology::eTriangleList;
		Bitset<VUK_MAX_ATTRIBUTES> set_attribute_descriptions = {};
		std::array<VertexInputAttributeDescription, VUK_MAX_ATTRIBUTES> attribute_descriptions;
		Bitset<VUK_MAX_ATTRIBUTES> set_binding_descriptions = {};
		std::array<VkVertexInputBindingDescription, VUK_MAX_ATTRIBUTES> binding_descriptions;

		// Specialization constant support
		struct SpecEntry {
			bool is_double;
			std::byte data[sizeof(double)];
		};
		robin_hood::unordered_flat_map<uint32_t, SpecEntry> spec_map_entries; // constantID -> SpecEntry

		// Individual pipeline states
		std::optional<PipelineRasterizationStateCreateInfo> rasterization_state;
		std::optional<PipelineDepthStencilStateCreateInfo> depth_stencil_state;
		bool broadcast_color_blend_attachment_0 = false;
		Bitset<VUK_MAX_COLOR_ATTACHMENTS> set_color_blend_attachments = {};
		fixed_vector<PipelineColorBlendAttachmentState, VUK_MAX_COLOR_ATTACHMENTS> color_blend_attachments;
		std::optional<std::array<float, 4>> blend_constants;
		float line_width = 1.0f;
		fixed_vector<VkViewport, VUK_MAX_VIEWPORTS> viewports;
		fixed_vector<VkRect2D, VUK_MAX_SCISSORS> scissors;

		// Push constants
		std::array<unsigned char, 128> push_constant_buffer;
		fixed_vector<VkPushConstantRange, VUK_MAX_PUSHCONSTANT_RANGES> pcrs;

		// Descriptor sets
		std::bitset<VUK_MAX_SETS> sets_used = {};
		std::array<SetBinding, VUK_MAX_SETS> set_bindings = {};
		std::bitset<VUK_MAX_SETS> persistent_sets_used = {};
		std::array<VkDescriptorSet, VUK_MAX_SETS> persistent_sets = {};


		// for rendergraph
		CommandBuffer(ExecutableRenderGraph& rg, Context& ctx, Allocator& alloc, VkCommandBuffer cb) : rg(&rg), ctx(ctx), allocator(&alloc), command_buffer(cb) {}
		CommandBuffer(ExecutableRenderGraph& rg, Context& ctx, Allocator& alloc, VkCommandBuffer cb, std::optional<RenderPassInfo> ongoing) : rg(&rg), ctx(ctx), allocator(&alloc), command_buffer(cb), ongoing_renderpass(ongoing) {}
	public:
		// for secondary cbufs
		CommandBuffer(ExecutableRenderGraph* rg, Context& ctx, VkCommandBuffer cb, std::optional<RenderPassInfo> ongoing) : rg(rg), ctx(ctx), command_buffer(cb), ongoing_renderpass(ongoing) {}

		Context& get_context() {
			return ctx;
		}
		const RenderPassInfo& get_ongoing_renderpass() const;
		Result<Buffer> get_resource_buffer(Name) const;
		Result<Image> get_resource_image(Name) const;
		Result<ImageView> get_resource_image_view(Name) const;

		// request dynamic state for the subsequent pipelines
		CommandBuffer& set_dynamic_state(DynamicStateFlags);
		// command buffer state setting
		// when a state is set it is persistent for a pass - similar to vulkan dynamic state
		CommandBuffer& set_viewport(unsigned index, Viewport vp);
		CommandBuffer& set_viewport(unsigned index, Rect2D area, float min_depth = 0.f, float max_depth = 1.f);
		CommandBuffer& set_scissor(unsigned index, Rect2D vp);

		CommandBuffer& set_rasterization(PipelineRasterizationStateCreateInfo);
		CommandBuffer& set_depth_stencil(PipelineDepthStencilStateCreateInfo);

		CommandBuffer& broadcast_color_blend(PipelineColorBlendAttachmentState);
		CommandBuffer& broadcast_color_blend(BlendPreset);
		CommandBuffer& set_color_blend(Name color_attachment, PipelineColorBlendAttachmentState);
		CommandBuffer& set_color_blend(Name color_attachment, BlendPreset);
		CommandBuffer& set_blend_constants(std::array<float, 4> constants);

		CommandBuffer& bind_graphics_pipeline(PipelineBaseInfo*);
		CommandBuffer& bind_graphics_pipeline(Name);

		CommandBuffer& bind_compute_pipeline(ComputePipelineBaseInfo*);
		CommandBuffer& bind_compute_pipeline(Name);

		CommandBuffer& set_primitive_topology(PrimitiveTopology);

		/// @brief Binds a vertex buffer to the given binding point and configures attributes sourced from this buffer based on a packed format list, the attribute locations are offset with first_location
		/// @param binding The binding point of the buffer
		/// @param buffer The buffer to be bound
		/// @param first_location First location assigned to the attributes
		/// @param format_list List of formats packed in buffer to generate attributes from
		CommandBuffer& bind_vertex_buffer(unsigned binding, const Buffer& buffer, unsigned first_location, Packed format_list);
		/// @brief Binds a vertex buffer to the given binding point and configures attributes sourced from this buffer based on a span of attribute descriptions and stride
		/// @param binding The binding point of the buffer
		/// @param buffer The buffer to be bound
		/// @param attribute_descriptions Attributes that are sourced from this buffer
		/// @param stride Stride of a vertex sourced from this buffer
		CommandBuffer& bind_vertex_buffer(unsigned binding, const Buffer& buffer, std::span<VertexInputAttributeDescription> attribute_descriptions, uint32_t stride);
		/// @brief Binds an index buffer with the given type
		/// @param buffer The buffer to be bound
		/// @param type The index type in the buffer
		CommandBuffer& bind_index_buffer(const Buffer& buffer, IndexType type);

		CommandBuffer& bind_sampled_image(unsigned set, unsigned binding, ImageView iv, SamplerCreateInfo sampler_create_info, ImageLayout = ImageLayout::eShaderReadOnlyOptimal);
		CommandBuffer& bind_sampled_image(unsigned set, unsigned binding, const Texture&, SamplerCreateInfo sampler_create_info, ImageLayout = ImageLayout::eShaderReadOnlyOptimal);
		CommandBuffer& bind_sampled_image(unsigned set, unsigned binding, Name, SamplerCreateInfo sampler_create_info);
		CommandBuffer& bind_sampled_image(unsigned set, unsigned binding, Name, ImageViewCreateInfo ivci, SamplerCreateInfo sampler_create_info);

		/// @brief Bind a persistent descriptor set to the command buffer
		/// @param set The set bind index to be used
		/// @param desc_set The persistent descriptor set to be bound
		CommandBuffer& bind_persistent(unsigned set, PersistentDescriptorSet& desc_set);

		CommandBuffer& push_constants(ShaderStageFlags stages, size_t offset, void* data, size_t size);
		template<class T>
		CommandBuffer& push_constants(ShaderStageFlags stages, size_t offset, std::span<T> span);
		template<class T>
		CommandBuffer& push_constants(ShaderStageFlags stages, size_t offset, T value);

		/// @brief Set specialization constants for the command buffer
		/// @param constant_id ID of the constant. All stages form a single namespace for IDs.
		/// @param value Value of the specialization constant
		CommandBuffer& specialize_constants(uint32_t constant_id, bool value);
		/// @brief Set specialization constants for the command buffer
		/// @param constant_id ID of the constant. All stages form a single namespace for IDs.
		/// @param value Value of the specialization constant
		CommandBuffer& specialize_constants(uint32_t constant_id, uint32_t value);
		/// @brief Set specialization constants for the command buffer
		/// @param constant_id ID of the constant. All stages form a single namespace for IDs.
		/// @param value Value of the specialization constant
		CommandBuffer& specialize_constants(uint32_t constant_id, int32_t value);
		/// @brief Set specialization constants for the command buffer
		/// @param constant_id ID of the constant. All stages form a single namespace for IDs.
		/// @param value Value of the specialization constant
		CommandBuffer& specialize_constants(uint32_t constant_id, float value);
		/// @brief Set specialization constants for the command buffer
		/// @param constant_id ID of the constant. All stages form a single namespace for IDs.
		/// @param value Value of the specialization constant
		CommandBuffer& specialize_constants(uint32_t constant_id, double value);

		/// @brief Bind a uniform buffer to the command buffer
		/// @param set The set bind index to be used
		/// @param binding The descriptor binding to bind the buffer to
		/// @param buffer The buffer to be bound
		CommandBuffer& bind_uniform_buffer(unsigned set, unsigned binding, const Buffer& buffer);
		/// @brief Bind a storage buffer to the command buffer
		/// @param set The set bind index to be used
		/// @param binding The descriptor binding to bind the buffer to
		/// @param buffer The buffer to be bound
		CommandBuffer& bind_storage_buffer(unsigned set, unsigned binding, const Buffer& buffer);

		CommandBuffer& bind_storage_image(unsigned set, unsigned binding, ImageView image_view);
		CommandBuffer& bind_storage_image(unsigned set, unsigned binding, Name);

		/// @brief Allocate some CPUtoGPU memory and bind it as a uniform. Return a pointer to the mapped memory.
		/// @param set The set bind index to be used
		/// @param binding The descriptor binding to bind the buffer to
		/// @param size Amount of memory to allocate
		/// @return pointer to the mapped host-visible memory. Null pointer if the command buffer has errored out previously or the allocation failed
		void* _map_scratch_uniform_binding(unsigned set, unsigned binding, size_t size);

		/// @brief Allocate some typed CPUtoGPU memory and bind it as a uniform. Return a pointer to the mapped memory.
		/// @tparam T Type of the uniform to write
		/// @param set The set bind index to be used
		/// @param binding The descriptor binding to bind the buffer to
		/// @return pointer to the mapped host-visible memory. Null pointer if the command buffer has errored out previously or the allocation failed
		template<class T>
		T* map_scratch_uniform_binding(unsigned set, unsigned binding);

		CommandBuffer& draw(size_t vertex_count, size_t instance_count, size_t first_vertex, size_t first_instance);
		CommandBuffer& draw_indexed(size_t index_count, size_t instance_count, size_t first_index, int32_t vertex_offset, size_t first_instance);

		CommandBuffer& draw_indexed_indirect(size_t command_count, const Buffer& indirect_buffer);
		CommandBuffer& draw_indexed_indirect(std::span<DrawIndexedIndirectCommand>);

		CommandBuffer& draw_indexed_indirect_count(size_t max_draw_count, const Buffer& indirect_buffer, const Buffer& count_buffer);

		CommandBuffer& dispatch(size_t group_count_x, size_t group_count_y = 1, size_t group_count_z = 1);
		// Perform a dispatch while specifying the minimum invocation count
		// Actual invocation count will be rounded up to be a multiple of local_size_{x,y,z}
		CommandBuffer& dispatch_invocations(size_t invocation_count_x, size_t invocation_count_y = 1, size_t invocation_count_z = 1);

		CommandBuffer& dispatch_indirect(const Buffer& indirect_buffer);

		Result<class SecondaryCommandBuffer> begin_secondary();
		CommandBuffer& execute(std::span<VkCommandBuffer>);

		// commands for renderpass-less command buffers
		CommandBuffer& clear_image(Name src, Clear);
		CommandBuffer& resolve_image(Name src, Name dst);
		CommandBuffer& blit_image(Name src, Name dst, ImageBlit region, Filter filter);
		CommandBuffer& copy_image_to_buffer(Name src, Name dst, BufferImageCopy);
		CommandBuffer& copy_buffer(Name src, Name dst, size_t size);

		// explicit synchronisation
		CommandBuffer& image_barrier(Name, Access src_access, Access dst_access, uint32_t mip_level = 0, uint32_t mip_count = VK_REMAINING_MIP_LEVELS);

		// queries
		CommandBuffer& write_timestamp(Query, PipelineStageFlagBits stage = PipelineStageFlagBits::eBottomOfPipe);

		bool has_error() const noexcept {
			return current_exception != nullptr;
		}

		[[nodiscard]] Exception& error()&;

		[[nodiscard]] Exception const& error() const&;

		[[nodiscard]] Exception&& error()&&;

	protected:
		[[nodiscard]] bool _bind_state(bool graphics);
		[[nodiscard]] bool _bind_compute_pipeline_state();
		[[nodiscard]] bool _bind_graphics_pipeline_state();

		CommandBuffer& specialize_constants(uint32_t constant_id, void* data, size_t size);
	};

	class SecondaryCommandBuffer : public CommandBuffer {
	public:
		using CommandBuffer::CommandBuffer;
		VkCommandBuffer get_buffer();

		~SecondaryCommandBuffer();
	};

	template<class T>
	inline CommandBuffer& CommandBuffer::push_constants(ShaderStageFlags stages, size_t offset, std::span<T> span) {
		return push_constants(stages, offset, (void*)span.data(), sizeof(T) * span.size());
	}

	template<class T>
	inline CommandBuffer& CommandBuffer::push_constants(ShaderStageFlags stages, size_t offset, T value) {
		return push_constants(stages, offset, (void*)&value, sizeof(T));
	}

	inline CommandBuffer& CommandBuffer::specialize_constants(uint32_t constant_id, bool value) {
		return specialize_constants(constant_id, (uint32_t)value);
	}

	inline CommandBuffer& CommandBuffer::specialize_constants(uint32_t constant_id, uint32_t value) {
		return specialize_constants(constant_id, (void*)&value, sizeof(uint32_t));
	}

	inline CommandBuffer& CommandBuffer::specialize_constants(uint32_t constant_id, int32_t value) {
		return specialize_constants(constant_id, (void*)&value, sizeof(int32_t));
	}

	inline CommandBuffer& CommandBuffer::specialize_constants(uint32_t constant_id, float value) {
		return specialize_constants(constant_id, (void*)&value, sizeof(float));
	}

	inline CommandBuffer& CommandBuffer::specialize_constants(uint32_t constant_id, double value) {
		return specialize_constants(constant_id, (void*)&value, sizeof(double));
	}

	template<class T>
	inline T* CommandBuffer::map_scratch_uniform_binding(unsigned set, unsigned binding) {
		return static_cast<T*>(_map_scratch_uniform_binding(set, binding, sizeof(T)));
	}

	struct TimedScope {
		TimedScope(CommandBuffer& cbuf, Query a, Query b) : cbuf(cbuf), a(a), b(b) {
			cbuf.write_timestamp(a, PipelineStageFlagBits::eBottomOfPipe);
		}

		~TimedScope() {
			cbuf.write_timestamp(b, PipelineStageFlagBits::eBottomOfPipe);
		}

		CommandBuffer& cbuf;
		Query a;
		Query b;
	};
}
