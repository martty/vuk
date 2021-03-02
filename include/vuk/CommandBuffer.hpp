#pragma once

#include <utility>
#include <optional>
#include <vuk/Config.hpp>
#include <Allocator.hpp>
#include <vuk/FixedVector.hpp>
#include <vuk/Types.hpp>
#include <vuk/Types.hpp>
#include <vuk/Image.hpp>
#include <vuk/Query.hpp>

namespace vuk {
	class Context;
	class PerThreadContext;

	struct Ignore {
		Ignore(size_t bytes) : format(vuk::Format::eUndefined), bytes((uint32_t)bytes) {}
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
		vuk::fixed_vector<FormatOrIgnore, VUK_MAX_ATTRIBUTES> list;
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

	class CommandBuffer {
	protected:
		friend struct ExecutableRenderGraph;
		ExecutableRenderGraph* rg = nullptr;
		vuk::PerThreadContext& ptc;
		VkCommandBuffer command_buffer;

		struct RenderPassInfo {
			VkRenderPass renderpass;
			uint32_t subpass;
			vuk::Extent2D extent;
			vuk::SampleCountFlagBits samples;
			std::span<const VkAttachmentReference> color_attachments;
		};
		std::optional<RenderPassInfo> ongoing_renderpass;
		PassInfo* current_pass = nullptr;
		vuk::PrimitiveTopology topology = vuk::PrimitiveTopology::eTriangleList;
		vuk::fixed_vector<vuk::VertexInputAttributeDescription, VUK_MAX_ATTRIBUTES> attribute_descriptions;
		vuk::fixed_vector<VkVertexInputBindingDescription, VUK_MAX_ATTRIBUTES> binding_descriptions;
		vuk::fixed_vector<VkPushConstantRange, VUK_MAX_PUSHCONSTANT_RANGES> pcrs;

		std::array<unsigned char, 128> push_constant_buffer;

		vuk::fixed_vector<std::pair<VkSpecializationMapEntry, VkShaderStageFlags>, VUK_MAX_SPECIALIZATIONCONSTANT_RANGES> smes;
		std::array<unsigned char, 64> specialization_constant_buffer;

		std::optional<vuk::PipelineColorBlendAttachmentState> blend_state_override;
		std::optional<std::array<float, 4>> blend_constants;

		vuk::PipelineBaseInfo* next_pipeline = nullptr;
		vuk::ComputePipelineInfo* next_compute_pipeline = nullptr;
		std::optional<vuk::PipelineInfo> current_pipeline;
		std::optional<vuk::ComputePipelineInfo> current_compute_pipeline;

		std::bitset<VUK_MAX_SETS> sets_used = {};
		std::array<SetBinding, VUK_MAX_SETS> set_bindings = {};
		std::bitset<VUK_MAX_SETS> persistent_sets_used = {};
		std::array<VkDescriptorSet, VUK_MAX_SETS> persistent_sets = {};

		// for rendergraph
		CommandBuffer(ExecutableRenderGraph& rg, vuk::PerThreadContext& ptc, VkCommandBuffer cb) : rg(&rg), ptc(ptc), command_buffer(cb) {}
		CommandBuffer(ExecutableRenderGraph& rg, vuk::PerThreadContext& ptc, VkCommandBuffer cb, std::optional<RenderPassInfo> ongoing) : rg(&rg), ptc(ptc), command_buffer(cb), ongoing_renderpass(ongoing) {}
	public:
		// for one shot
		CommandBuffer(vuk::PerThreadContext& ptc);
		// for secondary cbufs
		CommandBuffer(ExecutableRenderGraph* rg, vuk::PerThreadContext& ptc, VkCommandBuffer cb, std::optional<RenderPassInfo> ongoing) : rg(rg), ptc(ptc), command_buffer(cb), ongoing_renderpass(ongoing) {}

		vuk::PerThreadContext& get_context() {
			return ptc;
		}
		const RenderPassInfo& get_ongoing_renderpass() const;
		vuk::Buffer get_resource_buffer(Name) const;
		vuk::Image get_resource_image(Name) const;
		vuk::ImageView get_resource_image_view(Name) const;

		CommandBuffer& set_viewport(unsigned index, Viewport vp);
		CommandBuffer& set_viewport(unsigned index, Rect2D area, float min_depth = 0.f, float max_depth = 1.f);
		CommandBuffer& set_scissor(unsigned index, Rect2D vp);

		CommandBuffer& set_blend_state(vuk::PipelineColorBlendAttachmentState);
		CommandBuffer& set_blend_constants(std::array<float, 4> constants);

		CommandBuffer& bind_graphics_pipeline(vuk::PipelineBaseInfo*);
		CommandBuffer& bind_graphics_pipeline(Name);

		CommandBuffer& bind_compute_pipeline(vuk::ComputePipelineInfo*);
		CommandBuffer& bind_compute_pipeline(Name);

		CommandBuffer& set_primitive_topology(vuk::PrimitiveTopology);
		CommandBuffer& bind_vertex_buffer(unsigned binding, const Buffer&, unsigned first_location, Packed);
		CommandBuffer& bind_vertex_buffer(unsigned binding, const Buffer&, std::span<vuk::VertexInputAttributeDescription>, uint32_t stride);
		CommandBuffer& bind_index_buffer(const Buffer&, vuk::IndexType type);

		CommandBuffer& bind_sampled_image(unsigned set, unsigned binding, vuk::ImageView iv, vuk::SamplerCreateInfo sampler_create_info, vuk::ImageLayout = vuk::ImageLayout::eShaderReadOnlyOptimal);
		CommandBuffer& bind_sampled_image(unsigned set, unsigned binding, const vuk::Texture&, vuk::SamplerCreateInfo sampler_create_info, vuk::ImageLayout = vuk::ImageLayout::eShaderReadOnlyOptimal);
		CommandBuffer& bind_sampled_image(unsigned set, unsigned binding, Name, vuk::SamplerCreateInfo sampler_create_info);
		CommandBuffer& bind_sampled_image(unsigned set, unsigned binding, Name, vuk::ImageViewCreateInfo ivci, vuk::SamplerCreateInfo sampler_create_info);

		CommandBuffer& bind_persistent(unsigned set, PersistentDescriptorSet&);

		CommandBuffer& push_constants(vuk::ShaderStageFlags stages, size_t offset, void* data, size_t size);
		template<class T>
		CommandBuffer& push_constants(vuk::ShaderStageFlags stages, size_t offset, std::span<T> span);
		template<class T>
		CommandBuffer& push_constants(vuk::ShaderStageFlags stages, size_t offset, T value);

		CommandBuffer& specialization_constants(unsigned constant_id, vuk::ShaderStageFlags stages, size_t offset, void* data, size_t size);
		template<class T>
		CommandBuffer& specialization_constants(unsigned constant_id, vuk::ShaderStageFlags stages, size_t offset, std::span<T> span);
		template<class T>
		CommandBuffer& specialization_constants(unsigned constant_id, vuk::ShaderStageFlags stages, size_t offset, T value);

		CommandBuffer& bind_uniform_buffer(unsigned set, unsigned binding, Buffer buffer);
		CommandBuffer& bind_storage_buffer(unsigned set, unsigned binding, Buffer buffer);

		CommandBuffer& bind_storage_image(unsigned set, unsigned binding, vuk::ImageView image_view);
		CommandBuffer& bind_storage_image(unsigned set, unsigned binding, Name);

		void* _map_scratch_uniform_binding(unsigned set, unsigned binding, size_t size);

		template<class T>
		T* map_scratch_uniform_binding(unsigned set, unsigned binding);

		CommandBuffer& draw(size_t vertex_count, size_t instance_count, size_t first_vertex, size_t first_instance);
		CommandBuffer& draw_indexed(size_t index_count, size_t instance_count, size_t first_index, int32_t vertex_offset, size_t first_instance);

		CommandBuffer& draw_indexed_indirect(size_t command_count, Buffer indirect_buffer);
		CommandBuffer& draw_indexed_indirect(std::span<vuk::DrawIndexedIndirectCommand>);

		CommandBuffer& draw_indexed_indirect_count(size_t max_draw_count, Buffer indirect_buffer, Buffer count_buffer);

		CommandBuffer& dispatch(size_t group_count_x, size_t group_count_y = 1, size_t group_count_z = 1);
		// Perform a dispatch while specifying the minimum invocation count
		// Actual invocation count will be rounded up to be a multiple of local_size_{x,y,z}
		CommandBuffer& dispatch_invocations(size_t invocation_count_x, size_t invocation_count_y = 1, size_t invocation_count_z = 1);

		class SecondaryCommandBuffer begin_secondary();
		void execute(std::span<VkCommandBuffer>);

		// commands for renderpass-less command buffers
		void clear_image(Name src, Clear);
		void resolve_image(Name src, Name dst);
		void blit_image(Name src, Name dst, vuk::ImageBlit region, vuk::Filter filter);
		void copy_image_to_buffer(Name src, Name dst, vuk::BufferImageCopy);

		// explicit synchronisation
		void image_barrier(Name, vuk::Access src_access, vuk::Access dst_access);

		// queries
		void write_timestamp(Query, vuk::PipelineStageFlagBits stage = vuk::PipelineStageFlagBits::eBottomOfPipe);

	protected:
		void _bind_state(bool graphics);
		void _bind_compute_pipeline_state();
		void _bind_graphics_pipeline_state();
	};

	class SecondaryCommandBuffer : public CommandBuffer {
	public:
		using CommandBuffer::CommandBuffer;
		VkCommandBuffer get_buffer();

		~SecondaryCommandBuffer();
	};

	template<class T>
	inline CommandBuffer& CommandBuffer::push_constants(vuk::ShaderStageFlags stages, size_t offset, std::span<T> span) {
		return push_constants(stages, offset, (void*)span.data(), sizeof(T) * span.size());
	}

	template<class T>
	inline CommandBuffer& CommandBuffer::push_constants(vuk::ShaderStageFlags stages, size_t offset, T value) {
		return push_constants(stages, offset, (void*)&value, sizeof(T));
	}

	template<class T>
	inline CommandBuffer& CommandBuffer::specialization_constants(unsigned constant_id, vuk::ShaderStageFlags stages, size_t offset, std::span<T> span) {
		return specialization_constants(constant_id, stages, offset, (void*)span.data(), sizeof(T) * span.size());
	}

	template<class T>
	inline CommandBuffer& CommandBuffer::specialization_constants(unsigned constant_id, vuk::ShaderStageFlags stages, size_t offset, T value) {
		return specialization_constants(constant_id, stages, offset, (void*)&value, sizeof(T));
	}

	template<class T>
	inline T* CommandBuffer::map_scratch_uniform_binding(unsigned set, unsigned binding) {
		return static_cast<T*>(_map_scratch_uniform_binding(set, binding, sizeof(T)));
	}

	struct TimedScope {
		TimedScope(CommandBuffer& cbuf, Query a, Query b) : cbuf(cbuf), a(a), b(b) {
			cbuf.write_timestamp(a, vuk::PipelineStageFlagBits::eBottomOfPipe);
		}

		~TimedScope() {
			cbuf.write_timestamp(b, vuk::PipelineStageFlagBits::eBottomOfPipe);
		}

		CommandBuffer& cbuf;
		Query a;
		Query b;
	};
}

