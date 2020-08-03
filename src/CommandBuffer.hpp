#pragma once

#include <vulkan/vulkan.hpp>
#include <utility>
#include <optional>
#include "Allocator.hpp"
#include "FixedVector.hpp"
#define VUK_MAX_SETS 8
#define VUK_MAX_ATTRIBUTES 8
#define VUK_MAX_PUSHCONSTANT_RANGES 8

namespace vuk {
	class Context;
	class PerThreadContext;

	struct Area {
		Area(int32_t x, int32_t y, uint32_t width, uint32_t height) : offset{ x, y }, extent { width, height } {}

		struct Framebuffer {
			float x = 0.f;
			float y = 0.f;
			float width = 1.0f;
			float height = 1.0f;
		};

		vk::Offset2D offset;
		vk::Extent2D extent;
	};

	struct Ignore {
		Ignore(size_t bytes) : format(vk::Format::eUndefined), bytes((uint32_t)bytes) {}
		Ignore(vk::Format format) : format(format) {}
		vk::Format format;
		uint32_t bytes = 0;

		uint32_t to_size();
	};

	struct FormatOrIgnore {
		FormatOrIgnore(vk::Format format);
		FormatOrIgnore(Ignore ign);

		bool ignore;
		vk::Format format;
		uint32_t size;
	};

	struct Packed {
		Packed(std::initializer_list<FormatOrIgnore> ilist) : list(ilist) {}
		vuk::fixed_vector<FormatOrIgnore, VUK_MAX_ATTRIBUTES> list;
	};

	struct RenderGraph;
	class CommandBuffer {
    protected:
		friend struct RenderGraph;
		RenderGraph& rg;
		vuk::PerThreadContext& ptc;
		vk::CommandBuffer command_buffer;
		
		struct RenderPassInfo {
			vk::RenderPass renderpass;
			uint32_t subpass;
			vk::Extent2D extent;
			vk::SampleCountFlagBits samples;
			std::span<const vk::AttachmentReference> color_attachments;
		};
		std::optional<RenderPassInfo> ongoing_renderpass;
        vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList;
		vuk::fixed_vector<vk::VertexInputAttributeDescription, VUK_MAX_ATTRIBUTES> attribute_descriptions;
		vuk::fixed_vector<vk::VertexInputBindingDescription, VUK_MAX_ATTRIBUTES> binding_descriptions;
		vuk::fixed_vector<vk::PushConstantRange, VUK_MAX_PUSHCONSTANT_RANGES> pcrs;
		std::array<unsigned char, 64> push_constant_buffer;
		vuk::PipelineBaseInfo* next_pipeline = nullptr;
		vuk::ComputePipelineInfo* next_compute_pipeline = nullptr;
		std::optional<vuk::PipelineInfo> current_pipeline;
		std::optional<vuk::ComputePipelineInfo> current_compute_pipeline;
		std::bitset<VUK_MAX_SETS> sets_used = {};
		std::array<SetBinding, VUK_MAX_SETS> set_bindings = {};
	public:
		CommandBuffer(RenderGraph& rg, vuk::PerThreadContext& ptc, vk::CommandBuffer cb) : rg(rg), ptc(ptc), command_buffer(cb) {}
		CommandBuffer(RenderGraph& rg, vuk::PerThreadContext& ptc, vk::CommandBuffer cb, std::optional<RenderPassInfo> ongoing) : rg(rg), ptc(ptc), command_buffer(cb), ongoing_renderpass(ongoing) {}

		vuk::PerThreadContext& get_context() {
            return ptc;
        }

		const RenderPassInfo& get_ongoing_renderpass() const;
		vuk::Buffer get_resource_buffer(Name) const;
		vk::Image get_resource_image(Name) const;
		vuk::ImageView get_resource_image_view(Name) const;

		CommandBuffer& set_viewport(unsigned index, vk::Viewport vp);	
		CommandBuffer& set_viewport(unsigned index, Area area);
		CommandBuffer& set_viewport(unsigned index, Area::Framebuffer area);
		CommandBuffer& set_scissor(unsigned index, vk::Rect2D vp);
		CommandBuffer& set_scissor(unsigned index, Area area);
		CommandBuffer& set_scissor(unsigned index, Area::Framebuffer area);

		CommandBuffer& bind_graphics_pipeline(vuk::PipelineBaseInfo*);
		CommandBuffer& bind_graphics_pipeline(Name);

		CommandBuffer& bind_compute_pipeline(vuk::ComputePipelineInfo*);
		CommandBuffer& bind_compute_pipeline(Name);

		CommandBuffer& set_primitive_topology(vk::PrimitiveTopology);
		CommandBuffer& bind_vertex_buffer(unsigned binding, const Buffer&, unsigned first_location, Packed);
		CommandBuffer& bind_vertex_buffer(unsigned binding, const Buffer&, std::span<vk::VertexInputAttributeDescription>, uint32_t stride);
		CommandBuffer& bind_index_buffer(const Buffer&, vk::IndexType type);

		CommandBuffer& bind_sampled_image(unsigned set, unsigned binding, vuk::ImageView iv, vk::SamplerCreateInfo sampler_create_info);
		CommandBuffer& bind_sampled_image(unsigned set, unsigned binding, const vuk::Texture&, vk::SamplerCreateInfo sampler_create_info);
		CommandBuffer& bind_sampled_image(unsigned set, unsigned binding, Name, vk::SamplerCreateInfo sampler_create_info);
		CommandBuffer& bind_sampled_image(unsigned set, unsigned binding, Name, vk::ImageViewCreateInfo ivci, vk::SamplerCreateInfo sampler_create_info);
		
		CommandBuffer& push_constants(vk::ShaderStageFlags stages, size_t offset, void * data, size_t size);
		template<class T>
		CommandBuffer& push_constants(vk::ShaderStageFlags stages, size_t offset, std::span<T> span);
		template<class T>
		CommandBuffer& push_constants(vk::ShaderStageFlags stages, size_t offset, T value);

		CommandBuffer& bind_uniform_buffer(unsigned set, unsigned binding, Buffer buffer);
        CommandBuffer& bind_storage_buffer(unsigned set, unsigned binding, Buffer buffer);

		void* _map_scratch_uniform_binding(unsigned set, unsigned binding, size_t size);

		template<class T>
		T* map_scratch_uniform_binding(unsigned set, unsigned binding);

		CommandBuffer& draw(size_t vertex_count, size_t instance_count, size_t first_vertex, size_t first_instance);
		CommandBuffer& draw_indexed(size_t index_count, size_t instance_count, size_t first_index, int32_t vertex_offset, size_t first_instance);
		CommandBuffer& draw_indexed_indirect(std::span<vk::DrawIndexedIndirectCommand>);

		CommandBuffer& dispatch(size_t group_count_x, size_t group_count_y = 1, size_t group_count_z = 1);

		class SecondaryCommandBuffer begin_secondary();
        void execute(std::span<vk::CommandBuffer>);

		// commands for renderpass-less command buffers
		void resolve_image(Name src, Name dst);
		void blit_image(Name src, Name dst, vk::ImageBlit region, vk::Filter filter);
	protected:
		void _bind_state(bool graphics);
		void _bind_compute_pipeline_state();
		void _bind_graphics_pipeline_state();
	};

	class SecondaryCommandBuffer : public CommandBuffer {
    public:
		using CommandBuffer::CommandBuffer;
		vk::CommandBuffer get_buffer();

		~SecondaryCommandBuffer();
	};

	template<class T>
	inline CommandBuffer& CommandBuffer::push_constants(vk::ShaderStageFlags stages, size_t offset, std::span<T> span) {
		return push_constants(stages, offset, (void*)span.data(), sizeof(T) * span.size());
	}
	template<class T>
	inline CommandBuffer& CommandBuffer::push_constants(vk::ShaderStageFlags stages, size_t offset, T value) {
		return push_constants(stages, offset, (void*)&value, sizeof(T));
	}
	template<class T>
	inline T* CommandBuffer::map_scratch_uniform_binding(unsigned set, unsigned binding) {
		return static_cast<T*>(_map_scratch_uniform_binding(set, binding, sizeof(T)));
	}
}

