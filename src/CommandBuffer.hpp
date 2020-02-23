#pragma once

#include <vulkan/vulkan.hpp>
#include <utility>
#include <optional>
#include "Allocator.hpp"

namespace vuk {
	class Context;
	class InflightContext;
	class Buffer;

	struct CommandBuffer {
		Pass* current_pass;
		vk::CommandBuffer command_buffer;
		vuk::InflightContext& ifc;

		CommandBuffer(vuk::InflightContext& ifc, vk::CommandBuffer cb) : ifc(ifc), command_buffer(cb) {}

		std::optional<std::pair<vk::RenderPass, uint32_t>> ongoing_renderpass;
		std::optional<vk::Viewport> next_viewport;
		std::optional<vk::Rect2D> next_scissor;
		std::optional<vuk::create_info_t<vk::Pipeline>> next_graphics_pipeline;

		// global memory barrier
		bool global_memory_barrier_inserted_since_last_draw = false;
		unsigned src_access_mask = 0;
		unsigned dst_access_mask = 0;
		// buffer barriers
		struct QueueXFer {
			QueueID from;
			QueueID to;
		};
		std::vector<QueueXFer> queue_transfers;

		CommandBuffer& set_viewport(vk::Viewport vp) {
			next_viewport = vp;
			return *this;
		}

		CommandBuffer& set_scissor(vk::Rect2D vp) {
			next_scissor = vp;
			return *this;
		}

		CommandBuffer& bind_pipeline(vk::GraphicsPipelineCreateInfo gpci) {
			next_graphics_pipeline = gpci;
			return *this;
		}
		CommandBuffer& bind_pipeline(Name p);

		CommandBuffer& bind_vertex_buffer(Allocator::Buffer&);
		CommandBuffer& bind_index_buffer(Allocator::Buffer&);

		CommandBuffer& draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance);
		CommandBuffer& draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance);
	};
}

