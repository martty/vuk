#include "example_runner.hpp"
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

namespace {
	float angle = 0.f;
	auto box = util::generate_cube();

	vuk::Example x{
		.name = "05_deferred",
		.setup = [&](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			{
			vuk::PipelineCreateInfo pci;
			pci.shaders.push_back("../../examples/deferred.vert");
			pci.shaders.push_back("../../examples/deferred.frag");
			runner.context->named_pipelines.emplace("cube_deferred", pci);
			}

			{
			vuk::PipelineCreateInfo pci;
			pci.shaders.push_back("../../examples/fullscreen.vert");
			pci.shaders.push_back("../../examples/deferred_resolve.frag");
			runner.context->named_pipelines.emplace("deferred_resolve", pci);
			}

		},
		.render = [&](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			auto ptc = ifc.begin();

			auto [verts, stub1] = ptc.create_scratch_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eVertexBuffer, gsl::span(&box.first[0], box.first.size()));
			auto [inds, stub2] = ptc.create_scratch_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eIndexBuffer, gsl::span(&box.second[0], box.second.size()));
			struct VP {
				glm::mat4 view;
				glm::mat4 proj;
			} vp;
			auto cam_pos = glm::vec3(0, 1.5, 3.5);
			vp.view = glm::lookAt(cam_pos, glm::vec3(0), glm::vec3(0, 1, 0));
			vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);

			auto [uboVP, stub3] = ptc.create_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vk::BufferUsageFlagBits::eUniformBuffer, gsl::span(&vp, 1));
			ptc.wait_all_transfers();

			vuk::RenderGraph rg;
			// MRT pass
			rg.add_pass({
				.name = "05_deferred_MRT",
				.resources = {"05_position"_image(vuk::eColorWrite), "05_normal"_image(vuk::eColorWrite), "05_color"_image(vuk::eColorWrite), "05_depth"_image(vuk::eDepthStencilRW)},
				.execute = [verts, uboVP, inds](vuk::CommandBuffer& command_buffer) {
					command_buffer
					  .set_viewport(0, vuk::Area::Framebuffer{})
					  .set_scissor(0, vuk::Area::Framebuffer{})
					  .bind_vertex_buffer(0, verts, vuk::Packed{vk::Format::eR32G32B32Sfloat, vk::Format::eR32G32B32Sfloat, vuk::Ignore{offsetof(util::Vertex, uv_coordinates) - offsetof(util::Vertex, tangent)}, vk::Format::eR32G32Sfloat})
					  .bind_index_buffer(inds, vk::IndexType::eUint32)
					  .bind_pipeline("cube_deferred")
					  .bind_uniform_buffer(0, 0, uboVP);
					glm::mat4* model = command_buffer.map_scratch_uniform_binding<glm::mat4>(0, 1);
					*model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
					command_buffer
					  .draw_indexed(box.second.size(), 1, 0, 0, 0);
					}
				}
			);

			angle += 360.f * ImGui::GetIO().DeltaTime;

			rg.add_pass({
				.name = "05_deferred_resolve",
				.resources = {"05_deferred_final"_image(vuk::eColorWrite), "05_position"_image(vuk::eFragmentSampled), "05_normal"_image(vuk::eFragmentSampled), "05_color"_image(vuk::eFragmentSampled)},
				.execute = [=](vuk::CommandBuffer& command_buffer) {
					command_buffer
					  .set_viewport(0, vuk::Area::Framebuffer{})
					  .set_scissor(0, vuk::Area::Framebuffer{})
					  .bind_pipeline("deferred_resolve");
					*command_buffer.map_scratch_uniform_binding<glm::vec3>(0, 3) = cam_pos;
					vk::SamplerCreateInfo sci;
					sci.minFilter = sci.magFilter = vk::Filter::eNearest;
					command_buffer
					  .bind_sampled_image(0, 0, "05_position", sci)
					  .bind_sampled_image(0, 1, "05_normal", sci)
					  .bind_sampled_image(0, 2, "05_color", sci)
					  .draw(3, 1, 0, 0);
					}
				}
			);

			rg.mark_attachment_internal("05_position", vk::Format::eR16G16B16A16Sfloat, vuk::Extent2D{300, 300}, vuk::Samples::e1, vuk::ClearColor{ 1.f,0.f,0.f,0.f });
			rg.mark_attachment_internal("05_normal", vk::Format::eR16G16B16A16Sfloat, vuk::Extent2D::Framebuffer{}, vuk::Samples::e1, vuk::ClearColor{ 0.f, 1.f, 0.f, 0.f });
			rg.mark_attachment_internal("05_color", vk::Format::eR8G8B8A8Unorm, vuk::Extent2D::Framebuffer{}, vuk::Samples::e1, vuk::ClearColor{ 0.f, 0.f, 1.f, 0.f });
			rg.mark_attachment_internal("05_depth", vk::Format::eD32Sfloat, vuk::Extent2D::Framebuffer{}, vuk::Samples::Framebuffer{}, vuk::ClearDepthStencil{ 1.0f, 0 });

			return rg;
		}
	};

	REGISTER_EXAMPLE(x);
}