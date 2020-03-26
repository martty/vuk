#include "example_runner.hpp"
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <stb_image.h>

namespace {
	float angle = 0.f;
	auto box = util::generate_cube();
	vuk::Unique<vuk::ImageView> image_view;

	vuk::Example x{
		.name = "06_msaa",
		.setup = [&](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			{
			vuk::PipelineCreateInfo pci;
			pci.shaders.push_back("../../examples/ubo_test_tex.vert");
			pci.shaders.push_back("../../examples/triangle_depthshaded_tex.frag");
			runner.context->create_named_pipeline("textured_cube", pci);
			}

			int x, y, chans;
			auto doge_image = stbi_load("../../examples/doge.png", &x, &y, &chans, 4);

			auto ptc = ifc.begin();
			auto [img, iv, stub] = ptc.create_image(vk::Format::eR8G8B8A8Srgb, vk::Extent3D(x, y, 1), doge_image);
			image_view = std::move(iv);
			ptc.wait_all_transfers();
			stbi_image_free(doge_image);
		},
		.render = [&](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			auto ptc = ifc.begin();

			auto [verts, stub1] = ptc.create_scratch_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eVertexBuffer, gsl::span(&box.first[0], box.first.size()));
			auto [inds, stub2] = ptc.create_scratch_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eIndexBuffer, gsl::span(&box.second[0], box.second.size()));
			struct VP {
				glm::mat4 view;
				glm::mat4 proj;
			} vp;
			vp.view = glm::lookAt(glm::vec3(0, 1.5, 3.5), glm::vec3(0), glm::vec3(0, 1, 0));
			vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);

			auto [uboVP, stub3] = ptc.create_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vk::BufferUsageFlagBits::eUniformBuffer, gsl::span(&vp, 1));
			ptc.wait_all_transfers();

			vuk::RenderGraph rg;

			rg.add_pass({
				.resources = {"06_msaa_MS"_image(vuk::eColorWrite), "06_msaa_depth"_image(vuk::eDepthStencilRW)},
				.execute = [verts, uboVP, inds](vuk::CommandBuffer& command_buffer) {
					command_buffer
					  .set_viewport(0, vuk::Area::Framebuffer{})
					  .set_scissor(0, vuk::Area::Framebuffer{})
					  .bind_vertex_buffer(0, verts, vuk::Packed{vk::Format::eR32G32B32Sfloat, vuk::Ignore{offsetof(util::Vertex, uv_coordinates) - sizeof(util::Vertex::position)}, vk::Format::eR32G32Sfloat})
					  .bind_index_buffer(inds, vk::IndexType::eUint32)
					  .bind_sampled_image(0, 2, *image_view, vk::SamplerCreateInfo{})
					  .bind_pipeline("textured_cube")
					  .bind_uniform_buffer(0, 0, uboVP);
					glm::mat4* model = command_buffer.map_scratch_uniform_binding<glm::mat4>(0, 1);
					*model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
					command_buffer
					  .draw_indexed(box.second.size(), 1, 0, 0, 0);
					}
				}
			);

			angle += 180.f * ImGui::GetIO().DeltaTime;

			rg.mark_attachment_internal("06_msaa_MS", vk::Format::eR8G8B8A8Srgb, vuk::Extent2D::Framebuffer{}, vuk::Samples::e8, vuk::ClearColor{ 0.f, 0.f, 0.f, 0.f });
			rg.mark_attachment_internal("06_msaa_depth", vk::Format::eD32Sfloat, vuk::Extent2D::Framebuffer{}, vuk::Samples::Framebuffer{}, vuk::ClearDepthStencil{ 1.0f, 0 });
			rg.mark_attachment_resolve("06_msaa_final", "06_msaa_MS");
			return rg;
		},
		.cleanup = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			image_view.reset();
		}

	};

	REGISTER_EXAMPLE(x);
}