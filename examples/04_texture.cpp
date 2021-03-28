#include "example_runner.hpp"
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <stb_image.h>

/* 04_texture
* In this example we will build on the previous examples (02_cube and 03_multipass), but we will make the cube textured.
*
* These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
* Furthermore it allows launching individual examples and all examples with the example same code.
* Check out the framework (example_runner_*) files if interested!
*/

namespace {
	float angle = 0.f;
	auto box = util::generate_cube();
	// A vuk::Texture is an owned pair of Image and ImageView
	// An optional is used here so that we can reset this on cleanup, despite being a global (which is to simplify the code here)
	std::optional<vuk::Texture> texture_of_doge;

	vuk::Example x{
		.name = "04_texture",
		.setup = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			{
			vuk::PipelineBaseCreateInfo pci;
			pci.add_glsl(util::read_entire_file("../../examples/ubo_test_tex.vert"), "ubo_test_tex.vert");
			pci.add_glsl(util::read_entire_file("../../examples/triangle_depthshaded_tex.frag"), "triangle_depthshaded_tex.frag");
			runner.context->create_named_pipeline("textured_cube", pci);
			}

			// Use STBI to load the image
			int x, y, chans;
			auto doge_image = stbi_load("../../examples/doge.png", &x, &y, &chans, 4);
			
			auto ptc = ifc.begin();
			// Similarly to buffers, we allocate the image and enqueue the upload
			/*vuk::ImageCreateInfo ici;
			ici.format = vuk::Format::eR8G8B8A8Srgb;
			ici.extent = vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u };
			ici.samples = vuk::Samples::e1;
			ici.imageType = vuk::ImageType::e2D;
			ici.initialLayout = vuk::ImageLayout::eUndefined;
			ici.tiling = vuk::ImageTiling::eOptimal;
			ici.usage = vuk::ImageUsageFlagBits::eTransferSrc | vuk::ImageUsageFlagBits::eTransferDst | vuk::ImageUsageFlagBits::eSampled;
			ici.mipLevels = ici.arrayLayers = 1;
			auto tex = ptc.allocate_texture(ici);
			auto t = ptc.ctx. */
			auto [tex, _] = ptc.create_texture(vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, doge_image);
			texture_of_doge = std::move(tex);
			ptc.wait_all_transfers();
			stbi_image_free(doge_image);
		},
		.render = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			auto ptc = ifc.begin();

			// We set up the cube data, same as in example 02_cube
			auto verts = ptc.allocate_scratch_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eVertexBuffer | vuk::BufferUsageFlagBits::eTransferDst, box.first.size() * sizeof(box.first[0]), 1);
			auto inds = ptc.allocate_scratch_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eIndexBuffer | vuk::BufferUsageFlagBits::eTransferDst, box.second.size() * sizeof(box.second[0]), 1);

			// t1 is a Token with a reference to Context to allow chaining, but the Token can be extracted
			auto t1 = ptc.ctx.copy_to_buffer(vuk::Domain::eGraphics, verts, box.first.data(), box.first.size() * sizeof(box.first[0]));
			// += appends another token, but doesn't establish ordering
			t1 += ptc.ctx.copy_to_buffer(vuk::Domain::eGraphics, inds, box.second.data(), box.second.size() * sizeof(box.second[0]));
			
			struct VP {
				glm::mat4 view;
				glm::mat4 proj;
			} vp;
			vp.view = glm::lookAt(glm::vec3(0, 1.5, 3.5), glm::vec3(0), glm::vec3(0, 1, 0));
			vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);
			vp.proj[1][1] *= -1;

			auto [buboVP, _] = ptc.create_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vuk::BufferUsageFlagBits::eUniformBuffer, std::span(&vp, 1));
			auto uboVP = buboVP;
			
			// submit work (rg) bound to t1
			ptc.submit(t1, vuk::Domain::eHost);
			// to wait on host for the work of t1 to finish:
			// (instead of ptc.wait_all_transfers)
			// ptc.wait(t1);

			vuk::RenderGraph rg;

			// Set up the pass to draw the textured cube, with a color and a depth attachment
			rg.add_pass({
				.resources = {"04_texture_final"_image(vuk::eColorWrite), "04_texture_depth"_image(vuk::eDepthStencilRW)},
				// this pass waits for t1 to complete
				.waits = {t1},
				.execute = [verts, uboVP, inds](vuk::CommandBuffer& command_buffer) {
					command_buffer
					  .set_viewport(0, vuk::Rect2D::framebuffer())
					  .set_scissor(0, vuk::Rect2D::framebuffer())
					  .bind_vertex_buffer(0, verts, 0, vuk::Packed{vuk::Format::eR32G32B32Sfloat, vuk::Ignore{offsetof(util::Vertex, uv_coordinates) - sizeof(util::Vertex::position)}, vuk::Format::eR32G32Sfloat})
					  .bind_index_buffer(inds, vuk::IndexType::eUint32)
					  // Here we bind our vuk::Texture to (set = 0, binding = 2) with default sampler settings
					  .bind_sampled_image(0, 2, *texture_of_doge, vuk::SamplerCreateInfo{})
					  .bind_graphics_pipeline("textured_cube")
					  .bind_uniform_buffer(0, 0, uboVP);
					glm::mat4* model = command_buffer.map_scratch_uniform_binding<glm::mat4>(0, 1);
					*model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
					command_buffer
					  .draw_indexed(box.second.size(), 1, 0, 0, 0);
					}
				}
			);

			angle += 180.f * ImGui::GetIO().DeltaTime;

			rg.attach_managed("04_texture_depth", vuk::Format::eD32Sfloat, vuk::Dimension2D::framebuffer(), vuk::Samples::Framebuffer{}, vuk::ClearDepthStencil{ 1.0f, 0 });
			return rg;
		},
		// Perform cleanup for the example
		.cleanup = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			// We release the texture resources
			texture_of_doge.reset();
		}
	};

	REGISTER_EXAMPLE(x);
}
