#include "example_runner.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
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
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      vuk::Context& ctx = allocator.get_context();
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file("../../examples/ubo_test_tex.vert"), "ubo_test_tex.vert");
			      pci.add_glsl(util::read_entire_file("../../examples/triangle_depthshaded_tex.frag"), "triangle_depthshaded_tex.frag");
			      runner.context->create_named_pipeline("textured_cube", pci);
		      }

		      // Use STBI to load the image
		      int x, y, chans;
		      auto doge_image = stbi_load("../../examples/doge.png", &x, &y, &chans, 4);

		      // Similarly to buffers, we allocate the image and enqueue the upload
		      auto [tex, tex_fut] = create_texture(allocator, vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, doge_image, true);
		      texture_of_doge = std::move(tex);
		      runner.enqueue_setup(std::move(tex_fut));
		      stbi_image_free(doge_image);
		    },
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
		      vuk::Context& ctx = frame_allocator.get_context();

		      // We set up the cube data, same as in example 02_cube
		      auto [vert_buf, vert_fut] = create_buffer_gpu(frame_allocator, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.first));
		      auto verts = *vert_buf;
		      auto [ind_buf, ind_fut] = create_buffer_gpu(frame_allocator, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.second));
		      auto inds = *ind_buf;

		      struct VP {
			      glm::mat4 view;
			      glm::mat4 proj;
		      } vp;
		      vp.view = glm::lookAt(glm::vec3(0, 1.5, 3.5), glm::vec3(0), glm::vec3(0, 1, 0));
		      vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);
		      vp.proj[1][1] *= -1;

		      auto [buboVP, uboVP_fut] = create_buffer_cross_device(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, std::span(&vp, 1));
		      auto uboVP = *buboVP;

		      vuk::wait_for_futures(frame_allocator, vert_fut, ind_fut, uboVP_fut);

		      vuk::RenderGraph rg;

		      // Set up the pass to draw the textured cube, with a color and a depth attachment
		      rg.add_pass({ .resources = { "04_texture"_image >> vuk::eColorWrite >> "04_texture_final", "04_texture_depth"_image >> vuk::eDepthStencilRW },
		                    .execute = [verts, uboVP, inds](vuk::CommandBuffer& command_buffer) {
			                    command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
			                        .set_scissor(0, vuk::Rect2D::framebuffer())
			                        .set_rasterization({}) // Set the default rasterization state
			                        // Set the depth/stencil state
			                        .set_depth_stencil(vuk::PipelineDepthStencilStateCreateInfo{
			                            .depthTestEnable = true,
			                            .depthWriteEnable = true,
			                            .depthCompareOp = vuk::CompareOp::eLessOrEqual,
			                        })
			                        .broadcast_color_blend({}) // Set the default color blend state
			                        .bind_vertex_buffer(0,
			                                            verts,
			                                            0,
			                                            vuk::Packed{ vuk::Format::eR32G32B32Sfloat,
			                                                         vuk::Ignore{ offsetof(util::Vertex, uv_coordinates) - sizeof(util::Vertex::position) },
			                                                         vuk::Format::eR32G32Sfloat })
			                        .bind_index_buffer(inds, vuk::IndexType::eUint32)
			                        // Here we bind our vuk::Texture to (set = 0, binding = 2) with default sampler settings
			                        .bind_image(0, 2, *texture_of_doge->view)
			                        .bind_sampler(0, 2, {})
			                        .bind_graphics_pipeline("textured_cube")
			                        .bind_buffer(0, 0, uboVP);
			                    glm::mat4* model = command_buffer.map_scratch_uniform_binding<glm::mat4>(0, 1);
			                    *model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
			                    command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0);
		                    } });

		      angle += 180.f * ImGui::GetIO().DeltaTime;

		      rg.attach_managed(
		          "04_texture_depth", vuk::Format::eD32Sfloat, vuk::Dimension2D::framebuffer(), vuk::Samples::Framebuffer{}, vuk::ClearDepthStencil{ 1.0f, 0 });
		      return vuk::Future<vuk::ImageAttachment>{ frame_allocator, std::make_unique<vuk::RenderGraph>(std::move(rg)), "04_texture_final" };
		    },

		// Perform cleanup for the example
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      // We release the texture resources
		      texture_of_doge.reset();
		    }
	};

	REGISTER_EXAMPLE(x);
} // namespace