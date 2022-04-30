#include "example_runner.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <stb_image.h>

/* 06_msaa
 * In this example we will build on the previous example 04_texture, but we will render the cube to a multisampled texture,
 * which we will resolve to the final swapchain image.
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the example same code.
 * Check out the framework (example_runner_*) files if interested!
 */

namespace {
	float angle = 0.f;
	auto box = util::generate_cube();
	std::optional<vuk::Texture> texture_of_doge;

	vuk::Example x{
		.name = "06_msaa",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      // Same setup as for 04_texture, except we use spirv to create the pipeline
		      // This is a good option is if you don't want to ship shaderc or if you are caching or have your sl -> spirv pipeline
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_spirv(util::read_spirv("../../examples/ubo_test_tex.vert.spv"), "ubo_test_tex.vert");
			      pci.add_spirv(util::read_spirv("../../examples/triangle_depthshaded_tex.frag.spv"), "triangle_depthshaded_text.frag");
			      runner.context->create_named_pipeline("textured_cube", pci);
		      }

		      int x, y, chans;
		      auto doge_image = stbi_load("../../examples/doge.png", &x, &y, &chans, 4);

		      auto [tex, tex_fut] = create_texture(allocator, vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, doge_image, false);
		      texture_of_doge = std::move(tex);
		      runner.enqueue_setup(std::move(tex_fut));
		      stbi_image_free(doge_image);
		    },
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
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

		      // The rendering pass is unchanged by going to multisampled,
		      // but we will use an offscreen multisampled color attachment
		      rg.add_pass({ .resources = { "06_msaa_MS"_image >> vuk::eColorWrite, "06_msaa_depth"_image >> vuk::eDepthStencilRW },
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
			                        .bind_image(0, 2, *texture_of_doge->view)
			                        .bind_sampler(0, 2, {})
			                        .bind_graphics_pipeline("textured_cube")
			                        .bind_buffer(0, 0, uboVP);
			                    glm::mat4* model = command_buffer.map_scratch_uniform_binding<glm::mat4>(0, 1);
			                    *model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
			                    command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0);
		                    } });

		      angle += 180.f * ImGui::GetIO().DeltaTime;

		      // We mark our MS attachment as multisampled (8 samples)
		      // Since resolving requires equal sized images, we can actually infer the size of the MS attachment
		      // from the final image, and we don't need to specify here
		      // We use the swapchain format, since resolving needs identical formats
		      rg.attach_managed("06_msaa_MS", runner.swapchain->format, vuk::Dimension2D::framebuffer(), vuk::Samples::e8, vuk::ClearColor{ 0.f, 0.f, 0.f, 0.f });
		      rg.attach_managed(
		          "06_msaa_depth", vuk::Format::eD32Sfloat, vuk::Dimension2D::framebuffer(), vuk::Samples::Framebuffer{}, vuk::ClearDepthStencil{ 1.0f, 0 });
		      // We mark our final result "06_msaa_final" attachment to be a result of a resolve from "06_msaa_MS"
		      rg.resolve_resource_into("06_msaa", "06_msaa_final", "06_msaa_MS+");

		      return vuk::Future<vuk::ImageAttachment>{ frame_allocator, std::make_unique<vuk::RenderGraph>(std::move(rg)), "06_msaa_final" };
		    },
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
		      texture_of_doge.reset();
		    }

	};

	REGISTER_EXAMPLE(x);
} // namespace