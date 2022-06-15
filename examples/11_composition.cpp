#include "example_runner.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>

/* 11_deferred
 * In this example we will take our cube to the next level by rendering it deferred.
 * To achieve this, we will first render the cube to three offscreen textures -
 * one containing the world position, the second the world normals and the third containing colour.
 * We will also have depth buffering for this draw.
 * After this, we will compute the shading by using a fullscreen pass, where we sample from these textures.
 * To achieve this, we will need to let the rendergraph know of our image dependencies.
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the example same code.
 * Check out the framework (example_runner_*) files if interested!
 */

vuk::Future apply_fxaa(vuk::Future source, vuk::Future dst) {
	std::unique_ptr<vuk::RenderGraph> rgp = std::make_unique<vuk::RenderGraph>("fxaa");
	rgp->attach_in("jagged", std::move(source));
	rgp->attach_in("smooth", std::move(dst));
	rgp->add_pass({ .name = "fxaa",
	                .resources = { "jagged"_image >> vuk::eFragmentSampled, "smooth"_image >> vuk::eColorWrite },
	                .execute = [](vuk::CommandBuffer& command_buffer) {
		                command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
		                    .set_scissor(0, vuk::Rect2D::framebuffer())
		                    .set_rasterization({})     // Set the default rasterization state
		                    .broadcast_color_blend({}) // Set the default color blend state
		                    .bind_graphics_pipeline("fxaa");
		                auto res = *command_buffer.get_resource_image_attachment("smooth");
		                command_buffer.specialize_constants(0, (float)res.extent.extent.width).specialize_constants(1, (float)res.extent.extent.height);
		                command_buffer.bind_image(0, 0, "jagged").bind_sampler(0, 0, {}).draw(3, 1, 0, 0);
	                } });
	// bidirectional inference
	rgp->inference_rule("jagged", vuk::same_shape_as("smooth"));
	rgp->inference_rule("smooth", vuk::same_shape_as("jagged"));
	return { std::move(rgp), "smooth+" };
}

namespace {
	float angle = 0.f;
	auto box = util::generate_cube();
	vuk::BufferGPU verts, inds;

	vuk::Example x{ .name = "11_deferred",
		              .setup =
		                  [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		                    {
			                    vuk::PipelineBaseCreateInfo pci;
			                    pci.add_glsl(util::read_entire_file("../../examples/deferred.vert"), "deferred.vert");
			                    pci.add_glsl(util::read_entire_file("../../examples/deferred.frag"), "deferred.frag");
			                    runner.context->create_named_pipeline("cube_deferred", pci);
		                    }

		                    {
			                    vuk::PipelineBaseCreateInfo pci;
			                    pci.add_glsl(util::read_entire_file("../../examples/fullscreen.vert"), "fullscreen.vert");
			                    pci.add_glsl(util::read_entire_file("../../examples/deferred_resolve.frag"), "deferred_resolve.frag");
			                    runner.context->create_named_pipeline("deferred_resolve", pci);
		                    }

		                    {
			                    vuk::PipelineBaseCreateInfo pci;
			                    pci.add_glsl(util::read_entire_file("../../examples/fullscreen.vert"), "fullscreen.vert");
			                    pci.add_glsl(util::read_entire_file("../../examples/fxaa.frag"), "fxaa.frag");
			                    runner.context->create_named_pipeline("fxaa", pci);
		                    }

		                    // We set up the cube data, same as in example 02_cube
		                    auto [vert_buf, vert_fut] = create_buffer_gpu(allocator, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.first));
		                    verts = *vert_buf;
		                    auto [ind_buf, ind_fut] = create_buffer_gpu(allocator, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.second));
		                    inds = *ind_buf;
		                    // For the example, we just ask these that these uploads complete before moving on to rendering
		                    // In an engine, you would integrate these uploads into some explicit system
		                    runner.enqueue_setup(std::move(vert_fut));
		                    runner.enqueue_setup(std::move(ind_fut));
		                  },
		              .render =
		                  [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::Future target) {
		                    struct VP {
			                    glm::mat4 view;
			                    glm::mat4 proj;
		                    } vp;
		                    auto cam_pos = glm::vec3(0, 1.5, 3.5);
		                    vp.view = glm::lookAt(cam_pos, glm::vec3(0), glm::vec3(0, 1, 0));
		                    vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);
		                    vp.proj[1][1] *= -1;

		                    auto [buboVP, uboVP_fut] = create_buffer_cross_device(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, std::span(&vp, 1));
		                    auto uboVP = *buboVP;

		                    vuk::wait_for_futures(frame_allocator, uboVP_fut);

		                    std::shared_ptr<vuk::RenderGraph> rg = std::make_shared<vuk::RenderGraph>("MRT");
		                    // Here we will render the cube into 3 offscreen textures
		                    // The intermediate offscreen textures need to be bound
		                    // The "internal" rendering resolution is set here for one attachment, the rest infers from it
		                    rg->attach_and_clear_image("11_position",
		                                              { .format = vuk::Format::eR16G16B16A16Sfloat, .sample_count = vuk::Samples::e1 },
		                                              vuk::ClearColor{ 1.f, 0.f, 0.f, 0.f });
		                    rg->attach_and_clear_image("11_normal", { .format = vuk::Format::eR16G16B16A16Sfloat }, vuk::ClearColor{ 0.f, 1.f, 0.f, 0.f });
		                    rg->attach_and_clear_image("11_color", { .format = vuk::Format::eR8G8B8A8Srgb }, vuk::ClearColor{ 0.f, 0.f, 1.f, 0.f });
		                    rg->attach_and_clear_image("11_depth", { .format = vuk::Format::eD32Sfloat }, vuk::ClearDepthStencil{ 1.0f, 0 });
		                    rg->add_pass({ // Passes can be optionally named, this useful for visualization and debugging
		                                  .name = "deferred_MRT",
		                                  // Declare our framebuffer
		                                  .resources = { "11_position"_image >> vuk::eColorWrite,
		                                                 "11_normal"_image >> vuk::eColorWrite,
		                                                 "11_color"_image >> vuk::eColorWrite,
		                                                 "11_depth"_image >> vuk::eDepthStencilRW },
		                                  .execute = [uboVP](vuk::CommandBuffer& command_buffer) {
			                                  // Rendering is the same as in the case for forward
			                                  command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
			                                      .set_scissor(0, vuk::Rect2D::framebuffer())
			                                      .set_rasterization(vuk::PipelineRasterizationStateCreateInfo{}) // Set the default rasterization state
			                                      // Set the depth/stencil state
			                                      .set_depth_stencil(vuk::PipelineDepthStencilStateCreateInfo{
			                                          .depthTestEnable = true,
			                                          .depthWriteEnable = true,
			                                          .depthCompareOp = vuk::CompareOp::eLessOrEqual,
			                                      })
			                                      .broadcast_color_blend({})
			                                      .bind_vertex_buffer(
			                                          0,
			                                          verts,
			                                          0,
			                                          vuk::Packed{ vuk::Format::eR32G32B32Sfloat,
			                                                       vuk::Format::eR32G32B32Sfloat,
			                                                       vuk::Ignore{ offsetof(util::Vertex, uv_coordinates) - offsetof(util::Vertex, tangent) },
			                                                       vuk::Format::eR32G32Sfloat })
			                                      .bind_index_buffer(inds, vuk::IndexType::eUint32)
			                                      .bind_graphics_pipeline("cube_deferred")
			                                      .bind_buffer(0, 0, uboVP);
			                                  glm::mat4* model = command_buffer.map_scratch_uniform_binding<glm::mat4>(0, 1);
			                                  *model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
			                                  command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0);
		                                  } });

		                    std::shared_ptr<vuk::RenderGraph> rg_resolve = std::make_shared<vuk::RenderGraph>("11");
		                    std::vector futures = rg->split();
		                    rg_resolve->attach_in(futures);
		                    rg_resolve->attach_image("11_deferred", { .format = vuk::Format::eR8G8B8A8Srgb, .sample_count = vuk::Samples::e1 });
		                    rg_resolve->inference_rule("11_position+", vuk::same_extent_as("11_deferred"));
		                    // The shading pass for the deferred rendering
		                    rg_resolve->add_pass({ .name = "deferred_resolve",
		                                          // Declare that we are going to render to the final color image
		                                          // Declare that we are going to sample (in the fragment shader) from the previous attachments
		                                          .resources = { "11_deferred"_image >> vuk::eColorWrite >> "11_deferred+",
		                                                         "11_position+"_image >> vuk::eFragmentSampled,
		                                                         "11_normal+"_image >> vuk::eFragmentSampled,
		                                                         "11_color+"_image >> vuk::eFragmentSampled },
		                                          .execute = [cam_pos](vuk::CommandBuffer& command_buffer) {
			                                          command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
			                                              .set_scissor(0, vuk::Rect2D::framebuffer())
			                                              .set_rasterization({})     // Set the default rasterization state
			                                              .broadcast_color_blend({}) // Set the default color blend state
			                                              .bind_graphics_pipeline("deferred_resolve");
			                                          // Set camera position so we can do lighting
			                                          *command_buffer.map_scratch_uniform_binding<glm::vec3>(0, 3) = cam_pos;
			                                          // We will sample using nearest neighbour
			                                          vuk::SamplerCreateInfo sci;
			                                          sci.minFilter = sci.magFilter = vuk::Filter::eNearest;
			                                          // Bind the previous attachments as sampled images
			                                          command_buffer.bind_image(0, 0, "11_position+")
			                                              .bind_sampler(0, 0, sci)
			                                              .bind_image(0, 1, "11_normal+")
			                                              .bind_sampler(0, 1, sci)
			                                              .bind_image(0, 2, "11_color+")
			                                              .bind_sampler(0, 2, sci)
			                                              .draw(3, 1, 0, 0);
		                                          } });
		                    vuk::Future lit_fut = { std::move(rg_resolve), "11_deferred+" };
		                    vuk::Future sm_fut = apply_fxaa(std::move(lit_fut), std::move(target));

		                    angle += 20.f * ImGui::GetIO().DeltaTime;

		                    return sm_fut;
		                  } };

	REGISTER_EXAMPLE(x);
} // namespace
