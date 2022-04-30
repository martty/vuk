#include "example_runner.hpp"
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <stb_image.h>

/* 09_persistent_descriptorset
 * In this example we will see how to create persistent descriptorsets.
 * Normal descriptorsets are completely managed by vuk, and are cached based on their contents.
 * However, this behaviour is not helpful if you plan to keep the descriptorsets around, or if they have many elements (such as "bindless").
 * For these scenarios, you can create and explicitly manage descriptorsets.
 * Here we first generate two additional textures from the one we load: the first by Y flipping using blitting and the second by
 * running a compute shader on it. Afterwards we create the persistent set and write the three images into it.
 * Later, we draw three cubes and fetch the texture based on the base instance.
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the example same code.
 * Check out the framework (example_runner_*) files if interested!
 */

namespace {
	// The Y rotation angle of our cube
	float angle = 0.f;
	// Generate vertices and indices for the cube
	auto box = util::generate_cube();

	std::optional<vuk::Texture> texture_of_doge, variant1, variant2;
	vuk::Unique<vuk::PersistentDescriptorSet> pda;

	vuk::Example xample{
		.name = "09_persistent_descriptorset",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      vuk::Context& ctx = allocator.get_context();
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file("../../examples/bindless.vert"), "bindless.vert");
			      pci.add_glsl(util::read_entire_file("../../examples/triangle_tex_bindless.frag"), "triange_tex_bindless.frag");
			      // Flag this binding as partially bound, so that we don't need to set all the array elements
			      pci.set_binding_flags(1, 0, vuk::DescriptorBindingFlagBits::ePartiallyBound);
			      // Set the binding #0 in set #1 as a variable count binding, and set the maximum number of descriptors
			      pci.set_variable_count_binding(1, 0, 1024);
			      runner.context->create_named_pipeline("bindless_cube", pci);
		      }

		      // creating a compute pipeline that inverts an image
		      {
			      vuk::PipelineBaseCreateInfo pbci;
			      pbci.add_glsl(util::read_entire_file("../../examples/invert.comp"), "invert.comp");
			      runner.context->create_named_pipeline("invert", pbci);
		      }

		      // Use STBI to load the image
		      int x, y, chans;
		      auto doge_image = stbi_load("../../examples/doge.png", &x, &y, &chans, 4);

		      // Similarly to buffers, we allocate the image and enqueue the upload
		      auto [tex, tex_fut] = create_texture(allocator, vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, doge_image, false);
		      texture_of_doge = std::move(tex);
		      stbi_image_free(doge_image);

		      // Let's create two variants of the doge image
		      vuk::ImageCreateInfo ici;
		      ici.format = vuk::Format::eR8G8B8A8Srgb;
		      ici.extent = vuk::Extent3D{ (unsigned)x, (unsigned)y, 1 };
		      ici.samples = vuk::Samples::e1;
		      ici.imageType = vuk::ImageType::e2D;
		      ici.initialLayout = vuk::ImageLayout::eUndefined;
		      ici.tiling = vuk::ImageTiling::eOptimal;
		      ici.usage = vuk::ImageUsageFlagBits::eTransferWrite | vuk::ImageUsageFlagBits::eSampled;
		      ici.mipLevels = ici.arrayLayers = 1;
		      variant1 = ctx.allocate_texture(allocator, ici);
		      ici.format = vuk::Format::eR8G8B8A8Unorm;
		      ici.usage = vuk::ImageUsageFlagBits::eStorage | vuk::ImageUsageFlagBits::eSampled;
		      variant2 = ctx.allocate_texture(allocator, ici);
		      // Make a RenderGraph to process the loaded image
		      vuk::RenderGraph rg;
		      rg.add_pass({ .name = "09_preprocess",
		                    .resources = { "09_doge"_image >> vuk::eMemoryRead, "09_v1"_image >> vuk::eTransferWrite, "09_v2"_image >> vuk::eComputeWrite },
		                    .execute = [x, y](vuk::CommandBuffer& command_buffer) {
			                    // For the first image, flip the image on the Y axis using a blit
			                    vuk::ImageBlit blit;
			                    blit.srcSubresource.aspectMask = vuk::ImageAspectFlagBits::eColor;
			                    blit.srcSubresource.baseArrayLayer = 0;
			                    blit.srcSubresource.layerCount = 1;
			                    blit.srcSubresource.mipLevel = 0;
			                    blit.srcOffsets[0] = vuk::Offset3D{ 0, 0, 0 };
			                    blit.srcOffsets[1] = vuk::Offset3D{ x, y, 1 };
			                    blit.dstSubresource = blit.srcSubresource;
			                    blit.dstOffsets[0] = vuk::Offset3D{ x, y, 0 };
			                    blit.dstOffsets[1] = vuk::Offset3D{ 0, 0, 1 };
			                    command_buffer.blit_image("09_doge", "09_v1", blit, vuk::Filter::eLinear);
			                    // For the second image, invert the colours in compute
			                    command_buffer.bind_image(0, 0, "09_doge")
			                        .bind_sampler(0, 0, {})
			                        .bind_image(0, 1, "09_v2")
			                        .bind_compute_pipeline("invert")
			                        .dispatch_invocations(x, y);
		                    } });
		      // Bind the resources for the variant generation
		      // We specify the initial and final access
		      // The texture we have created is already in ShaderReadOptimal, but we need it in General during the pass, and we need it back to ShaderReadOptimal
		      // afterwards
		      rg.attach_in("09_doge", std::move(tex_fut));
		      rg.attach_image("09_v1", vuk::ImageAttachment::from_texture(*variant1), vuk::eNone, vuk::eFragmentSampled);
		      rg.attach_image("09_v2", vuk::ImageAttachment::from_texture(*variant2), vuk::eNone, vuk::eFragmentSampled);

		      // enqueue running the preprocessing rendergraph and force 09_doge to be sampleable later
		      auto fut = vuk::transition(vuk::Future<vuk::ImageAttachment>{ allocator, std::make_unique<vuk::RenderGraph>(std::move(rg)), "09_doge" },
		                                 vuk::eFragmentSampled);
		      runner.enqueue_setup(std::move(fut));

		      // Create persistent descriptorset for a pipeline and set index
		      pda = ctx.create_persistent_descriptorset(allocator, *runner.context->get_named_pipeline("bindless_cube"), 1, 64);
		      // Enqueue updates to the descriptors in the array
		      // This records the writes internally, but does not execute them
		      pda->update_combined_image_sampler(ctx, 0, 0, texture_of_doge->view.get(), {}, vuk::ImageLayout::eShaderReadOnlyOptimal);
		      pda->update_combined_image_sampler(ctx, 0, 1, variant1->view.get(), {}, vuk::ImageLayout::eShaderReadOnlyOptimal);
		      pda->update_combined_image_sampler(ctx, 0, 2, variant2->view.get(), {}, vuk::ImageLayout::eShaderReadOnlyOptimal);
		      // Execute the writes
		      ctx.commit_persistent_descriptorset(pda.get());
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

		      // Set up the pass to draw the textured cube, with a color and a depth attachment
		      rg.add_pass({ .resources = { "09_persistent_descriptorset"_image >> vuk::eColorWrite >> "09_persistent_descriptorset_final",
		                                   "09_depth"_image >> vuk::eDepthStencilRW },
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
			                        .bind_persistent(1, pda.get())
			                        .bind_graphics_pipeline("bindless_cube")
			                        .bind_buffer(0, 0, uboVP);
			                    glm::mat4* model = command_buffer.map_scratch_uniform_binding<glm::mat4>(0, 1);
			                    *model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
			                    // Draw 3 cubes, assign them different base instance to identify them in the shader
			                    command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0)
			                        .draw_indexed(box.second.size(), 1, 0, 0, 1)
			                        .draw_indexed(box.second.size(), 1, 0, 0, 2);
		                    } });

		      angle += 10.f * ImGui::GetIO().DeltaTime;

		      rg.attach_managed(
		          "09_depth", vuk::Format::eD32Sfloat, vuk::Dimension2D::framebuffer(), vuk::Samples::Framebuffer{}, vuk::ClearDepthStencil{ 1.0f, 0 });

		      return vuk::Future<vuk::ImageAttachment>{ frame_allocator, std::make_unique<vuk::RenderGraph>(std::move(rg)), "09_persistent_descriptorset_final" };
		    },

		// Perform cleanup for the example
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
		      // We release the resources manually
		      texture_of_doge.reset();
		      variant1.reset();
		      variant2.reset();
		      pda.reset();
		    }
	};

	REGISTER_EXAMPLE(xample);
} // namespace