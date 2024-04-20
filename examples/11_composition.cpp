#include "example_runner.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <stb_image.h>

/* 11_composition
 * We expand on example 5 by adding reflections to our deferred cube and FXAA.
 * To do this we showcase the composition features of vuk: we extensively use Futures to build up the final rendering. Rendering to and using cubemaps/array
 * images is also shown.
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the example same code.
 * Check out the framework (example_runner_*) files if interested!
 */

vuk::Value<vuk::ImageAttachment> apply_fxaa(vuk::Value<vuk::ImageAttachment> source, vuk::Value<vuk::ImageAttachment> dst) {
	auto pass = vuk::make_pass("fxaa", [](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eFragmentSampled) jagged, VUK_IA(vuk::eColorWrite) smooth) {
		command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
		    .set_scissor(0, vuk::Rect2D::framebuffer())
		    .set_rasterization({})     // Set the default rasterization state
		    .broadcast_color_blend({}) // Set the default color blend state
		    .bind_graphics_pipeline("fxaa");
		command_buffer.specialize_constants(0, (float)smooth->extent.width).specialize_constants(1, (float)smooth->extent.height);
		command_buffer.bind_image(0, 0, jagged).bind_sampler(0, 0, {}).draw(3, 1, 0, 0);
		return smooth;
	});
	// bidirectional inference
	source.same_shape_as(dst);
	return pass(source, dst);
}

std::pair<vuk::Unique<vuk::Image>, vuk::Value<vuk::ImageAttachment>> load_hdr(vuk::Allocator& allocator, const std::string& path) {
	int x, y, chans;
	stbi_set_flip_vertically_on_load(true);
	auto img = stbi_loadf(path.c_str(), &x, &y, &chans, STBI_rgb_alpha);
	stbi_set_flip_vertically_on_load(false);
	assert(img != nullptr);

	vuk::ImageAttachment ia;
	ia.format = vuk::Format::eR32G32B32A32Sfloat;
	ia.extent = vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u };
	ia.sample_count = vuk::Samples::e1;
	ia.image_type = vuk::ImageType::e2D;
	ia.view_type = vuk::ImageViewType::e2D;
	ia.tiling = vuk::ImageTiling::eOptimal;
	ia.usage = vuk::ImageUsageFlagBits::eTransferSrc | vuk::ImageUsageFlagBits::eTransferDst | vuk::ImageUsageFlagBits::eSampled;
	ia.base_level = 0;
	ia.level_count = 1;
	ia.base_layer = 0;
	ia.layer_count = 1;

	auto [image, fut] = vuk::create_image_with_data(allocator, vuk::DomainFlagBits::eTransferQueue, ia, img);

	stbi_image_free(img);

	return { std::move(image), std::move(fut) };
}

const glm::mat4 capture_projection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
const glm::mat4 capture_views[] = { glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
	                                  glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
	                                  glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
	                                  glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
	                                  glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
	                                  glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f)) };

namespace {
	float angle = 0.f;
	auto box = util::generate_cube();
	vuk::Unique<vuk::Buffer> verts, inds;
	vuk::Unique<vuk::Image> env_cubemap, hdr_image;
	vuk::ImageAttachment env_cubemap_ia;

	vuk::Example x{
		.name = "11_composition",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file((root / "examples/deferred.vert").generic_string()), (root / "deferred.vert").generic_string());
			      pci.add_glsl(util::read_entire_file((root / "examples/deferred_reflective.frag").generic_string()),
			                   (root / "deferred_reflective.frag").generic_string());
			      runner.runtime->create_named_pipeline("cube_deferred_reflective", pci);
		      }

		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file((root / "examples/fullscreen.vert").generic_string()), (root / "fullscreen.vert").generic_string());
			      pci.add_glsl(util::read_entire_file((root / "examples/deferred_resolve.frag").generic_string()), (root / "deferred_resolve.frag").generic_string());
			      runner.runtime->create_named_pipeline("deferred_resolve", pci);
		      }

		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file((root / "examples/fullscreen.vert").generic_string()), (root / "fullscreen.vert").generic_string());
			      pci.add_glsl(util::read_entire_file((root / "examples/fxaa.frag").generic_string()), (root / "fxaa.frag").generic_string());
			      runner.runtime->create_named_pipeline("fxaa", pci);
		      }

		      // We set up the cube data, same as in example 02_cube
		      auto [vert_buf, vert_fut] = create_buffer(allocator, vuk::MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.first));
		      verts = std::move(vert_buf);
		      auto [ind_buf, ind_fut] = create_buffer(allocator, vuk::MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.second));
		      inds = std::move(ind_buf);

		      auto [hdr_alloc, hdr_texture] = load_hdr(allocator, (root / "examples/the_sky_is_on_fire_1k.hdr").generic_string());
		      hdr_image = std::move(hdr_alloc);

		      // cubemap code from @jazzfool
		      // hdr_texture is a 2:1 equirectangular; it needs to be converted to a cubemap

		      env_cubemap_ia = { .image = *env_cubemap,
			                       .image_flags = vuk::ImageCreateFlagBits::eCubeCompatible,
			                       .image_type = vuk::ImageType::e2D,
			                       .usage = vuk::ImageUsageFlagBits::eSampled | vuk::ImageUsageFlagBits::eColorAttachment,
			                       .extent = { 1024, 1024, 1 },
			                       .format = vuk::Format::eR32G32B32A32Sfloat,
			                       .sample_count = vuk::Samples::e1,
			                       .view_type = vuk::ImageViewType::eCube,
			                       .layout = vuk::ImageLayout::eReadOnlyOptimal,
			                       .base_level = 0,
			                       .level_count = 1,
			                       .base_layer = 0,
			                       .layer_count = 6 };
		      env_cubemap = *vuk::allocate_image(allocator, env_cubemap_ia);
		      env_cubemap_ia.image = *env_cubemap;
		      auto env_cubemap_fut = vuk::declare_ia("env_cubemap", env_cubemap_ia);
		      vuk::PipelineBaseCreateInfo equirectangular_to_cubemap;
		      equirectangular_to_cubemap.add_glsl(util::read_entire_file((root / "examples/cubemap.vert").generic_string()),
		                                          (root / "examples/cubemap.vert").generic_string());
		      equirectangular_to_cubemap.add_glsl(util::read_entire_file((root / "examples/equirectangular_to_cubemap.frag").generic_string()),
		                                          (root / "examples/equirectangular_to_cubemap.frag").generic_string());
		      runner.runtime->create_named_pipeline("equirectangular_to_cubemap", equirectangular_to_cubemap);

		      // make cubemap by rendering to individual faces - we use a layered framebuffer to achieve this
		      // this only requires using an attachment with layers
		      {
			      auto render_texture_to_cubemap = vuk::make_pass(
			          "render_texture_to_cubemap",
			          [=](vuk::CommandBuffer& cbuf,
			              VUK_IA(vuk::eColorWrite) env_cubemap,
			              VUK_IA(vuk::eFragmentSampled) hdr_texture,
			              VUK_BA(vuk::eAttributeRead) verts,
			              VUK_BA(vuk::eIndexRead) inds) {
				          cbuf.set_viewport(0, vuk::Rect2D::framebuffer())
				              .set_scissor(0, vuk::Rect2D::framebuffer())
				              .broadcast_color_blend(vuk::BlendPreset::eOff)
				              .set_rasterization({})
				              .bind_vertex_buffer(
				                  0, verts, 0, vuk::Packed{ vuk::Format::eR32G32B32Sfloat, vuk::Ignore{ sizeof(util::Vertex) - sizeof(util::Vertex::position) } })
				              .bind_index_buffer(inds, vuk::IndexType::eUint32)
				              .bind_image(0, 2, hdr_texture)
				              .bind_sampler(0,
				                            2,
				                            vuk::SamplerCreateInfo{ .magFilter = vuk::Filter::eLinear,
				                                                    .minFilter = vuk::Filter::eLinear,
				                                                    .mipmapMode = vuk::SamplerMipmapMode::eLinear,
				                                                    .addressModeU = vuk::SamplerAddressMode::eClampToEdge,
				                                                    .addressModeV = vuk::SamplerAddressMode::eClampToEdge,
				                                                    .addressModeW = vuk::SamplerAddressMode::eClampToEdge })
				              .bind_graphics_pipeline("equirectangular_to_cubemap");
				          glm::mat4* projection = cbuf.scratch_buffer<glm::mat4>(0, 0);
				          *projection = capture_projection;
				          using mats = glm::mat4[6];
				          mats* view = cbuf.scratch_buffer<glm::mat4[6]>(0, 1);
				          memcpy(view, capture_views, sizeof(capture_views));
				          cbuf.draw_indexed(box.second.size(), 6, 0, 0, 0);

				          return env_cubemap;
			          });

			      // transition the cubemap for fragment sampling access to complete init
			      runner.enqueue_setup(render_texture_to_cubemap(env_cubemap_fut, hdr_texture, vert_fut, ind_fut)
			                               .as_released(vuk::Access::eFragmentSampled, vuk::DomainFlagBits::eGraphicsQueue));
		      }
		    },
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::Value<vuk::ImageAttachment> target) {
		      struct VP {
			      glm::mat4 view;
			      glm::mat4 proj;
		      } vp;
		      auto cam_pos = glm::vec3(0, 0.5, 7.5);
		      vp.view = glm::lookAt(cam_pos, glm::vec3(0), glm::vec3(0, 1, 0));
		      vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 0.1f, 30.f);
		      vp.proj[1][1] *= -1;

		      auto [buboVP, uboVP_fut] = create_buffer(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, vuk::DomainFlagBits::eTransferOnGraphics, std::span(&vp, 1));
		      auto uboVP = *buboVP;

		      // we are going to render the scene twice - once into a cubemap, then subsequently, we'll render it again while sampling from the cubemap

		      // this time we will show off rendering to individual cubemap faces
		      // create the cubemap in a separate rendergraph
		      auto cube_src = vuk::declare_ia("11_cube",
		                                      { .image_flags = vuk::ImageCreateFlagBits::eCubeCompatible,
		                                        .image_type = vuk::ImageType::e2D,
		                                        .extent = { 1024, 1024, 1 },
		                                        .format = vuk::Format::eR8G8B8A8Srgb,
		                                        .sample_count = vuk::Samples::e1,
		                                        .view_type = vuk::ImageViewType::eCube,
		                                        .level_count = 1,
		                                        .layer_count = 6 });

		      // for each face we do standard (example 05) deferred scene rendering
		      // but we resolve the deferred scene into the appropriate cubemap face
		      for (int i = 0; i < 6; i++) {
			      // Here we will render the cube into 3 offscreen textures
			      auto single_face_MRT = vuk::make_pass(
			          "single_face_MRT",
			          [i, cam_pos](vuk::CommandBuffer& command_buffer,
			                       VUK_IA(vuk::eColorWrite) position,
			                       VUK_IA(vuk::eColorWrite) normal,
			                       VUK_IA(vuk::eColorWrite) color,
			                       VUK_IA(vuk::eDepthStencilRW) depth) {
				          command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
				              .set_scissor(0, vuk::Rect2D::framebuffer())
				              .set_rasterization(
				                  vuk::PipelineRasterizationStateCreateInfo{ .cullMode = vuk::CullModeFlagBits::eBack }) // Set the default rasterization state
				              // Set the depth/stencil state
				              .set_depth_stencil(vuk::PipelineDepthStencilStateCreateInfo{
				                  .depthTestEnable = true,
				                  .depthWriteEnable = true,
				                  .depthCompareOp = vuk::CompareOp::eLessOrEqual,
				              })
				              .broadcast_color_blend({})
				              .bind_vertex_buffer(0,
				                                  *verts,
				                                  0,
				                                  vuk::Packed{ vuk::Format::eR32G32B32Sfloat,
				                                               vuk::Format::eR32G32B32Sfloat,
				                                               vuk::Ignore{ offsetof(util::Vertex, uv_coordinates) - offsetof(util::Vertex, tangent) },
				                                               vuk::Format::eR32G32Sfloat })
				              .bind_index_buffer(*inds, vuk::IndexType::eUint32);

				          command_buffer.push_constants(vuk::ShaderStageFlagBits::eFragment, 0, cam_pos).bind_graphics_pipeline("cube_deferred_reflective");
				          command_buffer.bind_image(0, 2, env_cubemap_ia).bind_sampler(0, 2, {});
				          VP* VP_data = command_buffer.scratch_buffer<VP>(0, 0);
				          VP_data->proj = capture_projection;
				          VP_data->view = capture_views[i];
				          for (auto j = 0; j < 64; j++) { // render all the cubes, except the center one
					          if (j == 36)
						          continue;

					          glm::mat4* model = command_buffer.scratch_buffer<glm::mat4>(0, 1);
					          *model = glm::scale(glm::mat4(1.f), glm::vec3(0.1f)) *
					                   glm::translate(glm::mat4(1.f), 4.f * glm::vec3(4 * (j % 8 - 4), sinf(0.1f * angle + j), 4 * (j / 8 - 4)));
					          command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0);
				          }

				          return std::tuple{ position, normal, color };
			          });
			      // The intermediate offscreen textures need to be bound
			      auto temp_position = vuk::clear_image(
			          vuk::declare_ia("temp_pos", { .format = vuk::Format::eR16G16B16A16Sfloat, .sample_count = vuk::Samples::e1, .layer_count = 1 }),
			          vuk::White<float>);
			      auto temp_normal = vuk::clear_image(vuk::declare_ia("temp_normal", { .format = vuk::Format::eR16G16B16A16Sfloat }), vuk::White<float>);
			      auto temp_color = vuk::clear_image(vuk::declare_ia("temp_color", { .format = vuk::Format::eR8G8B8A8Srgb }), vuk::White<float>);
			      auto temp_depth = vuk::clear_image(vuk::declare_ia("temp_depth", { .format = vuk::Format::eD32Sfloat }), vuk::DepthOne);

			      // run the MRT pass, filling out position, normal, color
			      std::tie(temp_position, temp_normal, temp_color) = single_face_MRT(temp_position, temp_normal, temp_color, temp_depth);

			      // The shading pass for the deferred rendering per face
			      auto resolve = vuk::make_pass("resolve",
			                                    // Declare that we are going to render to the final color image
			                                    // Declare that we are going to sample (in the fragment shader) from the previous attachments
			                                    [cam_pos](vuk::CommandBuffer& command_buffer,
			                                              VUK_IA(vuk::eColorWrite) cube_face,
			                                              VUK_IA(vuk::eFragmentSampled) position,
			                                              VUK_IA(vuk::eFragmentSampled) normal,
			                                              VUK_IA(vuk::eFragmentSampled) color) {
				                                    command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
				                                        .set_scissor(0, vuk::Rect2D::framebuffer())
				                                        .set_rasterization({})     // Set the default rasterization state
				                                        .broadcast_color_blend({}) // Set the default color blend state
				                                        .bind_graphics_pipeline("deferred_resolve");
				                                    // Set camera position so we can do lighting
				                                    *command_buffer.scratch_buffer<glm::vec3>(0, 3) = cam_pos;
				                                    // We will sample using nearest neighbour
				                                    vuk::SamplerCreateInfo sci;
				                                    sci.minFilter = sci.magFilter = vuk::Filter::eNearest;
				                                    // Bind the previous attachments as sampled images
				                                    command_buffer.bind_image(0, 0, position)
				                                        .bind_sampler(0, 0, sci)
				                                        .bind_image(0, 1, normal)
				                                        .bind_sampler(0, 1, sci)
				                                        .bind_image(0, 2, color)
				                                        .bind_sampler(0, 2, sci)
				                                        .draw(3, 1, 0, 0);

				                                    return cube_face;
			                                    });
			      auto cube_face = cube_src.layer(i);
			      temp_position.same_2D_extent_as(cube_face);
			      auto single_face = resolve(cube_face, temp_position, temp_normal, temp_color);
		      }
		      // time to render the final scene
		      // and do the standard deferred rendering as before, this time using the new cube map for the environment
		      auto final_MRT = vuk::make_pass(
		          "deferred_MRT",
		          [uboVP, cam_pos](vuk::CommandBuffer& command_buffer,
		                           VUK_IA(vuk::eColorWrite) position,
		                           VUK_IA(vuk::eColorWrite) normal,
		                           VUK_IA(vuk::eColorWrite) color,
		                           VUK_IA(vuk::eDepthStencilRW) depth,
		                           VUK_IA(vuk::eFragmentSampled) cube_refl) {
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
			              .bind_vertex_buffer(0,
			                                  *verts,
			                                  0,
			                                  vuk::Packed{ vuk::Format::eR32G32B32Sfloat,
			                                               vuk::Format::eR32G32B32Sfloat,
			                                               vuk::Ignore{ offsetof(util::Vertex, uv_coordinates) - offsetof(util::Vertex, tangent) },
			                                               vuk::Format::eR32G32Sfloat })
			              .bind_index_buffer(*inds, vuk::IndexType::eUint32);
			          command_buffer.push_constants(vuk::ShaderStageFlagBits::eFragment, 0, cam_pos).bind_graphics_pipeline("cube_deferred_reflective");
			          command_buffer.bind_image(0, 2, cube_refl).bind_sampler(0, 2, {}).bind_buffer(0, 0, uboVP);
			          glm::mat4* model = command_buffer.scratch_buffer<glm::mat4>(0, 1);
			          *model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
			          command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0);
			          for (auto i = 0; i < 64; i++) {
				          if (i == 36)
					          continue;
				          command_buffer.bind_image(0, 2, env_cubemap_ia).bind_sampler(0, 2, {}).bind_buffer(0, 0, uboVP);
				          glm::mat4* model = command_buffer.scratch_buffer<glm::mat4>(0, 1);
				          *model = glm::scale(glm::mat4(1.f), glm::vec3(0.1f)) *
				                   glm::translate(glm::mat4(1.f), 4.f * glm::vec3(4 * (i % 8 - 4), sinf(0.1f * angle + i), 4 * (i / 8 - 4)));
				          command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0);
			          }

			          return std::tuple{ position, normal, color };
		          });
		      auto final_position =
		          vuk::clear_image(vuk::declare_ia("final_pos", { .format = vuk::Format::eR16G16B16A16Sfloat, .sample_count = vuk::Samples::e1, .layer_count = 1 }),
		                           vuk::White<float>);
		      auto final_normal = vuk::clear_image(vuk::declare_ia("final_normal", { .format = vuk::Format::eR16G16B16A16Sfloat }), vuk::White<float>);
		      auto final_color = vuk::clear_image(vuk::declare_ia("final_color", { .format = vuk::Format::eR8G8B8A8Srgb }), vuk::White<float>);
		      auto final_depth = vuk::clear_image(vuk::declare_ia("final_depth", { .format = vuk::Format::eD32Sfloat }), vuk::DepthOne);

		      std::tie(final_position, final_normal, final_color) = final_MRT(final_position, final_normal, final_color, final_depth, cube_src);

		      // The shading pass for the deferred rendering
		      auto resolve = vuk::make_pass("final resolve",
		                                    // Declare that we are going to render to the final color image
		                                    // Declare that we are going to sample (in the fragment shader) from the previous attachments
		                                    [cam_pos](vuk::CommandBuffer& command_buffer,
		                                              VUK_IA(vuk::eColorWrite) cube_face,
		                                              VUK_IA(vuk::eFragmentSampled) position,
		                                              VUK_IA(vuk::eFragmentSampled) normal,
		                                              VUK_IA(vuk::eFragmentSampled) color) {
			                                    command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
			                                        .set_scissor(0, vuk::Rect2D::framebuffer())
			                                        .set_rasterization({})     // Set the default rasterization state
			                                        .broadcast_color_blend({}) // Set the default color blend state
			                                        .bind_graphics_pipeline("deferred_resolve");
			                                    // Set camera position so we can do lighting
			                                    *command_buffer.scratch_buffer<glm::vec3>(0, 3) = cam_pos;
			                                    // We will sample using nearest neighbour
			                                    vuk::SamplerCreateInfo sci;
			                                    sci.minFilter = sci.magFilter = vuk::Filter::eNearest;
			                                    // Bind the previous attachments as sampled images
			                                    command_buffer.bind_image(0, 0, position)
			                                        .bind_sampler(0, 0, sci)
			                                        .bind_image(0, 1, normal)
			                                        .bind_sampler(0, 1, sci)
			                                        .bind_image(0, 2, color)
			                                        .bind_sampler(0, 2, sci)
			                                        .draw(3, 1, 0, 0);

			                                    return cube_face;
		                                    });

		      auto final_deferred = vuk::declare_ia("final_deferred", { .format = vuk::Format::eR8G8B8A8Srgb, .sample_count = vuk::Samples::e1 });
		      final_position.same_extent_as(final_deferred);
		      final_deferred = resolve(final_deferred, final_position, final_normal, final_color);

		      auto post_processed_result = apply_fxaa(std::move(final_deferred), std::move(target));

		      angle += 10.f * ImGui::GetIO().DeltaTime;

		      return post_processed_result;
		    }, // Perform cleanup for the example
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
		      // We release the resources manually
		      verts.reset();
		      inds.reset();
		      env_cubemap.reset();
		      hdr_image.reset();
		    }
	};

	REGISTER_EXAMPLE(x);
} // namespace
