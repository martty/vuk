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

std::pair<vuk::Unique<vuk::Image>, vuk::Future> load_hdr_cubemap(vuk::Allocator& allocator, const std::string& path) {
	int x, y, chans;
	stbi_set_flip_vertically_on_load(true);
	auto img = stbi_loadf(path.c_str(), &x, &y, &chans, STBI_rgb_alpha);
	stbi_set_flip_vertically_on_load(false);
	assert(img != nullptr);

	vuk::ImageCreateInfo ici;
	ici.format = vuk::Format::eR32G32B32A32Sfloat;
	ici.extent = vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u };
	ici.samples = vuk::Samples::e1;
	ici.imageType = vuk::ImageType::e2D;
	ici.tiling = vuk::ImageTiling::eOptimal;
	ici.usage = vuk::ImageUsageFlagBits::eTransferSrc | vuk::ImageUsageFlagBits::eTransferDst | vuk::ImageUsageFlagBits::eSampled;
	ici.mipLevels = 1;
	ici.arrayLayers = 1;

	// TODO: not use create_texture
	auto [tex, tex_fut] = create_texture(allocator, vuk::Format::eR32G32B32A32Sfloat, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, img, false);
	stbi_image_free(img);

	return { std::move(tex.image), std::move(tex_fut) };
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
	vuk::BufferGPU verts, inds;
	vuk::Unique<vuk::Image> env_cubemap, hdr_image;
	vuk::ImageAttachment env_cubemap_ia;

	vuk::Example x{
		.name = "11_composition",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file("../../examples/deferred.vert"), "deferred.vert");
			      pci.add_glsl(util::read_entire_file("../../examples/deferred_reflective.frag"), "deferred_reflective.frag");
			      runner.context->create_named_pipeline("cube_deferred_reflective", pci);
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

		      auto hdr_texture = load_hdr_cubemap(allocator, "../../examples/the_sky_is_on_fire_1k.hdr");
		      hdr_image = std::move(hdr_texture.first);
		      // cubemap code from @jazzfool
		      // hdr_texture is a 2:1 equirectangular; it needs to be converted to a cubemap

		      env_cubemap_ia = { .image = *env_cubemap,
			                       .image_flags = vuk::ImageCreateFlagBits::eCubeCompatible,
			                       .image_type = vuk::ImageType::e2D,
			                       .usage = vuk::ImageUsageFlagBits::eSampled | vuk::ImageUsageFlagBits::eColorAttachment,
			                       .extent = vuk::Dimension3D::absolute(1024, 1024, 1),
			                       .format = vuk::Format::eR32G32B32A32Sfloat,
			                       .sample_count = vuk::Samples::e1,
			                       .view_type = vuk::ImageViewType::eCube,
			                       .base_level = 0,
			                       .level_count = 1,
			                       .base_layer = 0,
			                       .layer_count = 6 };
		      env_cubemap = *vuk::allocate_image(allocator, env_cubemap_ia);
		      env_cubemap_ia.image = *env_cubemap;

		      vuk::PipelineBaseCreateInfo equirectangular_to_cubemap;
		      equirectangular_to_cubemap.add_glsl(util::read_entire_file("../../examples/cubemap.vert"), "cubemap.vert");
		      equirectangular_to_cubemap.add_glsl(util::read_entire_file("../../examples/equirectangular_to_cubemap.frag"), "equirectangular_to_cubemap.frag");
		      runner.context->create_named_pipeline("equirectangular_to_cubemap", equirectangular_to_cubemap);

		      // make cubemap by rendering to individual faces - we use a layered framebuffer to achieve this
		      // this only requires using an attachment with layers
		      {
			      vuk::RenderGraph rg("cubegen");
			      rg.attach_in("hdr_texture", std::move(hdr_texture.second));
			      rg.attach_in("verts", std::move(vert_fut));
			      rg.attach_in("inds", std::move(ind_fut));
			      rg.attach_image("env_cubemap", env_cubemap_ia, vuk::Access::eNone, vuk::Access::eNone);
			      rg.add_pass({ .resources = { "env_cubemap"_image >> vuk::eColorWrite,
			                                   "hdr_texture"_image >> vuk::eFragmentSampled,
			                                   "verts"_buffer >> vuk::eAttributeRead,
			                                   "inds"_buffer >> vuk::eIndexRead },
			                    .execute = [=](vuk::CommandBuffer& cbuf) {
				                    cbuf.set_viewport(0, vuk::Rect2D::framebuffer())
				                        .set_scissor(0, vuk::Rect2D::framebuffer())
				                        .broadcast_color_blend(vuk::BlendPreset::eOff)
				                        .set_rasterization({})
				                        .bind_vertex_buffer(
				                            0,
				                            *cbuf.get_resource_buffer("verts"),
				                            0,
				                            vuk::Packed{ vuk::Format::eR32G32B32Sfloat, vuk::Ignore{ sizeof(util::Vertex) - sizeof(util::Vertex::position) } })
				                        .bind_index_buffer(*cbuf.get_resource_buffer("inds"), vuk::IndexType::eUint32)
				                        .bind_image(0, 2, "hdr_texture")
				                        .bind_sampler(0,
				                                      2,
				                                      vuk::SamplerCreateInfo{ .magFilter = vuk::Filter::eLinear,
				                                                              .minFilter = vuk::Filter::eLinear,
				                                                              .mipmapMode = vuk::SamplerMipmapMode::eLinear,
				                                                              .addressModeU = vuk::SamplerAddressMode::eClampToEdge,
				                                                              .addressModeV = vuk::SamplerAddressMode::eClampToEdge,
				                                                              .addressModeW = vuk::SamplerAddressMode::eClampToEdge })
				                        .bind_graphics_pipeline("equirectangular_to_cubemap");
				                    glm::mat4* projection = cbuf.map_scratch_uniform_binding<glm::mat4>(0, 0);
				                    *projection = capture_projection;
				                    using mats = glm::mat4[6];
				                    mats* view = cbuf.map_scratch_uniform_binding<glm::mat4[6]>(0, 1);
				                    memcpy(view, capture_views, sizeof(capture_views));
				                    cbuf.draw_indexed(box.second.size(), 6, 0, 0, 0);
			                    } });

			      // transition the cubemap for fragment sampling access to complete init
			      auto fut_in_layout = vuk::transition(vuk::Future{ std::make_unique<vuk::RenderGraph>(std::move(rg)), "env_cubemap+" }, vuk::eFragmentSampled);
			      runner.enqueue_setup(std::move(fut_in_layout));
		      }
		    },
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::Future target) {
		      struct VP {
			      glm::mat4 view;
			      glm::mat4 proj;
		      } vp;
		      auto cam_pos = glm::vec3(0, 0.5, 7.5);
		      vp.view = glm::lookAt(cam_pos, glm::vec3(0), glm::vec3(0, 1, 0));
		      vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 0.1f, 30.f);
		      vp.proj[1][1] *= -1;

		      auto [buboVP, uboVP_fut] = create_buffer_cross_device(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, std::span(&vp, 1));
		      auto uboVP = *buboVP;

		      // we are going to render the scene twice - once into a cubemap, then subsequently, we'll render it again while sampling from the cubemap

		      // this time we will show off rendering to individual cubemap faces
		      // create the cubemap in a separate rendergraph
		      std::shared_ptr<vuk::RenderGraph> cube_src = std::make_shared<vuk::RenderGraph>("cube_src");
		      cube_src->attach_image("11_cube",
		                             { .image_flags = vuk::ImageCreateFlagBits::eCubeCompatible,
		                               .image_type = vuk::ImageType::e2D,
		                               .extent = vuk::Dimension3D::absolute(1024, 1024, 1),
		                               .format = vuk::Format::eR8G8B8A8Srgb,
		                               .sample_count = vuk::Samples::e1,
		                               .view_type = vuk::ImageViewType::eCube,
		                               .layer_count = 6 });

		      // we split the cubemap into faces
		      for (uint32_t i = 0; i < 6; i++) {
			      cube_src->diverge_image(
			          "11_cube", vuk::Subrange::Image{ .base_layer = i, .layer_count = 1 }, vuk::Name("11_cube_face_").append(std::to_string(i)));
		      }

		      std::shared_ptr<vuk::RenderGraph> cube_refl = std::make_shared<vuk::RenderGraph>("scene");

		      // for each face we do standard (example 05) deferred scene rendering
		      // but we resolve the deferred scene into the appropriate cubemap face
		      std::vector<vuk::Name> cube_face_names;
		      for (int i = 0; i < 6; i++) {
			      std::shared_ptr<vuk::RenderGraph> rg = std::make_shared<vuk::RenderGraph>("single_face_MRT");
			      // Here we will render the cube into 3 offscreen textures
			      // The intermediate offscreen textures need to be bound
			      // The "internal" rendering resolution is set here for one attachment, the rest infers from it
			      rg->attach_and_clear_image("11_position", { .format = vuk::Format::eR16G16B16A16Sfloat, .sample_count = vuk::Samples::e1 }, vuk::White<float>);
			      rg->attach_and_clear_image("11_normal", { .format = vuk::Format::eR16G16B16A16Sfloat }, vuk::White<float>);
			      rg->attach_and_clear_image("11_color", { .format = vuk::Format::eR8G8B8A8Srgb }, vuk::White<float>);
			      rg->attach_and_clear_image("11_depth", { .format = vuk::Format::eD32Sfloat }, vuk::DepthOne);
			      rg->add_pass({ .name = "single_face_MRT",
			                     .resources = { "11_position"_image >> vuk::eColorWrite,
			                                    "11_normal"_image >> vuk::eColorWrite,
			                                    "11_color"_image >> vuk::eColorWrite,
			                                    "11_depth"_image >> vuk::eDepthStencilRW },
			                     .execute = [i, cam_pos](vuk::CommandBuffer& command_buffer) {
				                     command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
				                         .set_scissor(0, vuk::Rect2D::framebuffer())
				                         .set_rasterization(vuk::PipelineRasterizationStateCreateInfo{
				                             .cullMode = vuk::CullModeFlagBits::eBack }) // Set the default rasterization state
				                         // Set the depth/stencil state
				                         .set_depth_stencil(vuk::PipelineDepthStencilStateCreateInfo{
				                             .depthTestEnable = true,
				                             .depthWriteEnable = true,
				                             .depthCompareOp = vuk::CompareOp::eLessOrEqual,
				                         })
				                         .broadcast_color_blend({})
				                         .bind_vertex_buffer(0,
				                                             verts,
				                                             0,
				                                             vuk::Packed{ vuk::Format::eR32G32B32Sfloat,
				                                                          vuk::Format::eR32G32B32Sfloat,
				                                                          vuk::Ignore{ offsetof(util::Vertex, uv_coordinates) - offsetof(util::Vertex, tangent) },
				                                                          vuk::Format::eR32G32Sfloat })
				                         .bind_index_buffer(inds, vuk::IndexType::eUint32);
				                     command_buffer.push_constants(vuk::ShaderStageFlagBits::eFragment, 0, cam_pos).bind_graphics_pipeline("cube_deferred_reflective");
				                     for (auto j = 0; j < 64; j++) {
					                     if (j == 36)
						                     continue;
					                     command_buffer.bind_image(0, 2, env_cubemap_ia).bind_sampler(0, 2, {});
					                     VP* VP_data = command_buffer.map_scratch_uniform_binding<VP>(0, 0);
					                     VP_data->proj = capture_projection;
					                     VP_data->view = capture_views[i];
					                     glm::mat4* model = command_buffer.map_scratch_uniform_binding<glm::mat4>(0, 1);
					                     *model = glm::scale(glm::mat4(1.f), glm::vec3(0.1f)) *
					                              glm::translate(glm::mat4(1.f), 4.f * glm::vec3(4 * (j % 8 - 4), sinf(0.1f * angle + j), 4 * (j / 8 - 4)));
					                     command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0);
				                     }
			                     } });

			      rg->attach_in("11_cube_face", vuk::Future{ cube_src, vuk::Name("11_cube_face_").append(std::to_string(i)) });
			      rg->inference_rule("11_position+", vuk::same_2D_extent_as("11_cube_face"));
			      // The shading pass for the deferred rendering
			      rg->add_pass({ .name = "single_face_resolve",
			                     // Declare that we are going to render to the final color image
			                     // Declare that we are going to sample (in the fragment shader) from the previous attachments
			                     .resources = { "11_cube_face"_image >> vuk::eColorWrite >> "11_cube_face+",
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
			      vuk::Future lit_fut = { std::move(rg), "11_cube_face+" };
			      auto cube_face_name = vuk::Name("11_deferred_face_").append(std::to_string(i));
			      cube_face_names.emplace_back(cube_face_name);
			      // we attach the cubemap face into the final rendegraph
			      cube_refl->attach_in(cube_face_name, std::move(lit_fut));
		      }
		      // time to render the final scene
		      // we collect the cubemap faces
		      cube_refl->converge_image_explicit(cube_face_names, "11_cuberefl");
		      // and do the standard deferred rendering as before, this time using the new cube map for the environment
		      cube_refl->attach_and_clear_image("11_position", { .format = vuk::Format::eR16G16B16A16Sfloat, .sample_count = vuk::Samples::e1 }, vuk::White<float>);
		      cube_refl->attach_and_clear_image("11_normal", { .format = vuk::Format::eR16G16B16A16Sfloat }, vuk::White<float>);
		      cube_refl->attach_and_clear_image("11_color", { .format = vuk::Format::eR8G8B8A8Srgb }, vuk::White<float>);
		      cube_refl->attach_and_clear_image("11_depth", { .format = vuk::Format::eD32Sfloat }, vuk::DepthOne);
		      cube_refl->add_pass(
		          { .name = "deferred_MRT",
		            .resources = { "11_position"_image >> vuk::eColorWrite,
		                           "11_normal"_image >> vuk::eColorWrite,
		                           "11_color"_image >> vuk::eColorWrite,
		                           "11_depth"_image >> vuk::eDepthStencilRW,
		                           "11_cuberefl"_image >> vuk::eFragmentSampled },
		            .execute = [uboVP, cam_pos](vuk::CommandBuffer& command_buffer) {
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
			                                    verts,
			                                    0,
			                                    vuk::Packed{ vuk::Format::eR32G32B32Sfloat,
			                                                 vuk::Format::eR32G32B32Sfloat,
			                                                 vuk::Ignore{ offsetof(util::Vertex, uv_coordinates) - offsetof(util::Vertex, tangent) },
			                                                 vuk::Format::eR32G32Sfloat })
			                .bind_index_buffer(inds, vuk::IndexType::eUint32);
			            command_buffer.push_constants(vuk::ShaderStageFlagBits::eFragment, 0, cam_pos).bind_graphics_pipeline("cube_deferred_reflective");
			            command_buffer.bind_image(0, 2, "11_cuberefl").bind_sampler(0, 2, {}).bind_buffer(0, 0, uboVP);
			            glm::mat4* model = command_buffer.map_scratch_uniform_binding<glm::mat4>(0, 1);
			            *model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
			            command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0);
			            for (auto i = 0; i < 64; i++) {
				            if (i == 36)
					            continue;
				            command_buffer.bind_image(0, 2, env_cubemap_ia).bind_sampler(0, 2, {}).bind_buffer(0, 0, uboVP);
				            glm::mat4* model = command_buffer.map_scratch_uniform_binding<glm::mat4>(0, 1);
				            *model = glm::scale(glm::mat4(1.f), glm::vec3(0.1f)) *
				                     glm::translate(glm::mat4(1.f), 4.f * glm::vec3(4 * (i % 8 - 4), sinf(0.1f * angle + i), 4 * (i / 8 - 4)));
				            command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0);
			            }
		            } });
		      cube_refl->attach_image("11_deferred", { .format = vuk::Format::eR8G8B8A8Srgb, .sample_count = vuk::Samples::e1 });
		      cube_refl->inference_rule("11_position+", vuk::same_extent_as("11_deferred"));
		      // The shading pass for the deferred rendering
		      cube_refl->add_pass({ .name = "deferred_resolve",
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
		      vuk::Future lit_fut = { std::move(cube_refl), "11_deferred+" };
		      vuk::Future sm_fut = apply_fxaa(std::move(lit_fut), std::move(target));

		      angle += 10.f * ImGui::GetIO().DeltaTime;

		      return sm_fut;
		    }, // Perform cleanup for the example
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
		      // We release the resources manually
		      env_cubemap.reset();
		      hdr_image.reset();
		    }
	};

	REGISTER_EXAMPLE(x);
} // namespace
