#include "example_runner.hpp"
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <stb_image.h>

/* 10_baby_renderer
 *
 * In this example we make a small (baby) renderer (not for rendering babies!).
 * Here we use very simple (read: potentially not performant or convenient) abstractions.
 * The goal is that we can render diverse object with a single simple loop render loop.
 *
 * Generally resources can be bound individually to draws, here we show this for textures and a material buffer.
 * Or they can be aggregated into arrays and indexed in the shader, which is done here for model matrices.
 * This small example shows no state deduplication or sorting, which are very good optimizations for a real renderer.
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

	struct Mesh {
		vuk::Unique<vuk::Buffer> vertex_buffer;
		vuk::Unique<vuk::Buffer> index_buffer;
		uint32_t index_count;
	};

	struct Material {
		vuk::PipelineBaseInfo* pipeline;
		virtual void bind_parameters(vuk::CommandBuffer&){};
		virtual void bind_textures(vuk::CommandBuffer&){};
	};

	struct NormalMaterial : Material {
		vuk::ImageView texture;

		void bind_textures(vuk::CommandBuffer& cbuf) override {
			cbuf.bind_image(0, 2, texture).bind_sampler(0, 2, {});
		}
	};

	struct TintMaterial : Material {
		vuk::ImageView texture;
		glm::vec4 tint_color = glm::vec4(0);

		void bind_textures(vuk::CommandBuffer& cbuf) override {
			cbuf.bind_image(0, 2, texture).bind_sampler(0, 2, {});
		}

		void bind_parameters(vuk::CommandBuffer& cbuf) override {
			*cbuf.scratch_buffer<glm::vec4>(0, 3) = tint_color;
		}
	};

	std::optional<Mesh> cube_mesh, quad_mesh;

	struct Renderable {
		Mesh* mesh;
		Material* material;

		glm::vec3 position;
		glm::vec3 velocity = glm::vec3(0);
	};

	vuk::Unique<vuk::Image> image_of_doge;
	vuk::Unique<vuk::ImageView> image_view_of_doge;
	vuk::ImageAttachment texture_of_doge;
	vuk::Unique<vuk::Image> image_of_doge_v1;
	vuk::Unique<vuk::ImageView> image_view_of_doge_v1;
	vuk::ImageAttachment texture_of_doge_v1;
	vuk::Unique<vuk::Image> image_of_doge_v2;
	vuk::Unique<vuk::ImageView> image_view_of_doge_v2;
	vuk::ImageAttachment texture_of_doge_v2;

	std::vector<NormalMaterial> nmats;
	std::vector<TintMaterial> tmats;

	std::vector<Renderable> renderables;

	std::random_device rd;
	std::mt19937 g(rd());

	vuk::Example xample{
		.name = "10_baby_renderer",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator, vuk::Runtime& runtime) {
		      // Use STBI to load the image
		      int x, y, chans;
		      auto doge_image = stbi_load((root / "examples/doge.png").generic_string().c_str(), &x, &y, &chans, 4);

		      // Similarly to buffers, we allocate the image and enqueue the upload
		      texture_of_doge = vuk::ImageAttachment::from_preset(
		          vuk::ImageAttachment::Preset::eMap2D, vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, vuk::Samples::e1);
		      texture_of_doge.usage |= vuk::ImageUsageFlagBits::eTransferSrc;
		      texture_of_doge.level_count = 1;
		      auto [image, view, doge_src] = vuk::create_image_and_view_with_data(allocator, vuk::DomainFlagBits::eTransferOnTransfer, texture_of_doge, doge_image);
		      image_of_doge = std::move(image);
		      image_view_of_doge = std::move(view);
		      stbi_image_free(doge_image);

		      // Let's create two variants of the doge image (like in example 09)
		      // Creating a compute pipeline that inverts an image
		      {
			      vuk::PipelineBaseCreateInfo pbci;
			      pbci.add_glsl(util::read_entire_file((root / "examples/invert.comp").generic_string()), (root / "examples/invert.comp").generic_string());
			      runtime.create_named_pipeline("invert", pbci);
		      }
		      texture_of_doge_v1 = texture_of_doge;
		      texture_of_doge_v1.usage = vuk::ImageUsageFlagBits::eTransferDst | vuk::ImageUsageFlagBits::eSampled;
		      image_of_doge_v1 = *vuk::allocate_image(allocator, texture_of_doge_v1);
		      texture_of_doge_v1.image = *image_of_doge_v1;
		      image_view_of_doge_v1 = *vuk::allocate_image_view(allocator, texture_of_doge_v1);
		      texture_of_doge_v1.image_view = *image_view_of_doge_v1;
		      texture_of_doge_v2 = texture_of_doge;
		      texture_of_doge_v2.format = vuk::Format::eR8G8B8A8Unorm;
		      texture_of_doge_v2.usage = vuk::ImageUsageFlagBits::eStorage | vuk::ImageUsageFlagBits::eSampled;
		      image_of_doge_v2 = *vuk::allocate_image(allocator, texture_of_doge_v2);
		      texture_of_doge_v2.image = *image_of_doge_v2;
		      image_view_of_doge_v2 = *vuk::allocate_image_view(allocator, texture_of_doge_v2);
		      texture_of_doge_v2.image_view = *image_view_of_doge_v2;

		      // Make a RenderGraph to process the loaded image
		      auto doge_v1 = vuk::declare_ia("09_doge_v1", texture_of_doge_v1);
		      auto doge_v2 = vuk::declare_ia("09_doge_v2", texture_of_doge_v2);

		      auto preprocess = vuk::make_pass(
		          "preprocess",
		          [x, y](vuk::CommandBuffer& command_buffer,
		                 VUK_IA(vuk::eTransferRead | vuk::eComputeSampled) src,
		                 VUK_IA(vuk::eTransferWrite) v1,
		                 VUK_IA(vuk::eComputeWrite) v2) {
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
			          command_buffer.blit_image(src, v1, blit, vuk::Filter::eLinear);
			          // For the second image, invert the colours in compute
			          command_buffer.bind_image(0, 0, src).bind_sampler(0, 0, {}).bind_image(0, 1, v2).bind_compute_pipeline("invert").dispatch_invocations(x, y);

			          return std::make_tuple(src, v1, v2);
		          });
		      // Bind the resources for the variant generation
		      // We specify the initial and final access
		      // The texture we have created is already in ShaderReadOptimal, but we need it in General during the pass, and we need it back to ShaderReadOptimal
		      // afterwards
		      auto [src, v1, v2] = preprocess(std::move(doge_src), std::move(doge_v1), std::move(doge_v2));
		      src.release(vuk::Access::eFragmentSampled, vuk::DomainFlagBits::eGraphicsQueue);
		      v1.release(vuk::Access::eFragmentSampled, vuk::DomainFlagBits::eGraphicsQueue);
		      v2.release(vuk::Access::eFragmentSampled, vuk::DomainFlagBits::eGraphicsQueue);
		      // enqueue running the preprocessing rendergraph and force 09_doge to be sampleable later
		      runner.enqueue_setup(std::move(src));
		      runner.enqueue_setup(std::move(v1));
		      runner.enqueue_setup(std::move(v2));

		      // Set up the resources for our renderer

		      // Create meshes
		      cube_mesh.emplace();
		      auto [vert_buf, vert_fut] = create_buffer(allocator, vuk::MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnTransfer, std::span(box.first));
		      cube_mesh->vertex_buffer = std::move(vert_buf);
		      auto [idx_buf, idx_fut] = create_buffer(allocator, vuk::MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnTransfer, std::span(box.second));
		      cube_mesh->index_buffer = std::move(idx_buf);
		      cube_mesh->index_count = (uint32_t)box.second.size();

		      quad_mesh.emplace();
		      auto [vert_buf2, vert_fut2] =
		          create_buffer(allocator, vuk::MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnTransfer, std::span(&box.first[0], 6));
		      quad_mesh->vertex_buffer = std::move(vert_buf2);
		      auto [idx_buf2, idx_fut2] =
		          create_buffer(allocator, vuk::MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnTransfer, std::span(&box.second[0], 6));
		      quad_mesh->index_buffer = std::move(idx_buf2);
		      quad_mesh->index_count = 6;

		      runner.enqueue_setup(std::move(vert_fut));
		      runner.enqueue_setup(std::move(idx_fut));
		      runner.enqueue_setup(std::move(vert_fut2));
		      runner.enqueue_setup(std::move(idx_fut2));

		      // Create the pipelines
		      // A "normal" pipeline
		      vuk::PipelineBaseInfo* pipe1;
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file((root / "examples/baby_renderer.vert").generic_string()),
			                   (root / "examples/baby_renderer.vert").generic_string());
			      pci.add_glsl(util::read_entire_file((root / "examples/triangle_depthshaded_tex.frag").generic_string()),
			                   (root / "examples/triangle_depthshaded_tex.frag").generic_string());
			      pipe1 = runtime.get_pipeline(pci);
		      }

		      // A "tinted" pipeline
		      vuk::PipelineBaseInfo* pipe2;
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file((root / "examples/baby_renderer.vert").generic_string()),
			                   (root / "examples/baby_renderer.vert").generic_string());
			      pci.add_glsl(util::read_entire_file((root / "examples/triangle_tinted_tex.frag").generic_string()),
			                   (root / "examples/triangle_tinted_tex.frag").generic_string());
			      pipe2 = runtime.get_pipeline(pci);
		      }

		      // Create materials
		      nmats.resize(3);
		      nmats[0].texture = image_view_of_doge.get();
		      nmats[1].texture = image_view_of_doge_v1.get();
		      nmats[2].texture = image_view_of_doge_v2.get();
		      for (auto& mat : nmats) {
			      mat.pipeline = pipe1;
		      }

		      tmats.resize(3);
		      std::uniform_real_distribution<float> dist_tint(0, 1);
		      tmats[0].texture = image_view_of_doge.get();
		      tmats[1].texture = image_view_of_doge_v1.get();
		      tmats[2].texture = image_view_of_doge_v2.get();
		      for (auto& mat : tmats) {
			      mat.pipeline = pipe2;
			      mat.tint_color = glm::vec4(dist_tint(g), dist_tint(g), dist_tint(g), 1.f);
		      }

		      // Create objects
		      std::uniform_int_distribution<size_t> dist_mat(0, 1);
		      std::uniform_int_distribution<size_t> dist_tex(0, 2);

		      std::uniform_real_distribution<float> dist_pos(-10, 10);

		      // 64 quads
		      for (int i = 0; i < 64; i++) {
			      auto mat_id = dist_mat(g);
			      auto tex_id = dist_tex(g);

			      Material* m = mat_id == 0 ? (Material*)&nmats[tex_id] : (Material*)&tmats[tex_id];
			      glm::vec3 pos = glm::vec3(dist_pos(g), dist_pos(g), dist_pos(g));
			      renderables.emplace_back(Renderable{ .mesh = &*quad_mesh, .material = m, .position = pos });
		      }

		      // 16 cubes
		      for (int i = 0; i < 16; i++) {
			      auto mat_id = dist_mat(g);
			      auto tex_id = dist_tex(g);

			      Material* m = mat_id == 0 ? (Material*)&nmats[tex_id] : (Material*)&tmats[tex_id];
			      glm::vec3 pos = glm::vec3(dist_pos(g), dist_pos(g), dist_pos(g));
			      renderables.emplace_back(Renderable{ .mesh = &*cube_mesh, .material = m, .position = pos });
		      }
		    },
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::Value<vuk::ImageAttachment> target) {
		      // We set up VP data, same as in example 02_cube
		      struct VP {
			      glm::mat4 view;
			      glm::mat4 proj;
		      } vp;
		      vp.view = glm::lookAt(glm::vec3(0, 10.0, 11), glm::vec3(0), glm::vec3(0, 1, 0));
		      vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 100.f);
		      vp.proj[1][1] *= -1;

		      // Upload view & projection
		      auto [buboVP, uboVP_fut] = create_buffer(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, vuk::DomainFlagBits::eTransferOnGraphics, std::span(&vp, 1));
		      auto uboVP = *buboVP;

		      // Do a terrible simulation step
		      // All objects are attracted to the origin
		      for (auto& r : renderables) {
			      auto force_mag = 0.1f / glm::length(r.position);
			      r.velocity += force_mag * (-r.position) * ImGui::GetIO().DeltaTime;
			      r.position += r.velocity * ImGui::GetIO().DeltaTime;
		      }

		      // Upload model matrices to an array
		      auto modelmats = **allocate_buffer(frame_allocator, { vuk::MemoryUsage::eCPUtoGPU, sizeof(glm::mat4) * renderables.size(), 1 });
		      for (auto i = 0; i < renderables.size(); i++) {
			      glm::mat4 model_matrix = glm::translate(glm::mat4(1.f), renderables[i].position);
			      memcpy(reinterpret_cast<glm::mat4*>(modelmats.mapped_ptr) + i, &model_matrix, sizeof(glm::mat4));
		      }

		      auto forward_pass = vuk::make_pass(
		          "forward", [uboVP, modelmats](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eColorWrite) color, VUK_IA(vuk::eDepthStencilRW) depth) {
			          command_buffer.set_dynamic_state(vuk::DynamicStateFlagBits::eViewport | vuk::DynamicStateFlagBits::eScissor)
			              .set_viewport(0, vuk::Rect2D::framebuffer())
			              .set_scissor(0, vuk::Rect2D::framebuffer())
			              .set_rasterization({}) // Set the default rasterization state
			              // Set the depth/stencil state
			              .set_depth_stencil(vuk::PipelineDepthStencilStateCreateInfo{
			                  .depthTestEnable = true,
			                  .depthWriteEnable = true,
			                  .depthCompareOp = vuk::CompareOp::eLessOrEqual,
			              })
			              .broadcast_color_blend({}); // Set the default color blend state

			          // These binds don't change between meshes, so it is sufficient to bind them once
			          command_buffer.bind_buffer(0, 0, uboVP).bind_buffer(0, 1, modelmats);

			          for (auto i = 0; i < renderables.size(); i++) {
				          auto& r = renderables[i];

				          // Set up the draw state based on the mesh and material
				          command_buffer
				              .bind_vertex_buffer(0,
				                                  r.mesh->vertex_buffer.get(),
				                                  0,
				                                  vuk::Packed{ vuk::Format::eR32G32B32Sfloat,
				                                               vuk::Ignore{ offsetof(util::Vertex, uv_coordinates) - sizeof(util::Vertex::position) },
				                                               vuk::Format::eR32G32Sfloat })
				              .bind_index_buffer(r.mesh->index_buffer.get(), vuk::IndexType::eUint32)
				              .bind_graphics_pipeline(r.material->pipeline);

				          r.material->bind_parameters(command_buffer);
				          r.material->bind_textures(command_buffer);

				          // Draw the mesh, assign them different base instance to pick the correct transformation
				          command_buffer.draw_indexed(r.mesh->index_count, 1, 0, 0, i);
			          }

					  return color;
		          });

		      angle += 10.f * ImGui::GetIO().DeltaTime;

		      auto depth_img = vuk::declare_ia("09_depth");
		      depth_img->format = vuk::Format::eD32Sfloat;
		      depth_img = vuk::clear_image(std::move(depth_img), vuk::ClearDepthStencil{ 1.0f, 0 });

		      return forward_pass(std::move(target), std::move(depth_img));
		    },
		// Perform cleanup for the example
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
		      // We release the resources manually
		      cube_mesh.reset();
		      quad_mesh.reset();
		      image_of_doge.reset();
		      image_view_of_doge.reset();
		      image_of_doge_v1.reset();
		      image_view_of_doge_v1.reset();
		      image_of_doge_v2.reset();
		      image_view_of_doge_v2.reset();
		    }
	};

	REGISTER_EXAMPLE(xample);
} // namespace