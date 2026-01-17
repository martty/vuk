#include "example_runner.hpp"
#include "vuk/vsl/BindlessArray.hpp"
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <stb_image.h>

/* 09_persistent_descriptorset
 * In this example we will see how to use the BindlessArray utility class for managing bindless descriptors.
 *
 * Normal descriptorsets are completely managed by vuk and are cached based on their contents.
 * However, this behaviour is not helpful if you plan to keep the descriptor sets around, or if they have many elements (such as "bindless").
 *
 *
 * This example demonstrates:
 * - Creating a BindlessArray with combined image samplers
 * - Generating 3 texture variants (original, Y-flipped, color-inverted)
 * - Randomly assigning textures to 10 cubes
 * - Dynamically swapping textures at runtime (every 2 seconds)
 * - Using virtual address allocation for efficient sparse binding
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the same code.
 * Check out the framework (example_runner_*) files if interested!
 */

namespace {
	// The Y rotation angle of our cube
	float angle = 0.f;
	float time_accumulator = 0.f;
	// Generate vertices and indices for the cube
	auto box = util::generate_cube();
	vuk::Unique<vuk::Buffer> verts, inds;

	// Array of 3 doge texture variants
	std::array<vuk::Unique<vuk::Image>, 3> doge_images;
	std::array<vuk::Unique<vuk::ImageView>, 3> doge_image_views;
	std::array<vuk::ImageAttachment, 3> doge_textures;

	const size_t num_cubes = 10;
	std::vector<glm::vec3> cube_positions;

	std::uniform_int_distribution<size_t> rand_indices(0, 2);
	std::optional<vuk::BindlessArray> bindless_textures;

	std::random_device rd;
	std::mt19937 gen(rd());

	vuk::Example xample{
		.name = "09_persistent_descriptorset",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator, vuk::Runtime& runtime) {
		      // Create BindlessArray - it will create both the VirtualAddressSpace and PersistentDescriptorSet internally
		      bindless_textures.emplace(allocator, 1, vuk::BindlessArray::Bindings{ .combined_image_sampler = 0 }, 64);

		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file((root / "examples/bindless.vert").generic_string()), (root / "examples/bindless.vert").generic_string());
			      pci.add_glsl(util::read_entire_file((root / "examples/triangle_tex_bindless.frag").generic_string()),
			                   (root / "examples/triangle_tex_bindless.frag").generic_string());
			      // Use the descriptor set layout from BindlessArray instead of declaring it in the pipeline
			      pci.explicit_set_layouts.push_back(bindless_textures->get_descriptor_set_layout());
			      runtime.create_named_pipeline("bindless_cube", pci);
		      }

		      // creating a compute pipeline that inverts an image
		      {
			      vuk::PipelineBaseCreateInfo pbci;
			      pbci.add_glsl(util::read_entire_file((root / "examples/invert.comp").generic_string()), "examples/invert.comp");
			      runtime.create_named_pipeline("invert", pbci);
		      }

		      // Use STBI to load the image
		      int x, y, chans;
		      auto doge_image = stbi_load((root / "examples/doge.png").generic_string().c_str(), &x, &y, &chans, 4);

		      // Similarly to buffers, we allocate the image and enqueue the upload
		      doge_textures[0] = vuk::ImageAttachment::from_preset(
		          vuk::ImageAttachment::Preset::eMap2D, vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, vuk::Samples::e1);
		      doge_textures[0].usage |= vuk::ImageUsageFlagBits::eTransferSrc;
		      doge_textures[0].level_count = 1;
		      auto [image, view, doge_src] =
		          vuk::create_image_and_view_with_data(allocator, vuk::DomainFlagBits::eTransferOnTransfer, doge_textures[0], doge_image);
		      doge_images[0] = std::move(image);
		      doge_image_views[0] = std::move(view);
		      stbi_image_free(doge_image);

		      // We set up the cube data, same as in example 02_cube
		      auto [vert_buf, vert_fut] = create_buffer(allocator, vuk::MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.first));
		      verts = std::move(vert_buf);
		      auto [ind_buf, ind_fut] = create_buffer(allocator, vuk::MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.second));
		      inds = std::move(ind_buf);
		      // For the example, we just ask these that these uploads complete before moving on to rendering
		      // In an engine, you would integrate these uploads into some explicit system
		      runner.enqueue_setup(std::move(vert_fut));
		      runner.enqueue_setup(std::move(ind_fut));

		      // Let's create two variants of the doge image
		      // Variant 1: Y-flipped version
		      doge_textures[1] = doge_textures[0];
		      doge_textures[1].usage = vuk::ImageUsageFlagBits::eTransferDst | vuk::ImageUsageFlagBits::eSampled;
		      doge_images[1] = *vuk::allocate_image(allocator, doge_textures[1]);
		      doge_textures[1].image = *doge_images[1];
		      doge_image_views[1] = *vuk::allocate_image_view(allocator, doge_textures[1]);
		      doge_textures[1].image_view = *doge_image_views[1];

		      // Variant 2: Inverted colors version
		      doge_textures[2] = doge_textures[0];
		      doge_textures[2].format = vuk::Format::eR8G8B8A8Unorm;
		      doge_textures[2].usage = vuk::ImageUsageFlagBits::eStorage | vuk::ImageUsageFlagBits::eSampled;
		      doge_images[2] = *vuk::allocate_image(allocator, doge_textures[2]);
		      doge_textures[2].image = *doge_images[2];
		      doge_image_views[2] = *vuk::allocate_image_view(allocator, doge_textures[2]);
		      doge_textures[2].image_view = *doge_image_views[2];

		      // Make a RenderGraph to process the loaded image
		      auto doge_v1 = vuk::declare_ia("09_doge_v1", doge_textures[1]);
		      auto doge_v2 = vuk::declare_ia("09_doge_v2", doge_textures[2]);

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
		      auto [src, v1, v2] = preprocess(std::move(doge_src), std::move(doge_v1), std::move(doge_v2));
		      src.release(vuk::Access::eFragmentSampled, vuk::DomainFlagBits::eGraphicsQueue);
		      v1.release(vuk::Access::eFragmentSampled, vuk::DomainFlagBits::eGraphicsQueue);
		      v2.release(vuk::Access::eFragmentSampled, vuk::DomainFlagBits::eGraphicsQueue);
		      // enqueue running the preprocessing rendergraph
		      runner.enqueue_setup(std::move(src));
		      runner.enqueue_setup(std::move(v1));
		      runner.enqueue_setup(std::move(v2));

		      // Initially add all three textures
		      vuk::Sampler default_sampler = allocator.get_context().acquire_sampler({}, allocator.get_context().get_frame_count());

		      // Generate random textures for the cubes
		      for (size_t i = 0; i < num_cubes; i++) {
			      bindless_textures->push_back(*doge_image_views[rand_indices(gen)], default_sampler, vuk::ImageLayout::eReadOnlyOptimalKHR);
		      }

		      // Generate random positions for cubes

		      std::uniform_real_distribution<float> pos_dist(-5.0f, 5.0f);
		      std::uniform_real_distribution<float> y_dist(-2.0f, 2.0f);

		      cube_positions.reserve(num_cubes);
		      for (size_t i = 0; i < num_cubes; i++) {
			      cube_positions.push_back(glm::vec3(pos_dist(gen), y_dist(gen), pos_dist(gen)));
		      }
		    },
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::Value<vuk::ImageAttachment> target) {
		      struct VP {
			      glm::mat4 view;
			      glm::mat4 proj;
		      } vp;
		      vp.view = glm::lookAt(glm::vec3(0, 1.5, 5.5), glm::vec3(0), glm::vec3(0, 1, 0));
		      vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);
		      vp.proj[1][1] *= -1;

		      auto [buboVP, uboVP_fut] = create_buffer(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, vuk::DomainFlagBits::eTransferOnGraphics, std::span(&vp, 1));
		      auto uboVP = *buboVP;

		      float delta_time = ImGui::GetIO().DeltaTime;
		      time_accumulator += delta_time;

		      // Dynamically swap textures every 2 seconds to demonstrate bindless updates
		      static float last_toggle = 0.f;
		      if (time_accumulator - last_toggle > 2.0f) {
			      last_toggle = time_accumulator;
			      vuk::Sampler default_sampler = frame_allocator.get_context().acquire_sampler({}, frame_allocator.get_context().get_frame_count());

			      // Remove the first cube's texture and add a new random one
			      // This demonstrates that indices can be freed and reused efficiently
			      bindless_textures->erase(bindless_textures->get_active_indices()[0]);

			      // Add a texture back with a random variant
			      bindless_textures->push_back(*doge_image_views[rand_indices(gen)], default_sampler, vuk::ImageLayout::eReadOnlyOptimalKHR);
		      }
		      // Commit any pending descriptor updates before rendering
		      bindless_textures->commit();

		      // Set up the pass to draw the textured cubes
		      auto forward_pass =
		          vuk::make_pass("forward", [uboVP](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eColorWrite) color, VUK_IA(vuk::eDepthStencilRW) depth) {
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
			                                  *verts,
			                                  0,
			                                  vuk::Packed{ vuk::Format::eR32G32B32Sfloat,
			                                               vuk::Ignore{ offsetof(util::Vertex, uv_coordinates) - sizeof(util::Vertex::position) },
			                                               vuk::Format::eR32G32Sfloat })
			              .bind_index_buffer(*inds, vuk::IndexType::eUint32)
			              .bind_persistent(1, bindless_textures->get_persistent_set())
			              .bind_graphics_pipeline("bindless_cube")
			              .bind_buffer(0, 0, uboVP);
			          glm::mat4* model = command_buffer.scratch_buffer<glm::mat4>(0, 1);
			          *model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));

			          // Draw cubes at random positions with textures from the bindless array
			          auto indices = bindless_textures->get_active_indices();
			          for (size_t i = 0; i < indices.size(); i++) {
				          // Push the position for this cube
				          command_buffer.push_constants(vuk::ShaderStageFlagBits::eVertex, 0, cube_positions[i]);
				          // Draw the cube with the corresponding texture index
				          // We use the instance index as another "push constant" to index the textures
				          command_buffer.draw_indexed(box.second.size(), 1, 0, 0, indices[i]);
			          }

			          return color;
		          });

		      angle += 10.f * delta_time;

		      auto depth_img = vuk::declare_ia("09_depth");
		      depth_img->format = vuk::Format::eD32Sfloat;
		      depth_img = vuk::clear_image(std::move(depth_img), vuk::ClearDepthStencil{ 1.0f, 0 });

		      return forward_pass(std::move(target), std::move(depth_img));
		    },

		// Perform cleanup for the example
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
		      // We release the resources manually
		      verts.reset();
		      inds.reset();
		      for (auto& img : doge_images) {
			      img.reset();
		      }
		      for (auto& view : doge_image_views) {
			      view.reset();
		      }
		      bindless_textures = {};
		    }
	};

	REGISTER_EXAMPLE(xample);
} // namespace