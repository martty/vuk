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

	std::vector<vuk::Unique<vuk::Image>> doge_images;
	std::vector<vuk::Unique<vuk::ImageView>> doge_image_views;
	vuk::Value<vuk::ImageAttachment> initial_doge_texture;
	std::vector<vuk::Value<vuk::ImageAttachment>> pending_textures;

	const size_t max_cubes = 60;
	std::vector<glm::vec3> cube_positions;
	std::vector<uint32_t> texture_indices;

	std::optional<vuk::BindlessArray> bindless_textures;
	std::uniform_real_distribution<float> pos_dist(-5.0f, 5.0f);
	std::uniform_real_distribution<float> y_dist(-2.0f, 2.0f);
	std::uniform_real_distribution<float> color_dist(0.0f, 1.0f);

	std::random_device rd;
	std::mt19937 gen(rd());

	// Create a lifted compute pass for tinting
	auto tint_pass = vuk::lift_compute(vuk::PipelineBaseCreateInfo::from_inline_glsl(R"(#version 450
#pragma shader_stage(compute)

layout(binding = 0) uniform sampler2D inputImage;
layout(binding = 1, rgba8) uniform writeonly image2D outputImage;

layout(push_constant) uniform PushConstants {
	float r;
	float g;
	float b;
} pc;

layout(local_size_x = 8, local_size_y = 8) in;

void main() {
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	ivec2 imgSize = imageSize(outputImage);
	
	if (coord.x >= imgSize.x || coord.y >= imgSize.y) {
		return;
	}
	
	vec2 uv = (vec2(coord) + 0.5) / vec2(imgSize);
	vec4 color = texture(inputImage, uv);
	color.rgb *= vec3(pc.r, pc.g, pc.b);
	imageStore(outputImage, coord, color);
}
)"));
	// First pass: flip the image on the Y axis using a blit
	auto flip_pass = vuk::make_pass("flip", [](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eTransferRead) src, VUK_IA(vuk::eTransferWrite) dst) {
		vuk::ImageBlit blit;
		blit.srcSubresource.aspectMask = vuk::ImageAspectFlagBits::eColor;
		blit.srcSubresource.baseArrayLayer = 0;
		blit.srcSubresource.layerCount = 1;
		blit.srcSubresource.mipLevel = 0;
		blit.srcOffsets[0] = vuk::Offset3D{ 0, 0, 0 };
		blit.srcOffsets[1] = vuk::Offset3D{ (int)src->extent.width, (int)src->extent.height, 1 };
		blit.dstSubresource = blit.srcSubresource;
		blit.dstOffsets[0] = vuk::Offset3D{ (int)src->extent.width, (int)src->extent.height, 0 };
		blit.dstOffsets[1] = vuk::Offset3D{ 0, 0, 1 };
		command_buffer.blit_image(src, dst, blit, vuk::Filter::eLinear);
	});

	// Second pass: invert the colors using compute
	auto invert_pass = vuk::make_pass("invert", [](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eComputeSampled) src, VUK_IA(vuk::eComputeWrite) dst) {
		command_buffer.bind_image(0, 0, src).bind_sampler(0, 0, {}).bind_image(0, 1, dst).bind_compute_pipeline("invert").dispatch_invocations_per_pixel(dst);
	});

	vuk::Example xample{
		.name = "09_persistent_descriptorset",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator, vuk::Runtime& runtime) {
		      // Create BindlessArray - it will create both the VirtualAddressSpace and PersistentDescriptorSet internally
		      bindless_textures.emplace(allocator, 1, vuk::BindlessArray::Bindings{ .combined_image_sampler = 0 }, 1024);

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
		      auto doge_ia = vuk::ImageAttachment::from_preset(
		          vuk::ImageAttachment::Preset::eMap2D, vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, vuk::Samples::e1);
		      doge_ia.usage |= vuk::ImageUsageFlagBits::eTransferSrc;
		      doge_ia.level_count = 1;
		      auto [image, view, doge_src] = vuk::create_image_and_view_with_data(allocator, vuk::DomainFlagBits::eTransferOnTransfer, doge_ia, doge_image);
		      doge_images.push_back(std::move(image));
		      doge_image_views.push_back(std::move(view));
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

		      initial_doge_texture = std::move(doge_src);
		    },
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::Value<vuk::ImageAttachment> target) {
		      struct VP {
			      glm::mat4 view;
			      glm::mat4 proj;
		      } vp;
		      vp.view = glm::lookAt(glm::vec3(0, 2.5, 7.5), glm::vec3(0), glm::vec3(0, 1, 0));
		      vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);
		      vp.proj[1][1] *= -1;

		      auto [buboVP, uboVP_fut] = create_buffer(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, vuk::DomainFlagBits::eTransferOnGraphics, std::span(&vp, 1));
		      auto uboVP = *buboVP;

		      float delta_time = ImGui::GetIO().DeltaTime;
		      time_accumulator += delta_time;

		      // Dynamically swap textures every 2 seconds to demonstrate bindless updates
		      static float last_toggle = 0.f;
		      auto num_cubes = cube_positions.size();
		      if (time_accumulator - last_toggle > 1.0f || last_toggle == 0.f && num_cubes < max_cubes) {
			      last_toggle = time_accumulator;
			      // Obtain the superframe allocator to allocate images
			      auto sf_allocator = *runner.app->superframe_allocator;

			      vuk::ImageAttachment ia = vuk::ImageAttachment::from_preset(
			          vuk::ImageAttachment::Preset::eMap2D,
			          vuk::Format::eR8G8B8A8Unorm,
			          vuk::Extent3D{ (unsigned)initial_doge_texture->extent.width, (unsigned)initial_doge_texture->extent.height, 1u },
			          vuk::Samples::e1);
			      ia.usage = vuk::ImageUsageFlagBits::eStorage | vuk::ImageUsageFlagBits::eSampled | vuk::ImageUsageFlagBits::eTransferDst;
			      ia.level_count = 1;
			      // Store allocations
			      ia.image = *doge_images.emplace_back(*vuk::allocate_image(sf_allocator, ia));
			      ia.image_view = *doge_image_views.emplace_back(*vuk::allocate_image_view(sf_allocator, ia));

			      auto image_to_process = vuk::discard_ia("09_doge_i", ia);
			      // Randomly choose a processing operation to do on the image
			      std::uniform_int_distribution<size_t> process_to_do(0, 3); // 0: none, 1: flip, 2: invert, 3: tint
			      auto choice = process_to_do(gen);
			      switch (choice) {
			      case 0:
				      copy(initial_doge_texture, image_to_process);
				      break;
			      case 1:
				      flip_pass(initial_doge_texture, image_to_process);
				      break;
			      case 2:
				      invert_pass(initial_doge_texture, image_to_process);
				      break;
			      case 3: {
				      vuk::Value<float> tint_r = vuk::make_constant("r", color_dist(gen));
				      vuk::Value<float> tint_g = vuk::make_constant("g", color_dist(gen));
				      vuk::Value<float> tint_b = vuk::make_constant("b", color_dist(gen));
				      tint_pass(initial_doge_texture->extent.width / 8,
				                initial_doge_texture->extent.height / 8,
				                1,
				                vuk::combine_image_sampler("ci", initial_doge_texture, vuk::acquire_sampler("default_sampler", {})),
				                image_to_process,
				                tint_r,
				                tint_g,
				                tint_b);
			      } break;
			      }

			      image_to_process.release(vuk::Access::eFragmentSampled, vuk::DomainFlagBits::eGraphicsQueue);
			      image_to_process.submit(frame_allocator, runner.compiler);
			      pending_textures.push_back(image_to_process);
		      }

		      // Check if any pending textures have completed processing
		      for (auto it = pending_textures.begin(); it != pending_textures.end();) {
			      auto& tex = *it;
			      auto status = tex.poll();
			      if (status && status.value() == vuk::Signal::Status::eHostAvailable) {
				      vuk::Sampler default_sampler = frame_allocator.get_context().acquire_sampler({}, frame_allocator.get_context().get_frame_count());

				      // create a new cube with a random position and texture
				      cube_positions.push_back(glm::vec3(pos_dist(gen), y_dist(gen), pos_dist(gen)));
				      // Add the new texture to the bindless array
				      uint32_t new_idx = bindless_textures->push_back(tex->image_view, default_sampler, vuk::ImageLayout::eReadOnlyOptimalKHR);
				      texture_indices.push_back(new_idx);
				      // Remove from pending list
				      it = pending_textures.erase(it);
			      } else {
				      ++it;
			      }
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
			          // Use the tracked indices to index into the descriptor array
			          for (size_t i = 0; i < cube_positions.size(); i++) {
				          // Push the position for this cube
				          command_buffer.push_constants(vuk::ShaderStageFlagBits::eVertex, 0, cube_positions[i]);
				          // Draw the cube with the corresponding texture index
				          // The instance index is used to reference the texture in the bindless array
				          command_buffer.draw_indexed(box.second.size(), 1, 0, 0, texture_indices[i]);
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