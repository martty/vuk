#include "example_runner.hpp"
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <numeric>
#include <random>
#include <stb_image.h>

/* 08_pipelined_compute
 * In this example we will see how to run compute shaders on the graphics queue.
 * To showcases this, we will render a texture to a fullscreen framebuffer,
 * then display it, but scramble the pixels determined by indices in a storage buffer.
 * Between these two steps, we perform some iterations of bubble sort on the indices buffer in compute.
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the example same code.
 * Check out the framework (example_runner_*) files if interested!
 */

namespace {
	float time = 0.f;
	auto box = util::generate_cube();
	int x, y;
	uint32_t speed_count = 1;
	vuk::Unique<vuk::Image> image_of_doge;
	vuk::Unique<vuk::ImageView> image_view_of_doge;
	vuk::ImageAttachment texture_of_doge;
	vuk::Unique<vuk::Buffer> scramble_buf;
	std::random_device rd;
	std::mt19937 g(rd());
	vuk::Value<vuk::Buffer> scramble_buf_fut;
	vuk::Value<vuk::ImageAttachment> texture_of_doge_fut;

	vuk::Example xample{
		.name = "08_pipelined_compute",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator, vuk::Runtime& runtime) {
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file((root / "examples/fullscreen.vert").generic_string()), (root / "examples/fullscreen.vert").generic_string());
			      pci.add_glsl(util::read_entire_file((root / "examples/rtt.frag").generic_string()), (root / "examples/rtt.frag").generic_string());
			      runtime.create_named_pipeline("rtt", pci);
		      }

		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file((root / "examples/fullscreen.vert").generic_string()), (root / "examples/fullscreen.vert").generic_string());
			      pci.add_glsl(util::read_entire_file((root / "examples/scrambled_draw.frag").generic_string()),
			                   (root / "examples/scrambled_draw.frag").generic_string());
			      runtime.create_named_pipeline("scrambled_draw", pci);
		      }

		      // creating a compute pipeline is the same as creating a graphics pipeline
		      {
			      vuk::PipelineBaseCreateInfo pbci;
			      pbci.add_glsl(util::read_entire_file((root / "examples/stupidsort.comp").generic_string()), "examples/stupidsort.comp");
			      runtime.create_named_pipeline("stupidsort", pbci);
		      }

		      int chans;
		      auto doge_image = stbi_load((root / "examples/doge.png").generic_string().c_str(), &x, &y, &chans, 4);

		      texture_of_doge = vuk::ImageAttachment::from_preset(
		          vuk::ImageAttachment::Preset::eMap2D, vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, vuk::Samples::e1);
		      texture_of_doge.level_count = 1;
		      auto [image, view, future] = vuk::create_image_and_view_with_data(allocator, vuk::DomainFlagBits::eTransferOnTransfer, texture_of_doge, doge_image);
		      image_of_doge = std::move(image);
		      image_view_of_doge = std::move(view);
		      runner.enqueue_setup(future.as_released(vuk::Access::eFragmentSampled, vuk::DomainFlagBits::eGraphicsQueue));
		      stbi_image_free(doge_image);

		      // init scrambling buffer
		      std::vector<unsigned> indices(x * y);
		      std::iota(indices.begin(), indices.end(), 0);
		      std::shuffle(indices.begin(), indices.end(), g);

		      scramble_buf = *allocate_buffer(allocator, { vuk::MemoryUsage::eGPUonly, sizeof(unsigned) * x * y, 1 });

		      // make a GPU future
		      scramble_buf_fut = vuk::host_data_to_buffer(allocator, vuk::DomainFlagBits::eTransferOnTransfer, scramble_buf.get(), std::span(indices));
		      runner.enqueue_setup(scramble_buf_fut);
		    },
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::Value<vuk::ImageAttachment> target) {
		      auto rttf = vuk::clear_image(
		          vuk::declare_ia(
		              "08_rttf",
		              { .extent = { (unsigned)x, (unsigned)y }, .format = runner.app->swapchain->images[0].format, .sample_count = vuk::Samples::e1, .layer_count = 1 }),
		          vuk::ClearColor{ 0.f, 0.f, 0.f, 1.f });

		      // standard render to texture
		      auto rtt_pass = vuk::make_pass(
		          "rtt",
		          [](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eColorWrite) target) {
			          command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
			              .set_scissor(0, vuk::Rect2D::framebuffer())
			              .set_rasterization({})     // Set the default rasterization state
			              .broadcast_color_blend({}) // Set the default color blend state
			              .bind_image(0, 0, texture_of_doge.image_view)
			              .bind_sampler(0, 0, {})
			              .bind_graphics_pipeline("rtt")
			              .draw(3, 1, 0, 0);

			          return target;
		          },
		          vuk::DomainFlagBits::eGraphicsQueue);

		      rttf = rtt_pass(rttf);

		      // this pass executes outside of a renderpass
		      // we declare a buffer dependency and dispatch a compute shader
		      auto sort_pass = vuk::make_pass(
		          "sort",
		          [](vuk::CommandBuffer& command_buffer, VUK_BA(vuk::eComputeRW) scramble) {
			          command_buffer.bind_buffer(0, 0, scramble);
			          command_buffer.bind_compute_pipeline("stupidsort").specialize_constants(0, speed_count).dispatch(1);
			          // We can also customize pipelines by using specialization constants
			          // Here we will apply a tint based on the current frame
			          auto current_frame = command_buffer.get_context().get_frame_count();
			          auto mod_frame = current_frame % 100;
			          if (mod_frame == 99) {
				          speed_count += 256;
			          }

			          return scramble;
		          },
		          vuk::DomainFlagBits::eGraphicsQueue);

		      auto copy_pass = vuk::make_pass(
		          "copy",
		          [](vuk::CommandBuffer& command_buffer, VUK_BA(vuk::eTransferRead) src, VUK_BA(vuk::eTransferWrite) dst) {
			          command_buffer.copy_buffer(src, dst);
			          return dst;
		          },
		          vuk::DomainFlagBits::eTransferQueue);

		      auto scramble_temp = vuk::declare_buf("scramble temporary");
		      scramble_temp->memory_usage = vuk::MemoryUsage::eGPUonly;
		      scramble_temp.same_size(scramble_buf_fut);
		      auto sorted = sort_pass(scramble_buf_fut);
		      auto in_temp = copy_pass(sorted, scramble_temp);
		      scramble_buf_fut = copy_pass(in_temp, scramble_buf_fut);

		      // draw the scrambled image, with a buffer dependency on the scramble buffer
		      auto draw_pass = vuk::make_pass(
		          "draw",
		          [](vuk::CommandBuffer& command_buffer, VUK_BA(vuk::eFragmentRead) scramble, VUK_IA(vuk::eFragmentSampled) rtt, VUK_IA(vuk::eColorWrite) final) {
			          command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
			              .set_scissor(0, vuk::Rect2D::framebuffer())
			              .set_rasterization({})     // Set the default rasterization state
			              .broadcast_color_blend({}) // Set the default color blend state
			              .bind_image(0, 0, rtt)
			              .bind_sampler(0, 0, {})
			              .bind_buffer(0, 1, scramble)
			              .bind_graphics_pipeline("scrambled_draw")
			              .draw(3, 1, 0, 0);

			          return final;
		          },
		          vuk::DomainFlagBits::eGraphicsQueue);

		      time += ImGui::GetIO().DeltaTime;

		      return draw_pass(in_temp, rttf, target);
		    },
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
		      image_of_doge.reset();
		      image_view_of_doge.reset();
		      scramble_buf.reset();
		    }

	};

	REGISTER_EXAMPLE(xample);
} // namespace
