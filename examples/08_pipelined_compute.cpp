#include "example_runner.hpp"
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <stb_image.h>
#include <algorithm>
#include <random>
#include <numeric>

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

namespace vuk {
	Future<Buffer> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer buffer, void* src_data, size_t size) {
		// host-mapped buffers just get memcpys
		if (buffer.mapped_ptr) {
			memcpy(buffer.mapped_ptr, src_data, size);
			return { allocator, std::move(buffer) };
		}

		auto src = *allocate_buffer_cross_device(allocator, BufferCreateInfo{ vuk::MemoryUsage::eCPUonly, size, 1 });
		::memcpy(src->mapped_ptr, src_data, size);

		std::unique_ptr<RenderGraph> rgp = std::make_unique<RenderGraph>();
		rgp->add_pass({
			.name = "BUFFER COPY UPLOAD",
			.execute_on = copy_domain,
			.resources = {"_dst"_buffer >> vuk::Access::eTransferWrite, "_src"_buffer >> vuk::Access::eTransferRead},
			.execute = [size](vuk::CommandBuffer& command_buffer) {
				command_buffer.copy_buffer("_src", "_dst", size);
			} });
		rgp->attach_buffer("_src", *src, vuk::Access::eNone, vuk::Access::eNone);
		rgp->attach_buffer("_dst", buffer, vuk::Access::eNone, vuk::Access::eNone);
		return { allocator, std::move(rgp), "_dst"}; // TODO: we should allow this to be dst+
	}

	template<class T>
	Future<Buffer> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer dst, std::span<T> data) {
		return host_data_to_buffer(allocator, copy_domain, dst, data.data(), data.size_bytes());
	}
}

namespace {
	float time = 0.f;
	auto box = util::generate_cube();
	int x, y;
	uint32_t speed_count = 1;
	std::optional<vuk::Texture> texture_of_doge;
	vuk::Unique<vuk::BufferGPU> scramble_buf;
	std::random_device rd;
	std::mt19937 g(rd());
	vuk::Future<vuk::Buffer> scramble_buf_fut;

	vuk::Example xample{
		.name = "08_pipelined_compute",
		.setup = [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
			vuk::Context& ctx = allocator.get_context();

			{
			vuk::PipelineBaseCreateInfo pci;
			pci.add_glsl(util::read_entire_file("../../examples/fullscreen.vert"), "fullscreen.vert");
			pci.add_glsl(util::read_entire_file("../../examples/rtt.frag"), "rtt.frag");
			runner.context->create_named_pipeline("rtt", pci);
			}

			{
			vuk::PipelineBaseCreateInfo pci;
			pci.add_glsl(util::read_entire_file("../../examples/fullscreen.vert"), "fullscreen.vert");
			pci.add_glsl(util::read_entire_file("../../examples/scrambled_draw.frag"), "scrambled_draw.frag");
			runner.context->create_named_pipeline("scrambled_draw", pci);
			}

			// creating a compute pipeline is the same as creating a graphics pipeline
			{
			vuk::ComputePipelineBaseCreateInfo pbci;
			pbci.add_glsl(util::read_entire_file("../../examples/stupidsort.comp"), "stupidsort.comp");
			runner.context->create_named_pipeline("stupidsort", pbci);
			}

			int chans;
			auto doge_image = stbi_load("../../examples/doge.png", &x, &y, &chans, 4);

			auto [tex, stub] = ctx.create_texture(allocator, vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1 }, doge_image);
			texture_of_doge = std::move(tex);

			// init scrambling buffer
			scramble_buf = *allocate_buffer_gpu(allocator, { vuk::MemoryUsage::eGPUonly, sizeof(unsigned) * x * y, 1 });
			std::vector<unsigned> indices(x * y);
			std::iota(indices.begin(), indices.end(), 0);
			std::shuffle(indices.begin(), indices.end(), g);

			ctx.wait_all_transfers(allocator);

			//// <----------------->
			// make a GPU future
			// the copy (written above) is not performed yet, we just record the computation and bind to a result ("_dst")
			scramble_buf_fut = vuk::host_data_to_buffer(allocator, vuk::DomainFlagBits::eTransferOnTransfer, scramble_buf.get(), std::span(indices));

			stbi_image_free(doge_image);
		},
		.render = [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
			vuk::Context& ctx = frame_allocator.get_context();

			vuk::RenderGraph rgx;

			// standard render to texture
			rgx.add_pass({
				.name = "08_rtt",
				.execute_on = vuk::DomainFlagBits::eGraphicsQueue,
				.resources = {"08_rttf"_image(vuk::eColorWrite, runner.swapchain->format, vuk::Dimension2D::absolute((unsigned)x, (unsigned)y), vuk::Samples::e1, vuk::ClearColor{ 0.f, 0.f, 0.f, 0.f })},
				.execute = [](vuk::CommandBuffer& command_buffer) {
					command_buffer
						.set_viewport(0, vuk::Rect2D::framebuffer())
						.set_scissor(0, vuk::Rect2D::framebuffer())
						.set_rasterization({}) // Set the default rasterization state
						.broadcast_color_blend({}) // Set the default color blend state
						.bind_sampled_image(0, 0, *texture_of_doge, {})
						.bind_graphics_pipeline("rtt")
						.draw(3, 1, 0, 0);
				}
			});
			//// <-----------------> 
			// make a gpu future of the above graph (render to texture) and bind to an output (rttf)
			vuk::Future<vuk::Image> rttf{ frame_allocator, rgx, "08_rttf" };

			std::unique_ptr<vuk::RenderGraph> rgp = std::make_unique<vuk::RenderGraph>();
			auto& rg = *rgp;
			// this pass executes outside of a renderpass
			// we declare a buffer dependency and dispatch a compute shader
			rg.add_pass({
				.name = "08_sort",
				.execute_on = vuk::DomainFlagBits::eGraphicsQueue,
				.resources = {"08_scramble"_buffer >> vuk::eComputeRW >> "08_scramble+"},
				.execute = [](vuk::CommandBuffer& command_buffer) {
					command_buffer.bind_storage_buffer(0, 0, *command_buffer.get_resource_buffer("08_scramble"));
					command_buffer
						.bind_compute_pipeline("stupidsort")
						.specialize_constants(0, speed_count)
						.dispatch(1);
					// We can also customize pipelines by using specialization constants
					// Here we will apply a tint based on the current frame
					auto current_frame = command_buffer.get_context().frame_counter.load();
					auto mod_frame = current_frame % 100;
					if (mod_frame == 99) {
						speed_count += 256;
					}
				}
			});

			// draw the scrambled image, with a buffer dependency on the scramble buffer
			rg.add_pass({
				.name = "08_draw",
				.execute_on = vuk::DomainFlagBits::eGraphicsQueue,
				.resources = {"08_scramble+"_buffer >> vuk::eFragmentRead, "08_rtt"_image >> vuk::eFragmentSampled, "08_pipelined_compute"_image >> vuk::eColorWrite >> "08_pipelined_compute_final"},
				.execute = [](vuk::CommandBuffer& command_buffer) {
					command_buffer
						.set_viewport(0, vuk::Rect2D::framebuffer())
						.set_scissor(0, vuk::Rect2D::framebuffer())
						.set_rasterization({}) // Set the default rasterization state
						.broadcast_color_blend({}) // Set the default color blend state
						.bind_sampled_image(0, 0, "08_rtt", {})
						.bind_storage_buffer(0, 1, *command_buffer.get_resource_buffer("08_scramble"))
						.bind_graphics_pipeline("scrambled_draw")
						.draw(3, 1, 0, 0);
				}
			});

			time += ImGui::GetIO().DeltaTime;

			//// <-----------------> 
			// make the main rendergraph
			// our two inputs are the futures - they compile into the main rendergraph
			rg.attach_in("08_rtt", std::move(rttf), vuk::eNone);
			// the copy here in addition will execute on the transfer queue, and will signal the graphics to execute the rest
			// we created this future in the setup code, so on the first frame it will append the computation
			// but on the subsequent frames the future becomes ready (on the gpu) and this will only attach a buffer
			rg.attach_in("08_scramble", std::move(scramble_buf_fut), vuk::eNone);
			scramble_buf_fut = { *runner.global, std::move(rgp), "08_scramble" };
			rg.attach_out("08_scramble", scramble_buf_fut);

			return vuk::Future<vuk::Image>{frame_allocator, rg, "08_pipelined_compute_final"};
		},
		.cleanup = [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
			texture_of_doge.reset();
			scramble_buf.reset();
		}

	};

	REGISTER_EXAMPLE(xample);
}
