#include "bench_runner.hpp"
#include <stb_image.h>

namespace {
	struct V1 {
		std::string_view description = "10 iter";
		static constexpr unsigned n_iters = 10;
	};

	struct V2 {
		std::string_view description = "100000 iters";
		static constexpr unsigned n_iters = 100000;
	};

	std::optional<vuk::Texture> texture_of_doge;

	vuk::Bench<V1, V2> x{
		// The display name of this example
		.base = {
			.name = "Dependent vs. non-dependent texture fetch",
			// Setup code, ran once in the beginning
			.setup = [](vuk::BenchRunner& runner, vuk::InflightContext& ifc) {
			// Pipelines are created by filling out a vuk::PipelineCreateInfo
			// In this case, we only need the shaders, we don't care about the rest of the state
			{
				vuk::PipelineBaseCreateInfo pci;
				pci.add_shader(util::read_entire_file("../../benchmarks/fullscreen.vert"), "triangle.vert");
				pci.add_shader(util::read_entire_file("../../benchmarks/dependent_texture_fetch.frag"), "triangle.frag");

				pci.depth_stencil_state.depthCompareOp = vuk::CompareOp::eAlways;
				runner.context->create_named_pipeline("dependent", pci);
			}
			{
				vuk::PipelineBaseCreateInfo pci;
				pci.add_shader(util::read_entire_file("../../benchmarks/fullscreen.vert"), "triangle.vert");
				pci.add_shader(util::read_entire_file("../../benchmarks/nondependent_texture_fetch.frag"), "triangle.frag");

				pci.depth_stencil_state.depthCompareOp = vuk::CompareOp::eAlways;
				runner.context->create_named_pipeline("nondependent", pci);
			}

			auto ptc = ifc.begin();

			int x, y, chans;
			auto doge_image = stbi_load("../../examples/doge.png", &x, &y, &chans, 4);
			auto [tex, _] = ptc.create_texture(vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, doge_image);
			texture_of_doge = std::move(tex);
			ptc.wait_all_transfers();


			},
			.gui = [](vuk::BenchRunner& runner, vuk::InflightContext& ifc) {
			},
			.cleanup = [](vuk::BenchRunner& runner, vuk::InflightContext& ifc) {
				// We release the texture resources
				texture_of_doge.reset();
			}
		},
		.cases = {
			{"Dependent, small image", [](vuk::BenchRunner& runner, vuk::InflightContext& ifc, vuk::Query start, vuk::Query end, auto&& parameters) {
			auto ptc = ifc.begin();
			vuk::RenderGraph rg;
			rg.add_pass({
				.resources = {"_final"_image(vuk::eColorWrite)},
				.execute = [start, end, parameters](vuk::CommandBuffer& command_buffer) {
					command_buffer
						.set_viewport(0, vuk::Rect2D::framebuffer())
						.set_scissor(0, vuk::Rect2D::framebuffer()) // Set the scissor area to cover the entire framebuffer
						.bind_graphics_pipeline("dependent") // Recall pipeline for "triangle" and bind
						.bind_sampled_image(0, 0, *texture_of_doge, vuk::SamplerCreateInfo{.magFilter = vuk::Filter::eLinear, .minFilter = vuk::Filter::eLinear });
					vuk::TimedScope _{ command_buffer, start, end };
					command_buffer.draw(3 * parameters.n_iters, 1, 0, 0);
					}
				}
			);
			return rg;
		}},
				{"Non-dependent, small image", [](vuk::BenchRunner& runner, vuk::InflightContext& ifc, vuk::Query start, vuk::Query end, auto&& parameters) {
			auto ptc = ifc.begin();
			vuk::RenderGraph rg;
			rg.add_pass({
				.resources = {"_final"_image(vuk::eColorWrite)},
				.execute = [start, end, parameters](vuk::CommandBuffer& command_buffer) {
					command_buffer
						.set_viewport(0, vuk::Rect2D::framebuffer())
						.set_scissor(0, vuk::Rect2D::framebuffer())
						.bind_graphics_pipeline("nondependent")
						.bind_sampled_image(0, 0, *texture_of_doge, vuk::SamplerCreateInfo{.magFilter = vuk::Filter::eLinear, .minFilter = vuk::Filter::eLinear });
					vuk::TimedScope _{ command_buffer, start, end };
					command_buffer.draw(3 * parameters.n_iters, 1, 0, 0);
					}
				}
			);
			return rg;
		}},
			{"Dependent, small image again", [](vuk::BenchRunner& runner, vuk::InflightContext& ifc, vuk::Query start, vuk::Query end, auto&& parameters) {
			auto ptc = ifc.begin();
			vuk::RenderGraph rg;
			rg.add_pass({
				.resources = {"_final"_image(vuk::eColorWrite)},
				.execute = [start, end, parameters](vuk::CommandBuffer& command_buffer) {
					command_buffer
						.set_viewport(0, vuk::Rect2D::framebuffer())
						.set_scissor(0, vuk::Rect2D::framebuffer()) // Set the scissor area to cover the entire framebuffer
						.bind_graphics_pipeline("dependent") // Recall pipeline for "triangle" and bind
						.bind_sampled_image(0, 0, *texture_of_doge, vuk::SamplerCreateInfo{.magFilter = vuk::Filter::eLinear, .minFilter = vuk::Filter::eLinear });
					vuk::TimedScope _{ command_buffer, start, end };
					command_buffer.draw(3 * parameters.n_iters, 1, 0, 0);
					}
				}
			);
			return rg;
		}},
		}
	};

	REGISTER_BENCH(x);
}