#include "bench_runner.hpp"
#include <stb_image.h>

namespace {
	template<size_t Count>
	struct DrawCount {
		static constexpr unsigned n_draws = Count;
	};

	template<size_t Count>
	struct TriangleCount {
		static constexpr unsigned n_tris = Count;
	};

	struct V1 : DrawCount<10>, TriangleCount<1> {
		std::string_view description = "10 draws";
	};

	struct V2 : DrawCount<100>, TriangleCount<1> {
		std::string_view description = "100 draws";
	};

	struct V3 : TriangleCount<10>, DrawCount<1> {
		std::string_view description = "10 tris";
	};

	struct V4 : TriangleCount<100>, DrawCount<1> {
		std::string_view description = "100 tris";
	};

	template<class T>
	vuk::RenderGraph test_case(vuk::InflightContext& ifc, bool dependent, vuk::Texture& src, vuk::Texture& dst, vuk::Query start, vuk::Query end, T parameters) {
		auto ptc = ifc.begin();
		vuk::RenderGraph rg;
		rg.add_pass({
			.resources = {"_dst"_image(vuk::eColorWrite)},
			.execute = [start, end, parameters, &src, dependent](vuk::CommandBuffer& command_buffer) {
				command_buffer
					.set_viewport(0, vuk::Rect2D::framebuffer())
					.set_scissor(0, vuk::Rect2D::framebuffer())
					.bind_sampled_image(0, 0, src, vuk::SamplerCreateInfo{.magFilter = vuk::Filter::eLinear, .minFilter = vuk::Filter::eLinear });
				if (dependent) {
					command_buffer.bind_graphics_pipeline("dependent");
					command_buffer.push_constants<unsigned>(vuk::ShaderStageFlagBits::eFragment, 0, 112);
				} else {
					command_buffer.bind_graphics_pipeline("nondependent");
				}

				vuk::TimedScope _{ command_buffer, start, end };
				for (auto i = 0; i < parameters.n_draws; i++) {
					command_buffer.draw(3 * parameters.n_tris, 1, 0, 0);
				}
			}
			});
		rg.add_pass({
			.resources = {"_final"_image(vuk::eColorWrite), "_dst"_image(vuk::eFragmentSampled)},
			.execute = [](vuk::CommandBuffer& command_buffer) {
				command_buffer
					.set_viewport(0, vuk::Rect2D::framebuffer())
					.set_scissor(0, vuk::Rect2D::framebuffer())
					.bind_graphics_pipeline("blit")
					.bind_sampled_image(0, 0, "_dst", vuk::SamplerCreateInfo{.magFilter = vuk::Filter::eLinear, .minFilter = vuk::Filter::eLinear });
				command_buffer.draw(3, 1, 0, 0);
				}
			}
		);
		rg.attach_image("_dst", vuk::ImageAttachment::from_texture(dst), vuk::Access::eNone, vuk::Access::eNone);
		return rg;
	}

	void blit(vuk::PerThreadContext& ptc, vuk::Texture& src, vuk::Texture& dst) {
		vuk::RenderGraph rg;
		rg.add_pass({
			.resources = {"dst"_image(vuk::eColorWrite)},
			.execute = [&src](vuk::CommandBuffer& command_buffer) {
				command_buffer
					.set_viewport(0, vuk::Rect2D::framebuffer())
					.set_scissor(0, vuk::Rect2D::framebuffer())
					.bind_graphics_pipeline("blit")
					.bind_sampled_image(0, 0, src, vuk::SamplerCreateInfo{.magFilter = vuk::Filter::eLinear, .minFilter = vuk::Filter::eLinear });
				command_buffer.draw(3, 1, 0, 0);
				}
			}
		);
		rg.attach_image("dst", vuk::ImageAttachment::from_texture(dst), vuk::Access::eNone, vuk::Access::eFragmentSampled);
		vuk::execute_submit_and_wait(ptc, std::move(rg).link(ptc));
	}

	std::optional<vuk::Texture> texture_of_doge, tex2k, tex4k, tex8k;
	std::optional<vuk::Texture> dstsmall, dst2k, dst4k, dst8k;

	vuk::Bench<V1, V2, V3, V4> x{
		// The display name of this example
		.base = {
			.name = "Dependent vs. non-dependent texture fetch",
			// Setup code, ran once in the beginning
			.setup = [](vuk::BenchRunner& runner, vuk::InflightContext& ifc) {
			// Pipelines are created by filling out a vuk::PipelineCreateInfo
			// In this case, we only need the shaders, we don't care about the rest of the state
			{
				vuk::PipelineBaseCreateInfo pci;
				pci.add_glsl(util::read_entire_file("../../benchmarks/fullscreen.vert"), "fullscreen.vert");
				pci.add_glsl(util::read_entire_file("../../benchmarks/dependent_texture_fetch_explicit_lod.frag"), "dependent_texture_fetch_explicit_lod.frag");

				pci.depth_stencil_state.depthCompareOp = vuk::CompareOp::eAlways;
				runner.context->create_named_pipeline("dependent", pci);
			}
			{
				vuk::PipelineBaseCreateInfo pci;
				pci.add_glsl(util::read_entire_file("../../benchmarks/fullscreen.vert"), "fullscreen.vert");
				pci.add_glsl(util::read_entire_file("../../benchmarks/nondependent_texture_fetch_explicit_lod.frag"), "nondependent_texture_fetch_explicit_lod.frag");

				pci.depth_stencil_state.depthCompareOp = vuk::CompareOp::eAlways;
				runner.context->create_named_pipeline("nondependent", pci);
			}
			{
				vuk::PipelineBaseCreateInfo pci;
				pci.add_glsl(util::read_entire_file("../../benchmarks/fullscreen.vert"), "fullscreen.vert");
				pci.add_glsl(util::read_entire_file("../../benchmarks/blit.frag"), "blit.frag");

				pci.depth_stencil_state.depthCompareOp = vuk::CompareOp::eAlways;
				runner.context->create_named_pipeline("blit", pci);
			}

			auto ptc = ifc.begin();

			int x, y, chans;
			auto doge_image = stbi_load("../../examples/doge.png", &x, &y, &chans, 4);
			auto [tex, _] = ptc.create_texture(vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, doge_image);
			texture_of_doge = std::move(tex);
			ptc.wait_all_transfers();
			tex2k = ptc.allocate_texture(vuk::ImageCreateInfo{.format = vuk::Format::eR8G8B8A8Srgb, .extent = {.width = 2048, .height = 2048, .depth = 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled });
			tex4k = ptc.allocate_texture(vuk::ImageCreateInfo{.format = vuk::Format::eR8G8B8A8Srgb, .extent = {.width = 4096, .height = 4096, .depth = 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled });
			tex8k = ptc.allocate_texture(vuk::ImageCreateInfo{.format = vuk::Format::eR8G8B8A8Srgb, .extent = {.width = 8192, .height = 8192, .depth = 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled });
			blit(ptc, *texture_of_doge, *tex2k);
			blit(ptc, *texture_of_doge, *tex4k);
			blit(ptc, *texture_of_doge, *tex8k);
			dstsmall = ptc.allocate_texture(vuk::ImageCreateInfo{.format = vuk::Format::eR8G8B8A8Srgb, .extent = {.width = (unsigned)x, .height = (unsigned)y, .depth = 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled });
			dst2k = ptc.allocate_texture(vuk::ImageCreateInfo{.format = vuk::Format::eR8G8B8A8Srgb, .extent = {.width = 2048, .height = 2048, .depth = 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled });
			dst4k = ptc.allocate_texture(vuk::ImageCreateInfo{.format = vuk::Format::eR8G8B8A8Srgb, .extent = {.width = 4096, .height = 4096, .depth = 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled });
			dst8k = ptc.allocate_texture(vuk::ImageCreateInfo{.format = vuk::Format::eR8G8B8A8Srgb, .extent = {.width = 8192, .height = 8192, .depth = 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled });
			},
			.gui = [](vuk::BenchRunner& runner, vuk::InflightContext& ifc) {
			},
			.cleanup = [](vuk::BenchRunner& runner, vuk::InflightContext& ifc) {
				// We release the texture resources
				texture_of_doge.reset();
				tex2k.reset();
				tex4k.reset();
				tex8k.reset();
				dstsmall.reset();
				dst2k.reset();
				dst4k.reset();
				dst8k.reset();
			}
		},
		.cases = {
			{"Dependent 112x112", [](vuk::BenchRunner& runner, vuk::InflightContext& ifc, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(ifc, true, *texture_of_doge, *dstsmall, start, end, parameters);
		}},
			{"Non-dependent 112x112", [](vuk::BenchRunner& runner, vuk::InflightContext& ifc, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(ifc, false, *texture_of_doge, *dstsmall, start, end, parameters);
		}},
			{"Dependent 2K", [](vuk::BenchRunner& runner, vuk::InflightContext& ifc, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(ifc, true, *tex2k, *dst2k, start, end, parameters);
		}},
			{"Non-dependent 2K", [](vuk::BenchRunner& runner, vuk::InflightContext& ifc, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(ifc, false, *tex2k, *dst2k, start, end, parameters);
		}},
			{"Dependent 4K", [](vuk::BenchRunner& runner, vuk::InflightContext& ifc, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(ifc, true, *tex4k, *dst4k, start, end, parameters);
		}},
			{"Non-dependent 4K", [](vuk::BenchRunner& runner, vuk::InflightContext& ifc, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(ifc, false, *tex4k, *dst4k, start, end, parameters);
		}},
			{"Dependent 8K", [](vuk::BenchRunner& runner, vuk::InflightContext& ifc, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(ifc, true, *tex8k, *dst8k, start, end, parameters);
		}},
			{"Non-dependent 8K", [](vuk::BenchRunner& runner, vuk::InflightContext& ifc, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(ifc, false, *tex8k, *dst8k, start, end, parameters);
		}},
		}
	};

	REGISTER_BENCH(x);
}