#include "bench_runner.hpp"
#include "vuk/Partials.hpp"
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
	vuk::RenderGraph test_case(vuk::Allocator& allocator, bool dependent, vuk::Texture& src, vuk::Texture& dst, vuk::Query start, vuk::Query end, T parameters) {
		vuk::RenderGraph rg;
		rg.add_pass({ .resources = { "_dst"_image >> vuk::eColorWrite }, .execute = [start, end, parameters, &src, dependent](vuk::CommandBuffer& command_buffer) {
			             command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
			                 .set_scissor(0, vuk::Rect2D::framebuffer())
			                 .set_rasterization({})
			                 .broadcast_color_blend({})
			                 .bind_image(0, 0, *src.view)
			                 .bind_sampler(0, 0, vuk::SamplerCreateInfo{ .magFilter = vuk::Filter::eLinear, .minFilter = vuk::Filter::eLinear });
			             if (dependent) {
				             command_buffer.bind_graphics_pipeline("dependent");
				             command_buffer.push_constants<unsigned>(vuk::ShaderStageFlagBits::eFragment, 0, 1.0f / (float)src.extent.width);
			             } else {
				             command_buffer.bind_graphics_pipeline("nondependent");
			             }

			             vuk::TimedScope _{ command_buffer, start, end };
			             for (auto i = 0; i < parameters.n_draws; i++) {
				             command_buffer.draw(3 * parameters.n_tris, 1, 0, 0);
			             }
		             } });
		rg.add_pass(
		    { .resources = { "_final"_image >> vuk::eColorWrite, "_dst+"_image >> vuk::eFragmentSampled }, .execute = [](vuk::CommandBuffer& command_buffer) {
			     command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
			         .set_scissor(0, vuk::Rect2D::framebuffer())
			         .set_rasterization({})
			         .broadcast_color_blend({})
			         .bind_graphics_pipeline("blit")
			         .bind_image(0, 0, "_dst")
			         .bind_sampler(0, 0, vuk::SamplerCreateInfo{ .magFilter = vuk::Filter::eLinear, .minFilter = vuk::Filter::eLinear });
			     command_buffer.draw(3, 1, 0, 0);
		     } });
		rg.attach_image("_dst", vuk::ImageAttachment::from_texture(dst), vuk::Access::eNone, vuk::Access::eNone);
		return rg;
	}

	void blit(vuk::Allocator& allocator, vuk::Texture& src, vuk::Texture& dst) {
		std::shared_ptr<vuk::RenderGraph> rg = std::make_shared<vuk::RenderGraph>("blit");
		rg->add_pass({ .resources = { "dst"_image >> vuk::eColorWrite }, .execute = [&src](vuk::CommandBuffer& command_buffer) {
			              command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
			                  .set_scissor(0, vuk::Rect2D::framebuffer())
			                  .set_rasterization({})
			                  .broadcast_color_blend({})
			                  .bind_graphics_pipeline("blit")
			                  .bind_image(0, 0, *src.view)
			                  .bind_sampler(0, 0, { .magFilter = vuk::Filter::eLinear, .minFilter = vuk::Filter::eLinear });
			              command_buffer.draw(3, 1, 0, 0);
		              } });
		rg->attach_image("dst", vuk::ImageAttachment::from_texture(dst), vuk::Access::eNone, vuk::Access::eFragmentSampled);
		vuk::Compiler c;
		auto erg = c.link({ &rg, 1 }, {});
		vuk::execute_submit_and_wait(allocator, std::move(erg));
	}

	std::optional<vuk::Texture> texture_of_doge, tex2k, tex4k, tex8k;
	std::optional<vuk::Texture> dstsmall, dst2k, dst4k, dst8k;

	vuk::Bench<V1, V2, V3, V4> x{
		// The display name of this example
		.base = {
			.name = "Dependent vs. non-dependent texture fetch",
			// Setup code, ran once in the beginning
			.setup = [](vuk::BenchRunner& runner, vuk::Allocator& allocator) {
			auto& ctx = allocator.get_context();
			// Pipelines are created by filling out a vuk::PipelineCreateInfo
			// In this case, we only need the shaders, we don't care about the rest of the state
			{
				vuk::PipelineBaseCreateInfo pci;
				pci.add_glsl(util::read_entire_file("../../benchmarks/fullscreen.vert"), "fullscreen.vert");
				pci.add_glsl(util::read_entire_file("../../benchmarks/dependent_texture_fetch_explicit_lod.frag"), "dependent_texture_fetch_explicit_lod.frag");

				runner.context->create_named_pipeline("dependent", pci);
			}
			{
				vuk::PipelineBaseCreateInfo pci;
				pci.add_glsl(util::read_entire_file("../../benchmarks/fullscreen.vert"), "fullscreen.vert");
				pci.add_glsl(util::read_entire_file("../../benchmarks/nondependent_texture_fetch_explicit_lod.frag"), "nondependent_texture_fetch_explicit_lod.frag");

				runner.context->create_named_pipeline("nondependent", pci);
			}
			{
				vuk::PipelineBaseCreateInfo pci;
				pci.add_glsl(util::read_entire_file("../../benchmarks/fullscreen.vert"), "fullscreen.vert");
				pci.add_glsl(util::read_entire_file("../../benchmarks/blit.frag"), "blit.frag");

				runner.context->create_named_pipeline("blit", pci);
			}

			int x, y, chans;
			auto doge_image = stbi_load("../../examples/doge.png", &x, &y, &chans, 4);
			auto [tex, tex_fut] = create_texture(allocator, vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, doge_image, false);
			texture_of_doge = std::move(tex);
			vuk::Compiler c;
			tex_fut.wait(allocator, c);
			stbi_image_free(doge_image);
			tex2k = ctx.allocate_texture(allocator, vuk::ImageCreateInfo{.format = vuk::Format::eR8G8B8A8Srgb, .extent = {.width = 2048, .height = 2048, .depth = 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled });
			tex4k = ctx.allocate_texture(allocator, vuk::ImageCreateInfo{.format = vuk::Format::eR8G8B8A8Srgb, .extent = {.width = 4096, .height = 4096, .depth = 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled });
			tex8k = ctx.allocate_texture(allocator, vuk::ImageCreateInfo{.format = vuk::Format::eR8G8B8A8Srgb, .extent = {.width = 8192, .height = 8192, .depth = 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled });
			blit(allocator, *texture_of_doge, *tex2k);
			blit(allocator, *texture_of_doge, *tex4k);
			blit(allocator, *texture_of_doge, *tex8k);
			dstsmall = ctx.allocate_texture(allocator, vuk::ImageCreateInfo{.format = vuk::Format::eR8G8B8A8Srgb, .extent = {.width = (unsigned)x, .height = (unsigned)y, .depth = 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled });
			dst2k = ctx.allocate_texture(allocator, vuk::ImageCreateInfo{.format = vuk::Format::eR8G8B8A8Srgb, .extent = {.width = 2048, .height = 2048, .depth = 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled });
			dst4k = ctx.allocate_texture(allocator, vuk::ImageCreateInfo{.format = vuk::Format::eR8G8B8A8Srgb, .extent = {.width = 4096, .height = 4096, .depth = 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled });
			dst8k = ctx.allocate_texture(allocator, vuk::ImageCreateInfo{.format = vuk::Format::eR8G8B8A8Srgb, .extent = {.width = 8192, .height = 8192, .depth = 1}, .usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled });
			},
			.gui = [](vuk::BenchRunner& runner, vuk::Allocator& allocator) {
			},
			.cleanup = [](vuk::BenchRunner& runner, vuk::Allocator& allocator) {
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
			{"Dependent 112x112", [](vuk::BenchRunner& runner, vuk::Allocator& allocator, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(allocator, true, *texture_of_doge, *dstsmall, start, end, parameters);
		}},
			{"Non-dependent 112x112", [](vuk::BenchRunner& runner, vuk::Allocator& allocator, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(allocator, false, *texture_of_doge, *dstsmall, start, end, parameters);
		}},
			{"Dependent 2K", [](vuk::BenchRunner& runner, vuk::Allocator& allocator, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(allocator, true, *tex2k, *dst2k, start, end, parameters);
		}},
			{"Non-dependent 2K", [](vuk::BenchRunner& runner, vuk::Allocator& allocator, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(allocator, false, *tex2k, *dst2k, start, end, parameters);
		}},
			{"Dependent 4K", [](vuk::BenchRunner& runner, vuk::Allocator& allocator, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(allocator, true, *tex4k, *dst4k, start, end, parameters);
		}},
			{"Non-dependent 4K", [](vuk::BenchRunner& runner, vuk::Allocator& allocator, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(allocator, false, *tex4k, *dst4k, start, end, parameters);
		}},
			/*{"Dependent 8K", [](vuk::BenchRunner& runner, vuk::Allocator& allocator, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(allocator, true, *tex8k, *dst8k, start, end, parameters);
		}},
			{"Non-dependent 8K", [](vuk::BenchRunner& runner, vuk::Allocator& allocator, vuk::Query start, vuk::Query end, auto&& parameters) {
				return test_case(allocator, false, *tex8k, *dst8k, start, end, parameters);
		}},*/
		}
	};

	REGISTER_BENCH(x);
} // namespace