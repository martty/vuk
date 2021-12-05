#include "example_runner.hpp"
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <stb_image.h>
#include <algorithm>
#include <random>
#include <optional>
#include <numeric>

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
		virtual void bind_parameters(vuk::CommandBuffer&) {};
		virtual void bind_textures(vuk::CommandBuffer&) {};
	};

	struct NormalMaterial : Material {
		vuk::ImageView texture;

		void bind_textures(vuk::CommandBuffer& cbuf) override {
			cbuf.bind_sampled_image(0, 2, texture, {});
		}
	};

	struct TintMaterial : Material {
		vuk::ImageView texture;
		glm::vec4 tint_color = glm::vec4(0);

		void bind_textures(vuk::CommandBuffer& cbuf) override {
			cbuf.bind_sampled_image(0, 2, texture, {});
		}

		void bind_parameters(vuk::CommandBuffer& cbuf) override {
			*cbuf.map_scratch_uniform_binding<glm::vec4>(0, 3) = tint_color;
		}
	};

	std::optional<Mesh> cube_mesh, quad_mesh;

	struct Renderable {
		Mesh* mesh;
		Material* material;

		glm::vec3 position;
		glm::vec3 velocity = glm::vec3(0);
	};

	std::optional<vuk::Texture> texture_of_doge, variant1, variant2;

	std::vector<NormalMaterial> nmats;
	std::vector<TintMaterial> tmats;

	std::vector<Renderable> renderables;

	std::random_device rd;
	std::mt19937 g(rd());

	vuk::Example xample{
		.name = "10_baby_renderer",
		.setup = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			// Use STBI to load the image
			int x, y, chans;
			auto doge_image = stbi_load("../../examples/doge.png", &x, &y, &chans, 4);

			auto ptc = ifc.begin();
			// Similarly to buffers, we allocate the image and enqueue the upload
			auto [tex, _] = ptc.ctx.create_texture(vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, doge_image);
			texture_of_doge = std::move(tex);
			ptc.ctx.wait_all_transfers();
			stbi_image_free(doge_image);

			// Let's create two variants of the doge image (like in example 09)
			// Creating a compute pipeline that inverts an image 
			{
				vuk::ComputePipelineBaseCreateInfo pbci;
				pbci.add_glsl(util::read_entire_file("../../examples/invert.comp"), "invert.comp");
				runner.context->create_named_pipeline("invert", pbci);
			}
			vuk::ImageCreateInfo ici;
			ici.format = vuk::Format::eR8G8B8A8Srgb;
			ici.extent = vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u };
			ici.samples = vuk::Samples::e1;
			ici.imageType = vuk::ImageType::e2D;
			ici.initialLayout = vuk::ImageLayout::eUndefined;
			ici.tiling = vuk::ImageTiling::eOptimal;
			ici.usage = vuk::ImageUsageFlagBits::eTransferDst | vuk::ImageUsageFlagBits::eSampled;
			ici.mipLevels = ici.arrayLayers = 1;
			variant1 = ptc.ctx.allocate_texture(ici);
			ici.format = vuk::Format::eR8G8B8A8Unorm;
			ici.usage = vuk::ImageUsageFlagBits::eStorage | vuk::ImageUsageFlagBits::eSampled;
			variant2 = ptc.ctx.allocate_texture(ici);
			// Make a RenderGraph to process the loaded image
			vuk::RenderGraph rg;
			rg.add_pass({
						.name = "10_preprocess",
						.resources = {"10_doge"_image(vuk::eMemoryRead), "10_v1"_image(vuk::eTransferDst), "10_v2"_image(vuk::eComputeWrite)},
						.execute = [x, y](vuk::CommandBuffer& command_buffer) {
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
					command_buffer.blit_image("10_doge", "10_v1", blit, vuk::Filter::eLinear);
					// For the second image, invert the colours in compute
					command_buffer
						.bind_sampled_image(0, 0, "10_doge", {})
						.bind_storage_image(0, 1, "10_v2")
						.bind_compute_pipeline("invert")
						.dispatch_invocations(x, y);
			}
				});
			// Bind the resources for the variant generation
			// We specify the initial and final access
			// The texture we have created is already in ShaderReadOptimal, but we need it in General during the pass, and we need it back to ShaderReadOptimal afterwards
			rg.attach_image("10_doge", vuk::ImageAttachment::from_texture(*texture_of_doge), vuk::eFragmentSampled, vuk::eFragmentSampled);
			rg.attach_image("10_v1", vuk::ImageAttachment::from_texture(*variant1), vuk::eNone, vuk::eFragmentSampled);
			rg.attach_image("10_v2", vuk::ImageAttachment::from_texture(*variant2), vuk::eNone, vuk::eFragmentSampled);
			// The rendergraph is submitted and fence-waited on
			execute_submit_and_wait(ifc.ctx, ifc.ctx.get_vk_allocator(), std::move(rg).link(ptc, vuk::RenderGraph::CompileOptions{}));

			// Set up the resources for our renderer

			// Create meshes
			cube_mesh = Mesh{};
			cube_mesh->vertex_buffer = ptc.ctx.create_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eVertexBuffer, std::span(&box.first[0], box.first.size())).first;
			cube_mesh->index_buffer = ptc.ctx.create_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eIndexBuffer, std::span(&box.second[0], box.second.size())).first;
			cube_mesh->index_count = (uint32_t)box.second.size();

			quad_mesh = Mesh{};
			quad_mesh->vertex_buffer = ptc.ctx.create_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eVertexBuffer, std::span(&box.first[0], 6)).first;
			quad_mesh->index_buffer = ptc.ctx.create_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eIndexBuffer, std::span(&box.second[0], 6)).first;
			quad_mesh->index_count = 6;
			ptc.ctx.wait_all_transfers();

			// Create the pipelines
			// A "normal" pipeline
			vuk::PipelineBaseInfo* pipe1;
			{
				vuk::PipelineBaseCreateInfo pci;
				pci.add_glsl(util::read_entire_file("../../examples/baby_renderer.vert"), "baby_renderer.vert");
				pci.add_glsl(util::read_entire_file("../../examples/triangle_depthshaded_tex.frag"), "triangle_depthshaded_tex.frag");
				pipe1 = runner.context->get_pipeline(pci);
			}

			// A "tinted" pipeline
			vuk::PipelineBaseInfo* pipe2;
			{
				vuk::PipelineBaseCreateInfo pci;
				pci.add_glsl(util::read_entire_file("../../examples/baby_renderer.vert"), "baby_renderer.vert");
				pci.add_glsl(util::read_entire_file("../../examples/triangle_tinted_tex.frag"), "triangle_tinted_tex.frag");
				pipe2 = runner.context->get_pipeline(pci);
			}

			// Create materials
			nmats.resize(3);
			nmats[0].texture = texture_of_doge->view.get();
			nmats[1].texture = variant1->view.get();
			nmats[2].texture = variant2->view.get();
			for (auto& mat : nmats) {
				mat.pipeline = pipe1;
			}

			tmats.resize(3);
			std::uniform_real_distribution<float> dist_tint(0, 1);
			tmats[0].texture = texture_of_doge->view.get();
			tmats[1].texture = variant1->view.get();
			tmats[2].texture = variant2->view.get();
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
				renderables.emplace_back(Renderable{.mesh = &*quad_mesh, .material = m, .position = pos});
			}

			// 16 cubes
			for (int i = 0; i < 16; i++) {
				auto mat_id = dist_mat(g);
				auto tex_id = dist_tex(g);

				Material* m = mat_id == 0 ? (Material*)&nmats[tex_id] : (Material*)&tmats[tex_id];
				glm::vec3 pos = glm::vec3(dist_pos(g), dist_pos(g), dist_pos(g));
				renderables.emplace_back(Renderable{.mesh = &*cube_mesh, .material = m, .position = pos });
			}
	},
		.render = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			auto ptc = ifc.begin();

			// We set up VP data, same as in example 02_cube
			struct VP {
				glm::mat4 view;
				glm::mat4 proj;
			} vp;
			vp.view = glm::lookAt(glm::vec3(0, 10.0, 11), glm::vec3(0), glm::vec3(0, 1, 0));
			vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 100.f);
			vp.proj[1][1] *= -1;

			// Upload view & projection
			auto [buboVP, stub3] = ptc.ctx.create_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vuk::BufferUsageFlagBits::eUniformBuffer, std::span(&vp, 1));
			auto uboVP = buboVP;
			ptc.ctx.wait_all_transfers();

			// Do a terrible simulation step
			// All objects are attracted to the origin
			for (auto& r : renderables) {
				auto force_mag = 0.1f / glm::length(r.position);
				r.velocity += force_mag * (-r.position) * ImGui::GetIO().DeltaTime;
				r.position += r.velocity * ImGui::GetIO().DeltaTime;
			}

			// Upload model matrices to an array
			auto modelmats = ptc.ctx.allocate_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vuk::BufferUsageFlagBits::eStorageBuffer, sizeof(glm::mat4) * renderables.size(), 1);
			for (auto i = 0; i < renderables.size(); i++) {
				glm::mat4 model_matrix = glm::translate(glm::mat4(1.f), renderables[i].position);
				memcpy(reinterpret_cast<glm::mat4*>(modelmats.mapped_ptr) + i, &model_matrix, sizeof(glm::mat4));
			}

			vuk::RenderGraph rg;

			// Set up the pass to draw the renderables
			rg.add_pass({
				.resources = {"10_baby_renderer_final"_image(vuk::eColorWrite), "10_depth"_image(vuk::eDepthStencilRW)},
				.execute = [uboVP, modelmats](vuk::CommandBuffer& command_buffer) {
					command_buffer
						.set_dynamic_state(vuk::DynamicStateFlagBits::eViewport | vuk::DynamicStateFlagBits::eScissor)
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
					for (auto i = 0; i < renderables.size(); i++) {
						auto& r = renderables[i];

						// Set up the draw state based on the mesh and material
						command_buffer
							.bind_vertex_buffer(0, r.mesh->vertex_buffer.get(), 0, vuk::Packed{ vuk::Format::eR32G32B32Sfloat, vuk::Ignore{offsetof(util::Vertex, uv_coordinates) - sizeof(util::Vertex::position)}, vuk::Format::eR32G32Sfloat })
							.bind_index_buffer(r.mesh->index_buffer.get(), vuk::IndexType::eUint32)
							.bind_graphics_pipeline(r.material->pipeline)
							.bind_uniform_buffer(0, 0, uboVP)
							.bind_storage_buffer(0, 1, modelmats);

						r.material->bind_parameters(command_buffer);
						r.material->bind_textures(command_buffer);

						// Draw the mesh, assign them different base instance to pick the correct transformation
						command_buffer.draw_indexed(r.mesh->index_count, 1, 0, 0, i);
					}
				}
				});

			angle += 10.f * ImGui::GetIO().DeltaTime;

			rg.attach_managed("10_depth", vuk::Format::eD32Sfloat, vuk::Dimension2D::framebuffer(), vuk::Samples::Framebuffer{}, vuk::ClearDepthStencil{ 1.0f, 0 });
			return rg;
	},
		// Perform cleanup for the example
		.cleanup = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
		// We release the resources manually
		cube_mesh.reset();
		quad_mesh.reset();
		texture_of_doge.reset();
		variant1.reset();
		variant2.reset();
	}
	};

	REGISTER_EXAMPLE(xample);
}