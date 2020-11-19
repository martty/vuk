#include "example_runner.hpp"
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <stb_image.h>
#include <algorithm>
#include <random>
#include <optional>
#include <numeric>
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"
#include <unordered_set>
#include <span>

/* X1_tss
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
	float angle = 160.f;
	// Generate vertices and indices for the cube
	auto box = util::generate_cube();

	struct Mesh {
		bool interleaved = true;
		vuk::Packed attrs;
		vuk::Unique<vuk::Buffer> vertex_buffer;
		vuk::Unique<vuk::Buffer> index_buffer;
		vuk::IndexType index_type = vuk::IndexType::eUint32;
		uint32_t index_count;
		uint32_t vertex_count;

		vuk::Texture* bary_map;
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

	std::optional<vuk::Texture> bary_map;
	std::optional<Mesh> cube_mesh, quad_mesh;
	std::vector<Mesh> meshes;

	struct Renderable {
		Mesh* mesh;
		Material* material;

		glm::vec3 position;
		glm::quat orientation;
		glm::vec3 velocity = glm::vec3(0);
	};
	tinygltf::TinyGLTF loader;

	std::vector<uint32_t> index_map_cpu;
	vuk::Unique<vuk::Buffer> index_map;
	std::vector<uint32_t> mesh_info_cpu;
	vuk::Unique<vuk::Buffer> mesh_info;

	template<class T>
	std::vector<std::vector<T>> break_up_mesh(std::span<T> in) {
		std::vector<std::vector<T>> result(1);
		std::unordered_set<T> seen;
		for (unsigned i = 0; i < in.size(); i += 3) {
			if (seen.find(in[i]) != seen.end()) {
				if (seen.find(in[i + 1]) != seen.end()) {
					if (seen.find(in[i + 2]) != seen.end()) {
						// can't rotate anymore, start new mesh
						seen.clear();
						result.emplace_back();
						result.back().insert(result.back().end(), { in[i], in[i + 1], in[i + 2] });
						seen.insert(in[i]);
					} else {
						result.back().insert(result.back().end(), { in[i + 2], in[i], in[i + 1] });
						seen.insert(in[i + 2]);
					}
				} else {
					result.back().insert(result.back().end(), { in[i + 1], in[i + 2], in[i] });
					seen.insert(in[i + 1]);
				}
			} else {
				result.back().insert(result.back().end(), { in[i], in[i + 1], in[i + 2] });
				seen.insert(in[i]);
			}
		}
		// check
		seen.clear();
		uint32_t offset = 0;
		mesh_info_cpu.resize(result.size());
		for (auto mesh_id = 0; mesh_id < result.size(); mesh_id++) {
			auto& r = result[mesh_id];
			for (auto i = 0; i < r.size(); i += 3) {
				assert(seen.find(r[i]) == seen.end());
				seen.insert(r[i]);
				if (index_map_cpu.size() < (offset + (r[i] + 1) * 3))
					index_map_cpu.resize(offset + (r[i] + 1) * 3);
				index_map_cpu[offset + 3*r[i]] = r[i];
				index_map_cpu[offset + 3*r[i] + 1] = r[i + 1];
				index_map_cpu[offset + 3*r[i] + 2] = r[i + 2];
			}
			seen.clear();
			mesh_info_cpu[mesh_id] = offset;
			offset = index_map_cpu.size();
		}
		return result;
	}

	std::vector<Mesh> load_mesh(vuk::PerThreadContext& ptc, const std::string& file) {
		std::string err;
		std::string warn;

		tinygltf::Model model;
		bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, file.c_str());
		//bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, argv[1]); // for binary glTF(.glb)

		if (!warn.empty()) {
			printf("Warn: %s\n", warn.c_str());
		}

		if (!err.empty()) {
			printf("Err: %s\n", err.c_str());
		}

		if (!ret) {
			printf("Failed to parse glTF\n");
		}

		std::vector<Mesh> meshes;

		for (auto& m : model.meshes) {
			for (auto& p : m.primitives) {
				//p.mode -> PTS/LINES/TRIS
				{
					auto& acc = model.accessors[p.indices];
					auto& buffer_view = model.bufferViews[acc.bufferView];
					auto& buffer = model.buffers[buffer_view.buffer];
					auto data = std::span(buffer.data.data() + buffer_view.byteOffset + acc.byteOffset, buffer_view.byteLength);

					Mesh m;
					m.index_count = buffer_view.byteLength / acc.ByteStride(buffer_view);
					
					assert(acc.type == TINYGLTF_TYPE_SCALAR);
					m.index_type = acc.ByteStride(buffer_view) == 2 ? vuk::IndexType::eUint16 : vuk::IndexType::eUint32;

					if (m.index_type == vuk::IndexType::eUint16) {
						std::vector<std::vector<uint16_t>> res = break_up_mesh(std::span<uint16_t>((uint16_t*)(buffer.data.data() + buffer_view.byteOffset + acc.byteOffset), m.index_count));
						for(auto& r : res){
							meshes.emplace_back(std::move(m));
							auto& mesh = meshes.back();
							mesh.index_buffer = ptc.create_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eIndexBuffer, std::span(r.begin(), r.end())).first;
							mesh.index_count = r.size();
						}
					} else {
						std::vector<std::vector<uint32_t>> res = break_up_mesh(std::span<uint32_t>((uint32_t*)(buffer.data.data() + buffer_view.byteOffset + acc.byteOffset), m.index_count));
						for(auto& r : res){
							meshes.emplace_back(std::move(m));
							auto& mesh = meshes.back();
							mesh.index_buffer = ptc.create_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eIndexBuffer, std::span(r.begin(), r.end())).first;
							mesh.index_count = r.size();
						}
					}
					index_map = ptc.create_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eStorageBuffer, std::span(index_map_cpu.begin(), index_map_cpu.end())).first;
					mesh_info = ptc.create_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eStorageBuffer, std::span(mesh_info_cpu.begin(), mesh_info_cpu.end())).first;
				}

				for(auto& mesh: meshes){
					std::vector<unsigned char> vattrs;
					auto do_attr = [&](auto name, auto index) {
						auto& acc = model.accessors[index];
						auto& buffer_view = model.bufferViews[acc.bufferView];
						auto& buffer = model.buffers[buffer_view.buffer];
						auto data = std::span(buffer.data.data() + buffer_view.byteOffset + acc.byteOffset, buffer_view.byteLength);

						auto stride = acc.ByteStride(buffer_view);

						if (acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
							switch (acc.type) {
							case TINYGLTF_TYPE_VEC3: mesh.attrs.list.push_back(vuk::Format::eR32G32B32Sfloat); break;
							case TINYGLTF_TYPE_VEC2: mesh.attrs.list.push_back(vuk::Format::eR32G32Sfloat); break;
							}
						}
						mesh.vertex_count = buffer_view.byteLength / stride;

						vattrs.insert(vattrs.end(), data.begin(), data.end());
					};
					do_attr("POSITION", p.attributes["POSITION"]);
					//do_attr("NORMAL", p.attributes["NORMAL"]);
					do_attr("TEXCOORD_0", p.attributes["TEXCOORD_0"]);
					mesh.interleaved = false;
					mesh.vertex_buffer = ptc.create_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eVertexBuffer | vuk::BufferUsageFlagBits::eStorageBuffer, std::span(vattrs.begin(), vattrs.end())).first;
				}
			}
		}
		ptc.wait_all_transfers();

		return meshes;
	}

	std::optional<vuk::Texture> texture_of_doge, variant1, variant2;

	std::optional<vuk::Texture> TSS_dummy;

	std::vector<NormalMaterial> nmats;
	std::vector<TintMaterial> tmats;

	std::vector<Renderable> renderables;
	vuk::Unique<vuk::PersistentDescriptorSet> pds;
	std::vector<vuk::Unique<vuk::ImageView>> ivs;

	std::random_device rd;
	std::mt19937 g(rd());

	vuk::Example xample{
		.name = "X1_tss",
		.setup = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			// Use STBI to load the image
			int x, y, chans;
			auto doge_image = stbi_load("../../examples/doge.png", &x, &y, &chans, 4);

			auto ptc = ifc.begin();
			// Similarly to buffers, we allocate the image and enqueue the upload
			auto [tex, _] = ptc.create_texture(vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, doge_image);
			texture_of_doge = std::move(tex);
			ptc.wait_all_transfers();
			stbi_image_free(doge_image);

			// Let's create two variants of the doge image (like in example 09)
			// Creating a compute pipeline that inverts an image 
			{
				vuk::ComputePipelineCreateInfo pci;
				pci.add_shader(util::read_entire_file("../../examples/invert.comp"), "invert.comp");
				runner.context->create_named_pipeline("invert", pci);
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
			variant1 = ptc.allocate_texture(ici);
			ici.format = vuk::Format::eR8G8B8A8Unorm;
			ici.usage = vuk::ImageUsageFlagBits::eStorage | vuk::ImageUsageFlagBits::eSampled;
			variant2 = ptc.allocate_texture(ici);


			{
				vuk::ImageCreateInfo ici;
				ici.format = vuk::Format::eR32G32B32A32Sfloat;
				ici.extent = vuk::Extent3D{ 1024, 1024, 1u };
				ici.samples = vuk::Samples::e1;
				ici.imageType = vuk::ImageType::e2D;
				ici.initialLayout = vuk::ImageLayout::eUndefined;
				ici.tiling = vuk::ImageTiling::eOptimal;
				ici.usage = vuk::ImageUsageFlagBits::eSampled | vuk::ImageUsageFlagBits::eStorage | vuk::ImageUsageFlagBits::eTransferDst | vuk::ImageUsageFlagBits::eTransferSrc;

				auto mips = (uint32_t)std::ceil(std::min(std::log2f((float)ici.extent.width), std::log2f((float)ici.extent.height)));
				ici.mipLevels = mips;
				ici.arrayLayers = 1;
				TSS_dummy = ptc.allocate_texture(ici);
			}


			// Make a RenderGraph to process the loaded image
			vuk::RenderGraph rg;
			rg.add_pass({
						.name = "10_preprocess",
						.resources = {"10_doge"_image(vuk::eMemoryRead), "10_v1"_image(vuk::eTransferDst), "10_v2"_image(vuk::eComputeRead), "X_TD"_image(vuk::eFragmentSampled)},
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
			rg.bind_attachment("10_doge", vuk::Attachment::from_texture(*texture_of_doge), vuk::eFragmentSampled, vuk::eFragmentSampled);
			rg.bind_attachment("10_v1", vuk::Attachment::from_texture(*variant1), vuk::eNone, vuk::eFragmentSampled);
			rg.bind_attachment("10_v2", vuk::Attachment::from_texture(*variant2), vuk::eNone, vuk::eFragmentSampled);
			rg.bind_attachment("X_TD", vuk::Attachment::from_texture(*TSS_dummy), vuk::eNone, vuk::eFragmentSampled);
			rg.build();
			rg.build(ptc);
			// The rendergraph is submitted and fence-waited on
			execute_submit_and_wait(ptc, rg);

			// Set up the resources for our renderer

			// Create meshes
			cube_mesh = {};
			cube_mesh->interleaved = true;
			cube_mesh->index_type = vuk::IndexType::eUint32;
			cube_mesh->attrs = vuk::Packed{ vuk::Format::eR32G32B32Sfloat, vuk::Ignore{offsetof(util::Vertex, uv_coordinates) - sizeof(util::Vertex::position)}, vuk::Format::eR32G32Sfloat };
			cube_mesh->vertex_buffer = ptc.create_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eVertexBuffer, std::span(&box.first[0], box.first.size())).first;
			cube_mesh->index_buffer = ptc.create_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eIndexBuffer, std::span(&box.second[0], box.second.size())).first;
			cube_mesh->index_count = (uint32_t)box.second.size();

			meshes = load_mesh(ptc, "../../examples/randosph_smooth.glb");/*{};
			quad_mesh->vertex_buffer = ptc.create_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eVertexBuffer, std::span(&box.first[0], 6)).first;
			quad_mesh->index_buffer = ptc.create_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eIndexBuffer, std::span(&box.second[0], 6)).first;
			quad_mesh->index_count = 6;*/

			ptc.wait_all_transfers();

			{
				vuk::ComputePipelineCreateInfo pci;
				pci.add_shader(util::read_entire_file("../../examples/tss_shade.comp"), "tss_shade.comp");
				runner.context->create_named_pipeline("tss_shade", pci);
			}

			{
			vuk::PipelineBaseCreateInfo pci;
				pci.add_shader(util::read_entire_file("../../examples/tss_bary_pass.vert"), "tss_bary_pass.vert");
				pci.add_shader(util::read_entire_file("../../examples/tss_bary_pass.frag"), "tss_bary_pass.frag");
				//pci.depth_stencil_state.depthCompareOp = vuk::CompareOp::eLessEqual;
				runner.context->create_named_pipeline("tss_bary", pci);
			}

			{
			vuk::ImageCreateInfo ici;
			ici.format = vuk::Format::eR16G16Unorm;
			ici.extent = vuk::Extent3D{1024, 1024, 1u};
			ici.samples = vuk::Samples::e1;
			ici.imageType = vuk::ImageType::e2D;
			ici.initialLayout = vuk::ImageLayout::eUndefined;
			ici.tiling = vuk::ImageTiling::eOptimal;
			ici.usage = vuk::ImageUsageFlagBits::eColorAttachment | vuk::ImageUsageFlagBits::eSampled;

			auto mips = (uint32_t)std::min(std::log2f((float)ici.extent.width), std::log2f((float)ici.extent.height));
			ici.mipLevels = 1;//mips;
			ici.arrayLayers = 1;

			bary_map = ptc.allocate_texture(ici);
			}

		bool first_draw = true;
		for(auto& mesh : meshes){
			// Make a RenderGraph to make barycentric maps
			vuk::RenderGraph rg;
			rg.add_pass({.name = "X1_bary",
						.resources = {"X1_bary_out"_image(vuk::eColorWrite)},
						.execute = [&mesh](vuk::CommandBuffer &command_buffer) {

					command_buffer.set_viewport(0, vuk::Area::framebuffer()).set_scissor(0, vuk::Area::framebuffer());

						if (mesh.interleaved) {
							command_buffer.bind_vertex_buffer(0, mesh.vertex_buffer.get(), 0, mesh.attrs);
						} else {
							size_t offset = 0;
							for (unsigned i = 0; i < mesh.attrs.list.size(); i++) {
								vuk::Buffer single_attr = mesh.vertex_buffer.get().subrange(offset, mesh.vertex_count * vuk::format_to_size(mesh.attrs.list[i].format));
								command_buffer.bind_vertex_buffer(i, single_attr, i, vuk::Packed{ mesh.attrs.list[i] });
								offset += single_attr.size;
							}
						}

							command_buffer
								.bind_index_buffer(mesh.index_buffer.get(), mesh.index_type)
								.bind_graphics_pipeline("tss_bary");

							command_buffer.draw_indexed(mesh.index_count, 1, 0, 0, 0);
						}});
			mesh.bary_map = &*bary_map;
			vuk::ImageViewCreateInfo ivci{.image = mesh.bary_map->image.get(), .format = mesh.bary_map->format, 
				.subresourceRange = {.aspectMask = vuk::ImageAspectFlagBits::eColor } };

			auto iv = ptc.create_image_view(ivci);
			auto att = vuk::Attachment::from_texture(*bary_map);
			att.image_view = iv.get();
			rg.bind_attachment("X1_bary_out", att, first_draw ? vuk::eClear : vuk::eFragmentSampled, vuk::eFragmentSampled);
			rg.build();
			rg.build(ptc);
			execute_submit_and_wait(ptc, rg);
			first_draw = false;
		}
			

	

			// Create the pipelines
			// A "normal" pipeline
			vuk::PipelineBaseInfo *pipe1;
			{
				vuk::PipelineBaseCreateInfo pci;
				pci.add_shader(util::read_entire_file("../../examples/tss_depth_pass.vert"), "tss_depth_pass.vert");
				pci.add_shader(util::read_entire_file("../../examples/tss_depth_pass.frag"), "tss_depth_pass.frag");
				pci.rasterization_state.cullMode = vuk::CullModeFlagBits::eBack;
				pci.depth_stencil_state.depthCompareOp = vuk::CompareOp::eLess;
				pipe1 = runner.context->get_pipeline(pci);
			}

			// A "tinted" pipeline
			vuk::PipelineBaseInfo* pipe2;
			{
				vuk::PipelineBaseCreateInfo pci;
				pci.add_shader(util::read_entire_file("../../examples/tss_composite_pass.vert"), "tss_composite_pass.vert");
				pci.add_shader(util::read_entire_file("../../examples/tss_composite_pass.frag"), "tss_composite_pass.frag");
				pci.depth_stencil_state.depthCompareOp = vuk::CompareOp::eEqual;
				runner.context->create_named_pipeline("tss_composite", pci);
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
			/*for (auto& mat : tmats) {
				mat.pipeline = pipe2;
				mat.tint_color = glm::vec4(dist_tint(g), dist_tint(g), dist_tint(g), 1.f);
			}*/

			// Create objects
			std::uniform_int_distribution<size_t> dist_mat(0, 1);
			std::uniform_int_distribution<size_t> dist_tex(0, 2);

			std::uniform_real_distribution<float> dist_pos(-10, 10);
			
			// 64 quads
			for (int i = 0; i < 0; i++) {
				auto mat_id = dist_mat(g);
				auto tex_id = dist_tex(g);

				Material* m = mat_id == 0 ? (Material*)&nmats[tex_id] : (Material*)&tmats[tex_id];
				glm::vec3 pos = glm::vec3(dist_pos(g), dist_pos(g), dist_pos(g));
				renderables.emplace_back(Renderable{.mesh = &*quad_mesh, .material = m, .position = pos});
			}

			for(auto& mesh : meshes){
				Material* m = &nmats[0];
				glm::vec3 pos = glm::vec3(0);
				renderables.emplace_back(Renderable{.mesh = &mesh, .material = m, .position = pos});
			}

			pds = ptc.create_persistent_descriptorset(*runner.context->get_named_compute_pipeline("tss_shade"), 1, 10);
			ivs.resize(10);
			for (unsigned i = 0; i < 10; i++) {
				vuk::ImageViewCreateInfo ivci{.image = TSS_dummy->image.get(), .format = TSS_dummy->format,
				.subresourceRange = {.aspectMask = vuk::ImageAspectFlagBits::eColor, .baseMipLevel = i, .levelCount = 1 } };

				ivs[i] = ptc.create_image_view(ivci);
				pds->update_storage_image(ptc, 0, i, ivs[i].get());
			}
			ptc.commit_persistent_descriptorset(*pds);


			// 16 cubes
			/*for (int i = 0; i < 16; i++) {
				auto mat_id = dist_mat(g);
				auto tex_id = dist_tex(g);

				Material* m = mat_id == 0 ? (Material*)&nmats[tex_id] : (Material*)&tmats[tex_id];
				glm::vec3 pos = glm::vec3(dist_pos(g), dist_pos(g), dist_pos(g));
				renderables.emplace_back(Renderable{ .mesh = &*cube_mesh, .material = m, .position = pos });
			}*/
		},
		.render = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			auto ptc = ifc.begin();

			// We set up VP data, same as in example 02_cube
			struct VP {
				glm::mat4 view;
				glm::mat4 proj;
			} vp;
			vp.view = glm::lookAt(glm::vec3(0, 0.5, 2), glm::vec3(0), glm::vec3(0, 1, 0));
			vp.proj = glm::perspective(glm::degrees(70.f), 1024.f/1024.f, 1.f, 100.f);

			// Upload view & projection
			auto [buboVP, stub3] = ptc.create_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vuk::BufferUsageFlagBits::eUniformBuffer, std::span(&vp, 1));
			auto uboVP = buboVP;
			ptc.wait_all_transfers();

			// Do a terrible simulation step
			// All objects are attracted to the origin
			for (auto& r : renderables) {
				/*
				auto force_mag = 0.1f/glm::length(r.position);
				r.velocity += force_mag * (-r.position) * ImGui::GetIO().DeltaTime;
				r.position += r.velocity * ImGui::GetIO().DeltaTime;
				*/
				r.orientation = glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f));
			}
			

			// Upload model matrices to an array
			auto modelmats = ptc._allocate_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vuk::BufferUsageFlagBits::eStorageBuffer, sizeof(glm::mat4) * renderables.size(), 1, true);
			for (auto i = 0; i < renderables.size(); i++) {
				glm::mat4 model_matrix = glm::translate(glm::mat4(1.f), renderables[i].position) * static_cast<glm::mat4>(renderables[i].orientation);
				memcpy(reinterpret_cast<glm::mat4*>(modelmats.mapped_ptr) + i, &model_matrix, sizeof(glm::mat4));
			}

			vuk::RenderGraph rg;

			// visibility pass
			rg.add_pass({
				.resources = {"X1_depth_result_1"_image(vuk::eColorWrite), "X1_depth_result_2"_image(vuk::eColorWrite), "X1_depth"_image(vuk::eDepthStencilRW)},
				.execute = [uboVP, modelmats](vuk::CommandBuffer& command_buffer) {
					command_buffer
					  .set_viewport(0, vuk::Area::relative(0,0,0.75,0.75))
					  .set_scissor(0, vuk::Area::relative(0,0,0.75,0.75));

					for (auto i = 0; i < renderables.size(); i++) {
						auto& r = renderables[i];

						// Set up the draw state based on the mesh and material
						if (r.mesh->interleaved) {
							command_buffer.bind_vertex_buffer(0, r.mesh->vertex_buffer.get(), 0, r.mesh->attrs);
						} else {
							size_t offset = 0;
							for (unsigned i = 0; i < r.mesh->attrs.list.size(); i++) {
								vuk::Buffer single_attr = r.mesh->vertex_buffer.get().subrange(offset, r.mesh->vertex_count * vuk::format_to_size(r.mesh->attrs.list[i].format));
								command_buffer.bind_vertex_buffer(i, single_attr, i, vuk::Packed{ r.mesh->attrs.list[i] });
								offset += single_attr.size;
							}
						}
						command_buffer
							.bind_index_buffer(r.mesh->index_buffer.get(), r.mesh->index_type)
							.bind_graphics_pipeline(r.material->pipeline)
							.bind_uniform_buffer(0, 0, uboVP)
							.bind_storage_buffer(0, 1, modelmats);

						r.material->bind_parameters(command_buffer);
						command_buffer.bind_sampled_image(0, 2, *TSS_dummy, {});

						// Draw the mesh, assign them different base instance to pick the correct transformation
						command_buffer.draw_indexed(r.mesh->index_count, 1, 0, 0, i);
						//break;
					}
				}
			});

			// shade evaluation pass
			rg.add_pass({
				.resources = {"X1_depth_result_1"_image(vuk::eComputeSampled), "X1_depth_result_2"_image(vuk::eComputeSampled), "X1_TSS"_image(vuk::eComputeWrite)},
				.execute = [&](vuk::CommandBuffer& command_buffer) {
					command_buffer.clear_image("X1_TSS", vuk::ClearColor(0.f, 0.f, 0.f, 0.f));
					command_buffer.image_barrier("X1_TSS", vuk::eTransferClear, vuk::eComputeSampled);
					
					command_buffer
						.bind_compute_pipeline("tss_shade")
						.bind_sampled_image(0, 0, "X1_depth_result_1", {})
						.bind_sampled_image(0, 1, "X1_depth_result_2", {})
						.bind_storage_buffer(0, 2, meshes[0].vertex_buffer.get());

						command_buffer.bind_storage_buffer(0, 3, *index_map);
						command_buffer.bind_storage_buffer(0, 4, *mesh_info);

						command_buffer.bind_persistent(1, *pds);
						command_buffer.dispatch_invocations(1024, 1024);
					}
				});

			// compositing pass
			rg.add_pass({
				.resources = {"X1_tss_final"_image(vuk::eColorWrite), "X1_depth"_image(vuk::eDepthStencilRW), "X1_TSS"_image(vuk::eFragmentSampled), "X1_db"_image(vuk::eColorWrite)},
				.execute = [uboVP, modelmats](vuk::CommandBuffer& command_buffer) {
					command_buffer
					  .set_viewport(0, vuk::Area::relative(0,0,0.75,0.75))
					  .set_scissor(0, vuk::Area::relative(0,0,0.75,0.75));

					for (auto i = 0; i < renderables.size(); i++) {
						auto& r = renderables[i];

						// Set up the draw state based on the mesh and material
						if (r.mesh->interleaved) {
							command_buffer.bind_vertex_buffer(0, r.mesh->vertex_buffer.get(), 0, r.mesh->attrs);
						} else {
							size_t offset = 0;
							for (unsigned i = 0; i < r.mesh->attrs.list.size(); i++) {
								vuk::Buffer single_attr = r.mesh->vertex_buffer.get().subrange(offset, r.mesh->vertex_count * vuk::format_to_size(r.mesh->attrs.list[i].format));
								command_buffer.bind_vertex_buffer(i, single_attr, i, vuk::Packed{ r.mesh->attrs.list[i] });
								offset += single_attr.size;
							}
						}
						command_buffer
							.bind_index_buffer(r.mesh->index_buffer.get(), r.mesh->index_type)
							.bind_graphics_pipeline("tss_composite")
							.bind_uniform_buffer(0, 0, uboVP)
							.bind_storage_buffer(0, 1, modelmats)
							.bind_sampled_image(0, 2, "X1_TSS", {});

						// Draw the mesh, assign them different base instance to pick the correct transformation
						command_buffer.draw_indexed(r.mesh->index_count, 1, 0, 0, i);
						//break;
					}
				}
			});

			rg.add_pass({
			.resources = {"X1_tss_final"_image(vuk::eTransferDst), "X1_TSS"_image(vuk::eTransferSrc), "X1_db"_image(vuk::eTransferSrc)},
			.execute = [uboVP, modelmats](vuk::CommandBuffer& command_buffer) {
					vuk::ImageBlit blit;
					int offset = 0;
					for (unsigned i = 0; i < 10; i++) {
						blit.srcOffsets = { vuk::Offset3D{0,0,0}, vuk::Offset3D{1024,1024,1} };
						blit.srcSubresource = { .aspectMask = vuk::ImageAspectFlagBits::eColor, .mipLevel = i };
						blit.srcOffsets = { vuk::Offset3D{0,0,0}, vuk::Offset3D{1024>>i,1024>>i,1} };
						blit.dstOffsets = { vuk::Offset3D{768, offset, 0}, vuk::Offset3D{768+(256>>i), offset+(256>>i), 1} };
						blit.dstSubresource = { .aspectMask = vuk::ImageAspectFlagBits::eColor };

						if (blit.dstOffsets[0].x == blit.dstOffsets[1].x) break;
						command_buffer.blit_image("X1_TSS", "X1_tss_final", blit, vuk::Filter::eNearest);
						offset += 256 >> i;
					}
			} });
			angle += 3.f * ImGui::GetIO().DeltaTime;

			rg.bind_attachment("X1_TSS", vuk::Attachment::from_texture(*TSS_dummy, vuk::ClearColor(0.f, 0.f, 0.f, 0.f)), vuk::eClear, vuk::eFragmentSampled);
			rg.mark_attachment_internal("X1_depth", vuk::Format::eD32Sfloat, vuk::Extent2D::Framebuffer{}, vuk::Samples::Framebuffer{}, vuk::ClearDepthStencil{ 1.0f, 0 });

			rg.mark_attachment_internal("X1_depth_result_1", vuk::Format::eR32G32Uint, runner.swapchain->extent, vuk::Samples::e1, vuk::ClearColor(0.f, 0.f, 0.f, 0.f));
			rg.mark_attachment_internal("X1_depth_result_2", vuk::Format::eR32G32B32A32Uint, vuk::Extent2D::Framebuffer{}, vuk::Samples::Framebuffer{}, vuk::ClearColor(0.f, 0.f, 0.f, 0.f));
			rg.mark_attachment_internal("X1_db", vuk::Format::eR32G32B32A32Uint, vuk::Extent2D::Framebuffer{}, vuk::Samples::Framebuffer{}, vuk::ClearColor(0.f, 0.f, 0.f, 0.f));
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