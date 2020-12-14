#include "example_runner.hpp"
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <stb_image.h>
#include <algorithm>
#include <random>
#include <numeric>


using glm::vec3;
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

enum class SDF_CMD_type : unsigned {
	end = 0,
	transform = 1,
	sphere = 2,
	smooth_combine = 3
};

struct end_cmd {
	alignas(16) SDF_CMD_type type = SDF_CMD_type::end;
};

struct transform_cmd {
	alignas(16) SDF_CMD_type type = SDF_CMD_type::transform;
	alignas(16) glm::mat4 inv_tf;

	transform_cmd(glm::mat4 tf) : inv_tf(glm::inverse(tf)) {}
};

struct sphere_cmd {
	alignas(16) SDF_CMD_type type = SDF_CMD_type::sphere;
	float radius;
	alignas(16) vec3 material;

	sphere_cmd(float r, vec3 material) : radius(r), material(material) {}
};

struct smooth_combine_cmd {
	alignas(16) SDF_CMD_type type = SDF_CMD_type::smooth_combine;
	float k;

	smooth_combine_cmd(float k) : k(k) {}
};

struct SDF_commands {
	std::vector<std::byte> data;

	template<class T>
	void push(const T& cmd) {
		auto size = data.size();
		data.resize(data.size() + sizeof(T));
		auto ptr = data.data() + size;
		memcpy(ptr, &cmd, sizeof(T));
	}

	void clear() {
		data.clear();
	}
};

// cubemap code from @jazzfool			
vuk::Texture load_cubemap_texture(std::string path, vuk::PerThreadContext& ptc) {
	int x, y, chans;
	stbi_set_flip_vertically_on_load(true);
	auto img = stbi_loadf(path.c_str(), &x, &y, &chans, STBI_rgb_alpha);
	stbi_set_flip_vertically_on_load(false);
	assert(img != nullptr);

	vuk::ImageCreateInfo ici;
	ici.format = vuk::Format::eR32G32B32A32Sfloat;
	ici.extent = vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u };
	ici.samples = vuk::Samples::e1;
	ici.imageType = vuk::ImageType::e2D;
	ici.initialLayout = vuk::ImageLayout::eUndefined;
	ici.tiling = vuk::ImageTiling::eOptimal;
	ici.usage = vuk::ImageUsageFlagBits::eTransferSrc | vuk::ImageUsageFlagBits::eTransferDst | vuk::ImageUsageFlagBits::eSampled;
	ici.mipLevels = 1;
	ici.arrayLayers = 1;

	auto tex = ptc.ctx.allocate_texture(ici);

	ptc.upload(*tex.image, ici.format, ici.extent, 0, std::span(&img[0], x * y * 4), false);

	ptc.wait_all_transfers();

	stbi_image_free(img);

	return tex;
}

namespace {
	float time = 0.f;
	auto box = util::generate_cube();
	int x, y;
	float dx1 = 0.f, dx2 = 0.f;
	bool d1 = true, d2 = false;
	std::optional<vuk::Texture> texture_of_doge;
	std::random_device rd;
	std::mt19937 g(rd());
	std::vector<vec3> poss;
	std::vector<vec3> vels;
	glm::vec3 max = glm::vec3(5.f, 5.f, 5.f);
	glm::vec3 min = glm::vec3(-5.f, -5.f, -5.f);
	glm::vec3 vox = glm::vec3(0.1f);
	enum class VertexPlacement : uint32_t {
		surface_net = 0,
		linear = 1
	};
	static VertexPlacement placement_method = VertexPlacement::linear;
	static glm::uvec3 count = glm::uvec3((max - min) / vox);
	static SDF_commands cmds;
	vuk::Texture env_cubemap;
	vuk::Texture hdr_texture;
	bool use_smooth_normals = true;
	bool view_space_grid = false;

	vuk::Example xample{
		.name = "08_pipelined_compute",
		.setup = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			{
			vuk::PipelineBaseCreateInfo pci;
			pci.add_shader(util::read_entire_file("../../examples/ubo_test.vert"), "ubo_test.vert");
			pci.add_shader(util::read_entire_file("../../examples/triangle_depthshaded.frag"), "triangle_depthshaded.frag");
			runner.context->create_named_pipeline("fwd", pci);
			}

			// creating a compute pipeline is very similar to creating a graphics pipeline
			// but here we can compile immediately
			{
			vuk::ComputePipelineCreateInfo pci;
			pci.add_shader(util::read_entire_file("../../examples/sdf.comp"), "sdf.comp");
			runner.context->create_named_pipeline("sdf", pci);
			}

			int chans;
			auto doge_image = stbi_load("../../examples/doge.png", &x, &y, &chans, 4);

			auto ptc = ifc.begin();
			auto [tex, stub] = ptc.create_texture(vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1 }, doge_image);
			texture_of_doge = std::move(tex);
			std::uniform_real_distribution<float> dist_pos(-3, 3);

			for (int i = 0; i < 64; i++) {
				glm::vec3 pos = glm::vec3(dist_pos(g), dist_pos(g), dist_pos(g));
				vels.push_back(0.5f * glm::normalize(glm::cross(pos, vec3(0, 1, 0))));
				poss.push_back(pos);
			}
			ptc.wait_all_transfers();
			stbi_image_free(doge_image);

			// cubemap code from @jazzfool
			// https://github.com/jazzfool/vuk-pbr/blob/main/Source/Renderer.cpp

			hdr_texture = load_cubemap_texture("../../examples/the_sky_is_on_fire_2k.hdr", ptc);

			// m_hdr_texture is a 2:1 equirectangular; it needs to be converted to a cubemap

			vuk::ImageCreateInfo cube_ici{
				.flags = vuk::ImageCreateFlagBits::eCubeCompatible,
				.imageType = vuk::ImageType::e2D,
				.format = vuk::Format::eR32G32B32A32Sfloat,
				.extent = {1024,1024,1},
				.arrayLayers = 6,
				.usage = vuk::ImageUsageFlagBits::eSampled | vuk::ImageUsageFlagBits::eColorAttachment
			};
			env_cubemap.image = ptc.ctx.allocate_texture(cube_ici).image;
			vuk::ImageViewCreateInfo cube_ivci{
				.image = *env_cubemap.image,
				.viewType = vuk::ImageViewType::eCube,
				.format = vuk::Format::eR32G32B32A32Sfloat,
				.subresourceRange = vuk::ImageSubresourceRange{.aspectMask = vuk::ImageAspectFlagBits::eColor, .layerCount = 6}
			};
			env_cubemap.view = ptc.ctx.create_image_view(cube_ivci);
			env_cubemap.format = vuk::Format::eR32G32B32A32Sfloat;
			env_cubemap.extent = { 1024,1024,1 };

			const glm::mat4 capture_projection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
			const glm::mat4 capture_views[] = { glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
											   glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
											   glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
											   glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
											   glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
											   glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f)) };


			// upload the cube mesh
			auto [bverts, stub1] = ptc.create_scratch_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eVertexBuffer, std::span{ box.first });
			auto verts = std::move(bverts);
			auto [binds, stub2] = ptc.create_scratch_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eIndexBuffer, std::span{ box.second });
			auto inds = std::move(binds);

			ptc.wait_all_transfers();

			vuk::PipelineBaseCreateInfo equirectangular_to_cubemap;
			equirectangular_to_cubemap.add_shader(util::read_entire_file("../../examples/cubemap.vert"), "cubemap.vert");
			equirectangular_to_cubemap.add_shader(util::read_entire_file("../../examples/equirectangular_to_cubemap.frag"), "equirectangular_to_cubemap.frag");
			ptc.ctx.create_named_pipeline("equirectangular_to_cubemap", equirectangular_to_cubemap);

			{
				for (unsigned i = 0; i < 6; ++i) {
					vuk::RenderGraph rg;
					rg.add_pass({.resources = {"env_cubemap_face"_image(vuk::eColorWrite)}, .execute = [&](vuk::CommandBuffer& cbuf) {
									 cbuf.set_viewport(0, vuk::Rect2D::framebuffer())
										 .set_scissor(0, vuk::Rect2D::framebuffer())
										 .bind_vertex_buffer(0, verts, 0,
															 vuk::Packed{vuk::Format::eR32G32B32Sfloat, vuk::Ignore{sizeof(util::Vertex) - sizeof(util::Vertex::position)}})
										 .bind_index_buffer(inds, vuk::IndexType::eUint32)
										 .bind_sampled_image(0, 2, hdr_texture,
															 vuk::SamplerCreateInfo{.magFilter = vuk::Filter::eLinear,
																					.minFilter = vuk::Filter::eLinear,
																					.mipmapMode = vuk::SamplerMipmapMode::eLinear,
																					.addressModeU = vuk::SamplerAddressMode::eClampToEdge,
																					.addressModeV = vuk::SamplerAddressMode::eClampToEdge,
																					.addressModeW = vuk::SamplerAddressMode::eClampToEdge})
										 .bind_graphics_pipeline("equirectangular_to_cubemap");
									 glm::mat4* projection = cbuf.map_scratch_uniform_binding<glm::mat4>(0, 0);
									 *projection = capture_projection;
									 glm::mat4* view = cbuf.map_scratch_uniform_binding<glm::mat4>(0, 1);
									 *view = capture_views[i];
									 cbuf.draw_indexed(box.second.size(), 1, 0, 0, 0);
								 } });

					vuk::ImageAttachment ia{
							.image = *env_cubemap.image,
							.image_view = *env_cubemap.view.layer_subrange(i, 1).view_as(vuk::ImageViewType::e2D).apply(),
							.extent = vuk::Extent2D{1024, 1024},
							.format = vuk::Format::eR32G32B32A32Sfloat,
					};
					rg.attach_image("env_cubemap_face",	ia, vuk::Access::eNone, vuk::Access::eFragmentSampled);

					auto erg = std::move(rg).link(ptc);
					vuk::execute_submit_and_wait(ptc, std::move(erg));
				}
			}
		},
		.render = [&](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			auto ptc = ifc.begin();
			float resolution = vox.x;
			ImGui::Checkbox("Smooth normals", &use_smooth_normals);
			ImGui::Checkbox("Viewspace grid", &view_space_grid);
			ImGui::DragFloat("Resolution", &resolution, 0.01f, 0.f, 5.f, "%.3f", 1.f);
			
			const char* items[] = { "Surface net", "Linear contouring" };
			ImGui::Combo("Meshing", (int32_t*)&placement_method, items, (int)std::size(items));
			// init vtx_buf
			auto vtx_buf = ptc._allocate_scratch_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eStorageBuffer | vuk::BufferUsageFlagBits::eVertexBuffer, sizeof(glm::vec3) * 3 * 150000, 1, false);
			auto idx_buf = ptc._allocate_scratch_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eStorageBuffer | vuk::BufferUsageFlagBits::eIndexBuffer, sizeof(glm::uint) * 200 * 4096, 1, false);
			auto idcmd_buf = ptc._allocate_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vuk::BufferUsageFlagBits::eStorageBuffer | vuk::BufferUsageFlagBits::eIndirectBuffer, sizeof(vuk::DrawIndexedIndirectCommand), sizeof(vuk::DrawIndexedIndirectCommand), true);
			vuk::DrawIndexedIndirectCommand di{};
			di.instanceCount = 1;
			memcpy(idcmd_buf.mapped_ptr, &di, sizeof(vuk::DrawIndexedIndirectCommand));
			cmds.clear();
			cmds.push(sphere_cmd(1, vec3(1.00, 0.71, 0.29)));
			for (auto i = 0; i < poss.size(); i++) {
				cmds.push(transform_cmd(glm::translate(glm::mat4(1.f), poss[i])));
				cmds.push(sphere_cmd(0.2f, vec3(0.95, 0.93, 0.88)));
				cmds.push(smooth_combine_cmd(1));
			}
			cmds.push(end_cmd());

			for (auto i = 0; i < vels.size(); i++) {
				auto force_mag = 0.1f / glm::length(poss[i]);
				vels[i] += force_mag * (-poss[i]) * ImGui::GetIO().DeltaTime;
				poss[i] += vels[i] * ImGui::GetIO().DeltaTime;
			}

			auto vmcmd_buf = ptc._allocate_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vuk::BufferUsageFlagBits::eUniformBuffer, cmds.data.size(), 1, true);
			memcpy(vmcmd_buf.mapped_ptr, cmds.data.data(), cmds.data.size());
			struct VP {
				glm::mat4 view;
				glm::mat4 proj;
			} vp;

			vec3 cam_pos = glm::vec3(0, 1.5f, 3.5f);
			vp.view = glm::lookAt(cam_pos, glm::vec3(0), glm::vec3(0, 1, 0));
			vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 0.01f, 10.f);
			vp.proj[1][1] *= -1;

			vox = glm::vec3(resolution);
			vec3 mmin = min, mmax = max;
			if (view_space_grid) {
				vox = vec3(vp.view * glm::vec4(vox, 0.f));
				mmin = vec3(vp.view * glm::vec4(min, 1.f));
				mmax = vec3(vp.view * glm::vec4(max, 1.f));
			}
			auto mmmin = glm::min(mmin, mmax);
			auto mmmax = glm::max(mmin, mmax);
			count = glm::uvec3((mmmax - mmmin) / vox);

			vuk::RenderGraph rg;

			struct PC {
				glm::vec3 min;
				float px1 = 0.f;
				glm::vec3 vox_size;
				float px2 = 0.f;
				VertexPlacement placement_method;
			}pc = {mmin, dx1, vox, dx2, placement_method};

			rg.add_pass({
				.resources = {"vtx"_buffer(vuk::eComputeWrite), "idx"_buffer(vuk::eComputeWrite), "cmd"_buffer(vuk::eComputeWrite)},
				.execute = [pc, vmcmd_buf](vuk::CommandBuffer& command_buffer) {
					command_buffer
						.bind_storage_buffer(0, 0, command_buffer.get_resource_buffer("vtx"))
						.bind_storage_buffer(0, 1, command_buffer.get_resource_buffer("idx"))
						.bind_uniform_buffer(0, 2, vmcmd_buf)
						.bind_storage_buffer(0, 3, command_buffer.get_resource_buffer("cmd"))
						.bind_compute_pipeline("sdf")
						.push_constants(vuk::ShaderStageFlagBits::eCompute, 0, pc)
						.dispatch_invocations(count.x, count.y, count.z);
				}
			});

			auto uboVP = ptc._allocate_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vuk::BufferUsageFlagBits::eUniformBuffer, sizeof(VP), 1, true);
			memcpy(uboVP.mapped_ptr, &vp, sizeof(VP));

			struct FwdConstants {
				vec3 camPos;
				unsigned use_smooth_normals;
				unsigned view_space_grid;
			} fwd_pc = { cam_pos, use_smooth_normals, view_space_grid };

			rg.add_pass({
				.resources = {"08_pipelined_compute_final"_image(vuk::eColorWrite), "08_depth"_image(vuk::eDepthStencilRW), "vtx"_buffer(vuk::eAttributeRead), "idx"_buffer(vuk::eIndexRead), "cmd"_buffer(vuk::eIndirectRead)},
				.execute = [uboVP, fwd_pc](vuk::CommandBuffer& command_buffer) {
					command_buffer
						.set_viewport(0, vuk::Rect2D::framebuffer())
						.set_scissor(0, vuk::Rect2D::framebuffer())

						.bind_vertex_buffer(0, command_buffer.get_resource_buffer("vtx"), 0, vuk::Packed{vuk::Format::eR32G32B32Sfloat, vuk::Format::eR8G8B8A8Unorm, vuk::Format::eR32G32B32Sfloat})
						.bind_index_buffer(command_buffer.get_resource_buffer("idx"), vuk::IndexType::eUint32)
						.bind_graphics_pipeline("fwd")
						.bind_uniform_buffer(0, 0, uboVP)
						.bind_sampled_image(0, 1, *env_cubemap.view, {})
						.push_constants(vuk::ShaderStageFlagBits::eVertex | vuk::ShaderStageFlagBits::eFragment, 0, fwd_pc)
						.draw_indexed_indirect(1, command_buffer.get_resource_buffer("cmd"));
				}
			});

			time += ImGui::GetIO().DeltaTime;

			if (d1) {
				dx1 += 0.001f;
			} else {
				dx1 -= 0.001f;
			}
			if (abs(dx1) > 5.f) {
				d1 = !d1;
			}

			rg.attach_managed("08_depth", vuk::Format::eD32Sfloat, vuk::Dimension2D::framebuffer(), vuk::Samples::e1, vuk::ClearDepthStencil{ 1.f, 0 });
			rg.attach_buffer("vtx", vtx_buf, vuk::eNone, vuk::eNone);
			rg.attach_buffer("idx", idx_buf, vuk::eNone, vuk::eNone);
			rg.attach_buffer("cmd", idcmd_buf, vuk::eNone, vuk::eNone);
			return rg;
		},
		.cleanup = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			texture_of_doge.reset();
		}

	};

	REGISTER_EXAMPLE(xample);
}
