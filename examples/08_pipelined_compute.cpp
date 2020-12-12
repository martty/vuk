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
	float color;

	sphere_cmd(float r, float color) : radius(r), color(color) {}
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
			std::uniform_real_distribution<float> dist_pos(-5, 5);

			for (int i = 0; i < 32; i++) {
				glm::vec3 pos = glm::vec3(dist_pos(g), dist_pos(g), dist_pos(g));
				vels.push_back(0.01f * glm::normalize(glm::cross(pos, vec3(0, 1, 0))));
				poss.push_back(pos);
			}
			ptc.wait_all_transfers();
			stbi_image_free(doge_image);
		},
		.render = [&](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			auto ptc = ifc.begin();
			float resolution = vox.x;
			ImGui::DragFloat("Resolution", &resolution, 0.01f, 0.f, 5.f, "%.3f", 1.f);
			vox = glm::vec3(resolution);
			count = glm::uvec3((max - min) / vox);
			const char* items[] = { "Surface net", "Linear contouring" };
			ImGui::Combo("Meshing", (int32_t*)&placement_method, items, std::size(items));
			// init vtx_buf
			auto vtx_buf = ptc._allocate_scratch_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eStorageBuffer | vuk::BufferUsageFlagBits::eVertexBuffer, sizeof(glm::vec3) * 3 * 150000, 1, false);
			auto idx_buf = ptc._allocate_scratch_buffer(vuk::MemoryUsage::eGPUonly, vuk::BufferUsageFlagBits::eStorageBuffer | vuk::BufferUsageFlagBits::eIndexBuffer, sizeof(glm::uint) * 200 * 4096, 1, false);
			auto idcmd_buf = ptc._allocate_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vuk::BufferUsageFlagBits::eStorageBuffer | vuk::BufferUsageFlagBits::eIndirectBuffer, sizeof(vuk::DrawIndexedIndirectCommand), sizeof(vuk::DrawIndexedIndirectCommand), true);
			vuk::DrawIndexedIndirectCommand di{};
			di.instanceCount = 1;
			memcpy(idcmd_buf.mapped_ptr, &di, sizeof(vuk::DrawIndexedIndirectCommand));
			cmds.clear();
			cmds.push(sphere_cmd(1, 1.f));
			for (auto i = 0; i < poss.size(); i++) {
				cmds.push(transform_cmd(glm::translate(glm::mat4(1.f), poss[i])));
				cmds.push(sphere_cmd(0.2, 0.1f));
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
			vp.view = glm::lookAt(glm::vec3(0, 1.5, 3.5), glm::vec3(0), glm::vec3(0, 1, 0));
			vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);
			vp.proj[1][1] *= -1;

			vuk::RenderGraph rg;
			// this pass executes outside of a renderpass
			// we declare a buffer dependency and dispatch a compute shader

			struct PC {
				glm::vec3 min;
				float px1 = 0.f;
				glm::vec3 vox_size;
				float px2 = 0.f;
				VertexPlacement placement_method;
			}pc = {min, dx1, vox, dx2, placement_method};

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

			// draw the scrambled image, with a buffer dependency on the scramble buffer
			rg.add_pass({
				.resources = {"08_pipelined_compute_final"_image(vuk::eColorWrite), "08_depth"_image(vuk::eDepthStencilRW), "vtx"_buffer(vuk::eAttributeRead), "idx"_buffer(vuk::eIndexRead), "cmd"_buffer(vuk::eIndirectRead)},
				.execute = [uboVP](vuk::CommandBuffer& command_buffer) {
					command_buffer
						.set_viewport(0, vuk::Rect2D::framebuffer())
						.set_scissor(0, vuk::Rect2D::framebuffer())

						.bind_vertex_buffer(0, command_buffer.get_resource_buffer("vtx"), 0, vuk::Packed{vuk::Format::eR32G32B32Sfloat, vuk::Format::eR32G32B32Sfloat})
						.bind_index_buffer(command_buffer.get_resource_buffer("idx"), vuk::IndexType::eUint32)
						.bind_graphics_pipeline("fwd")
						.bind_uniform_buffer(0, 0, uboVP)
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
