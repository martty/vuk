#include "example_runner.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>

/* 03_multipass
 * In this example we will build on the previous example (02_cube),
 * but we will add in a second resource (a depth buffer). Furthermore we will see how to add multiple passes.
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the example same code.
 * Check out the framework (example_runner_*) files if interested!
 */

namespace {
	float angle = 0.f;
	auto box = util::generate_cube();
	vuk::Unique<vuk::Buffer> verts, inds;

	vuk::Example x{
		.name = "03_multipass",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file((root / "examples/triangle.vert").generic_string()), (root / "examples/triangle.vert").generic_string());
			      pci.add_glsl(util::read_entire_file((root / "examples/triangle.frag").generic_string()), (root / "examples/triangle.frag").generic_string());
			      runner.context->create_named_pipeline("triangle", pci);
		      }
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file((root / "examples/ubo_test.vert").generic_string()), (root / "examples/ubo_test.vert").generic_string());
			      pci.add_glsl(util::read_entire_file((root / "examples/triangle_depthshaded.frag").generic_string()),
			                   (root / "examples/triangle_depthshaded.frag").generic_string());
			      runner.context->create_named_pipeline("cube", pci);
		      }

		      // We set up the cube data, same as in example 02_cube
		      auto [vert_buf, vert_fut] = create_buffer(allocator, vuk::MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.first));
		      verts = std::move(vert_buf);
		      auto [ind_buf, ind_fut] = create_buffer(allocator, vuk::MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.second));
		      inds = std::move(ind_buf);
		      // For the example, we just ask these that these uploads complete before moving on to rendering
		      // In an engine, you would integrate these uploads into some explicit system
		      runner.enqueue_setup(std::move(vert_fut));
		      runner.enqueue_setup(std::move(ind_fut));
		    },
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::TypedFuture<vuk::Image> target) {
		      struct VP {
			      glm::mat4 view;
			      glm::mat4 proj;
		      } vp;
		      vp.view = glm::lookAt(glm::vec3(0, 1.5, 3.5), glm::vec3(0), glm::vec3(0, 1, 0));
		      vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);
		      vp.proj[1][1] *= -1;

		      auto [buboVP, uboVP_fut] = create_buffer(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, vuk::DomainFlagBits::eTransferOnGraphics, std::span(&vp, 1));
		      auto uboVP = *buboVP;

		      // Add a pass to draw a triangle (from the first example) into the top left corner

		      // In this example we want to use this resource after our write, but resource names are consumed by writes
		      // To be able to refer to this resource with the write completed, we assign it a new name ("03_multipass+")
		      auto small_tri_generic = [](vuk::Rect2D position) {
			      return vuk::make_pass("03_small_tri", [=](vuk::CommandBuffer& command_buffer, vuk::IA<vuk::eColorWrite, decltype([]() {})> color_rt) {
				      command_buffer.set_viewport(0, position)
				          .set_scissor(0, position)
				          .set_rasterization({})     // Set the default rasterization state
				          .broadcast_color_blend({}) // Set the default color blend state
				          .bind_graphics_pipeline("triangle")
				          .draw(3, 1, 0, 0);
				      return std::make_tuple(color_rt);
			      });
		      };

		      // Add a pass to draw a triangle (from the first example) into the bottom right corner
		      auto tl_tri = small_tri_generic(vuk::Rect2D::relative(0.0f, 0.0f, 0.2f, 0.2f));
		      auto br_tri = small_tri_generic(vuk::Rect2D::relative(0.8f, 0.8f, 0.2f, 0.2f));

			  // Add a pass to draw a cube (from the second example) in the middle, but with depth buffering
		      // Here a second resource is added: a depth attachment
		      // The example framework took care of our color image, but this attachment we will need bind later
		      // Depth attachments are denoted by the use vuk::eDepthStencilRW
			  auto cube_pass = vuk::make_pass("03_cube", [uboVP](vuk::CommandBuffer& command_buffer, vuk::IA<vuk::eColorWrite, decltype([]() {})> color_rt, vuk::IA<vuk::eDepthStencilRW, decltype([]() {})> depth_rt) {
			          command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
			              .set_scissor(0, vuk::Rect2D::framebuffer())
			              .set_rasterization({}) // Set the default rasterization state
			              // Set the depth/stencil state
			              .set_depth_stencil(vuk::PipelineDepthStencilStateCreateInfo{
			                  .depthTestEnable = true, .depthWriteEnable = true, .depthCompareOp = vuk::CompareOp::eLessOrEqual })
			              .broadcast_color_blend({}) // Set the default color blend state
			              .bind_index_buffer(*inds, vuk::IndexType::eUint32)
			              .bind_graphics_pipeline("cube")
			              .bind_vertex_buffer(
			                  0, *verts, 0, vuk::Packed{ vuk::Format::eR32G32B32Sfloat, vuk::Ignore{ sizeof(util::Vertex) - sizeof(util::Vertex::position) } })
			              .bind_buffer(0, 0, uboVP);
			          glm::mat4* model = command_buffer.map_scratch_buffer<glm::mat4>(0, 1);
			          *model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
			          command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0);
			          return std::make_tuple(color_rt);
				  });
		      
		      angle += 360.f * ImGui::GetIO().DeltaTime;

		      // The rendergraph has a reference to "03_depth" resource, so we must provide the attachment
		      // In this case, the depth attachment is an "internal" attachment:
		      // we don't provide an input texture, nor do we want to save the results later
		      // This depth attachment will have extents matching the framebuffer (deduced from the color attachment)
		      // but we will need to provide the format
		      auto depth_img = vuk::declare_ia("03_depth");
		      depth_img->format = vuk::Format::eD32Sfloat;
		      depth_img = vuk::clear(depth_img, vuk::ClearDepthStencil{ 1.0f, 0 });

		      return cube_pass(tl_tri(br_tri(std::move(target))), std::move(depth_img));
		    },
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
		      verts.reset();
		      inds.reset();
		    }
	};

	REGISTER_EXAMPLE(x);
} // namespace