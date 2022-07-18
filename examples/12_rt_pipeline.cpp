#include "example_runner.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <stb_image.h>

/* 04_texture
 * In this example we will build on the previous examples (02_cube and 03_multipass), but we will make the cube textured.
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the example same code.
 * Check out the framework (example_runner_*) files if interested!
 */

namespace {
	float angle = 0.f;
	auto box = util::generate_cube();
	vuk::BufferGPU verts, inds;
	// A vuk::Texture is an owned pair of Image and ImageView
	// An optional is used here so that we can reset this on cleanup, despite being a global (which is to simplify the code here)
	std::optional<vuk::Texture> texture_of_doge;
	vuk::Unique<VkAccelerationStructureKHR> nulltlas, tlas;
	vuk::Unique<vuk::BufferGPU> nulltlas_buf;

	vuk::Example x{
		.name = "12_rt_pipeline",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      auto& ctx = allocator.get_context();
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file("../../examples/rt.rgen"), "rt.rgen");
			      pci.add_glsl(util::read_entire_file("../../examples/rt.rmiss"), "rt.rmiss");
			      runner.context->create_named_pipeline("raytracing", pci);
		      }

		      // Use STBI to load the image
		      int x, y, chans;
		      auto doge_image = stbi_load("../../examples/doge.png", &x, &y, &chans, 4);

		      // Similarly to buffers, we allocate the image and enqueue the upload
		      auto [tex, tex_fut] = create_texture(allocator, vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, doge_image, true);
		      texture_of_doge = std::move(tex);
		      runner.enqueue_setup(std::move(tex_fut));
		      stbi_image_free(doge_image);

		      // We set up the cube data, same as in example 02_cube
		      auto [vert_buf, vert_fut] = create_buffer_gpu(allocator, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.first));
		      verts = *vert_buf;
		      auto [ind_buf, ind_fut] = create_buffer_gpu(allocator, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.second));
		      inds = *ind_buf;

		      // BLAS building
		      uint32_t maxPrimitiveCount = box.second.size() / 3;

		      // Describe buffer as array of VertexObj.
		      VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
		      triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; // vec3 vertex position data.
		      triangles.vertexData.deviceAddress = verts.device_address;
		      triangles.vertexStride = sizeof(util::Vertex);
		      // Describe index data (32-bit unsigned int)
		      triangles.indexType = VK_INDEX_TYPE_UINT32;
		      triangles.indexData.deviceAddress = inds.device_address;
		      // Indicate identity transform by setting transformData to null device pointer.
		      // triangles.transformData = {};
		      triangles.maxVertex = box.first.size();

		      // Identify the above data as containing opaque triangles.
		      VkAccelerationStructureGeometryKHR asGeom{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
		      asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		      asGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		      asGeom.geometry.triangles = triangles;

		      // The entire array will be used to build the BLAS.
		      VkAccelerationStructureBuildRangeInfoKHR offset;
		      offset.firstVertex = 0;
		      offset.primitiveCount = maxPrimitiveCount;
		      offset.primitiveOffset = 0;
		      offset.transformOffset = 0;

		      // TLAS building

		      VkAccelerationStructureGeometryInstancesDataKHR instancesVk{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
		      instancesVk.data.deviceAddress = {};

		      // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label it as instance data.
		      VkAccelerationStructureGeometryKHR topASGeometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
		      topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		      topASGeometry.geometry.instances = instancesVk;

		      // Find sizes
		      VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
		      buildInfo.flags = 0;
		      buildInfo.geometryCount = 1;
		      buildInfo.pGeometries = &topASGeometry;
		      buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		      buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		      buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

		      uint32_t countInstance = 0;

		      VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
		      ctx.vkGetAccelerationStructureBuildSizesKHR(
		          allocator.get_context().device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &countInstance, &sizeInfo);

		      VkAccelerationStructureCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		      createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		      createInfo.size = sizeInfo.accelerationStructureSize;
		      nulltlas_buf = *vuk::allocate_buffer_gpu(allocator, { .mem_usage = vuk::MemoryUsage::eGPUonly, .size = sizeInfo.accelerationStructureSize });
		      createInfo.buffer = nulltlas_buf->buffer;

		      nulltlas = vuk::Unique<VkAccelerationStructureKHR>(allocator);
		      allocator.allocate_acceleration_structures({ &*nulltlas, 1 }, { &createInfo, 1 });

		      // Allocate the scratch memory
		      auto scratchBuffer =
		          *vuk::allocate_buffer_gpu(allocator, vuk::BufferCreateInfo{ .mem_usage = vuk::MemoryUsage::eGPUonly, .size = sizeInfo.buildScratchSize });

		      // Update build information
		      buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
		      buildInfo.dstAccelerationStructure = *nulltlas;
		      buildInfo.scratchData.deviceAddress = scratchBuffer->device_address;

		      // Build Offsets info: n instances
		      VkAccelerationStructureBuildRangeInfoKHR buildOffsetInfo{ countInstance, 0, 0, 0 };
		      const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

		      // Build the TLAS
		      vuk::RenderGraph tlas_build("tlas_build");
		      tlas_build.add_pass({ .execute = [&ctx, buildInfo, pBuildOffsetInfo](vuk::CommandBuffer& command_buffer) {
			      ctx.vkCmdBuildAccelerationStructuresKHR(command_buffer.get_underlying(), 1, &buildInfo, &pBuildOffsetInfo);
		      } });
		      vuk::wait_for_futures(allocator, vuk::Future(std::make_shared<vuk::RenderGraph>(std::move(tlas_build)), ""));
		      // For the example, we just ask these that these uploads complete before moving on to rendering
		      // In an engine, you would integrate these uploads into some explicit system
		      runner.enqueue_setup(std::move(vert_fut));
		      runner.enqueue_setup(std::move(ind_fut));
		    },
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::Future target) {
		      struct VP {
			      glm::mat4 inv_view;
			      glm::mat4 inv_proj;
		      } vp;
		      vp.inv_view = glm::lookAt(glm::vec3(0, 1.5, 3.5), glm::vec3(0), glm::vec3(0, 1, 0));
		      vp.inv_proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);
		      vp.inv_proj[1][1] *= -1;
		      vp.inv_view = glm::inverse(vp.inv_view);
		      vp.inv_proj = glm::inverse(vp.inv_proj);

		      auto [buboVP, uboVP_fut] = create_buffer_cross_device(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, std::span(&vp, 1));
		      auto uboVP = *buboVP;

		      vuk::wait_for_futures(frame_allocator, uboVP_fut);

		      vuk::RenderGraph rg("12");
		      rg.attach_in("12_final", std::move(target));
		      //  Set up the pass to draw the textured cube, with a color and a depth attachment
		      rg.add_pass(
		          { .resources = { "12_final"_image >> vuk::eRayTracingWrite >> "04_texture_final" }, .execute = [uboVP](vuk::CommandBuffer& command_buffer) {
			           command_buffer.bind_acceleration_structure(0, 0, *nulltlas)
			               .bind_image(0, 1, "12_final")
			               .bind_buffer(0, 2, uboVP)
			               .bind_ray_tracing_pipeline("raytracing");
			           /*glm::mat4* model = command_buffer.map_scratch_uniform_binding<glm::mat4>(0, 1);
			            *model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));*/
			           command_buffer.trace_rays(1024, 1024, 1);
		           } });

		      angle += 180.f * ImGui::GetIO().DeltaTime;

		      return vuk::Future{ std::make_unique<vuk::RenderGraph>(std::move(rg)), "04_texture_final" };
		    },

		// Perform cleanup for the example
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      // We release the texture resources
		      texture_of_doge.reset();
		      nulltlas.reset();
		      nulltlas_buf.reset();
		    }
	};

	REGISTER_EXAMPLE(x);
} // namespace