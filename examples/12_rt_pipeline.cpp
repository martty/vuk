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
	vuk::Unique<VkAccelerationStructureKHR> tlas, blas;
	vuk::Unique<vuk::BufferGPU> tlas_buf, blas_buf;

	vuk::Example x{
		.name = "12_rt_pipeline",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      auto& ctx = allocator.get_context();
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file("../../examples/rt.rgen"), "rt.rgen");
			      pci.add_glsl(util::read_entire_file("../../examples/rt.rmiss"), "rt.rmiss");
			      pci.add_glsl(util::read_entire_file("../../examples/rt.rchit"), "rt.rchit");
			      pci.add_hit_group(vuk::HitGroup{ .type = vuk::HitGroupType::eTriangles, .closest_hit = 2 });
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

		      vuk::wait_for_futures(allocator, vert_fut, ind_fut);

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
		      triangles.transformData = {};
		      triangles.maxVertex = box.first.size();

		      // Identify the above data as containing opaque triangles.
		      VkAccelerationStructureGeometryKHR asGeom{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
		      asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		      asGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		      asGeom.geometry.triangles = triangles;

		      // The entire array will be used to build the BLAS.
		      VkAccelerationStructureBuildRangeInfoKHR blas_offset;
		      blas_offset.firstVertex = 0;
		      blas_offset.primitiveCount = maxPrimitiveCount;
		      blas_offset.primitiveOffset = 0;
		      blas_offset.transformOffset = 0;
		      const VkAccelerationStructureBuildRangeInfoKHR* pblas_offset = &blas_offset;

		      VkAccelerationStructureBuildGeometryInfoKHR blas_build_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
		      blas_build_info.dstAccelerationStructure = *blas;
		      blas_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		      blas_build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		      blas_build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		      blas_build_info.geometryCount = 1;
		      blas_build_info.pGeometries = &asGeom;

		      VkAccelerationStructureBuildSizesInfoKHR blas_size_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };

		      ctx.vkGetAccelerationStructureBuildSizesKHR(
		          ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blas_build_info, &maxPrimitiveCount, &blas_size_info);

		      VkAccelerationStructureCreateInfoKHR blas_ci{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		      blas_ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		      blas_ci.size = blas_size_info.accelerationStructureSize; // Will be used to allocate memory.
		      blas_buf = *vuk::allocate_buffer_gpu(allocator, { .mem_usage = vuk::MemoryUsage::eGPUonly, .size = blas_size_info.accelerationStructureSize });
		      blas_ci.buffer = blas_buf->buffer;
		      blas_ci.offset = blas_buf->offset;

		      blas = vuk::Unique<VkAccelerationStructureKHR>(allocator);
		      allocator.allocate_acceleration_structures({ &*blas, 1 }, { &blas_ci, 1 });

		      // Allocate the scratch memory
		      auto blas_scratch_buffer =
		          *vuk::allocate_buffer_gpu(allocator, vuk::BufferCreateInfo{ .mem_usage = vuk::MemoryUsage::eGPUonly, .size = blas_size_info.buildScratchSize });

		      // Update build information
		      blas_build_info.srcAccelerationStructure = VK_NULL_HANDLE;
		      blas_build_info.dstAccelerationStructure = *blas;
		      blas_build_info.scratchData.deviceAddress = blas_scratch_buffer->device_address;

		      // TLAS building
		      VkAccelerationStructureInstanceKHR rayInst{};
		      rayInst.transform = {};
		      rayInst.transform.matrix[0][0] = 1.f;
		      rayInst.transform.matrix[1][1] = 1.f;
		      rayInst.transform.matrix[2][2] = 1.f;
		      rayInst.instanceCustomIndex = 0;                                                                // gl_InstanceCustomIndexEXT
		      rayInst.accelerationStructureReference = blas_buf->device_address;
		      rayInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		      rayInst.mask = 0xFF;                                //  Only be hit if rayMask & instance.mask != 0
		      rayInst.instanceShaderBindingTableRecordOffset = 0; // We will use the same hit group for all objects

		      auto instances_buffer = vuk::create_buffer_cross_device(allocator, vuk::MemoryUsage::eCPUtoGPU, std::span{ &rayInst, 1 });
		      vuk::wait_for_futures(allocator, instances_buffer.second); // no-op

		      VkAccelerationStructureGeometryInstancesDataKHR instancesVk{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
		      instancesVk.data.deviceAddress = instances_buffer.first->device_address;

		      // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label it as instance data.
		      VkAccelerationStructureGeometryKHR topASGeometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
		      topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		      topASGeometry.geometry.instances = instancesVk;

		      // Find sizes
		      VkAccelerationStructureBuildGeometryInfoKHR tlas_build_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
		      tlas_build_info.flags = 0;
		      tlas_build_info.geometryCount = 1;
		      tlas_build_info.pGeometries = &topASGeometry;
		      tlas_build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		      tlas_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

		      uint32_t countInstance = 1;

		      VkAccelerationStructureBuildSizesInfoKHR tlas_size_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
		      ctx.vkGetAccelerationStructureBuildSizesKHR(
		          allocator.get_context().device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlas_build_info, &countInstance, &tlas_size_info);

		      VkAccelerationStructureCreateInfoKHR tlas_ci{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		      tlas_ci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		      tlas_ci.size = tlas_size_info.accelerationStructureSize;
		      tlas_buf = *vuk::allocate_buffer_gpu(allocator, { .mem_usage = vuk::MemoryUsage::eGPUonly, .size = tlas_size_info.accelerationStructureSize });
		      tlas_ci.buffer = tlas_buf->buffer;
		      tlas_ci.offset = tlas_buf->offset;

		      tlas = vuk::Unique<VkAccelerationStructureKHR>(allocator);
		      allocator.allocate_acceleration_structures({ &*tlas, 1 }, { &tlas_ci, 1 });

		      // Allocate the scratch memory
		      auto tlas_scratch_buffer =
		          *vuk::allocate_buffer_gpu(allocator, vuk::BufferCreateInfo{ .mem_usage = vuk::MemoryUsage::eGPUonly, .size = tlas_size_info.buildScratchSize });

		      // Update build information
		      tlas_build_info.srcAccelerationStructure = VK_NULL_HANDLE;
		      tlas_build_info.dstAccelerationStructure = *tlas;
		      tlas_build_info.scratchData.deviceAddress = tlas_scratch_buffer->device_address;

		      // Build Offsets info: n instances
		      VkAccelerationStructureBuildRangeInfoKHR tlas_offset{ countInstance, 0, 0, 0 };
		      const VkAccelerationStructureBuildRangeInfoKHR* ptlas_offset = &tlas_offset;

		      // Build the TLAS
		      vuk::RenderGraph blas_build("blas_build");
		      blas_build.add_pass({ .execute = [&ctx, tlas_build_info, ptlas_offset, blas_build_info, pblas_offset](vuk::CommandBuffer& command_buffer) {
			      ctx.vkCmdBuildAccelerationStructuresKHR(command_buffer.get_underlying(), 1, &blas_build_info, &pblas_offset);
		      } });
		      vuk::wait_for_futures(allocator, vuk::Future(std::make_shared<vuk::RenderGraph>(std::move(blas_build)), ""));
		      vuk::RenderGraph tlas_build("tlas_build");
		      tlas_build.add_pass({ .execute = [&ctx, tlas_build_info, ptlas_offset, blas_build_info, pblas_offset](vuk::CommandBuffer& command_buffer) {
			      ctx.vkCmdBuildAccelerationStructuresKHR(command_buffer.get_underlying(), 1, &tlas_build_info, &ptlas_offset);
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
		      vp.inv_proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 100.f);
		      vp.inv_proj[1][1] *= -1;
		      vp.inv_view = glm::inverse(vp.inv_view);
		      vp.inv_proj = glm::inverse(vp.inv_proj);

		      auto [buboVP, uboVP_fut] = create_buffer_cross_device(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, std::span(&vp, 1));
		      auto uboVP = *buboVP;

		      vuk::wait_for_futures(frame_allocator, uboVP_fut);

		      vuk::RenderGraph rg("12");
		      rg.attach_in("12_rt", std::move(target));
		      //  Set up the pass to draw the textured cube, with a color and a depth attachment
		      rg.add_pass({ .resources = { "12_rt"_image >> vuk::eRayTracingWrite >> "12_rt_final" }, .execute = [uboVP](vuk::CommandBuffer& command_buffer) {
			                   command_buffer.bind_acceleration_structure(0, 0, *tlas)
			                       .bind_image(0, 1, "12_rt")
			                       .bind_buffer(0, 2, uboVP)
			                       .bind_ray_tracing_pipeline("raytracing");
			                   /*glm::mat4* model = command_buffer.map_scratch_uniform_binding<glm::mat4>(0, 1);
			                    *model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));*/
			                   command_buffer.trace_rays(1024, 1024, 1);
		                   } });

		      angle += 180.f * ImGui::GetIO().DeltaTime;

		      return vuk::Future{ std::make_unique<vuk::RenderGraph>(std::move(rg)), "12_rt_final" };
		    },

		// Perform cleanup for the example
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      // We release the texture resources
		      texture_of_doge.reset();
		      tlas.reset();
		      tlas_buf.reset();
		      blas.reset();
		      blas_buf.reset();
		    }
	};

	REGISTER_EXAMPLE(x);
} // namespace