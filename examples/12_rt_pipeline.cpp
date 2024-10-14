#include "example_runner.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <stb_image.h>

/* 12_rt_pipeline
 * This example demonstrates how to build acceleration structures and trace against them using RT pipelines. This example requires that your driver supports
 * VK_KHR_ray_tracing. While there is no tight integration yet for building acceleration structures, you can already synchronize their building and raytracing
 * just as graphics and compute workloads.
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the example same code.
 * Check out the framework (example_runner_*) files if interested!
 */

namespace {
	float angle = 0.f;
	auto box = util::generate_cube();
	vuk::Unique<vuk::Buffer> verts, inds;
	vuk::Unique<VkAccelerationStructureKHR> tlas, blas;
	vuk::Unique<vuk::Buffer> tlas_buf_alloc, blas_buf_alloc, tlas_scratch_buffer_alloc;

	vuk::Example x{
		.name = "12_rt_pipeline",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      auto& ctx = allocator.get_context();

		      // If the runner has detected that there is no RT support, this example won't run
		      if (!runner.has_rt) {
			      return;
		      }

		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file((root / "examples/rt.rgen").generic_string()), (root / "examples/rt.rgen").generic_string());
			      pci.add_glsl(util::read_entire_file((root / "examples/rt.rmiss").generic_string()), (root / "examples/rt.rmiss").generic_string());
			      pci.add_glsl(util::read_entire_file((root / "examples/rt.rchit").generic_string()), (root / "examples/rt.rchit").generic_string());
			      // new for RT: a hit group is a collection of shaders identified by their index in the PipelineBaseCreateInfo
			      // 2 => rt.rchit
			      pci.add_hit_group(vuk::HitGroup{ .type = vuk::HitGroupType::eTriangles, .closest_hit = 2 });
			      runner.runtime->create_named_pipeline("raytracing", pci);
		      }

		      // We set up the cube data, same as in example 02_cube
		      auto [vert_buf, vert_fut] = create_buffer(allocator, vuk::MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.first));
		      verts = std::move(vert_buf);
		      auto [ind_buf, ind_fut] = create_buffer(allocator, vuk::MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.second));
		      inds = std::move(ind_buf);

		      // BLAS building
		      // We build a BLAS out of our cube.
		      uint32_t maxPrimitiveCount = (uint32_t)box.second.size() / 3;

		      // Describe the mesh
		      VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
		      triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; // vec3 vertex position data.
		      triangles.vertexData.deviceAddress = verts->device_address;
		      triangles.vertexStride = sizeof(util::Vertex);
		      // Describe index data (32-bit unsigned int)
		      triangles.indexType = VK_INDEX_TYPE_UINT32;
		      triangles.indexData.deviceAddress = inds->device_address;
		      // Indicate identity transform by setting transformData to null device pointer.
		      triangles.transformData = {};
		      triangles.maxVertex = (uint32_t)box.first.size();

		      // Identify the above data as containing opaque triangles.
		      VkAccelerationStructureGeometryKHR as_geom{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
		      as_geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		      as_geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		      as_geom.geometry.triangles = triangles;

		      // Find sizes
		      VkAccelerationStructureBuildGeometryInfoKHR blas_build_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
		      blas_build_info.dstAccelerationStructure = *blas;
		      blas_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		      blas_build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		      blas_build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		      blas_build_info.geometryCount = 1;
		      blas_build_info.pGeometries = &as_geom;

		      VkAccelerationStructureBuildSizesInfoKHR blas_size_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };

		      ctx.vkGetAccelerationStructureBuildSizesKHR(
		          ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blas_build_info, &maxPrimitiveCount, &blas_size_info);

		      // Allocate the BLAS object and a buffer that holds the data
		      VkAccelerationStructureCreateInfoKHR blas_ci{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		      blas_ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		      blas_ci.size = blas_size_info.accelerationStructureSize; // Will be used to allocate memory.
		      blas_buf_alloc =
		          *vuk::allocate_buffer(allocator, { .mem_usage = vuk::MemoryUsage::eGPUonly, .size = blas_size_info.accelerationStructureSize, .alignment = 256 });
		      blas_ci.buffer = blas_buf_alloc->buffer;
		      blas_ci.offset = blas_buf_alloc->offset;

		      blas = vuk::Unique<VkAccelerationStructureKHR>(allocator);
		      allocator.allocate_acceleration_structures({ &*blas, 1 }, { &blas_ci, 1 });

		      // Allocate the scratch memory for the BLAS build
		      auto blas_scratch_buffer =
		          *vuk::allocate_buffer(allocator,
		                                vuk::BufferCreateInfo{ .mem_usage = vuk::MemoryUsage::eGPUonly,
		                                                       .size = blas_size_info.buildScratchSize,
		                                                       .alignment = ctx.as_properties.minAccelerationStructureScratchOffsetAlignment });

		      // Update build information
		      blas_build_info.srcAccelerationStructure = VK_NULL_HANDLE;
		      blas_build_info.dstAccelerationStructure = *blas;
		      blas_build_info.scratchData.deviceAddress = blas_scratch_buffer->device_address;

		      // TLAS building
		      // We build a TLAS that refers to the BLAS we build before.
		      VkAccelerationStructureInstanceKHR rayInst{};
		      rayInst.transform = {};
		      rayInst.transform.matrix[0][0] = 1.f;
		      rayInst.transform.matrix[1][1] = 1.f;
		      rayInst.transform.matrix[2][2] = 1.f;
		      rayInst.instanceCustomIndex = 0; // gl_InstanceCustomIndexEXT
		      rayInst.accelerationStructureReference = blas_buf_alloc->device_address;
		      rayInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		      rayInst.mask = 0xFF;                                //  Only be hit if rayMask & instance.mask != 0
		      rayInst.instanceShaderBindingTableRecordOffset = 0; // We will use the same hit group for all objects

		      auto instances_buffer =
		          vuk::create_buffer(allocator, vuk::MemoryUsage::eCPUtoGPU, vuk::DomainFlagBits::eTransferOnGraphics, std::span{ &rayInst, 1 });

		      VkAccelerationStructureGeometryInstancesDataKHR instancesVk{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
		      instancesVk.data.deviceAddress = instances_buffer.first->device_address;

		      // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label it as instance data.
		      VkAccelerationStructureGeometryKHR topASGeometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
		      topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		      topASGeometry.geometry.instances = instancesVk;

		      // Find sizes
		      VkAccelerationStructureBuildGeometryInfoKHR tlas_build_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
		      tlas_build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
		      tlas_build_info.geometryCount = 1;
		      tlas_build_info.pGeometries = &topASGeometry;
		      tlas_build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		      tlas_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

		      uint32_t countInstance = 1;

		      VkAccelerationStructureBuildSizesInfoKHR tlas_size_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
		      ctx.vkGetAccelerationStructureBuildSizesKHR(
		          allocator.get_context().device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlas_build_info, &countInstance, &tlas_size_info);

		      // Allocate the TLAS object and a buffer that holds the data
		      VkAccelerationStructureCreateInfoKHR tlas_ci{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		      tlas_ci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		      tlas_ci.size = tlas_size_info.accelerationStructureSize;
		      tlas_buf_alloc =
		          *vuk::allocate_buffer(allocator, { .mem_usage = vuk::MemoryUsage::eGPUonly, .size = tlas_size_info.accelerationStructureSize, .alignment = 256 });
		      tlas_ci.buffer = tlas_buf_alloc->buffer;
		      tlas_ci.offset = tlas_buf_alloc->offset;

		      tlas = vuk::Unique<VkAccelerationStructureKHR>(allocator);
		      allocator.allocate_acceleration_structures({ &*tlas, 1 }, { &tlas_ci, 1 });

		      // Allocate the scratch memory
		      tlas_scratch_buffer_alloc =
		          *vuk::allocate_buffer(allocator,
		                                vuk::BufferCreateInfo{ .mem_usage = vuk::MemoryUsage::eGPUonly,
		                                                       .size = tlas_size_info.buildScratchSize,
		                                                       .alignment = ctx.as_properties.minAccelerationStructureScratchOffsetAlignment });

		      // Update build information
		      tlas_build_info.srcAccelerationStructure = VK_NULL_HANDLE;
		      tlas_build_info.dstAccelerationStructure = *tlas;
		      tlas_build_info.scratchData.deviceAddress = tlas_scratch_buffer_alloc->device_address;

		      // Build the BLAS & TLAS

		      // Synchronization happens against the AS buffers
		      auto blas_buf = vuk::discard_buf("blas_buf", *blas_buf_alloc);
		      auto tlas_buf = vuk::discard_buf("tlas_buf", *tlas_buf_alloc);
		      auto build_blas = vuk::make_pass("BLAS build",
		                                       [maxPrimitiveCount, as_geom, blas_build_info](vuk::CommandBuffer& command_buffer,
		                                                                                     VUK_BA(vuk::eAccelerationStructureBuildWrite) blas_buf,
		                                                                                     VUK_BA(vuk::eAccelerationStructureBuildRead) verts,
		                                                                                     VUK_BA(vuk::eAccelerationStructureBuildRead) inds) mutable {
			                                       // We make a copy of the AS geometry to not dangle when this runs.
			                                       blas_build_info.pGeometries = &as_geom;

			                                       // Describe what we are building.
			                                       VkAccelerationStructureBuildRangeInfoKHR blas_offset;
			                                       blas_offset.primitiveCount = maxPrimitiveCount;
			                                       blas_offset.firstVertex = 0;
			                                       blas_offset.primitiveOffset = 0;
			                                       blas_offset.transformOffset = 0;
			                                       const VkAccelerationStructureBuildRangeInfoKHR* pblas_offset = &blas_offset;
			                                       command_buffer.build_acceleration_structures(1, &blas_build_info, &pblas_offset);

			                                       return blas_buf;
		                                       });
		      auto build_tlas = vuk::make_pass("TLAS build",
		                                       [countInstance, topASGeometry, tlas_build_info](vuk::CommandBuffer& command_buffer,
		                                                                                       VUK_BA(vuk::eAccelerationStructureBuildRead) blas_buf,
		                                                                                       VUK_BA(vuk::eAccelerationStructureBuildWrite) tlas_buf) mutable {
			                                       // We make a copy of the AS geometry to not dangle when this runs.
			                                       tlas_build_info.pGeometries = &topASGeometry;

			                                       // Describe what we are building.
			                                       VkAccelerationStructureBuildRangeInfoKHR tlas_offset{ countInstance, 0, 0, 0 };
			                                       const VkAccelerationStructureBuildRangeInfoKHR* ptlas_offset = &tlas_offset;
			                                       command_buffer.build_acceleration_structures(1, &tlas_build_info, &ptlas_offset);

			                                       return tlas_buf;
		                                       });

		      // For the example, we just ask these that these uploads and AS building complete before moving on to rendering
		      // In an engine, you would integrate these uploads into some explicit system
		      auto blas = build_blas(blas_buf, vert_fut, ind_fut);
		      auto tlas = build_tlas(std::move(blas), tlas_buf);
		      runner.enqueue_setup(std::move(tlas));
		    },
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::Value<vuk::ImageAttachment> target) {
		      if (!runner.has_rt) {
			      return target;
		      }

		      struct VP {
			      glm::mat4 inv_view;
			      glm::mat4 inv_proj;
		      } vp;
		      vp.inv_view = glm::lookAt(glm::vec3(0, 1.5, 3.5), glm::vec3(0), glm::vec3(0, 1, 0));
		      vp.inv_proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 100.f);
		      vp.inv_proj[1][1] *= -1;
		      vp.inv_view = glm::inverse(vp.inv_view);
		      vp.inv_proj = glm::inverse(vp.inv_proj);

		      auto [buboVP, uboVP_fut] = create_buffer(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, vuk::DomainFlagBits::eTransferOnGraphics, std::span(&vp, 1));
		      auto uboVP = *buboVP;

		      // TLAS update - we make a new buffer of BLAS instances, which we use to update the TLAS later
		      VkAccelerationStructureInstanceKHR rayInst{};
		      rayInst.transform = {};
		      glm::mat4 model_transform = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
		      glm::mat3x4 reduced_model_transform = static_cast<glm::mat3x4>(model_transform);
		      memcpy(&rayInst.transform.matrix, &reduced_model_transform, sizeof(glm::mat3x4));
		      rayInst.instanceCustomIndex = 0; // gl_InstanceCustomIndexEXT
		      rayInst.accelerationStructureReference = blas_buf_alloc->device_address;
		      rayInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		      rayInst.mask = 0xFF;                                //  Only be hit if rayMask & instance.mask != 0
		      rayInst.instanceShaderBindingTableRecordOffset = 0; // We will use the same hit group for all objects

		      auto [instances_buffer, instances_fut] =
		          vuk::create_buffer(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, vuk::DomainFlagBits::eTransferOnGraphics, std::span{ &rayInst, 1 });

		      // TLAS update pass
		      auto update_tlas = vuk::make_pass(
		          "TLAS update", [inst_buf = *instances_buffer](vuk::CommandBuffer& command_buffer, VUK_BA(vuk::eAccelerationStructureBuildWrite) tlas_buf) {
			          // TLAS update
			          VkAccelerationStructureGeometryInstancesDataKHR instancesVk{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
			          instancesVk.data.deviceAddress = inst_buf.device_address;

			          VkAccelerationStructureGeometryKHR topASGeometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
			          topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
			          topASGeometry.geometry.instances = instancesVk;

			          VkAccelerationStructureBuildGeometryInfoKHR tlas_build_info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
			          tlas_build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
			          tlas_build_info.geometryCount = 1;
			          tlas_build_info.pGeometries = &topASGeometry;
			          tlas_build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
			          tlas_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

			          tlas_build_info.srcAccelerationStructure = *tlas;
			          tlas_build_info.dstAccelerationStructure = *tlas;
			          tlas_build_info.scratchData.deviceAddress = tlas_scratch_buffer_alloc->device_address;

			          VkAccelerationStructureBuildRangeInfoKHR tlas_offset{ 1, 0, 0, 0 };
			          const VkAccelerationStructureBuildRangeInfoKHR* ptlas_offset = &tlas_offset;
			          command_buffer.build_acceleration_structures(1, &tlas_build_info, &ptlas_offset);

			          return tlas_buf;
		          });

		      auto tlas_buf = vuk::declare_buf("tlas_buf", *tlas_buf_alloc);
		      auto updated_tlas = update_tlas(std::move(tlas_buf));
		      // We use a eR8G8B8A8Unorm, as the swapchain is in sRGB which does not support storage use
		      auto rt_target = vuk::declare_ia(
		          "12_rt_target", vuk::ImageAttachment{ .format = vuk::Format::eR8G8B8A8Unorm, .sample_count = vuk::SampleCountFlagBits::e1, .layer_count = 1 });
		      // This intermediate image is the same shape as the swapchain image
		      rt_target.same_shape_as(target);
		      // Synchronize against the TLAS buffer to run this pass after the TLAS update has completed
		      auto trace_rays = vuk::make_pass(
		          "trace rays", [uboVP](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eRayTracingWrite) rt_target, VUK_BA(vuk::eRayTracingRead) tlas_buf) {
			          command_buffer.bind_acceleration_structure(0, 0, *tlas)
			              .bind_image(0, 1, rt_target)
			              .bind_buffer(0, 2, uboVP)
			              .bind_ray_tracing_pipeline("raytracing");
			          // Launch one ray per pixel in the intermediate image
			          auto extent = rt_target->extent;
			          command_buffer.trace_rays(extent.width, extent.height, 1);

			          return rt_target;
		          });
		      auto intermediate_image = trace_rays(std::move(rt_target), std::move(updated_tlas));
		      // Perform a blit of the intermediate image onto the swapchain (this will also do the non-linear encoding for us, although we lost some precision
		      // when we rendered into Unorm)
		      auto blit_unorm =
		          vuk::make_pass("blit unorm", [](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eTransferRead) rt_target, VUK_IA(vuk::eTransferWrite) rt_final) {
			          vuk::ImageBlit blit;
			          blit.srcSubresource.aspectMask = vuk::ImageAspectFlagBits::eColor;
			          blit.srcSubresource.baseArrayLayer = 0;
			          blit.srcSubresource.layerCount = 1;
			          blit.srcSubresource.mipLevel = 0;
			          blit.dstSubresource = blit.srcSubresource;
			          blit.srcOffsets[1] = vuk::Offset3D{ static_cast<int>(rt_target->extent.width), static_cast<int>(rt_target->extent.height), 1 };
			          blit.dstOffsets[1] = blit.srcOffsets[1];
			          command_buffer.blit_image(rt_target, rt_final, blit, vuk::Filter::eNearest);
			          return rt_final;
		          });
		      angle += 20.f * ImGui::GetIO().DeltaTime;

		      return blit_unorm(std::move(intermediate_image), std::move(target));
		    },

		// Perform cleanup for the example
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      verts.reset();
		      inds.reset();
		      tlas.reset();
		      tlas_buf_alloc.reset();
		      blas.reset();
		      blas_buf_alloc.reset();
		      tlas_scratch_buffer_alloc.reset();
		    }
	};

	REGISTER_EXAMPLE(x);
} // namespace