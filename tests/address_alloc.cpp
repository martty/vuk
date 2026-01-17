#include "TestContext.hpp"
#include "vuk/runtime/vk/Address.hpp"
#include "vuk/runtime/vk/DeviceVkResource.hpp"
#include <doctest/doctest.h>

using namespace vuk;

// ============================================================================
// VirtualAllocation and VirtualAddressSpace Allocation Tests
// ============================================================================

TEST_CASE("virtual_allocation_from_address_space") {
	// Create virtual address space using device resource directly
	auto& runtime = test_context.allocator->get_context();
	auto& device_resource = runtime.get_vk_resource();
	Allocator direct_alloc(device_resource);

	VirtualAddressSpaceCreateInfo space_ci{ .size = 1024 * 1024 };
	VirtualAddressSpace space;

	auto space_result = direct_alloc.allocate(std::span{ &space, 1 }, std::span{ &space_ci, 1 });
	REQUIRE(space_result);

	// Allocate a single virtual allocation from the address space
	VirtualAllocationCreateInfo alloc_ci{ .size = 256, .alignment = 64, .address_space = &space };

	VirtualAllocation alloc;
	auto alloc_result = direct_alloc.allocate(std::span{ &alloc, 1 }, std::span{ &alloc_ci, 1 });

	REQUIRE(alloc_result);
	REQUIRE(alloc);
	REQUIRE(alloc.address_space == &space);
	REQUIRE(alloc.offset % 64 == 0); // Check alignment

	// Test the operator[] indexing
	REQUIRE(alloc[0] == alloc.offset);
	REQUIRE(alloc[64] == alloc.offset + 64);

	// Cleanup
	direct_alloc.deallocate(std::span{ &alloc, 1 });
	direct_alloc.deallocate(std::span{ &space, 1 });
}

TEST_CASE("multiple_virtual_allocations_from_address_space") {
	// Create virtual address space using device resource directly
	auto& runtime = test_context.allocator->get_context();
	auto& device_resource = runtime.get_vk_resource();
	Allocator direct_alloc(device_resource);

	VirtualAddressSpaceCreateInfo space_ci{ .size = 4096 };
	VirtualAddressSpace space;

	auto space_result = direct_alloc.allocate(std::span{ &space, 1 }, std::span{ &space_ci, 1 });
	REQUIRE(space_result);

	// Allocate multiple virtual allocations with different sizes and alignments
	VirtualAllocationCreateInfo alloc_cis[] = { { .size = 256, .alignment = 64, .address_space = &space },
		                                          { .size = 512, .alignment = 128, .address_space = &space },
		                                          { .size = 128, .alignment = 32, .address_space = &space } };

	VirtualAllocation allocations[3];
	auto alloc_result = direct_alloc.allocate(std::span{ allocations, 3 }, std::span{ alloc_cis, 3 });

	REQUIRE(alloc_result);

	// Verify all allocations
	for (size_t i = 0; i < 3; i++) {
		REQUIRE(allocations[i]);
		REQUIRE(allocations[i].address_space == &space);
		REQUIRE(allocations[i].offset % alloc_cis[i].alignment == 0);
	}

	// Verify that offsets are unique (no overlap)
	REQUIRE(allocations[0].offset != allocations[1].offset);
	REQUIRE(allocations[0].offset != allocations[2].offset);
	REQUIRE(allocations[1].offset != allocations[2].offset);

	// Cleanup
	direct_alloc.deallocate(std::span{ allocations, 3 });
	direct_alloc.deallocate(std::span{ &space, 1 });
}

TEST_CASE("virtual_allocation_failure_null_address_space") {
	// Try to allocate a virtual allocation without a valid address space using device resource directly
	auto& runtime = test_context.allocator->get_context();
	auto& device_resource = runtime.get_vk_resource();
	Allocator direct_alloc(device_resource);

	VirtualAllocationCreateInfo alloc_ci{ .size = 256, .alignment = 64, .address_space = nullptr };

	VirtualAllocation alloc;
	auto result = direct_alloc.allocate(std::span{ &alloc, 1 }, std::span{ &alloc_ci, 1 });

	REQUIRE(!result.holds_value()); // Should fail
}

TEST_CASE("virtual_allocation_named_methods") {
	// Test the named allocation methods using device resource directly
	auto& runtime = test_context.allocator->get_context();
	auto& device_resource = runtime.get_vk_resource();
	Allocator direct_alloc(device_resource);

	VirtualAddressSpaceCreateInfo space_ci{ .size = 2048 };
	VirtualAddressSpace space;

	auto space_result = direct_alloc.allocate_virtual_address_spaces(std::span{ &space, 1 }, std::span{ &space_ci, 1 });
	REQUIRE(space_result);

	VirtualAllocationCreateInfo alloc_ci{ .size = 512, .alignment = 256, .address_space = &space };

	VirtualAllocation alloc;
	auto alloc_result = direct_alloc.allocate_virtual_allocations(std::span{ &alloc, 1 }, std::span{ &alloc_ci, 1 });

	REQUIRE(alloc_result);
	REQUIRE(alloc);
	REQUIRE(alloc.offset % 256 == 0);

	// Cleanup using deallocate overloads
	direct_alloc.deallocate(std::span{ &alloc, 1 });
	direct_alloc.deallocate(std::span{ &space, 1 });
}

TEST_CASE("virtual_address_space_exhaustion") {
	// Create a small address space and try to allocate more than it can hold using device resource directly
	auto& runtime = test_context.allocator->get_context();
	auto& device_resource = runtime.get_vk_resource();
	Allocator direct_alloc(device_resource);

	VirtualAddressSpaceCreateInfo space_ci{ .size = 512 };
	VirtualAddressSpace space;

	auto space_result = direct_alloc.allocate(std::span{ &space, 1 }, std::span{ &space_ci, 1 });
	REQUIRE(space_result);

	// First allocation should succeed
	VirtualAllocationCreateInfo alloc_ci1{ .size = 256, .alignment = 1, .address_space = &space };

	VirtualAllocation alloc1;
	auto alloc1_result = direct_alloc.allocate(std::span{ &alloc1, 1 }, std::span{ &alloc_ci1, 1 });
	REQUIRE(alloc1_result);

	// Second allocation should succeed
	VirtualAllocationCreateInfo alloc_ci2{ .size = 256, .alignment = 1, .address_space = &space };

	VirtualAllocation alloc2;
	auto alloc2_result = direct_alloc.allocate(std::span{ &alloc2, 1 }, std::span{ &alloc_ci2, 1 });
	REQUIRE(alloc2_result);

	// Third allocation should fail (address space exhausted)
	VirtualAllocationCreateInfo alloc_ci3{ .size = 256, .alignment = 1, .address_space = &space };

	VirtualAllocation alloc3;
	auto alloc3_result = direct_alloc.allocate(std::span{ &alloc3, 1 }, std::span{ &alloc_ci3, 1 });
	REQUIRE(!alloc3_result.holds_value()); // Should fail

	// Cleanup
	direct_alloc.deallocate(std::span{ &alloc1, 1 });
	direct_alloc.deallocate(std::span{ &alloc2, 1 });
	direct_alloc.deallocate(std::span{ &space, 1 });
}

TEST_CASE("virtual_allocation_reallocation_after_free") {
	// Test that virtual allocations can be reallocated after being freed using device resource directly
	auto& runtime = test_context.allocator->get_context();
	auto& device_resource = runtime.get_vk_resource();
	Allocator direct_alloc(device_resource);

	VirtualAddressSpaceCreateInfo space_ci{ .size = 1024 };
	VirtualAddressSpace space;

	auto space_result = direct_alloc.allocate(std::span{ &space, 1 }, std::span{ &space_ci, 1 });
	REQUIRE(space_result);

	VirtualAllocationCreateInfo alloc_ci{ .size = 512, .alignment = 64, .address_space = &space };

	// Allocate, free, and reallocate
	VirtualAllocation alloc1;
	auto result1 = direct_alloc.allocate(std::span{ &alloc1, 1 }, std::span{ &alloc_ci, 1 });
	REQUIRE(result1);

	uint64_t first_offset = alloc1.offset;
	direct_alloc.deallocate(std::span{ &alloc1, 1 });

	VirtualAllocation alloc2;
	auto result2 = direct_alloc.allocate(std::span{ &alloc2, 1 }, std::span{ &alloc_ci, 1 });
	REQUIRE(result2);

	// Should get the same or a valid offset
	REQUIRE(alloc2.offset % 64 == 0);

	// Cleanup
	direct_alloc.deallocate(std::span{ &alloc2, 1 });
	direct_alloc.deallocate(std::span{ &space, 1 });
}

TEST_CASE("virtual_allocation_large_alignment") {
	// Test allocation with large alignment requirements using device resource directly
	auto& runtime = test_context.allocator->get_context();
	auto& device_resource = runtime.get_vk_resource();
	Allocator direct_alloc(device_resource);

	VirtualAddressSpaceCreateInfo space_ci{ .size = 1024 * 1024 };
	VirtualAddressSpace space;

	auto space_result = direct_alloc.allocate(std::span{ &space, 1 }, std::span{ &space_ci, 1 });
	REQUIRE(space_result);

	VirtualAllocationCreateInfo alloc_ci{ .size = 256,
		                                    .alignment = 4096, // 4KB alignment
		                                    .address_space = &space };

	VirtualAllocation alloc;
	auto alloc_result = direct_alloc.allocate(std::span{ &alloc, 1 }, std::span{ &alloc_ci, 1 });

	REQUIRE(alloc_result);
	REQUIRE(alloc.offset % 4096 == 0);

	// Cleanup
	direct_alloc.deallocate(std::span{ &alloc, 1 });
	direct_alloc.deallocate(std::span{ &space, 1 });
}

TEST_CASE("virtual_allocation_frame_allocator_automatic_cleanup") {
	// Test that DeviceFrameResource automatically cleans up virtual allocations when the frame is recycled
	auto& runtime = test_context.allocator->get_context();
	DeviceSuperFrameResource super_frame_allocator(runtime, 3); // 3 frames in flight

	// Get a frame allocator
	auto& frame_allocator = super_frame_allocator.get_next_frame();
	Allocator frame_alloc(frame_allocator);

	// Allocate a virtual address space from the frame allocator
	VirtualAddressSpaceCreateInfo space_ci{ .size = 4096 };
	VirtualAddressSpace space;
	auto space_result = frame_alloc.allocate(std::span{ &space, 1 }, std::span{ &space_ci, 1 });
	REQUIRE(space_result);

	// Allocate virtual allocations from the address space
	VirtualAllocationCreateInfo alloc_cis[] = { { .size = 512, .alignment = 64, .address_space = &space },
		                                          { .size = 256, .alignment = 32, .address_space = &space } };

	VirtualAllocation allocations[2];
	auto alloc_result = frame_alloc.allocate(std::span{ allocations, 2 }, std::span{ alloc_cis, 2 });
	REQUIRE(alloc_result);

	// Verify allocations
	REQUIRE(allocations[0]);
	REQUIRE(allocations[1]);
	REQUIRE(allocations[0].address_space == &space);
	REQUIRE(allocations[1].address_space == &space);

	// Advance frames to trigger cleanup
	// With 3 frames in flight, we need to advance 3 times to recycle the first frame
	for (int i = 0; i < 3; i++) {
		super_frame_allocator.get_next_frame();
	}

	// At this point, the original frame should have been recycled and resources cleaned up
	// No explicit deallocation needed - frame allocator handles it automatically
}
