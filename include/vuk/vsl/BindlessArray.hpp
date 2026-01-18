#pragma once

#include "vuk/runtime/vk/Address.hpp"
#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/runtime/vk/Image.hpp"
#include "vuk/Types.hpp"
#include <vector>

namespace vuk {
	/// @brief A vector-like container for bindless descriptor arrays.
	///
	/// @note This class is NOT thread-safe. All calls should be protected.
	///
	/// Requirements:
	/// - Allocator must wrap a DeviceSuperFrameResource for persistent resource lifetime
	/// - Vulkan device must support VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
	/// - Vulkan device must support VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
	///
	/// Example usage:
	/// @code
	/// // Create array with combined image samplers
	/// BindlessArray textures(allocator, 1, {.combined_image_sampler = 0}, 1024);
	///
	/// // Add resources
	/// uint32_t idx = textures.push_back(image_view, sampler, ImageLayout::eReadOnlyOptimalKHR);
	///
	/// // Update resources
	/// textures.set(idx, new_view, new_sampler, ImageLayout::eReadOnlyOptimalKHR);
	///
	/// // Commit changes and use in pipeline
	/// textures.commit();
	/// command_buffer.bind_persistent(1, textures.get_persistent_set());
	/// @endcode
	class BindlessArray {
	public:
		BindlessArray() = default;

		/// @brief Configuration struct for BindlessArray binding indices
		struct Bindings {
			uint32_t sampler = ~0U;
			uint32_t sampled_image = ~0U;
			uint32_t combined_image_sampler = ~0U;
			uint32_t storage_image = ~0U;
			uint32_t uniform_buffer = ~0U;
			uint32_t storage_buffer = ~0U;
			uint32_t acceleration_structure = ~0U;
		};

		/// @brief Constructs a BindlessArray for bindless descriptor arrays.
		/// @param allocator The allocator used for managing Vulkan resources and virtual address spaces. Must be wrapping a DeviceSuperFrameResource.
		/// @param set_index The descriptor set index where this bindless array will be bound.
		/// @param indices Binding indices for each descriptor type (~0U to omit).
		/// @param max_descriptors The maximum number of descriptors allowed in each binding array.
		BindlessArray(Allocator& allocator, size_t set_index, const Bindings& indices, uint32_t max_descriptors) : allocator(allocator) {
			assert(dynamic_cast<DeviceSuperFrameResource*>(&allocator.get_device_resource()) != nullptr);

			auto& runtime = allocator.get_context();

			// Build bindings and flags arrays, skipping ~0U indices
			std::vector<VkDescriptorSetLayoutBinding> bindings;
			std::vector<VkDescriptorBindingFlags> binding_flags;
			std::vector<VkDescriptorPoolSize> descriptor_sizes;

			constexpr static auto bindless_flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

			// Initialize binding indices to ~0U
			binding_indices.fill(~0U);

			// Add sampler binding if requested
			if (indices.sampler != ~0U) {
				binding_indices[static_cast<size_t>(DescriptorType::eSampler)] = indices.sampler;
				bindings.push_back({ .binding = indices.sampler,
				                     .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
				                     .descriptorCount = max_descriptors,
				                     .stageFlags = VK_SHADER_STAGE_ALL,
				                     .pImmutableSamplers = nullptr });
				binding_flags.push_back(bindless_flags);
				descriptor_sizes.emplace_back(VK_DESCRIPTOR_TYPE_SAMPLER, max_descriptors);
			}

			// Add sampled image binding if requested
			if (indices.sampled_image != ~0U) {
				binding_indices[static_cast<size_t>(DescriptorType::eSampledImage)] = indices.sampled_image;
				bindings.push_back({ .binding = indices.sampled_image,
				                     .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
				                     .descriptorCount = max_descriptors,
				                     .stageFlags = VK_SHADER_STAGE_ALL,
				                     .pImmutableSamplers = nullptr });
				binding_flags.push_back(bindless_flags);
				descriptor_sizes.emplace_back(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, max_descriptors);
			}

			// Add combined image sampler binding if requested
			if (indices.combined_image_sampler != ~0U) {
				binding_indices[static_cast<size_t>(DescriptorType::eCombinedImageSampler)] = indices.combined_image_sampler;
				bindings.push_back({ .binding = indices.combined_image_sampler,
				                     .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				                     .descriptorCount = max_descriptors,
				                     .stageFlags = VK_SHADER_STAGE_ALL,
				                     .pImmutableSamplers = nullptr });
				binding_flags.push_back(bindless_flags);
				descriptor_sizes.emplace_back(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_descriptors);
			}

			// Add storage image binding if requested
			if (indices.storage_image != ~0U) {
				binding_indices[static_cast<size_t>(DescriptorType::eStorageImage)] = indices.storage_image;
				bindings.push_back({ .binding = indices.storage_image,
				                     .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				                     .descriptorCount = max_descriptors,
				                     .stageFlags = VK_SHADER_STAGE_ALL,
				                     .pImmutableSamplers = nullptr });
				binding_flags.push_back(bindless_flags);
				descriptor_sizes.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, max_descriptors);
			}

			// Add uniform buffer binding if requested
			if (indices.uniform_buffer != ~0U) {
				binding_indices[static_cast<size_t>(DescriptorType::eUniformBuffer)] = indices.uniform_buffer;
				bindings.push_back({ .binding = indices.uniform_buffer,
				                     .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				                     .descriptorCount = max_descriptors,
				                     .stageFlags = VK_SHADER_STAGE_ALL,
				                     .pImmutableSamplers = nullptr });
				binding_flags.push_back(bindless_flags);
				descriptor_sizes.emplace_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, max_descriptors);
			}

			// Add storage buffer binding if requested
			if (indices.storage_buffer != ~0U) {
				binding_indices[static_cast<size_t>(DescriptorType::eStorageBuffer)] = indices.storage_buffer;
				bindings.push_back({ .binding = indices.storage_buffer,
				                     .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				                     .descriptorCount = max_descriptors,
				                     .stageFlags = VK_SHADER_STAGE_ALL,
				                     .pImmutableSamplers = nullptr });
				binding_flags.push_back(bindless_flags);
				descriptor_sizes.emplace_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_descriptors);
			}

			// Add acceleration structure binding if requested
			if (indices.acceleration_structure != ~0U) {
				binding_indices[static_cast<size_t>(DescriptorType::eAccelerationStructureKHR)] = indices.acceleration_structure;
				bindings.push_back({ .binding = indices.acceleration_structure,
				                     .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
				                     .descriptorCount = max_descriptors,
				                     .stageFlags = VK_SHADER_STAGE_ALL,
				                     .pImmutableSamplers = nullptr });
				binding_flags.push_back(bindless_flags);
				descriptor_sizes.emplace_back(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, max_descriptors);
			}

			assert(!bindings.empty());

			auto pool_flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
			auto pool_info = VkDescriptorPoolCreateInfo{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.pNext = nullptr,
				.flags = static_cast<VkDescriptorPoolCreateFlags>(pool_flags),
				.maxSets = 1,
				.poolSizeCount = static_cast<uint32_t>(descriptor_sizes.size()),
				.pPoolSizes = descriptor_sizes.data(),
			};
			VkDescriptorPool pool;
			runtime.vkCreateDescriptorPool(runtime.device, &pool_info, nullptr, &pool);

			auto set_layout_binding_flags_info = VkDescriptorSetLayoutBindingFlagsCreateInfo{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
				.pNext = nullptr,
				.bindingCount = static_cast<uint32_t>(binding_flags.size()),
				.pBindingFlags = binding_flags.data(),
			};

			auto set_layout_info = VkDescriptorSetLayoutCreateInfo{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.pNext = &set_layout_binding_flags_info,
				.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
				.bindingCount = static_cast<uint32_t>(bindings.size()),
				.pBindings = bindings.data(),
			};

			auto dslci = vuk::DescriptorSetLayoutCreateInfo{
				.dslci = set_layout_info,
				.index = set_index,
				.bindings = bindings,
				.flags = binding_flags,
			};

			auto& dslai = runtime.acquire_descriptor_set_layout(dslci);

			auto set_alloc_info = VkDescriptorSetAllocateInfo{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.pNext = nullptr,
				.descriptorPool = pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &dslai.layout,
			};

			VkDescriptorSet descriptor_set;
			runtime.vkAllocateDescriptorSets(runtime.device, &set_alloc_info, &descriptor_set);

			persistent_set = Unique<PersistentDescriptorSet>(Allocator(runtime.get_vk_resource()),
			                                                 PersistentDescriptorSet{
			                                                     .backing_pool = pool,
			                                                     .set_layout_create_info = dslci,
			                                                     .set_layout = dslai.layout,
			                                                     .backing_set = descriptor_set,
			                                                     .wdss = {},
			                                                     .descriptor_bindings = {},
			                                                 });
			for (unsigned i = 0; i < bindings.size(); i++) {
				persistent_set->descriptor_bindings[i].resize(max_descriptors);
			}

			// Create virtual address space
			VirtualAddressSpaceCreateInfo space_ci{ .size = max_descriptors };
			VirtualAddressSpace space;
			auto space_result = allocator.allocate(std::span{ &space, 1 }, std::span{ &space_ci, 1 });
			if (!space_result) {
				return;
			}
			address_space = Unique<VirtualAddressSpace>(allocator, space);
		}

		/// @brief Add a sampler to the array
		/// @param sampler The sampler to add
		/// @return The index where the resource was added
		uint32_t push_back(Sampler sampler) {
			assert(binding_indices[static_cast<size_t>(DescriptorType::eSampler)] != ~0U && "Sampler binding not configured");
			uint32_t index = push();

			DescriptorBinding db;
			db.type = DescriptorType::eSampler;
			db.image.set_sampler(sampler);
			add_write(binding_indices[static_cast<size_t>(DescriptorType::eSampler)], index, db);

			return index;
		}

		/// @brief Add a sampled image to the array
		/// @param image_view The image view to add
		/// @param layout The image layout
		/// @return The index where the resource was added
		uint32_t push_back(ImageView image_view, ImageLayout layout) {
			assert(binding_indices[static_cast<size_t>(DescriptorType::eSampledImage)] != ~0U && "Sampled image binding not configured");
			uint32_t index = push();

			DescriptorBinding db;
			db.type = DescriptorType::eSampledImage;
			db.image = DescriptorImageInfo({}, image_view, layout);
			add_write(binding_indices[static_cast<size_t>(DescriptorType::eSampledImage)], index, db);

			return index;
		}

		/// @brief Add a combined image sampler to the array
		/// @param image_view The image view to add
		/// @param sampler The sampler to use
		/// @param layout The image layout
		/// @return The index where the resource was added
		uint32_t push_back(ImageView image_view, Sampler sampler, ImageLayout layout) {
			assert(binding_indices[static_cast<size_t>(DescriptorType::eCombinedImageSampler)] != ~0U && "Combined image sampler binding not configured");
			uint32_t index = push();

			DescriptorBinding db;
			db.type = DescriptorType::eCombinedImageSampler;
			db.image = DescriptorImageInfo(sampler, image_view, layout);
			add_write(binding_indices[static_cast<size_t>(DescriptorType::eCombinedImageSampler)], index, db);

			return index;
		}

		/// @brief Add a storage image to the array
		/// @param image_view The image view to add
		/// @return The index where the resource was added
		uint32_t push_back(ImageView image_view) {
			assert(binding_indices[static_cast<size_t>(DescriptorType::eStorageImage)] != ~0U && "Storage image binding not configured");
			uint32_t index = push();

			DescriptorBinding db;
			db.type = DescriptorType::eStorageImage;
			db.image = DescriptorImageInfo({}, image_view, ImageLayout::eGeneral);
			add_write(binding_indices[static_cast<size_t>(DescriptorType::eStorageImage)], index, db);

			return index;
		}

		/// @brief Add a uniform buffer to the array
		/// @param buffer The buffer to add
		/// @return The index where the resource was added
		uint32_t push_back_uniform_buffer(Buffer buffer) {
			assert(binding_indices[static_cast<size_t>(DescriptorType::eUniformBuffer)] != ~0U && "Uniform buffer binding not configured");
			uint32_t index = push();

			DescriptorBinding db;
			db.type = DescriptorType::eUniformBuffer;
			db.buffer.buffer = buffer.buffer;
			db.buffer.offset = buffer.offset;
			db.buffer.range = buffer.size;
			add_write(binding_indices[static_cast<size_t>(DescriptorType::eUniformBuffer)], index, db);

			return index;
		}

		/// @brief Add a storage buffer to the array
		/// @param buffer The buffer to add
		/// @return The index where the resource was added
		uint32_t push_back_storage_buffer(Buffer buffer) {
			assert(binding_indices[static_cast<size_t>(DescriptorType::eStorageBuffer)] != ~0U && "Storage buffer binding not configured");
			uint32_t index = push();

			DescriptorBinding db;
			db.type = DescriptorType::eStorageBuffer;
			db.buffer.buffer = buffer.buffer;
			db.buffer.offset = buffer.offset;
			db.buffer.range = buffer.size;
			add_write(binding_indices[static_cast<size_t>(DescriptorType::eStorageBuffer)], index, db);

			return index;
		}

		/// @brief Add an acceleration structure to the array
		/// @param accel_struct The acceleration structure to add
		/// @return The index where the resource was added
		uint32_t push_back(VkAccelerationStructureKHR accel_struct) {
			assert(binding_indices[static_cast<size_t>(DescriptorType::eAccelerationStructureKHR)] != ~0U && "Acceleration structure binding not configured");
			uint32_t index = push();

			DescriptorBinding db;
			db.type = DescriptorType::eAccelerationStructureKHR;
			db.as.as = accel_struct;
			add_write(binding_indices[static_cast<size_t>(DescriptorType::eAccelerationStructureKHR)], index, db);

			return index;
		}

		/// @brief Remove a resource at the given index
		/// @param index The index to remove
		void erase(uint32_t index) {
			// Find the allocation with this offset
			auto it = std::find_if(allocations.begin(), allocations.end(), [index](const VirtualAllocation& a) { return a.offset == static_cast<uint64_t>(index); });

			assert(it != allocations.end());

			// Deallocate the virtual address
			allocator.deallocate(std::span{ &*it, 1 });

			// Remove from our tracking
			allocations.erase(it);
		}

		/// @brief Clear all resources from the array
		void clear() {
			// Deallocate all virtual addresses
			if (!allocations.empty()) {
				allocator.deallocate(std::span{ allocations.data(), allocations.size() });
				allocations.clear();
			}

			// Clear any pending descriptor updates
			persistent_set->wdss.clear();
			descriptors.clear();
		}

		/// @brief Commit all pending descriptor updates to the GPU.
		///
		/// Must be called after any push_back() or set() operations before the descriptor set is used in rendering.
		/// Internally calls vkUpdateDescriptorSets to apply all queued descriptor writes.
		/// After commit(), all pending updates are cleared and the descriptor set is ready to use.
		///
		/// @note Can be called even when there are no pending updates (safe to call every frame).
		void commit() {
			for (size_t i = 0; i < persistent_set->wdss.size(); i++) {
				auto& wds = persistent_set->wdss[i];
				auto& db = descriptors[i];
				if (db.type == DescriptorType::eAccelerationStructureKHR) {
					db.as.wds = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
					db.as.wds.accelerationStructureCount = 1;
					db.as.wds.pAccelerationStructures = &db.as.as;
				}
				wds.pNext = db.type == DescriptorType::eAccelerationStructureKHR ? &db.as.wds : nullptr;
				wds.pImageInfo = &db.image.dii;
				wds.pBufferInfo = &db.buffer;
			}
			auto& runtime = allocator.get_context();
			runtime.vkUpdateDescriptorSets(runtime.device, (uint32_t)persistent_set->wdss.size(), persistent_set->wdss.data(), 0, nullptr);
			persistent_set->wdss.clear();
			descriptors.clear();
		}

		/// @brief Get the number of active resources
		size_t size() const {
			return allocations.size();
		}

		/// @brief Check if the array is empty
		bool empty() const {
			return allocations.empty();
		}

		/// @brief Check if a specific index is currently allocated
		/// @param index The index to check
		/// @return true if the index is allocated, false otherwise
		bool is_allocated(uint32_t index) const {
			return std::any_of(allocations.begin(), allocations.end(), [index](const VirtualAllocation& a) { return a.offset == static_cast<uint64_t>(index); });
		}

		/// @brief Get the persistent descriptor set (non-const)
		PersistentDescriptorSet& get_persistent_set() {
			return *persistent_set;
		}

		/// @brief Get the persistent descriptor set (const)
		const PersistentDescriptorSet& get_persistent_set() const {
			return *persistent_set;
		}

		/// @brief Gets the descriptor set layout creation information. To be passed as explicit set layout when creating pipelines.
		DescriptorSetLayoutCreateInfo get_descriptor_set_layout() const {
			return persistent_set->set_layout_create_info;
		}

		/// @brief Get all active indices
		std::vector<uint32_t> get_active_indices() const {
			std::vector<uint32_t> indices;
			indices.reserve(allocations.size());
			for (const auto& alloc : allocations) {
				indices.push_back(static_cast<uint32_t>(alloc.offset));
			}
			return indices;
		}

	private:
		uint32_t push() {
			// Allocate a virtual address (index) from the address space
			VirtualAllocationCreateInfo alloc_ci{ .size = 1, .alignment = 1, .address_space = &address_space.get() };
			VirtualAllocation alloc;

			auto result = allocator.allocate(std::span{ &alloc, 1 }, std::span{ &alloc_ci, 1 });
			assert(result);
			if (!result) {
				return static_cast<uint32_t>(-1);
			}

			// The offset is our index
			uint32_t index = static_cast<uint32_t>(alloc.offset);

			// Store the allocation
			allocations.push_back(alloc);

			return index;
		}

		void add_write(uint32_t binding, uint32_t index, DescriptorBinding db) {
			persistent_set->wdss.push_back(VkWriteDescriptorSet{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			                                                     .pNext = nullptr,
			                                                     .dstSet = persistent_set->backing_set,
			                                                     .dstBinding = binding,
			                                                     .dstArrayElement = index,
			                                                     .descriptorCount = 1,
			                                                     .descriptorType = DescriptorBinding::vk_descriptor_type(db.type) });
			descriptors.push_back(db);
		}

		Allocator allocator;
		Unique<PersistentDescriptorSet> persistent_set;

		Unique<VirtualAddressSpace> address_space;
		std::vector<VirtualAllocation> allocations;

		// Store binding indices indexed by DescriptorType enum value
		std::array<uint32_t, 16> binding_indices;
		std::vector<DescriptorBinding> descriptors;
	};
} // namespace vuk
