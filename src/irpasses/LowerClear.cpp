#include "vuk/ir/IRPasses.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/SyncLowering.hpp"

#include "compute_clear_comp_spv_shader.h"

namespace vuk {
	std::pair<ImageViewEntry, ImageEntry> lower_clear::evaluate_imageview_from_ir(Ref imageview_ref) {
		ImageViewEntry ve{};
		ImageEntry ie{};

		// First, try to evaluate directly if the ImageView is already allocated
		auto iv_eval = eval<ImageView<>>(imageview_ref);
		if (iv_eval) {
			ve = iv_eval->get_meta();
			ie = ve.image.get_meta();
			return { ve, ie }; // ImageEntry not needed for allocated views
		}
		(void)iv_eval.error();

		Ref current = to_def(imageview_ref);

		// Handle SLICE nodes - they modify view parameters
		std::pmr::vector<Node*> slice_stack(allocator);
		while (current.node->kind == Node::SLICE) {
			slice_stack.push_back(current.node);
			current = to_def(current.node->slice.src);
		}

		// Try to get from ALLOCATE node
		if (current.node->kind == Node::ALLOCATE && Type::stripped(current.node->type[0])->is_imageview()) {
			auto ivci_ref = to_def(current.node->allocate.src);

			// Should be a CONSTRUCT node (IVCI)
			if (ivci_ref.node->kind == Node::CONSTRUCT) {
				// IVCI fields: base_level, level_count, base_layer, layer_count, image, format, offset, extent
				auto& args = ivci_ref.node->construct.args;
				assert(args.size() == 9);

				// Extract base_level (index 1, after memory at 0)
				if (auto bl_eval = eval<uint16_t>(args[1])) {
					ve.base_level = *bl_eval;
				}

				// Extract level_count (index 2)
				if (auto lc_eval = eval<uint16_t>(args[2])) {
					ve.level_count = *lc_eval;
				}

				// Extract base_layer (index 3)
				if (auto blay_eval = eval<uint16_t>(args[3])) {
					ve.base_layer = *blay_eval;
				}

				// Extract layer_count (index 4)
				auto layc_eval = eval<uint16_t>(args[4]);
				if (layc_eval) {
					ve.layer_count = *layc_eval;
				}

				// Extract image pointer (index 5) and trace to Image allocation
				auto image_ref = to_def(args[5]);

				// Try to evaluate the Image directly if already allocated
				auto img_eval = eval<Image<>>(image_ref);
				if (img_eval && *img_eval) {
					ie = img_eval->get_meta();
				} else if (image_ref.node->kind == Node::ALLOCATE) {
					(void)img_eval.error();
					// Image not yet allocated, extract from ICI
					auto ici_ref = to_def(image_ref.node->allocate.src);

					// ICI is always evaluatable, so eval it directly
					if (auto ici_eval = eval<ICI>(ici_ref)) {
						ie = ImageEntry(*ici_eval);
					}
				}

				// Extract format from IVCI (index 6)
				if (auto format_eval = eval<Format>(args[6])) {
					ve.format = *format_eval;
				}

				// Extract offset (index 7)
				if (auto offset_eval = eval<Offset3D>(args[7])) {
					ve.offset = *offset_eval;
				}

				// Extract extent (index 8)
				if (auto extent_eval = eval<Extent3D>(args[8])) {
					ve.extent = *extent_eval;
				}
			}
		}

		// Apply SLICE modifications in reverse order (bottom-up)
		for (auto it = slice_stack.rbegin(); it != slice_stack.rend(); ++it) {
			auto slice_node = *it;
			auto axis = slice_node->slice.axis;

			auto start_eval = eval_as_size_t(slice_node->slice.start);
			auto count_eval = eval_as_size_t(slice_node->slice.count);

			if (!start_eval || !count_eval) {
				continue; // Can't apply slice without knowing start/count
			}

			switch (axis) {
			case Node::MIP:
				ve.base_level += static_cast<uint16_t>(*start_eval);
				ve.level_count = static_cast<uint16_t>(*count_eval);
				break;
			case Node::LAYER:
				ve.base_layer += static_cast<uint16_t>(*start_eval);
				ve.layer_count = static_cast<uint16_t>(*count_eval);
				break;
			case Node::X:
				ve.offset.x += static_cast<int32_t>(*start_eval);
				ve.extent.width = static_cast<uint32_t>(*count_eval);
				break;
			case Node::Y:
				ve.offset.y += static_cast<int32_t>(*start_eval);
				ve.extent.height = static_cast<uint32_t>(*count_eval);
				break;
			case Node::Z:
				ve.offset.z += static_cast<int32_t>(*start_eval);
				ve.extent.depth = static_cast<uint32_t>(*count_eval);
				break;
			default:
				break;
			}
		}

		return { ve, ie };
	}

	Result<void> lower_clear::operator()() {
		// Process all CLEAR nodes and transform them to CALL nodes
		for (auto node : impl.nodes) {
			if (node->kind != Node::CLEAR) {
				continue;
			}

			// Create the clear_image pass function
			auto clear_value = *node->clear.cv;

			// Get the scheduled domain from the node (set by queue inference)
			auto domain = node->scheduled_item->scheduled_domain;

			// Evaluate the ImageView to determine format/aspect
			auto [ve, ie] = evaluate_imageview_from_ir(node->clear.dst);

			// Determine aspect from format
			ImageAspectFlags aspect = ImageAspectFlagBits::eColor;

			if (ve.format != Format::eUndefined) {
				aspect = format_to_aspect(ve.format);
			}

			assert(domain != DomainFlagBits::eDevice || domain != DomainFlagBits::eAny);

			// Determine the effective operation type
			// If no operation is explicitly set, infer from the queue type
			DomainFlags effective_domain = domain;
			if (!(domain & DomainFlagBits::eOpMask)) {
				// No operation set, infer from queue
				if (domain & DomainFlagBits::eGraphicsQueue) {
					effective_domain = domain | DomainFlagBits::eGraphicsOperation;
				} else if (domain & DomainFlagBits::eComputeQueue) {
					effective_domain = domain | DomainFlagBits::eComputeOperation;
				} else if (domain & DomainFlagBits::eTransferQueue) {
					effective_domain = domain | DomainFlagBits::eTransferOperation;
				}
			}

			// Determine the clear access based on domain and aspect
			Access clear_access;

			// Check the operation type, not just the queue type
			if (effective_domain & DomainFlagBits::eTransferOperation) {
				return { expected_error,
					       RenderGraphException("Clear operations are not supported on transfer queues. "
					                            "Use graphics queue or compute queue.") };
			} else if (effective_domain & DomainFlagBits::eGraphicsOperation) {
				// Graphics operation: use renderpass-based clear
				// ColorWrite for color images, DSWrite for depth/stencil images
				if (aspect & (ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil)) {
					clear_access = Access::eDepthStencilWrite;
				} else {
					clear_access = Access::eColorWrite;
				}
			} else if (effective_domain & DomainFlagBits::eComputeOperation) {
				// Compute operation: use compute shader to clear
				// Works for both spanning and non-spanning views
				clear_access = Access::eComputeWrite;
			} else {
				assert(false && "Clear must have a specific operation type (graphics or compute)");
				clear_access = Access::eClear;
			}

			auto imageview_ty = current_module->types.make_imageview_ty();
			auto imbued_ty = current_module->types.make_imbued_ty(imageview_ty, clear_access);
			auto aliased_ty = current_module->types.make_aliased_ty(imageview_ty, 1);

			std::vector<std::shared_ptr<Type>> args;
			args.push_back(imbued_ty);

			std::vector<std::shared_ptr<Type>> return_types;
			return_types.push_back(aliased_ty);

			// Create a callback that performs the clear operation
			auto callback = [clear_value, effective_domain](CommandBuffer* cbuf, std::span<void*> in, std::span<void*> /* meta */, std::span<void*> out) {
				// Get the ImageView from inout[0]
				auto& iv = *reinterpret_cast<ImageView<>*>(in[0]);
				out[0] = in[0];

				DomainFlags scheduled_domain = cbuf->get_scheduled_domain();
				assert((effective_domain & DomainFlagBits::eQueueMask) == scheduled_domain);
				if (!(scheduled_domain & DomainFlagBits::eOpMask)) {
					// No operation set, infer from queue
					if (scheduled_domain & DomainFlagBits::eGraphicsQueue) {
						scheduled_domain |= DomainFlagBits::eGraphicsOperation;
					} else if (scheduled_domain & DomainFlagBits::eComputeQueue) {
						scheduled_domain |= DomainFlagBits::eComputeOperation;
					} else if (scheduled_domain & DomainFlagBits::eTransferQueue) {
						scheduled_domain |= DomainFlagBits::eTransferOperation;
					}
				}

				// Domain-specific validation and clearing based on operation type
				if (scheduled_domain & DomainFlagBits::eGraphicsOperation) {
					// Graphics: use renderpass-based clear (vkCmdClearAttachments or load op clear)
					cbuf->clear_image(iv, clear_value);
				} else if (scheduled_domain & DomainFlagBits::eComputeOperation) {
					// Compute operation: use compute shader to clear the image
					auto& ve = iv.get_meta();

					// Get the pipeline for compute clear
					// The pipeline should be created once and cached
					auto& runtime = cbuf->get_context();
					static std::once_flag once;
					std::call_once(once, [&runtime]() {
						PipelineBaseCreateInfo pbci;
						pbci.add_static_spirv((uint32_t*)compute_clear_comp_spv_shader, sizeof(compute_clear_comp_spv_shader) / 4, "compute_clear.comp");
						runtime.create_named_pipeline("_vuk_compute_clear", pbci);
					});

					// Set up push constants for the clear
					struct ClearPushConstants {
						float clear_color[4];
						int32_t offset[4];
						int32_t extent[3];
						uint32_t format_index;
					} pc;

					pc.clear_color[0] = clear_value.c.color.float32[0];
					pc.clear_color[1] = clear_value.c.color.float32[1];
					pc.clear_color[2] = clear_value.c.color.float32[2];
					pc.clear_color[3] = clear_value.c.color.float32[3];
					pc.offset[0] = ve.offset.x;
					pc.offset[1] = ve.offset.y;
					pc.offset[2] = ve.offset.z;
					pc.extent[0] = ve.extent.width;
					pc.extent[1] = ve.extent.height;
					pc.extent[2] = ve.extent.depth;
					pc.format_index = uint32_t(ve.format);

					// Calculate dispatch size (8x8x1 local size)
					uint32_t num_workgroups_x = (ve.extent.width + 7) / 8;
					uint32_t num_workgroups_y = (ve.extent.height + 7) / 8;
					uint32_t num_workgroups_z = (ve.extent.depth + 0) / 1;

					// Bind the pipeline and image, then dispatch
					cbuf->bind_compute_pipeline("_vuk_compute_clear")
					    .bind_image(0, 0, iv)
					    .push_constants(ShaderStageFlagBits::eCompute, 0, pc)
					    .dispatch(num_workgroups_x, num_workgroups_y, num_workgroups_z);
				} else {
					// Transfer operation should never reach here - already caught in IR lowering
					assert(false && "Clear operations are not supported on transfer queues.");
				}
			};

			auto fn_ty = current_module->types.make_opaque_fn_ty(args, return_types, domain, typeid(callback).hash_code(), callback, "ffn");

			// Create the function declaration node
			Ref fn_ref = current_module->make_declare_fn(fn_ty);
			add_node(fn_ref.node);

			// Save the dst reference before modifying the node
			Ref dst = node->clear.dst;

			// Directly mutate the CLEAR node into a CALL node
			Ref* args_ptr = new Ref[2]{ fn_ref, dst };
			auto call_ty = new std::shared_ptr<Type>[1]{ imbued_ty };

			node->kind = Node::CALL;
			node->type = std::span{ call_ty, 1 };
			node->call = { .args = std::span(args_ptr, 2) };

			// Determine required usage flags from access and add them to the image
			access_to_usage(impl.image_usage_flags[dst], clear_access);
		}

		return { expected_value };
	}
} // namespace vuk
