#include "vuk/ir/GraphDumper.hpp"
#include "vuk/ir/IR.hpp"
#include "vuk/ir/IRPasses.hpp"
#include "vuk/Result.hpp"
#include "vuk/SyncLowering.hpp"
#include <stack>
#include <unordered_map>

namespace vuk {
	Result<void> reify_inference::operator()() {
		// Walk through all nodes to find CALL nodes with framebuffer attachments
		for (auto node : impl.nodes) {
			if (node->kind != Node::CALL) {
				continue;
			}

			// Check if the function type has parameters
			auto fn_type = node->call.args[0].type();
			size_t first_parm = fn_type->kind == Type::OPAQUE_FN_TY ? 1 : 4;
			auto& args = fn_type->kind == Type::OPAQUE_FN_TY ? fn_type->opaque_fn.args : fn_type->shader_fn.args;

			// Collect all framebuffer attachment image view arguments
			std::vector<Ref> fb_attachments;
			for (size_t i = first_parm; i < node->call.args.size(); i++) {
				auto& arg_ty = args[i - first_parm];
				if (arg_ty->kind == Type::IMBUED_TY) {
					auto access = arg_ty->imbued.access;
					if (is_framebuffer_attachment(access)) {
						fb_attachments.push_back(node->call.args[i]);
					}
				}
			}

			// If we have multiple framebuffer attachments, we need to enforce constraints
			if (fb_attachments.size() < 2) {
				continue;
			}

			// For each framebuffer attachment, walk back to find the Image allocation
			std::vector<Ref> image_refs;
			for (auto& iv_ref : fb_attachments) {
				// Walk to definition
				Ref current = to_def(iv_ref);

				// Should be an ImageView ALLOCATE
				if (current.node->kind == Node::ALLOCATE && Type::stripped(current.node->type[0])->is_imageview()) {
					auto ivci_ref = to_def(current.node->allocate.src);

					// Should be a CONSTRUCT node (IVCI)
					if (ivci_ref.node->kind == Node::CONSTRUCT) {
						// The image pointer should be at index 4 (base_level, level_count, base_layer, layer_count, image, format)
						if (ivci_ref.node->construct.args.size() >= 6) {
							auto image_ref = to_def(ivci_ref.node->construct.args[5]);

							// Check if this is an Image allocation
							if (image_ref.node->kind == Node::ALLOCATE) {
								image_refs.push_back(image_ref);
							}
						}
					}
				}
			}

			// Now add SET nodes to enforce constraints between all pairs
			if (image_refs.size() >= 2) {
				for (size_t i = 1; i < image_refs.size(); i++) {
					auto ref_image = image_refs[0];
					auto cur_image = image_refs[i];

					// Get the ICI from both images
					auto ref_ici = current_module->make_get_ci(ref_image);
					add_node(ref_ici.node);
					auto cur_ici = current_module->make_get_ci(cur_image);
					add_node(cur_ici.node);

					// Extract and enforce: extent (index 4)
					auto ref_extent = current_module->make_extract(ref_ici, 4);
					add_node(ref_extent.node);
					auto cur_extent = current_module->make_extract(cur_ici, 4);
					add_node(cur_extent.node);

					// Extract width and height (2D extent)
					auto ref_width = current_module->make_extract(ref_extent, 0);
					add_node(ref_width.node);
					auto cur_width = current_module->make_extract(cur_extent, 0);
					add_node(cur_width.node);

					auto ref_height = current_module->make_extract(ref_extent, 1);
					add_node(ref_height.node);
					auto cur_height = current_module->make_extract(cur_extent, 1);
					add_node(cur_height.node);

					// SET: extent.width must match
					auto width_set = current_module->set_value(cur_width, ref_width);
					impl.set_nodes.push_back(width_set);

					// SET: extent.height must match
					auto height_set = current_module->set_value(cur_height, ref_height);
					impl.set_nodes.push_back(height_set);

					// Extract and SET: extent.depth = 1
					auto cur_depth = current_module->make_extract(cur_extent, 2);
					add_node(cur_depth.node);
					auto depth_one = current_module->make_constant<uint32_t>(1);
					auto depth_set = current_module->set_value(cur_depth, depth_one);
					impl.set_nodes.push_back(depth_set);
					if (i == 1) {
						// Only need to set this once for reference
						auto ref_depth = current_module->make_extract(ref_extent, 2);
						add_node(ref_depth.node);
						auto ref_depth_set = current_module->set_value(ref_depth, depth_one);
						impl.set_nodes.push_back(ref_depth_set);
					}

					// Extract sample_count from reference ICI (index 6)
					auto ref_samples = current_module->make_extract(ref_ici, 6);
					add_node(ref_samples.node);
					auto cur_samples = current_module->make_extract(cur_ici, 6);
					add_node(cur_samples.node);

					// SET: sample_count must match
					auto samples_set = current_module->set_value(cur_samples, ref_samples);
					impl.set_nodes.push_back(samples_set);

					// Extract and SET: level_count = 1 (index 7)
					auto cur_levels = current_module->make_extract(cur_ici, 7);
					add_node(cur_levels.node);
					auto level_one = current_module->make_constant<uint32_t>(1);
					auto levels_set = current_module->set_value(cur_levels, level_one);
					impl.set_nodes.push_back(levels_set);
					if (i == 1) {
						// Only need to set this once for reference
						auto ref_levels = current_module->make_extract(ref_ici, 7);
						add_node(ref_levels.node);
						auto ref_levels_set = current_module->set_value(ref_levels, level_one);
						impl.set_nodes.push_back(ref_levels_set);
					}

					// Extract layer_count from reference ICI (index 8)
					auto ref_layers = current_module->make_extract(ref_ici, 8);
					add_node(ref_layers.node);
					auto cur_layers = current_module->make_extract(cur_ici, 8);
					add_node(cur_layers.node);

					// SET: layer_count must match
					auto layers_set = current_module->set_value(cur_layers, ref_layers);
					impl.set_nodes.push_back(layers_set);
				}
			}
		}

		return { expected_value };
	}
} // namespace vuk