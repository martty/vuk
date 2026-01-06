#include "vuk/ir/GraphDumper.hpp"
#include "vuk/ir/IRPasses.hpp"
#include "vuk/RenderGraph.hpp"

namespace vuk {
	Result<void> expand_default_view::operator()() {
		rewrite([this](Node* node, Replacer& r) {
			if (node->kind == Node::ALLOCATE) {
				auto alloc_type = Type::stripped(node->type[0]);
				auto src_type = Type::stripped(node->allocate.src.type());

				// Check if this is an ImageView allocation from an Image pointer (not IVCI)
				if (alloc_type->is_imageview() && src_type->kind == Type::IMAGE_TY) {
					// Get the ICI from the image allocation
					auto ici_ref = current_module->make_get_ci(node->allocate.src);
					add_node(ici_ref.node);

					// Construct IVCI from ICI
					// IVCI fields: base_level, level_count, base_layer, layer_count, image, format
					// ICI fields: image_flags, image_type, tiling, usage, extent, format, sample_count, level_count, layer_count

					// Extract format from ICI (index 5)
					auto format_ref = current_module->make_extract(ici_ref, 5);
					add_node(format_ref.node);

					// Create IVCI with default values matching get_default_view_create_info
					// base_level = 0
					// level_count = VK_REMAINING_MIP_LEVELS (0xFFFF)
					// base_layer = 0
					// layer_count = VK_REMAINING_ARRAY_LAYERS (0xFFFF)
					// image = the image pointer
					// format = from ICI

					auto base_level = current_module->make_constant<uint16_t>(0);
					auto level_count = current_module->make_extract(ici_ref, 7);
					auto base_layer = current_module->make_constant<uint16_t>(0);
					auto layer_count = current_module->make_extract(ici_ref, 8);

					// Construct IVCI
					std::array<Ref, 6> ivci_args = { base_level, level_count, base_layer, layer_count, node->allocate.src, format_ref };

					auto ivci_ref = current_module->make_construct(to_IR_type<IVCI>(), nullptr, std::span(ivci_args));
					add_node(ivci_ref.node);

					// Create new ALLOCATE node with IVCI
					auto new_alloc = current_module->make_allocate(node->type[0], ivci_ref, node->allocate.allocator);
					add_node(new_alloc.node);

					// Move debug info
					if (node->debug_info) {
						new_alloc.node->debug_info = node->debug_info;
						node->debug_info = nullptr;
					}

					r.replace({ node, 0 }, new_alloc);
				}
			}
		});

		return { expected_value };
	}
} // namespace vuk
