#include "vuk/ir/GraphDumper.hpp"
#include "vuk/ir/IRPasses.hpp"
#include "vuk/RenderGraph.hpp"

// This pass takes the image usage flags collected in RGCImpl::image_usage_flags
// during link building and propagates them to the Image Create Info (ICI) allocation nodes.

namespace vuk {
	Ref to_def(Ref ref) {
		auto link = &ref.link();
		while (link->prev && link->prev->def) {
			link = link->prev;
		}
		return link->def;
	}

	Result<void> propagate_usage_flags::operator()() {
		// Iterate through all image_usage_flags collected during link building
		for (auto [ref, usage_flags] : impl.image_usage_flags) {
			Ref current = to_def(ref);

			// First, find the ImageView allocation
			if (current.node->kind == Node::ALLOCATE && Type::stripped(current.node->type[0])->is_imageview()) {
				auto ivci_ref = to_def(current.node->allocate.src);

				// Check if this is a construct node (IVCI)
				if (ivci_ref.node->kind == Node::CONSTRUCT) {
					// The image pointer should be at index 5 (nullptr, base_level, level_count, base_layer, layer_count, image, format)
					assert(ivci_ref.node->construct.args.size() >= 5);
					auto image_ref = to_def(ivci_ref.node->construct.args[5]);

					// Check if this is an Image allocation
					if (image_ref.node->kind == Node::ALLOCATE) {
						// Get the ICI source
						auto ici_ref = to_def(image_ref.node->allocate.src);

						// We usually fold this away
						if (ici_ref.node->kind == Node::CONSTANT) {
							auto& ici = constant<ICI>(ici_ref);

							ici.usage |= usage_flags;
						} else {
							assert(false);
						}
					}
				}
			}
		}

		return { expected_value };
	}
} // namespace vuk
