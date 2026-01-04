#include "vuk/ir/GraphDumper.hpp"
#include "vuk/ir/IRPasses.hpp"

namespace vuk {
	// Make sure this table is updated if Node::Kind changes!
	constexpr DomainFlags op_compute_class[] = {
		/*PLACEHOLDER*/ DomainFlagBits::ePlaceholder,
		/*CONSTANT*/ DomainFlagBits::eConstant,
		/*CONSTRUCT*/ DomainFlagBits::eConstant,
		/*SLICE*/ DomainFlagBits::eConstant,
		/*CONVERGE*/ DomainFlagBits::eConstant,
		/*IMPORT*/ DomainFlagBits::eHost,
		/*CALL*/ DomainFlagBits::eHost,
		/*CLEAR*/ DomainFlagBits::eDevice,
		/*ACQUIRE*/ DomainFlagBits::eHost,
		/*RELEASE*/ DomainFlagBits::eHost,
		/*ACQUIRE_NEXT_IMAGE*/ DomainFlagBits::eHost,
		/*USE*/ DomainFlagBits::eConstant,
		/*LOGICAL_COPY*/ DomainFlagBits::eConstant,
		/*SET*/ DomainFlagBits::ePlaceholder,
		/*CAST*/ DomainFlagBits::eConstant,
		/*MATH_BINARY*/ DomainFlagBits::eConstant,
		/*COMPILE_PIPELINE*/ DomainFlagBits::eConstant,
		/*ALLOCATE*/ DomainFlagBits::eHost,
		/*GET_ALLOCATION_SIZE*/ DomainFlagBits::eConstant,
		/*GET_CI*/ DomainFlagBits::eConstant,
		/*GARBAGE*/ DomainFlagBits::ePlaceholder
	};

	static_assert(sizeof(op_compute_class) == Node::Kind::NODE_KIND_MAX * sizeof(DomainFlags));

	Result<void> constant_folding::operator()() {
		rewrite([this](Node* node, Replacer& r) {
			switch (node->kind) {
			case Node::SLICE: {
				if (!node->type[0]->is_synchronized()) {
					// direct slicing of a composite
					if (node->slice.src.node->kind == Node::CONSTRUCT && node->slice.axis == Node::NamedAxis::FIELD) {
						auto field_idx = constant<uint64_t>(node->slice.start);
						r.replace({ node, 0 }, node->slice.src.node->construct.args[field_idx + 1]);
					}
					// slicing a slice
					else if (node->slice.src.node->kind == Node::SLICE && node->slice.src.index <= 1 && node->slice.axis == Node::NamedAxis::FIELD) {
						auto field_idx = constant<uint64_t>(node->slice.start);
						auto new_slice = current_module->make_extract(node->slice.src.node->slice.src, field_idx);
						add_node(new_slice.node);
						r.replace({ node, 0 }, new_slice);
					} else if (node->slice.src.node->kind == Node::CALL) {
						auto field_idx = constant<uint64_t>(node->slice.start);
						auto new_slice = current_module->make_extract(node->slice.src.link().prev->def, field_idx);
						add_node(new_slice.node);
						r.replace({ node, 0 }, new_slice);
					}
				}
			} break;
			case Node::CONVERGE: {
				// if all args are the same, replace with that arg
				bool all_same = true;
				auto first = node->converge.diverged[0].node;
				for (auto& arg : node->converge.diverged) {
					if (!(arg.node == first)) {
						all_same = false;
						break;
					}
				}
				if (all_same && node->converge.diverged[0].node->kind == Node::SLICE) {
					r.replace({ node, 0 }, node->converge.diverged[0].node->slice.src);
				}
			} break;
			default:
				break;
			}
		});

		// compute class assignments & perform constant folding
		visit_all_postorder([this](Node* node) {
			DomainFlags op_class = op_compute_class[node->kind];

			node->compute_class = op_class;

			switch (node->kind) {
			case Node::CALL: {
				auto fn_type = node->call.args[0].type();
				// TODO: this is always eAny now
				if (fn_type->kind == Type::OPAQUE_FN_TY && fn_type->opaque_fn.execute_on != DomainFlagBits::eAny) {
					node->compute_class = fn_type->opaque_fn.execute_on;
				} else if (fn_type->shader_fn.execute_on != DomainFlagBits::eAny) {
					node->compute_class = fn_type->shader_fn.execute_on;
				}
				node->compute_class = DomainFlagBits::eDevice;
			}
				[[fallthrough]];
			default:
				apply_generic_args(
				    [&, this](Ref& arg) {
					    DomainFlags input_class = arg.node->compute_class;
					    if (input_class.m_mask > node->compute_class.m_mask) {
						    node->compute_class = input_class;
					    }
					    // fold away logical copies
					    if (arg.node->kind == Node::LOGICAL_COPY) {
						    arg = arg.node->logical_copy.src;
					    } else if (arg.node->compute_class == DomainFlagBits::eConstant && arg.node->kind != Node::CONSTANT &&
					               arg.node->kind != Node::PLACEHOLDER) { // do constant folding here
						    auto result = eval(arg);
						    if (result.holds_value()) {
							    arg = current_module->make_constant(arg.type(), *result);
							    add_node(arg.node);
						    }
					    } else if (!arg.type()->is_synchronized() && arg.node->kind != Node::CONSTANT) {
						    auto result = eval(arg);
						    if (result.holds_value()) {
							    arg = current_module->make_constant(arg.type(), *result);
							    add_node(arg.node);
						    }
					    }
				    },
				    node);
			}
		});

		if (impl.set_nodes.size() > 0) {
			// apply SETs
			rewrite([](Node* node, Replacer& r) {
				if (node->kind == Node::SET) {
					auto& set = node->set;
					if (set.value.node->kind != Node::PLACEHOLDER) {
						r.replace(set.dst, set.value);
					}
				}
			});
			/*
			GraphDumper::begin_graph(true, "Before Constant Folding");
			GraphDumper::dump_graph(impl.nodes, false, false);
			GraphDumper::end_graph();
			printf("");
			*/
		}

		impl.set_nodes.clear();

		return { expected_value };
	}
} // namespace vuk