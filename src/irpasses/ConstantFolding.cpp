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
		/*GARBAGE*/ DomainFlagBits::ePlaceholder
	};

	Result<void> constant_folding::operator()() {
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
					    // do constant folding here
					    if (arg.node->compute_class == DomainFlagBits::eConstant && arg.node->kind != Node::CONSTANT) {
						    auto result = eval(arg);
						    if (result) {
							    arg = current_module->make_constant(arg.type(), *result);
							    add_node(arg.node);
						    }
					    }
				    },
				    node);
			}
		});
		return { expected_value };
	}
} // namespace vuk