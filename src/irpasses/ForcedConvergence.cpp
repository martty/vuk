#include "vuk/ir/IRPasses.hpp"
#include <array>

namespace vuk {
	Result<void> forced_convergence::operator()() {
		for (auto& [def, lr] : impl.live_ranges) {
			if (lr.def_link->def.node->kind == Node::SLICE) { // subchains - not important
				continue;
			}
			while (lr.undef_link->next) {
				lr.undef_link = lr.undef_link->next;
			}
			if (lr.undef_link->undef && lr.undef_link->undef.node->kind == Node::SLICE &&
			    nth(lr.undef_link->undef.node, 2).type()->kind != Type::UNION_TY) { // main chain that ends in SLICE..
				// make force reconvergence node
				auto slice_node = lr.undef_link->undef.node;
				std::array tails{ nth(slice_node, 2), nth(slice_node, 0), nth(slice_node, 1) };
				auto f_converge = current_module->make_converge(slice_node->slice.src.type(), tails);
				add_node(f_converge.node);
				// add use node
				auto use_node = current_module->make_use(f_converge, Access::eNone);
				add_node(use_node.node);
				// make the ref node depend on it
				assert(impl.ref_nodes.back()->kind == Node::RELEASE);
				auto& release_node = impl.ref_nodes.back();
				std::array wrapping{ release_node->release.src[0], use_node };
				release_node->release.src[0].link().undef = {};
				release_node->release.src[0].link().next = {};
				release_node->release.src[0] = current_module->make_converge(release_node->release.src[0].type(), wrapping);
				release_node->release.src[0].node->index = release_node->index;
				add_node(release_node->release.src[0].node);
				allocate_node_links(release_node);
				process_node_links(release_node);
			}
		}
		return { expected_value };
	}
} // namespace vuk