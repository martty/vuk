#include "vuk/ir/GraphDumper.hpp"
#include "vuk/ir/IR.hpp"
#include "vuk/ir/IRPasses.hpp"
#include "vuk/Result.hpp"
#include "vuk/SyncLowering.hpp"
#include <stack>
#include <unordered_map>

namespace vuk {
	Result<void> reify_inference::operator()() {
		auto is_placeholder = [](Ref r) {
			return r.node->kind == Node::PLACEHOLDER;
		};

		bool progress = false;

		auto placeholder_to_constant = [&progress]<class T>(Ref r, T value) {
			if (r.node->kind == Node::PLACEHOLDER) {
				r.node->kind = Node::CONSTANT;
				assert(sizeof(T) == r.type()->size);
				r.node->constant.value = new char[sizeof(T)];
				new (r.node->constant.value) T(value);
				r.node->constant.owned = true;
				progress = true;
			}
		};

		auto placeholder_to_ptr = [](Ref r, void* ptr) {
			if (r.node->kind == Node::PLACEHOLDER) {
				r.node->kind = Node::CONSTANT;
				r.node->constant.value = ptr;
				r.node->constant.owned = false;
			}
		};

		std::unordered_set<Node*> inference_graph;
		// compute uses - direct & indirect of placeholders

		auto traverse = [&](const auto& self, Ref r, std::vector<size_t>& type_path) {
			switch (r.node->kind) {
			case Node::PLACEHOLDER:
				[[fallthrough]];
			case Node::MATH_BINARY:
				[[fallthrough]];
			case Node::CONSTRUCT:
				[[fallthrough]];
			case Node::LOGICAL_COPY:
				[[fallthrough]];
			case Node::SLICE: {
				break;
			}
			default:
				return;
			}
			if (!inference_graph.emplace(r.node).second) {
				return;
			}

			for_each_use(r, [&](Ref use) {
				if (use.node->kind == Node::CONSTRUCT) {
					type_path.push_back(use.index);
					self(self, first(use.node), type_path); // reads are rrefs
					type_path.pop_back();
				} else if (use.node->kind == Node::SLICE) {
					auto& slice = use.node->slice;
					if (!type_path.empty() && constant<uint64_t>(slice.start) == type_path.back()) {
						auto t = type_path.back();
						type_path.pop_back();
						self(self, first(use.node), type_path);
						type_path.push_back(t);
					} else {
						self(self, nth(use.node, 1), type_path);
					}
				} else {
					self(self, use, type_path);
				}
			});
		};
		/*
		for (auto node : impl.nodes) {
		  switch (node->kind) {
		  case Node::PLACEHOLDER: {
		    std::vector<size_t> type_path;
		    traverse(traverse, Ref{ node, 0 }, type_path);
		  } break;
		  }
		}*/

		/*
		// construct reification - if there were later setting of fields, then remove placeholders
		// TODO: PAV: constructs with placeholders inside them - to be redone
		for (auto node : impl.nodes) {
		  auto ty = node->type[0];
		  switch (node->kind) {
		  case Node::CONSTRUCT: {
		    auto args_ptr = node->construct.args.data();
		    if (ty->hash_value == current_module->types.builtin_image) {
		      auto ptr = &constant<ImageAttachment>(args_ptr[0]);
		      auto& value = constant<ImageAttachment>(args_ptr[0]);
		      if (value.extent.width > 0) {
		        placeholder_to_ptr(args_ptr[1], &ptr->extent.width);
		      }
		      if (value.extent.height > 0) {
		        placeholder_to_ptr(args_ptr[2], &ptr->extent.height);
		      }
		      if (value.extent.depth > 0) {
		        placeholder_to_ptr(args_ptr[3], &ptr->extent.depth);
		      }
		      if (value.format != Format::eUndefined) {
		        placeholder_to_ptr(args_ptr[4], &ptr->format);
		      }
		      if (value.sample_count != Samples::eInfer) {
		        placeholder_to_ptr(args_ptr[5], &ptr->sample_count);
		      }
		      if (value.base_layer != VK_REMAINING_ARRAY_LAYERS) {
		        placeholder_to_ptr(args_ptr[6], &ptr->base_layer);
		      }
		      if (value.layer_count != VK_REMAINING_ARRAY_LAYERS) {
		        placeholder_to_ptr(args_ptr[7], &ptr->layer_count);
		      }
		      if (value.base_level != VK_REMAINING_MIP_LEVELS) {
		        placeholder_to_ptr(args_ptr[8], &ptr->base_level);
		      }
		      if (value.level_count != VK_REMAINING_MIP_LEVELS) {
		        placeholder_to_ptr(args_ptr[9], &ptr->level_count);
		      }
		    } else if (ty->kind == Type::COMPOSITE_TY) {
		      // special casing for buffer views
		      if (ty->is_bufferlike_view()) {
		        // case 1: view has a known size, but the allocate is a placeholder
		        if (node->construct.args[2].node->kind != Node::PLACEHOLDER) {
		          auto def = eval(node->construct.args[1]);
		          if (def && def->is_ref && def->ref.node->kind == Node::CONSTRUCT && def->ref.node->construct.args[2].node->kind == Node::PLACEHOLDER) {
		            def->ref.node->construct.args[2] = node->construct.args[2];
		          }
		        }
		      }

		      for (size_t i = 1; i < node->construct.args.size(); i++) {
		        bool is_default = ty->composite.is_default(base, i - 1);
		        if (!is_default) {
		          placeholder_to_ptr(args_ptr[i], ty->composite.get(base, i - 1));
		        }
		      }
		    }

		  } break;
		  default:
		    break;
		  }
		}*/
		/*
		// framebuffer inference
		do {
		  progress = false;
		  for (auto node : impl.nodes) {
		    switch (node->kind) {
		    case Node::CALL: {
		      if (node->call.args[0].type()->kind != Type::OPAQUE_FN_TY) {
		        continue;
		      }

		      // args
		      std::optional<Extent2D> extent;
		      std::optional<Samples> samples;
		      std::optional<uint32_t> layer_count;
		      for (size_t i = 1; i < node->call.args.size(); i++) {
		        auto& arg_ty = node->call.args[0].type()->opaque_fn.args[i - 1];
		        auto& parm = node->call.args[i];
		        if (arg_ty->kind == Type::IMBUED_TY) {
		          auto access = arg_ty->imbued.access;
		          if (is_framebuffer_attachment(access)) {
		            auto def = eval(parm);
		            if (!def.holds_value() || !def->ref) {
		              continue;
		            }
		            if (def->ref.node->kind == Node::CONSTRUCT) {
		              auto& args = def->ref.node->construct.args;
		              if (is_placeholder(args[9])) {
		                placeholder_to_constant(args[9], 1U); // can only render to a single mip level
		              }
		              if (is_placeholder(args[3])) {
		                placeholder_to_constant(args[3], 1U); // depth must be 1
		              }
		              if (!samples && !is_placeholder(args[5])) { // known sample count
		                samples = constant<Samples>(args[5]);
		              } else if (samples && is_placeholder(args[5])) {
		                placeholder_to_constant(args[5], *samples);
		              }
		              if (!extent && !is_placeholder(args[1]) && !is_placeholder(args[2])) { // known extent2D
		                auto e1 = eval<uint32_t>(args[1]);
		                auto e2 = eval<uint32_t>(args[2]);
		                if (e1.holds_value() && e2.holds_value()) {
		                  extent = Extent2D{ *e1, *e2 };
		                }
		              } else if (extent && is_placeholder(args[1]) && is_placeholder(args[2])) {
		                placeholder_to_constant(args[1], extent->width);
		                placeholder_to_constant(args[2], extent->height);
		              }
		              if (!layer_count && !is_placeholder(args[7])) { // known layer count
		                auto e = eval<uint32_t>(args[7]);
		                if (e.holds_value()) {
		                  layer_count = *e;
		                }
		              } else if (layer_count && is_placeholder(args[7])) {
		                placeholder_to_constant(args[7], *layer_count);
		              }
		            }
		          }
		        } else {
		          assert(0);
		        }
		      }
		      break;
		    }
		    case Node::CONSTRUCT: {
		      auto& args = node->construct.args;
		      if (node->type[0]->hash_value == current_module->types.builtin_image) {
		        if (constant<ImageAttachment>(args[0]).image.image == VK_NULL_HANDLE) { // if there is no image, we will use base layer 0 and base mip 0
		          placeholder_to_constant(args[6], 0U);
		          placeholder_to_constant(args[8], 0U);
		        }
		      }
		      break;
		    }
		    default:
		      break;
		    }
		  }
		} while (progress);
		*/
		return { expected_value };
	}
} // namespace vuk