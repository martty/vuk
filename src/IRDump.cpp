#include "vuk/ir/IRProcess.hpp"

namespace vuk {
	std::string domain_to_string(DomainFlagBits domain) {
		domain = (DomainFlagBits)(domain & DomainFlagBits::eDomainMask).m_mask;
		std::string result;

		if (domain == DomainFlagBits::eNone)
			return "None";
		else if (domain == DomainFlagBits::eAny)
			return "Any";
		else if (domain == DomainFlagBits::eDevice)
			return "Device";

		if (domain & DomainFlagBits::eHost)
			result += "Host |";
		if (domain & DomainFlagBits::ePE)
			result += "PE |";
		if (domain & DomainFlagBits::eGraphicsQueue)
			result += "Graphics |";
		if (domain & DomainFlagBits::eComputeQueue)
			result += "Compute |";
		if (domain & DomainFlagBits::eTransferQueue)
			result += "Transfer |";
		return result.substr(0, result.size() - 2);
	}

	std::string format_source_location(SourceLocationAtFrame& source) {
		return fmt::format("{}({}): ", source.location.file_name(), source.location.line());
	}

	std::string format_source_location(Node* node) {
		if (node->debug_info) {
			std::string msg = "";
			for (int i = 0; i < node->debug_info->trace.size(); i++) {
				auto& source = node->debug_info->trace[i];
				msg += fmt::format("{}({}): ", source.file_name(), source.line());
				if (i < (node->debug_info->trace.size() - 1)) {
					msg += "\n";
				}
			}
			return msg;
		} else {
			return "?: ";
		}
	}

	void parm_to_string(Ref parm, std::string& msg) {
		if (parm.node->debug_info && parm.node->debug_info->result_names.size() > parm.index) {
			fmt::format_to(std::back_inserter(msg), "%{}", parm.node->debug_info->result_names[parm.index]);
		} else if (parm.node->kind == Node::CONSTANT) {
			Type* ty = parm.node->type[0].get();
			if (ty->kind == Type::INTEGER_TY) {
				switch (ty->scalar.width) {
				case 32:
					fmt::format_to(std::back_inserter(msg), "{}", constant<uint32_t>(parm));
					break;
				case 64:
					fmt::format_to(std::back_inserter(msg), "{}", constant<uint64_t>(parm));
					break;
				}
			} else if (ty->kind == Type::MEMORY_TY) {
				fmt::format_to(std::back_inserter(msg), "<mem>");
			} else if (ty->kind == Type::ENUM_TY) {
				if (ty->format_to) {
					ty->format_to(parm.node->constant.value, msg);
				} else {
					fmt::format_to(std::back_inserter(msg), "<enum>");
				}
			} else if (ty->kind == Type::ENUM_VALUE_TY) {
				std::string formatted;
				if (ty->enum_value.enum_type->get()->format_to) {
					ty->enum_value.enum_type->get()->format_to((void*)&ty->enum_value.value, formatted);
					fmt::format_to(std::back_inserter(msg), "{}", formatted);
				} else {
					fmt::format_to(std::back_inserter(msg), "{}", ty->enum_value.value);
				}
			} else if (ty->kind == Type::COMPOSITE_TY) {
				if (ty->format_to) {
					ty->format_to(parm.node->constant.value, msg);
				} else {
					fmt::format_to(std::back_inserter(msg), "<composite>");
				}
			}
		} else if (parm.node->kind == Node::PLACEHOLDER) {
			fmt::format_to(std::back_inserter(msg), "?");
		} else if (parm.node->execution_info) {
			fmt::format_to(
			    std::back_inserter(msg), "%{}_{}", Node::kind_to_sv(parm.node->execution_info->kind), parm.node->execution_info->naming_index + parm.index);
		} else {
			fmt::format_to(std::back_inserter(msg), "%{}_{}", Node::kind_to_sv(parm.node->kind), parm.node->scheduled_item->naming_index + parm.index);
		}
	};

	void print_args_to_string(std::span<Ref> args, std::string& msg) {
		for (size_t i = 0; i < args.size(); i++) {
			if (i > 0) {
				fmt::format_to(std::back_inserter(msg), ", ");
			}
			auto& parm = args[i];

			parm_to_string(parm, msg);
		}
	};

	void print_args(std::span<Ref> args) {
		std::string msg;
		print_args_to_string(args, msg);
		fmt::print("{}", msg);
	};

	std::string print_args_to_string_with_arg_names(std::span<const std::string_view> arg_names, std::span<Ref> args) {
		std::string msg = "";
		for (size_t i = 0; i < args.size(); i++) {
			if (i > 0) {
				msg += fmt::format(", ");
			}
			auto& parm = args[i];

			msg += fmt::format("{}:", arg_names[i]);
			parm_to_string(parm, msg);
		}
		return msg;
	};

	std::string node_to_string(Node* node) {
		if (node->kind == Node::CONSTRUCT) {
			return fmt::format("construct<{}> ", Type::to_string(node->type[0].get()));
		} else {
			return fmt::format("{} ", Node::kind_to_sv(node->kind));
		}
	};

	using namespace std::literals;
	std::vector<std::string_view> arg_names(Type* t) { // TODO: decommission
		if (t->kind == Type::COMPOSITE_TY) {
			std::vector<std::string_view> result;
			for (auto& m : t->member_names) {
				result.push_back(m);
			}
			return result;
		} else {
			assert(0);
			return {};
		}
	};

	std::string format_graph_message(Level level, Node* node, std::string err) {
		std::string msg = "";
		msg += format_source_location(node);
		msg += fmt::format("{}: {}", level == Level::eError ? "error" : "other", node_to_string(node));
		msg += err;
		return msg;
	};

	void print_results_to_string(ScheduledItem& item, std::string& msg) {
		auto& node = item.execable;
		for (size_t i = 0; i < node->type.size(); i++) {
			if (i > 0) {
				fmt::format_to(std::back_inserter(msg), ", ");
			}
			if (node->debug_info && !node->debug_info->result_names.empty() && node->debug_info->result_names.size() > i) {
				fmt::format_to(std::back_inserter(msg), "%{}", node->debug_info->result_names[i]);
			} else {
				fmt::format_to(std::back_inserter(msg), "%{}_{}", Node::kind_to_sv(node->kind), item.naming_index + i);
			}
		}
	};

	void print_results(ScheduledItem& item) {
		std::string msg = "";
		print_results_to_string(item, msg);
		fmt::print("{}", msg);
	};

	void format_args(ScheduledItem& item, std::string& line) {
		Node* node = item.execable;
		switch (node->kind) {
		case Node::GARBAGE:
		case Node::PLACEHOLDER:
		case Node::CONSTANT:
		case Node::IMPORT:
		case Node::CLEAR:
		case Node::SET:
		case Node::CAST:
		case Node::MATH_BINARY: {
			assert(0);
		}
		case Node::CONSTRUCT: {
			if (node->type[0]->kind == Type::ARRAY_TY) {
				auto array_size = node->type[0]->array.count;
				auto elem_ty = *node->type[0]->array.T;
				fmt::format_to(std::back_inserter(line), "construct<{}[{}]> ", elem_ty->debug_info.name, array_size);
			} else if (node->type[0]->kind == Type::UNION_TY) {
				fmt::format_to(std::back_inserter(line), "construct<union> ");
			} else {
				fmt::format_to(std::back_inserter(line), "construct<{}> ", node->type[0]->debug_info.name);
			}
			print_args_to_string(node->construct.args.subspan(1), line);
		} break;
		case Node::CALL: {
			auto fn_type = node->call.args[0].type();
			fmt::format_to(std::back_inserter(line), "call ${} ", domain_to_string(item.scheduled_domain));
			if (!fn_type->debug_info.name.empty()) {
				fmt::format_to(std::back_inserter(line), "<{}> ", fn_type->debug_info.name);
			}
			print_args_to_string(node->call.args.subspan(1), line);
		} break;
		case Node::RELEASE: {
			DomainFlagBits dst_domain = node->release.dst_domain;
			if (node->release.dst_domain == DomainFlagBits::eDevice) {
				dst_domain = item.scheduled_domain;
			}
			DomainFlagBits sched_domain = item.scheduled_domain;

			fmt::format_to(std::back_inserter(line), "release ${} -> ${} ", domain_to_string(sched_domain), domain_to_string(dst_domain));
			print_args_to_string(node->release.src, line);
		} break;
		case Node::ACQUIRE: {
			fmt::format_to(std::back_inserter(line), "acquire<");
			for (size_t i = 0; i < node->acquire.values.size(); i++) {
				if (node->type[i]->is_bufferlike_view()) {
					fmt::format_to(std::back_inserter(line), "buffer");
				} else if (node->type[i]->is_imageview()) {
					fmt::format_to(std::back_inserter(line), "image");
				} else if (node->type[0]->kind == Type::ARRAY_TY) {
					fmt::format_to(std::back_inserter(line), "{}[]", (*node->type[0]->array.T)->is_bufferlike_view() ? "buffer" : "image");
				}
				if (i + 1 < node->acquire.values.size()) {
					fmt::format_to(std::back_inserter(line), ", ");
				}
			}
			fmt::format_to(std::back_inserter(line), ">");
		} break;
		case Node::ACQUIRE_NEXT_IMAGE: {
			fmt::format_to(std::back_inserter(line), "acquire_next_image ");
			print_args_to_string(std::span{ &node->acquire_next_image.swapchain, 1 }, line);
		} break;
		case Node::SLICE: {
			auto axis = node->slice.axis;
			// these must have run by now, so we can just eval
			auto start = *get_value<uint64_t>(node->slice.start);
			auto count = *get_value<uint64_t>(node->slice.count);

			print_args_to_string(std::span{ &node->slice.src, 1 }, line);
			if (start > 0 || count != Range::REMAINING) {
				if (node->slice.axis != 0) {
					if (count > 1) {
						fmt::format_to(std::back_inserter(line), "[{}->{}:{}]", node->slice.axis, start, start + count - 1);
					} else if (axis == Node::NamedAxis::FIELD) {
						fmt::format_to(std::back_inserter(line), ".{}", start);
					} else {
						fmt::format_to(std::back_inserter(line), "[{}->{}]", node->slice.axis, start);
					}
				} else {
					if (count > 1) {
						fmt::format_to(std::back_inserter(line), "[{}:{}]", start, start + count - 1);
					} else {
						fmt::format_to(std::back_inserter(line), "[{}]", start);
					}
				}
			}
		} break;
		case Node::CONVERGE: {
			print_args_to_string(node->converge.diverged.subspan(0, 1), line);
			fmt::format_to(std::back_inserter(line), "{{");
			print_args_to_string(node->converge.diverged.subspan(1), line);
			fmt::format_to(std::back_inserter(line), "}}");
		} break;
		case Node::USE: {
			print_args_to_string(std::span(&node->use.src, 1), line);
			fmt::format_to(std::back_inserter(line), ": {}", Type::to_sv(node->use.access));
		} break;
		case Node::LOGICAL_COPY: {
			print_args_to_string(std::span(&node->logical_copy.src, 1), line);
		} break;
		case Node::COMPILE_PIPELINE: {
			print_args_to_string(std::span(&node->compile_pipeline.src, 1), line);
		} break;
		case Node::ALLOCATE: {
			fmt::format_to(std::back_inserter(line), "allocate ");
			print_args_to_string(std::span(&node->allocate.src, 1), line);
		} break;
		case Node::GET_ALLOCATION_SIZE: {
			print_args_to_string({ &node->get_allocation_size.ptr, 1 }, line);
		} break;
		case Node::GET_CI: {
			print_args_to_string({ &node->get_ci.src, 1 }, line);
		} break;
		}
	}

	std::string format_message(Level level, ScheduledItem& item, std::string err) {
		Node* node = item.execable;
		std::string msg = "";
		fmt::format_to(std::back_inserter(msg), "{}{}: '", format_source_location(node), level == Level::eError ? "error" : "other");
		print_results_to_string(item, msg);
		fmt::format_to(std::back_inserter(msg), " = ");
		if (node->kind == Node::CONSTRUCT) {
			msg += node_to_string(node);
			auto names = arg_names(node->type[0].get());
			msg += print_args_to_string_with_arg_names(names, item.execable->construct.args.subspan(1));
		} else {
			format_args(item, msg);
		}
		msg += err;
		return msg;
	};

	std::string exec_to_string(ScheduledItem& item) {
		std::string line;
		print_results_to_string(item, line);
		fmt::format_to(std::back_inserter(line), " = ");
		format_args(item, line);
		return line;
	}
} // namespace vuk