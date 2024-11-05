#include <fstream>
#include <sstream>
#include <string>
#ifdef _WINDOWS
#include <Windows.h>
#include <shellapi.h>
#endif

#include "vuk/IR.hpp"
#include <fmt/format.h>

#include "GraphDumper.hpp"

namespace vuk {
	struct GraphDumperImpl {
		bool enable;
		std::stringstream ss;
		std::string current_cluster;
		std::string last_cluster;

		void begin_graph(std::string label) {
			ss << "digraph vuk {\n";
			ss << "graph[rankdir=\"TB\", newrank = false, compound = true]\nnode[shape = rectangle width = 0 height = 0 margin = 0]\n";
			ss << "label = \"" << label << "\";\n";
		}

		void begin_cluster(std::string label) {
			current_cluster = label;
			ss << "subgraph cluster_" << label << "{\n";
			ss << "cl_" << label << "[shape = point style = invis];\n";
			ss << "label = \"" << label << "\";\n";
		}

		void end_cluster() {
			ss << "}\n";
		}

		void next_cluster(std::string label) {
			end_cluster();
			last_cluster = current_cluster;
			begin_cluster(label);
			if (!last_cluster.empty()) {
				ss << "cl_" << last_cluster << "->" << "cl_" << label << "[lhead = cluster_" << label << ", ltail = cluster_" << last_cluster << ", minlen = 1];\n";
			}
		}

		void dump_node(const Node* node, bool bridge_splices, bool bridge_slices) {
			if (node->kind == Node::GARBAGE) {
				return;
			}
			if (node->kind == Node::CONSTANT) {
				if (node->type[0]->kind == Type::INTEGER_TY || node->type[0]->kind == Type::MEMORY_TY || node->type[0]->kind == Type::OPAQUE_FN_TY) {
					return;
				}
			}
			if (node->kind == Node::PLACEHOLDER || (bridge_splices && node->kind == Node::SPLICE) || (bridge_slices && node->kind == Node::SLICE)) {
				return;
			}

			auto arg_count = node->generic_node.arg_count == (uint8_t)~0u ? node->variable_node.args.size() : node->generic_node.arg_count;
			auto result_count = node->type.size();
			ss << current_cluster << uintptr_t(node) << " [label=<\n";
			ss << "<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\"";
			ss << "><TR>\n ";

			for (size_t i = 0; i < result_count; i++) {
				ss << "<TD PORT= \"r" << i << "\">";
				ss << "<FONT FACE=\"Courier New\">";
				if (node->debug_info && node->debug_info->result_names.size() > i) {
					ss << "%" << node->debug_info->result_names[i] << ":";
				}
				ss << Type::to_string(node->type[i].get());
				ss << "</FONT>";
				ss << "</TD>";
			}
			ss << "<TD>";
			ss << node->kind_to_sv();
			if (node->kind == Node::CALL) {
				auto opaque_fn_ty = node->call.args[0].type()->opaque_fn;

				if (!node->call.args[0].type()->debug_info.name.empty()) {
					ss << " <B>";
					ss << node->call.args[0].type()->debug_info.name;
					ss << "</B>";
				}
			}
			ss << "</TD>";

			for (size_t i = 0; i < arg_count; i++) {
				auto arg = node->generic_node.arg_count != (uint8_t)~0u ? node->fixed_node.args[i] : node->variable_node.args[i];

				ss << "<TD PORT= \"a" << i << "\">";
				if (arg.node->kind == Node::CONSTANT) {
					if (arg.type()->kind == Type::INTEGER_TY) {
						if (arg.type()->integer.width == 32) {
							ss << constant<uint32_t>(arg);
						} else {
							ss << constant<uint64_t>(arg);
						}
					} else if (arg.type()->kind == Type::MEMORY_TY) {
						ss << "&lt;mem&gt;";
					}
				} else if (arg.node->kind == Node::PLACEHOLDER) {
					ss << "?";
				} else {
					if (node->kind == Node::CALL) {
						if (i == 0) { // don't render the fn parm
							continue;
						}
						auto fn_type = node->call.args[0].type();
						size_t first_parm = fn_type->kind == Type::OPAQUE_FN_TY ? 1 : 4;
						auto& args = fn_type->kind == Type::OPAQUE_FN_TY ? fn_type->opaque_fn.args : fn_type->shader_fn.args;
						if (args[i - first_parm]->kind == Type::IMBUED_TY) {
							ss << "<FONT FACE=\"Courier New\">";
							ss << ":" << Type::to_sv(args[i - first_parm]->imbued.access);
							ss << "</FONT>";
						}
					} else {
						ss << "&bull;";
					}
				}
				ss << "</TD>";
			}

			ss << "</TR></TABLE>>];\n";

			// connections
			for (size_t i = 0; i < arg_count; i++) {
				auto arg = node->generic_node.arg_count != (uint8_t)~0u ? node->fixed_node.args[i] : node->variable_node.args[i];
				if (arg.node->kind == Node::CONSTANT) {
					if (arg.type()->kind == Type::INTEGER_TY || arg.type()->kind == Type::MEMORY_TY || arg.type()->kind == Type::OPAQUE_FN_TY) {
						continue;
					}
				}
				if (arg.node->kind == Node::PLACEHOLDER) {
					continue;
				}
				if (bridge_splices && arg.node->kind == Node::SPLICE && arg.node->splice.rel_acq &&
				    arg.node->splice.rel_acq->status == Signal::Status::eDisarmed) { // bridge splices
					auto bridged_arg = arg.node->splice.src[arg.index];
					ss << current_cluster << uintptr_t(bridged_arg.node) << " :r" << bridged_arg.index << " -> " << current_cluster << uintptr_t(node) << " :a" << i
					   << " :n [color=red]\n";
				} else if (bridge_splices && arg.node->kind == Node::SPLICE && arg.node->splice.rel_acq) {
					ss << current_cluster << "EXT\n";
					ss << current_cluster << "EXT -> " << current_cluster << uintptr_t(node) << " :a" << i << " :n [color=red]\n";
				} else if (bridge_splices && arg.node->kind == Node::SPLICE) { // disabled
					auto bridged_arg = arg.node->splice.src[arg.index];
					ss << current_cluster << uintptr_t(bridged_arg.node) << " :r" << bridged_arg.index << " -> " << current_cluster << uintptr_t(node) << " :a" << i
					   << " :n [color=blue]\n";
				} else if (bridge_slices && arg.node->kind == Node::SLICE) { // bridge slices
					auto bridged_arg = arg.node->slice.image;
					if (bridged_arg.node->kind == Node::SPLICE) {
						bridged_arg = bridged_arg.node->splice.src[arg.index];
					}
					Subrange::Image r = { constant<uint32_t>(arg.node->slice.base_level),
						                    constant<uint32_t>(arg.node->slice.level_count),
						                    constant<uint32_t>(arg.node->slice.base_layer),
						                    constant<uint32_t>(arg.node->slice.layer_count) };
					ss << current_cluster << uintptr_t(bridged_arg.node) << " :r" << bridged_arg.index << " -> " << current_cluster << uintptr_t(node) << " :a" << i
					   << " :n [color=green, label=\"";
					if (r.base_level > 0 || r.level_count != VK_REMAINING_MIP_LEVELS) {
						ss << fmt::format("[m{}:{}]", r.base_level, r.base_level + r.level_count - 1);
					}
					if (r.base_layer > 0 || r.layer_count != VK_REMAINING_ARRAY_LAYERS) {
						ss << fmt::format("[l{}:{}]", r.base_layer, r.base_layer + r.layer_count - 1);
					}
					ss << "\"]\n";
				} else {
					ss << current_cluster << uintptr_t(arg.node) << " :r" << arg.index << " -> " << current_cluster << uintptr_t(node) << " :a" << i << " :n\n";
				}
			}
		}

		void end_graph() {
			ss << "}\n";
			std::string temp_file = std::tmpnam(nullptr);
			std::ofstream out(temp_file);
			auto str = ss.str();
			out << str;
			out.close();
			ss.str("");
			ss.clear();
#ifdef _WINDOWS
			std::string png_temp_file = std::tmpnam(nullptr);
			png_temp_file += ".png";
			auto cmd = std::string("\"C:\\Program Files\\Graphviz\\bin\\dot.exe\" -Tpng -o") + png_temp_file + " " + temp_file + "  >nul 2>nul";
			auto res = std::system(cmd.c_str());
			SHELLEXECUTEINFO ShExecInfo = { 0 };
			ShExecInfo.cbSize = sizeof(SHELLEXECUTEINFO);
			ShExecInfo.fMask = 0; // SEE_MASK_NOCLOSEPROCESS;
			ShExecInfo.hwnd = NULL;
			ShExecInfo.lpVerb = NULL;
			ShExecInfo.lpFile = png_temp_file.c_str();
			ShExecInfo.lpParameters = "";
			ShExecInfo.lpDirectory = NULL;
			ShExecInfo.nShow = SW_SHOW;
			ShExecInfo.hInstApp = NULL;
			ShellExecuteExA(&ShExecInfo);
			Sleep(1000); // bit of schlep otherwise the window might not open
#endif
		}
	};

	thread_local GraphDumperImpl dumper;

	void GraphDumper::begin_graph(bool enable, std::string label) {
		dumper.enable = enable;
		if (dumper.enable) {
			dumper.begin_graph(label);
		}
	}

	void GraphDumper::begin_cluster(std::string label) {
		if (dumper.enable) {
			dumper.begin_cluster(label);
		}
	}

	void GraphDumper::next_cluster(std::string label) {
		if (dumper.enable) {
			dumper.next_cluster(label);
		}
	}

	void GraphDumper::next_cluster(std::string prev, std::string label) {
		if (dumper.enable) {
			dumper.current_cluster = prev;
			dumper.next_cluster(label);
		}
	}

	void GraphDumper::dump_node(const struct Node* node, bool bridge_splices, bool bridge_slices) {
		if (dumper.enable) {
			dumper.dump_node(node, bridge_splices, bridge_slices);
		}
	}

	void GraphDumper::end_cluster() {
		if (dumper.enable) {
			dumper.end_cluster();
		}
	}

	void GraphDumper::end_graph() {
		if (dumper.enable) {
			dumper.end_graph();
		}
	}
} // namespace vuk