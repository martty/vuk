#include "vuk/ir/GraphSnapshots.hpp"
#include "vuk/ir/IR.hpp"
#include "EmbeddedHtmlTemplate.hpp"
#include <sstream>
#include <fstream>

#if VUK_OS_WINDOWS
#include <Windows.h>
#include <shellapi.h>
#endif

namespace {
	std::string load_html_template() {
		// Use compile-time embedded HTML template
		return std::string(
			reinterpret_cast<const char*>(vuk::embedded::html_template_data),
			vuk::embedded::html_template_size
		);
	}
}

namespace vuk {
	void GraphSnapshotCollector::add_snapshot(GraphSnapshot snapshot) {
		snapshot.global_index = global_snapshot_counter++;
		snapshots.push_back(std::move(snapshot));
	}

	std::string escape_json(const std::string& str) {
		std::string result;
		result.reserve(str.size());
		for (char c : str) {
			switch (c) {
			case '"': result += "\\\""; break;
			case '\\': result += "\\\\"; break;
			case '\n': result += "\\n"; break;
			case '\r': result += "\\r"; break;
			case '\t': result += "\\t"; break;
			default: result += c;
			}
		}
		return result;
	}

	std::string GraphSnapshotCollector::serialize_snapshots_to_json() const {
		std::stringstream ss;
		ss << "[\n";
		
		for (size_t snap_idx = 0; snap_idx < snapshots.size(); ++snap_idx) {
			const auto& snapshot = snapshots[snap_idx];
			if (snap_idx > 0) ss << ",\n";
			
			ss << "  {\n";
			ss << "    \"globalIndex\": " << snapshot.global_index << ",\n";
			ss << "    \"hierarchicalName\": \"" << escape_json(snapshot.hierarchical_name) << "\",\n";
			ss << "    \"passName\": \"" << escape_json(snapshot.pass_name) << "\",\n";
			ss << "    \"label\": \"" << escape_json(snapshot.label) << "\",\n";
			
			// Nodes
			ss << "    \"nodes\": [\n";
			for (size_t i = 0; i < snapshot.nodes.size(); ++i) {
				const auto& node = snapshot.nodes[i];
				if (i > 0) ss << ",\n";
				
				ss << "      {\n";
				ss << "        \"id\": " << node.id << ",\n";
				ss << "        \"kind\": \"" << escape_json(node.kind) << "\",\n";
				ss << "        \"debugName\": \"" << escape_json(node.debug_name) << "\",\n";
			ss << "        \"computeClass\": \"" << escape_json(node.compute_class) << "\",\n";
			ss << "        \"constantValue\": \"" << escape_json(node.constant_value) << "\",\n";
			ss << "        \"sliceAxis\": " << (int)node.slice_axis << ",\n";
			ss << "        \"sliceFieldName\": \"" << escape_json(node.slice_field_name) << "\",\n";
			ss << "        \"argAccesses\": [";
			for (size_t i = 0; i < node.arg_accesses.size(); i++) {
				if (i > 0) ss << ", ";
				ss << "\"" << escape_json(node.arg_accesses[i]) << "\"";
			}
			ss << "],\n";
			ss << "        \"types\": [";
				for (size_t j = 0; j < node.types.size(); ++j) {
					if (j > 0) ss << ", ";
					ss << "\"" << escape_json(node.types[j]) << "\"";
				}
				ss << "],\n";
				ss << "        \"typeDebugNames\": [";
				for (size_t j = 0; j < node.type_debug_names.size(); ++j) {
					if (j > 0) ss << ", ";
					ss << "\"" << escape_json(node.type_debug_names[j]) << "\"";
				}
				ss << "],\n";
				ss << "        \"args\": [";
				for (size_t j = 0; j < node.args.size(); ++j) {
					if (j > 0) ss << ", ";
					ss << node.args[j];
				}
				ss << "]\n";
				ss << "      }";
			}
			ss << "\n    ],\n";
			
			// Edges
			ss << "    \"edges\": [\n";
			for (size_t i = 0; i < snapshot.edges.size(); ++i) {
				const auto& edge = snapshot.edges[i];
				if (i > 0) ss << ",\n";
				
				ss << "      {\n";
				ss << "        \"from\": " << edge.from << ",\n";
				ss << "        \"fromIndex\": " << edge.from_index << ",\n";
				ss << "        \"to\": " << edge.to << ",\n";
				ss << "        \"toIndex\": " << edge.to_index << "\n";
				ss << "      }";
			}
			ss << "\n    ]\n";
			
			ss << "  }";
		}
		
		ss << "\n]";
		return ss.str();
	}

	std::string GraphSnapshotCollector::generate_d3_html() const {
		std::string json_data = serialize_snapshots_to_json();
		std::string html_template = load_html_template();
		
		// Replace placeholder with actual JSON data
		size_t pos = html_template.find("{{SNAPSHOTS_JSON_DATA}}");
		if (pos != std::string::npos) {
			html_template.replace(pos, 23, json_data);
		}
		
		return html_template;
	}

	std::string GraphSnapshotCollector::generate_html() const {
		return generate_d3_html();
	}

	void GraphSnapshotCollector::write_to_disk(const std::string& filepath) const {
		std::ofstream out(filepath);
		if (!out) {
			return;
		}
		out << generate_html();
		out.close();

#if VUK_OS_WINDOWS
		SHELLEXECUTEINFOA ShExecInfo = { 0 };
		ShExecInfo.cbSize = sizeof(SHELLEXECUTEINFOA);
		ShExecInfo.fMask = 0;
		ShExecInfo.hwnd = NULL;
		ShExecInfo.lpVerb = NULL;
		ShExecInfo.lpFile = filepath.c_str();
		ShExecInfo.lpParameters = "";
		ShExecInfo.lpDirectory = NULL;
		ShExecInfo.nShow = SW_SHOW;
		ShExecInfo.hInstApp = NULL;
		ShellExecuteExA(&ShExecInfo);
#endif
	}
}

