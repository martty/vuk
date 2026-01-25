#pragma once

#include <string>
#include <vector>
#include <memory>

namespace vuk {
	struct Node;
	struct Ref;

	/// @brief Represents a snapshot of the IR graph at a specific point in time
	/// Captures the state of nodes and edges for visualization and debugging
	struct GraphSnapshot {
		/// @brief Data representation of a node in the snapshot
		struct NodeData {
			uintptr_t id;                         ///< Unique identifier (node pointer)
			std::string kind;                     ///< Node kind (e.g., "CALL", "SLICE")
			std::string debug_name;               ///< Debug name if available
			std::vector<std::string> types;       ///< Type information
			std::vector<std::string> type_debug_names; ///< Debug names for types (from TypeDebugInfo)
			std::vector<uintptr_t> args;          ///< Arguments (node IDs)
			std::string compute_class;            ///< Compute domain (Host/Device/Constant)
			std::string constant_value;           ///< For CONSTANT nodes, the actual value as string
			uint8_t slice_axis = 0;               ///< For SLICE nodes, the axis value
			std::string slice_field_name;         ///< For SLICE nodes with FIELD axis, the field name
			std::vector<std::string> arg_accesses; ///< For CALL nodes with IMBUED_TY args, the access patterns
		};

		/// @brief Data representation of an edge between nodes
		struct EdgeData {
			uintptr_t from;      ///< Source node ID
			size_t from_index;   ///< Output index from source
			uintptr_t to;        ///< Target node ID
			size_t to_index;     ///< Input index to target
		};

		size_t global_index;              ///< Global chronological index across all passes
		std::string hierarchical_name;    ///< Full name: "PassName/Label"
		std::string pass_name;            ///< Name of the pass that created this snapshot
		std::string label;                ///< User-provided or auto-generated label
		std::vector<NodeData> nodes;      ///< All nodes in the graph at this point
		std::vector<EdgeData> edges;      ///< All edges in the graph at this point
	};

	/// @brief Collects snapshots from all passes and generates interactive HTML visualization
	/// Used by RGCImpl to accumulate snapshots throughout compilation pipeline
	class GraphSnapshotCollector {
	public:
		/// @brief Add a snapshot to the collection
		void add_snapshot(GraphSnapshot snapshot);
		
		/// @brief Generate a complete HTML visualization of all snapshots
		/// @return Self-contained HTML string with embedded D3.js and data
		std::string generate_html() const;
		
		/// @brief Write HTML to disk and optionally open in browser
		/// @param filepath Path where HTML should be written
		void write_to_disk(const std::string& filepath) const;
		
		/// @brief Check if any snapshots have been collected
		bool has_snapshots() const { return !snapshots.empty(); }
		
		/// @brief Get all collected snapshots
		const std::vector<GraphSnapshot>& get_snapshots() const { return snapshots; }

	private:
		std::vector<GraphSnapshot> snapshots;
		size_t global_snapshot_counter = 0;

		std::string generate_d3_html() const;
		std::string serialize_snapshots_to_json() const;
	};
}
