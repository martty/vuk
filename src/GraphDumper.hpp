#include <string>
#include <memory>

namespace vuk {
	struct GraphDumper {
		static void begin_graph(bool enable, std::string label);

		static void begin_cluster(std::string label);

		static void next_cluster(std::string label);
		static void next_cluster(std::string prev, std::string label);

		static void end_cluster();

		static void dump_node(const struct Node*, bool bridge_splices = true, bool bridge_slices = true);

		template<class T>
		static void dump_graph_op(const T& nodes, bool bridge_splices = true, bool bridge_slices = true) {
			for (auto& node : nodes) {
				dump_node(&node, bridge_splices, bridge_slices);
			}
		}

		template<class T>
		static void dump_graph(const T& nodes, bool bridge_splices = true, bool bridge_slices = true) {
			for (auto& node : nodes) {
				dump_node(node, bridge_splices, bridge_slices);
			}
		}

		static void end_graph();
	};
} // namespace vuk