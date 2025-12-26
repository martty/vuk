#pragma once

#include "vuk/ir/IR.hpp"
#include "vuk/ir/IRPass.hpp"

#include <string>
#include <vector>

namespace vuk {
	struct IRAnalysisPass : IRPass {
		using IRPass::IRPass;

		bool node_set_modified() override {
			return false;
		}
		bool node_connections_modified() override {
			return false;
		}
	};

	struct link_building : IRPass {
		using IRPass::IRPass;

		Result<void> implicit_linking(std::pmr::vector<Node*>& nodes);
		Result<void> operator()() override;
	};

	struct reify_inference : IRPass {
		using IRPass::IRPass;

		Result<void> operator()() override;
	};
	struct constant_folding : IRPass {
		using IRPass::IRPass;

		Result<void> operator()() override;
	};
	struct forced_convergence : IRPass {
		using IRPass::IRPass;

		Result<void> operator()() override;
	};
	struct validate_duplicated_resource_ref : IRAnalysisPass {
		using IRAnalysisPass::IRAnalysisPass;

		Result<void> operator()() override;
	};
	struct linearization : IRPass {
		using IRPass::IRPass;

		Result<void> operator()() override;
	};
} // namespace vuk