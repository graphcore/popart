// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <typeinfo>
#include <vector>
#include <popart/analysis/replicaequal/replicaequalanalysis.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/transforms/stochasticrounding.hpp>

#include "popart/basicoptionals.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensorindex.hpp"
#include "popart/transforms/transform.hpp"
#include "popart/util.hpp"

namespace popart {

std::size_t StochasticRounding::id() {
  return typeid(StochasticRounding).hash_code();
}

bool StochasticRounding::apply(Graph &graph) const {

  // TODO(T48752): Remove _enableRngStateManagement (always enable it).
  if (graph.getIr().getSessionOptions()._enableRngStateManagement) {
    ReplicaEqualAnalysis analysis{graph.getIr()};
    analysis.apply();

    for (auto op : graph.getIr().getAllOps()) {
      // Check if all outputs of the Op are replica equal.
      auto outputTensors = op->output->tensorMap();
      bool identical =
          std::all_of(outputTensors.begin(), outputTensors.end(), [&](auto &t) {
            return analysis.isOpOutputEqual(op, t.first);
          });

      // Use this to determine what random state to use.
      auto stochasticRoundingMethod =
          (identical) ? StochasticRoundingMethod::IdenticalSeeds
                      : StochasticRoundingMethod::DifferingSeeds;
      op->setStochasticRoundingMethod(stochasticRoundingMethod);

      // Log our choice.
      logging::debug("[StochasticRounding] {} will use '{}'",
                     op->str(),
                     stochasticRoundingMethod);
    }
  } else {
    // TODO(T48752): Remove _enableRngStateManagement (always enable it).
    for (auto op : graph.getIr().getAllOps()) {
      op->settings.stochasticRoundingMethod =
          StochasticRoundingMethod::IdenticalSeeds;
    }
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new StochasticRounding);
}

} // namespace popart
