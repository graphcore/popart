// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/transforms/stochasticrounding.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>

namespace popart {

std::size_t StochasticRounding::id() {
  return typeid(StochasticRounding).hash_code();
}

bool StochasticRounding::apply(Graph &graph) const {

  // TODO T49922: Use data flow analysis to set these properties in a way that
  // stops weights from drifting inadvertently.

  for (auto op : graph.getIr().getAllOps()) {
    op->settings.stochasticRoundingMethod =
        StochasticRoundingMethod::IdenticalSeeds;
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new StochasticRounding);
}

} // namespace popart
