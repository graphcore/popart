// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/timepartitionlogger.hpp>
#include <popart/graph.hpp>
#include <popart/op/call.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/subgraphx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

namespace popart {
namespace popx {

SubgraphOpx::SubgraphOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}

std::vector<std::tuple<TensorId, TensorId, bool>>
SubgraphOpx::getInputsToPrepare() const {

  const auto addGraphTask = op_p->getIr().timePartitionLogger().scopedStopwatch(
      "SubgraphOpx::getInputsToPrepare");

  auto &subgraphop = getOp<SubgraphOp>();
  std::vector<std::tuple<TensorId, TensorId, bool>> inputs;
  auto sgInIds = subgraphop.getCalledGraph().getInputIds();
  for (InIndex sgInIdx = 0; sgInIdx < sgInIds.size(); ++sgInIdx) {
    InIndex opInIdx = subgraphop.subgraphInToOpInIndex(sgInIdx);
    if (subgraphop.hasInput(opInIdx)) {
      inputs.emplace_back(
          subgraphop.input->id(opInIdx), sgInIds.at(sgInIdx), false);
    } else {
      inputs.emplace_back("", sgInIds.at(sgInIdx), false);
    }
  }
  return inputs;
}

std::vector<std::tuple<TensorId, TensorId, bool>>
SubgraphOpx::getOutputsToPrepare() const {

  const auto addGraphTask = op_p->getIr().timePartitionLogger().scopedStopwatch(
      "SubgraphOpx::getOutputsToPrepare");

  auto &subgraphop = getOp<SubgraphOp>();
  std::vector<std::tuple<TensorId, TensorId, bool>> outputs;
  for (auto &output : subgraphop.output->tensorMap()) {
    OutIndex outIdx = output.first;

    OutIndex sgOutIdx = subgraphop.opOutToSubgraphOutIndex(outIdx);
    TensorId prepOutId;

    if (sgOutIdx >= 0 &&
        sgOutIdx < subgraphop.getCalledGraph().getOutputIds().size()) {
      prepOutId = subgraphop.getCalledGraph().getOutputIds().at(sgOutIdx);
    }

    bool aliased = false;
    for (auto &input : subgraphop.input->tensorMap()) {
      InIndex inIdx = input.first;
      // Fully aliased & shape did not change
      auto aliasRegions = subgraphop.aliases(inIdx, outIdx);
      bool alias        = aliasRegions.size() == 1 &&
                   aliasRegions.front().nelms() ==
                       subgraphop.output->tensor(outIdx)->info.nelms() &&
                   subgraphop.output->tensor(outIdx)->info.shape() ==
                       subgraphop.input->tensor(inIdx)->info.shape();
      aliased |= alias;
      if (alias) {
        prepOutId = subgraphop.input->id(inIdx);
      }
    }

    TensorId callOutId = subgraphop.output->tensor(outIdx)->id;

    logging::opx::trace(
        "To prepare op output {}, aliased: {}", callOutId, aliased);
    outputs.emplace_back(prepOutId, callOutId, aliased);
  }
  return outputs;
}

} // namespace popx
} // namespace popart
