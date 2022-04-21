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

SubgraphOpx::SubgraphOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {}

PreparedTensorInfos SubgraphOpx::getInputsToPrepare() const {

  const auto addGraphTask = op_p->getIr().timePartitionLogger().scopedStopwatch(
      "SubgraphOpx::getInputsToPrepare");

  auto &subgraphop = getOp<SubgraphOp>();
  PreparedTensorInfos inputs;
  auto sgInIds = subgraphop.getCalledGraph().getInputIds();
  for (InIndex sgInIdx = 0; sgInIdx < sgInIds.size(); ++sgInIdx) {
    InIndex opInIdx = subgraphop.subgraphInToOpInIndex(sgInIdx);
    if (subgraphop.hasInput(opInIdx)) {
      inputs.emplace_back(subgraphop.input->id(opInIdx),
                          sgInIds.at(sgInIdx),
                          CanAlias::No,
                          RequireParallelWritable::Yes);
    } else {
      inputs.emplace_back(
          "", sgInIds.at(sgInIdx), CanAlias::No, RequireParallelWritable::Yes);
    }
  }
  return inputs;
}

PreparedTensorInfos SubgraphOpx::getOutputsToPrepare() const {

  const auto addGraphTask = op_p->getIr().timePartitionLogger().scopedStopwatch(
      "SubgraphOpx::getOutputsToPrepare");

  auto &subgraphop = getOp<SubgraphOp>();
  PreparedTensorInfos outputs;
  for (auto &output : subgraphop.output->tensorMap()) {
    OutIndex outIdx = output.first;

    OutIndex sgOutIdx = subgraphop.opOutToSubgraphOutIndex(outIdx);
    TensorId prepOutId;

    if (sgOutIdx >= 0 &&
        sgOutIdx < subgraphop.getCalledGraph().getOutputIds().size()) {
      prepOutId = subgraphop.getCalledGraph().getOutputIds().at(sgOutIdx);
    }

    // Note that this type of aliasing will alias the Op input to the Op output
    // Can still fall back to outplace if initializing the tensor by aliasing
    // fails
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
    outputs.emplace_back(prepOutId,
                         callOutId,
                         aliased ? CanAlias::Yes : CanAlias::No,
                         RequireParallelWritable::Yes);
  }
  return outputs;
}

} // namespace popx
} // namespace popart
