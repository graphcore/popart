// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/init.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/reshape.hpp>
#include <popart/patterns/loopscanoutpattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/util.hpp>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/scope.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensors.hpp"

namespace popart {

bool LoopScanOutPattern::matches(Op *op) const {
  return op->isConvertibleTo<LoopOp>() &&
         dynamic_cast<LoopOp *>(op)->getNumImplicitScanOutputs() > 0;
}

std::vector<const Tensor *> LoopScanOutPattern::touches(Op *) const {
  return {};
}

bool LoopScanOutPattern::apply(Op *op) const {

  if (LoopOp *loopOp = dynamic_cast<LoopOp *>(op)) {
    logging::pattern::debug(
        "[LoopScanOutPattern] Converting {} implicit scan outputs of {}",
        loopOp->getNumImplicitScanOutputs(),
        loopOp->debugName());

    auto &graph   = loopOp->getGraph();
    auto &ir      = graph.getIr();
    auto settings = loopOp->settings;

    auto numImplicitScanOutputs = loopOp->getNumImplicitScanOutputs();

    auto &oldLoopSubgraph = loopOp->getCalledGraph();
    auto &newLoopSubgraph = ir.createGraph(ir.createUniqueSubgraphId({"loop"}));

    newLoopSubgraph.copyFrom(oldLoopSubgraph);
    auto loopSubgraphScope = newLoopSubgraph.getScope();

    loopOp->setCalledGraph(newLoopSubgraph);

    // Create InitOps for the scan outputs, update scan outputs
    for (int i = 0; i < numImplicitScanOutputs; ++i) {
      OutIndex loopScanOutIdx =
          loopOp->output->n() - numImplicitScanOutputs + i;
      OutIndex loopScanSgOutIdx =
          loopOp->opOutToSubgraphOutIndex(loopScanOutIdx);
      InIndex loopScanInIdx = loopScanOutIdx + 2;

      logging::trace(
          "[LoopScanOutPattern] Converting implicit scan output {}->({}->){}",
          loopScanInIdx,
          loopScanSgOutIdx,
          loopScanOutIdx);

      TensorId loopScanOutId = loopOp->outId(loopScanOutIdx);
      Tensor *loopScanOut    = graph.getTensors().get(loopScanOutId);

      TensorId loopScanSgOutId =
          newLoopSubgraph.getOutputIds().at(loopScanSgOutIdx);
      Tensor *loopScanSgOut = newLoopSubgraph.getTensors().get(loopScanSgOutId);
      TensorId loopSgItId   = newLoopSubgraph.getInputIds().at(
          LoopOp::getLoopGraphIterationInIndex());

      TensorId initId          = ir.createIntermediateTensorId(loopScanOutId);
      TensorId loopSgInitId    = addScope(newLoopSubgraph, initId);
      TensorId loopSgUpdatedId = ir.createIntermediateTensorId(loopScanSgOutId);

      // InitOp as a "producer" Op
      auto initOpUp  = std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                               loopScanOut->info,
                                               loopScanOut->tensorType(),
                                               InitType::NoInit,
                                               settings);
      InitOp *initOp = initOpUp.get();
      graph.moveIntoGraph(std::move(initOpUp));
      initOp->createAndConnectOutTensor(InitOp::getOutIndex(), initId);
      initOp->setup();

      loopOp->addLoopInput(loopScanInIdx, initId, loopSgInitId, false);
      auto loopSgInit = newLoopSubgraph.getTensors().get(loopSgInitId);

      Op::Settings updateSettings = settings;
      if (loopScanSgOut->hasProducer()) {
        updateSettings = loopScanSgOut->getProducer()->settings;
      }
      updateSettings.scope = loopSubgraphScope;

      std::vector<int64_t> axesv(1, 0);
      std::vector<int64_t> sizesv(1, 1);

      // Reshape the original output
      auto loopScanSgOutTmpId = ir.createIntermediateTensorId(loopScanSgOutId);
      auto loopScanSgOutTmpShape = loopScanSgOut->info.shape();
      loopScanSgOutTmpShape.insert(loopScanSgOutTmpShape.begin(), 1);
      logging::pattern::trace("[LoopScanOutPattern] Reshaping {} {} -> {} {}",
                              loopScanSgOutId,
                              loopScanSgOut->info.shape(),
                              loopScanSgOutTmpId,
                              loopScanSgOutTmpShape);
      std::unique_ptr<ReshapeOp> reshapeOpUp =
          std::make_unique<ReshapeOp>(Onnx::AiOnnx::OpSet11::Reshape,
                                      loopScanSgOutTmpShape,
                                      updateSettings);
      Op *reshapeOp = reshapeOpUp.get();
      newLoopSubgraph.moveIntoGraph(std::move(reshapeOpUp));

      reshapeOp->connectInTensor(ReshapeOp::getInIndex(), loopScanSgOutId);
      reshapeOp->createAndConnectOutTensor(ReshapeOp::getOutIndex(),
                                           loopScanSgOutTmpId);
      reshapeOp->setup();

      // Update the scan output
      std::unique_ptr<DynamicUpdateOp> updateOpUp =
          std::make_unique<DynamicUpdateOp>(
              Onnx::CustomOperators::DynamicUpdate_1,
              axesv,
              sizesv,
              true,
              updateSettings);
      DynamicUpdateOp *updateOp = updateOpUp.get();
      newLoopSubgraph.moveIntoGraph(std::move(updateOpUp));

      updateOp->connectInTensor(DynamicUpdateOp::getInIndex(),
                                loopScanSgOutTmpId);
      updateOp->connectInTensor(DynamicUpdateOp::getIndexInIndex(), loopSgItId);
      updateOp->connectInTensor(DynamicUpdateOp::getUpdateInIndex(),
                                loopSgInitId);
      updateOp->createAndConnectOutTensor(DynamicUpdateOp::getOutIndex(),
                                          loopSgUpdatedId);
      updateOp->setup();
      auto loopSgUpdated = newLoopSubgraph.getTensors().get(loopSgUpdatedId);

      logging::pattern::trace("[LoopScanOutPattern] LoopOp implicit scan "
                              "output {} ({}) {} ({}) -> {} ({})",
                              loopScanSgOutId,
                              loopScanSgOut->info,
                              loopSgInitId,
                              loopSgInit->info,
                              loopSgUpdatedId,
                              loopSgUpdated->info);

      loopOp->addLoopOutput(
          loopScanOutIdx, loopScanOutId, loopSgUpdatedId, true);
    }

    // The implicit outputs are now explicit input/output pairs
    loopOp->setNumImplicitScanOutputs(0);

    if (oldLoopSubgraph.getCallSiteOps().empty()) {
      ir.removeGraph(oldLoopSubgraph.id);
    }

    loopOp->setup();

    for (auto &output : loopOp->output->tensorMap()) {
      logging::pattern::trace(
          "[LoopScanOutPattern] LoopOp output {}: {} (info {})",
          output.first,
          output.second->id,
          output.second->info);
    }

    return true;
  } else {
    throw error("[LoopScanOutPattern] Op {} is not a LoopOp.", op->debugName());
  }
  return false;
}

namespace {
static PatternCreator<LoopScanOutPattern>
    LoopScanOutPattern("LoopScanOut",
                       /* enabled = */ true,
                       /* mandatory = */ true);

}

} // namespace popart
