// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/slice.hpp>
#include <popart/opsharding.hpp>
#include <popart/shardingplan.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/decomposeloops.hpp>
#include <popart/transforms/overlapio.hpp>

namespace popart {

std::size_t OverlapIO::id() { return typeid(OverlapIO).hash_code(); }

namespace {

std::set<ExchangeStrategy> overlapIORequired(Ir &ir) {
  std::set<ExchangeStrategy> required;

  auto hostLoadTensors = ir.getHostLoadTensors();
  for (auto &loadTensor : hostLoadTensors) {
    for (Tensor *loadOutTensor : loadTensor.second) {
      auto strategy =
          ir.getTensor(loadTensor.first)->inputSettings.exchangeStrategy();
      if (loadOutTensor->hasProducer()) {
        auto op     = loadOutTensor->getProducer();
        auto loadOp = dynamic_cast<HostLoadOp *>(op);
        if (loadOp && loadOp->settings.tileSet == TileSet::IO) {
          logging::transform::trace(
              "[OverlapIO] Op {} {}", op->debugName(), strategy);
          required.insert(strategy);
        }
      }
    }
  }

  auto hostStoreTensors = ir.getHostStoreTensors();
  for (auto &storeTensor : hostStoreTensors) {
    auto art = ir.getDataFlow().getAnchorReturnTypeMap().at(storeTensor.first);
    for (Tensor *storeInTensor : storeTensor.second) {
      auto ops = storeInTensor->consumers.getOps();
      for (auto op : ops) {
        auto storeOp = dynamic_cast<HostStoreOp *>(op);
        if (storeOp && storeOp->settings.tileSet == TileSet::IO) {
          logging::transform::trace(
              "[OverlapIO] Op {} {}", op->debugName(), art.exchangeStrategy());
          required.insert(art.exchangeStrategy());
        }
      }
    }
  }

  return required;
}

} // namespace

bool OverlapIO::apply(Graph &graph) const {
  logging::transform::debug("[OverlapIO] Started.");

  auto &ir             = graph.getIr();
  Graph &innerSubgraph = MainLoops::getInnerLoopSubgraph(ir);
  Graph &outerSubgraph = MainLoops::getOuterLoopSubgraph(ir);

  logging::transform::debug("[OverlapIO] Inner loop subgraph: {}",
                            innerSubgraph.id.str());
  logging::transform::debug("[OverlapIO] Outer loop subgraph: {}",
                            outerSubgraph.id.str());

  DecomposeLoops decompose;

  auto overlapRequired = overlapIORequired(ir);

  bool overlapInnerLoop =
      overlapRequired.find(ExchangeStrategy::OverlapInnerLoop) !=
      overlapRequired.end();
  bool overlapLoops = overlapRequired.find(ExchangeStrategy::OverlapLoops) !=
                      overlapRequired.end();
  bool overlapStep = overlapRequired.find(ExchangeStrategy::OverlapStep) !=
                     overlapRequired.end();

  bool overlapOuter =
      (overlapLoops || overlapStep) &&
      innerSubgraph.getGraphId() != outerSubgraph.getGraphId() &&
      outerSubgraph.getGraphId() != ir.getMainGraph().getGraphId();

  bool overlapInner =
      (overlapInnerLoop || overlapLoops || overlapStep) &&
      outerSubgraph.getGraphId() != ir.getMainGraph().getGraphId();

  if (overlapOuter || overlapInner) {
    logging::transform::debug("[OverlapIO] Overlap IO is required.");

    ir.verifyTensorInfos();

    std::set<ExchangeStrategy> computeLikeExchangeStrategies = {
        ExchangeStrategy::JustInTime};

    if (overlapInner) {
      logging::transform::debug("[OverlapIO] Decomposing inner main loop {}.",
                                innerSubgraph.id.str());
      for (auto *callSiteOp : innerSubgraph.getCallSiteOps()) {
        if (auto loopOp = dynamic_cast<LoopOp *>(callSiteOp)) {
          decompose.decomposeLoop(
              loopOp->getGraph(),
              loopOp,
              DecomposeLoopOverlapModel(
                  overlapOuter ? DecomposeTopoConLevel::None
                               : DecomposeTopoConLevel::Full,
                  DecomposeTopoConLevel::Full,
                  overlapOuter ? DecomposeTopoConLevel::None
                               : DecomposeTopoConLevel::Full,
                  computeLikeExchangeStrategies));
          ir.removeIsolatedTensors(true);
          computeLikeExchangeStrategies.insert(
              ExchangeStrategy::OverlapInnerLoop);
        }
      }
      ir.verifyTensorInfos();
    }

    if (overlapOuter) {
      logging::transform::debug("[OverlapIO] Decomposing outer main loop {}.",
                                outerSubgraph.id.str());
      for (auto *callSiteOp : outerSubgraph.getCallSiteOps()) {
        if (auto loopOp = dynamic_cast<LoopOp *>(callSiteOp)) {
          decompose.decomposeLoop(
              loopOp->getGraph(),
              loopOp,
              DecomposeLoopOverlapModel(DecomposeTopoConLevel::Full,
                                        DecomposeTopoConLevel::Full,
                                        DecomposeTopoConLevel::Full,
                                        computeLikeExchangeStrategies));
          ir.removeIsolatedTensors(true);
        }
      }
      ir.verifyTensorInfos();
    }

  } else {
    logging::transform::debug("[OverlapIO] Overlap IO is not required.");
  }

  logging::transform::debug("[OverlapIO] Done.");
  return true;
}

namespace {
// OverlapIO
bool init = Transform::registerTransform(new OverlapIO());
} // namespace

} // namespace popart
