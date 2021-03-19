// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/init.hpp>
#include <popart/op/remote.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

#include <popart/transforms/mergeremote.hpp>

namespace popart {

std::size_t MergeRemote::id() { return typeid(MergeRemote).hash_code(); }

void MergeRemote::insertRemoteExchange(Graph &graph,
                                       std::vector<Op *> remoteOps) const {
  // Strip topocons that would be blocking
  for (Op *op0 : remoteOps) {
    for (Op *op1 : remoteOps) {
      if (graph.topoCons->contains(op0, op1)) {
        logging::transform::info(
            "[RemoteExchange] Removed topological constraint {} -> {}",
            op0->debugName(),
            op1->debugName());
        graph.topoCons->remove(op0, op1);
      }
    }
  }

  Op::Settings settings = remoteOps.back()->settings;
  settings.name.clear();

  std::vector<RemoteBufferId> remoteBufferIds;
  std::vector<std::pair<OptionalVGraphId, TileSet>> vgids;

  // Add all loads first
  for (Op *op : remoteOps) {
    RemoteLoadOp *rl = dynamic_cast<RemoteLoadOp *>(op);
    if (rl) {
      remoteBufferIds.push_back(rl->getRemoteBufferId());
      vgids.push_back({rl->settings.vgraphId, rl->settings.tileSet});
    }
  }

  // Add all stores second
  for (Op *op : remoteOps) {
    RemoteStoreOp *rs = dynamic_cast<RemoteStoreOp *>(op);
    if (rs) {
      remoteBufferIds.push_back(rs->getRemoteBufferId());
      vgids.push_back({rs->settings.vgraphId, rs->settings.tileSet});
    }
  }

  // Insert replacement Op
  auto remoteExchangeOpUp = std::make_unique<RemoteExchangeOp>(
      Onnx::CustomOperators::RemoteExchange, settings, remoteBufferIds, vgids);
  RemoteExchangeOp *remoteExchangeOp = remoteExchangeOpUp.get();
  graph.moveIntoGraph(std::move(remoteExchangeOpUp));

  // Connect all load inputs first
  int loadIdx = 0;
  for (Op *op : remoteOps) {
    RemoteLoadOp *rl = dynamic_cast<RemoteLoadOp *>(op);
    if (rl) {
      remoteExchangeOp->connectInTensor(
          loadIdx,
          rl->input->tensor(RemoteLoadOp::getLocalTensorInIndex())->id);
      remoteExchangeOp->connectInTensor(
          remoteOps.size() + loadIdx,
          rl->input->tensor(RemoteLoadOp::getRemoteBufferOffsetInIndex())->id);
      TensorId outId =
          rl->output->tensor(RemoteLoadOp::getLocalTensorOutIndex())->id;
      rl->disconnectAllInputs();
      rl->disconnectAllOutputs();
      remoteExchangeOp->connectOutTensor(loadIdx, outId);
      ++loadIdx;
    }
  }

  // Connect all store inputs second
  int storeIdx = 0;
  for (Op *op : remoteOps) {
    RemoteStoreOp *rs = dynamic_cast<RemoteStoreOp *>(op);
    if (rs) {
      remoteExchangeOp->connectInTensor(
          loadIdx + storeIdx,
          rs->input->tensor(RemoteStoreOp::getLocalTensorInIndex())->id);
      remoteExchangeOp->connectInTensor(
          remoteOps.size() + loadIdx + storeIdx,
          rs->input->tensor(RemoteStoreOp::getRemoteBufferOffsetInIndex())->id);
      rs->disconnectAllInputs();
      rs->disconnectAllOutputs();
      ++storeIdx;
    }
  }

  for (Op *op : remoteOps) {
    logging::transform::trace("[MergeRemote] Op {} merged into {}.",
                              op->debugName(),
                              remoteExchangeOp->debugName());
    graph.topoCons->transfer(op, remoteExchangeOp);
    graph.eraseOp(op->id);
  }
  remoteExchangeOp->setup();
}

void MergeRemote::conditionallyInsertRemoteExchange(Graph &graph,
                                                    std::vector<Op *> remoteOps,
                                                    bool phaseMerge,
                                                    bool bspMerge) const {
  if (remoteOps.size() > 1) {
    if (remoteOps.back()->settings.executionContext ==
            ExecutionContext::AccumulateOuterFragment ||
        phaseMerge ||
        (bspMerge && remoteOps.back()->hasBatchSerializedPhase())) {
      insertRemoteExchange(graph, remoteOps);
    }
  }
}

bool MergeRemote::apply(Graph &graph) const {
  // Keep a record of which tensors have been copied to/from IO tiles
  std::set<TensorId> copiedTensors;
  std::set<TensorId> processedTensors;

  auto &opts    = graph.getIr().getSessionOptions();
  auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::Yes);

  // Merge batch serialized RemoteLoad/RemoteStore together
  bool bspMerge = opts.batchSerializationSettings.factor > 1 &&
                  (opts.batchSerializationSettings.batchSchedule ==
                       BatchSerializationBatchSchedule::OverlapOnCompute ||
                   opts.batchSerializationSettings.batchSchedule ==
                       BatchSerializationBatchSchedule::OverlapOnIo);

  // Merge execution phase RemoteLoad/RemoteStore together
  bool phaseMerge =
      opts.virtualGraphMode == VirtualGraphMode::ExecutionPhases &&
      opts.executionPhaseSettings.phases > 1 &&
      opts.executionPhaseSettings.schedule ==
          ExecutionPhaseSchedule::BatchClusteredIO;

  std::vector<Op *> remoteOps;

  bool seenRemoteLoads  = false;
  bool seenRemoteStores = false;
  auto aofSchedule      = graph.getIr()
                         .getSessionOptions()
                         .accumulateOuterFragmentSettings.schedule;
  bool inhibitMerging =
      (aofSchedule == AccumulateOuterFragmentSchedule::OverlapMemoryOptimized);

  // For each op (in schedule order)
  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op     = schedule.at(i);
    Op *prevOp = nullptr;
    if (i > 0) {
      prevOp = schedule.at(i - 1);
    }

    bool isInit   = op->isConvertibleTo<InitOp>();
    bool isRemote = op->isConvertibleTo<RemoteLoadOp>() ||
                    op->isConvertibleTo<RemoteStoreOp>();

    bool contextChanged = prevOp && op->settings.executionContext !=
                                        prevOp->settings.executionContext;
    bool bspChanged = prevOp && op->hasBatchSerializedPhase() !=
                                    prevOp->hasBatchSerializedPhase();
    bool isMerge = (seenRemoteLoads && op->isConvertibleTo<RemoteStoreOp>()) ||
                   (seenRemoteStores && op->isConvertibleTo<RemoteLoadOp>());
    bool isAof = (op->settings.executionContext ==
                  ExecutionContext::AccumulateOuterFragment);

    if (contextChanged || !(isInit || isRemote) || (bspChanged && bspMerge) ||
        (inhibitMerging && isAof && isMerge)) {
      conditionallyInsertRemoteExchange(graph, remoteOps, phaseMerge, bspMerge);
      remoteOps.clear();
      seenRemoteLoads  = false;
      seenRemoteStores = false;
    }

    seenRemoteLoads  = seenRemoteLoads || op->isConvertibleTo<RemoteLoadOp>();
    seenRemoteStores = seenRemoteStores || op->isConvertibleTo<RemoteStoreOp>();

    if (isRemote) {
      remoteOps.push_back(op);
    }
  }
  conditionallyInsertRemoteExchange(graph, remoteOps, phaseMerge, bspMerge);

  return true;
}

namespace {
bool init = Transform::registerTransform(new MergeRemote);
}

} // namespace popart
