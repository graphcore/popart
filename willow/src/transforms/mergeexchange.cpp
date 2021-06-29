// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/op/init.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

#include <popart/transforms/mergeexchange.hpp>

namespace popart {

std::size_t MergeExchange::id() { return typeid(MergeExchange).hash_code(); }

void MergeExchange::insertMultiExchange(
    Graph &graph,
    std::vector<ExchangeBaseOp *> exchangeOps) const {
  // Strip topocons that would be blocking
  for (Op *op0 : exchangeOps) {
    for (Op *op1 : exchangeOps) {
      if (graph.topoCons->contains(op0, op1)) {
        logging::transform::info(
            "[MergeExchange] Removed topological constraint {} -> {}",
            op0->debugName(),
            op1->debugName());
        graph.topoCons->remove(op0, op1);
      }
    }
  }

  Op::Settings settings = exchangeOps.back()->settings;
  settings.name.clear();

  std::vector<ExchangeDescriptor> descriptors;
  descriptors.reserve(exchangeOps.size());

  // Add all loads first
  for (ExchangeBaseOp *op : exchangeOps) {
    for (int index = 0; index < op->getNumExchanges(); ++index) {
      auto descriptor = op->getExchangeDescriptor(index);
      if (descriptor.getDirection() == ExchangeDirection::Load) {
        descriptors.push_back(descriptor);
      }
    }
  }

  // Add all stores second
  for (ExchangeBaseOp *op : exchangeOps) {
    for (int index = 0; index < op->getNumExchanges(); ++index) {
      auto descriptor = op->getExchangeDescriptor(index);
      if (descriptor.getDirection() == ExchangeDirection::Store) {
        descriptors.push_back(descriptor);
      }
    }
  }

  // Insert replacement Op
  auto multiExchangeOpUp = std::make_unique<MultiExchangeOp>(
      Onnx::CustomOperators::MultiExchange, settings, descriptors);
  MultiExchangeOp *multiExchangeOp = multiExchangeOpUp.get();
  graph.moveIntoGraph(std::move(multiExchangeOpUp));

  InIndex multiInIdx   = 0;
  OutIndex multiOutIdx = 0;

  auto connectInputsAndOutputs =
      [&multiExchangeOp, &multiInIdx, &multiOutIdx](ExchangeBaseOp *op) {
        auto inputs = op->input->tensorMap();
        for (auto &input : inputs) {
          op->disconnectInTensor(input.first);
          logging::transform::trace(
              "[MergeExchange] Moving op {} input {} to op {} input {}",
              op->debugName(),
              input.first,
              multiExchangeOp->debugName(),
              multiInIdx);
          multiExchangeOp->connectInTensor(multiInIdx++, input.second->id);
        }
        auto outputs = op->output->tensorMap();
        for (auto &output : outputs) {
          op->disconnectOutTensor(output.second);
          logging::transform::trace(
              "[MergeExchange] Moving op {} output {} to op {} output {}",
              op->debugName(),
              output.first,
              multiExchangeOp->debugName(),
              multiOutIdx);
          multiExchangeOp->connectOutTensor(multiOutIdx++, output.second->id);
        }
      };

  // Connect inputs and outputs
  // Add all load inputs / outputs first
  for (ExchangeBaseOp *op : exchangeOps) {
    for (int index = 0; index < op->getNumExchanges(); ++index) {
      auto descriptor = op->getExchangeDescriptor(index);
      if (descriptor.getDirection() == ExchangeDirection::Load) {
        connectInputsAndOutputs(op);
      }
    }
  }

  // Add all store inputs / outputs second
  for (ExchangeBaseOp *op : exchangeOps) {
    for (int index = 0; index < op->getNumExchanges(); ++index) {
      auto descriptor = op->getExchangeDescriptor(index);
      if (descriptor.getDirection() == ExchangeDirection::Store) {
        connectInputsAndOutputs(op);
      }
    }
  }

  for (Op *op : exchangeOps) {
    logging::transform::trace("[MergeExchange] Op {} merged into {}.",
                              op->debugName(),
                              multiExchangeOp->debugName());
    graph.topoCons->transfer(op, multiExchangeOp);
    graph.eraseOp(op->id);
  }
  multiExchangeOp->setup();

  logging::transform::debug("[MergeExchange] MultiExchangeOp {} added.",
                            multiExchangeOp->debugName());
}

void MergeExchange::conditionallyInsertMultiExchange(
    Graph &graph,
    std::vector<ExchangeBaseOp *> exchangeOps,
    bool phaseMerge,
    bool bspMerge) const {
  if (exchangeOps.size() > 1) {
    if (exchangeOps.back()->settings.executionContext ==
            ExecutionContext::AccumulateOuterFragment ||
        phaseMerge ||
        (bspMerge && exchangeOps.back()->hasBatchSerializedPhase())) {
      insertMultiExchange(graph, exchangeOps);
    }
  }
}

bool MergeExchange::apply(Graph &graph) const {
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

  std::vector<ExchangeBaseOp *> exchangeOps;

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
    bool isHost =
        op->isConvertibleTo<HostLoadOp>() || op->isConvertibleTo<HostStoreOp>();

    bool isMulti = op->isConvertibleTo<MultiExchangeOp>();

    bool contextChanged = prevOp && op->settings.executionContext !=
                                        prevOp->settings.executionContext;
    bool bspChanged = prevOp && op->hasBatchSerializedPhase() !=
                                    prevOp->hasBatchSerializedPhase();
    bool isMerge = (seenRemoteLoads && op->isConvertibleTo<RemoteStoreOp>()) ||
                   (seenRemoteStores && op->isConvertibleTo<RemoteLoadOp>());
    bool isAof = (op->settings.executionContext ==
                  ExecutionContext::AccumulateOuterFragment);

    if (contextChanged || !(isInit || isRemote || isHost || isMulti) ||
        (bspChanged && bspMerge) || (inhibitMerging && isAof && isMerge)) {
      conditionallyInsertMultiExchange(
          graph, exchangeOps, phaseMerge, bspMerge);
      exchangeOps.clear();
      seenRemoteLoads  = false;
      seenRemoteStores = false;
    }

    seenRemoteLoads  = seenRemoteLoads || op->isConvertibleTo<RemoteLoadOp>();
    seenRemoteStores = seenRemoteStores || op->isConvertibleTo<RemoteStoreOp>();

    if (isRemote || isHost || isMulti) {
      if (ExchangeBaseOp *exchOp = dynamic_cast<ExchangeBaseOp *>(op)) {
        exchangeOps.push_back(exchOp);
      }
    }
  }
  conditionallyInsertMultiExchange(graph, exchangeOps, phaseMerge, bspMerge);

  return true;
}

namespace {
bool init = Transform::registerTransform(new MergeExchange);
}

} // namespace popart
