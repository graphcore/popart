// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
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

using ExchangeOps = std::vector<std::pair<int, ExchangeBaseOp *>>;

namespace {
bool hasDataDependency(Op *const op,
                       const ExchangeOps &exchangeOps,
                       const std::map<OpId, int> &opToPosition) {
  if (exchangeOps.empty()) {
    return false;
  }

  std::set<OpId> exchangeOpIds;
  for (auto &exchangeOp : exchangeOps) {
    exchangeOpIds.insert(exchangeOp.second->id);
  }

  std::vector<graphutils::TensorAndCallStack> inputs;

  for (auto input : op->input->tensorMap()) {
    inputs.push_back({input.second, {}});
  }

  // Walk back from the current exchange Op and ensure we do not encounter
  // any other exchange Op currently scheduled for merging.
  //
  //  InitOp
  //     |
  //  RemoteLoad                        } even if scheduled adjacently,
  //     |         <- data dependency   } cannot be merged to
  //  RemoteStore                       } MultiExchangeOp

  bool dataDependency = false;
  graphutils::traverse(
      inputs,
      [&exchangeOpIds, &dataDependency](Tensor *t) {
        if (t->hasProducer()) {
          if (exchangeOpIds.find(t->getProducer()->id) != exchangeOpIds.end()) {
            // The current exchange Op depends on data/tensors
            // from a previous exchange Op, and so cannot be merged
            dataDependency = true;
            return false;
          }
        }
        return true;
      },
      [&exchangeOps, &opToPosition, &op](Op *top, Tensor *t0, Tensor *t1) {
        if (op->getGraph().id != top->getGraph().id) {
          return true;
        } else {
          auto it = opToPosition.find(top->id);
          if (it != opToPosition.end()) {
            return exchangeOps.front().first < it->second;
          }
        }
        return false;
      },
      graphutils::TraversalType::DepthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::Backward);

  return dataDependency;
}

OpsBeforeKey initOpConstraints(std::vector<Op *> &initOps,
                               ExchangeOps exchangeOps) {
  // Ensure the scheduler allows InitOps to be scheduled before
  // RemoteLoad/HostLoad/RemoteStore/HostStore/MultiExchange
  //
  //  InitOp0                   valid if both InitOp0 and InitOp1 can be
  //     |         InitOp1      scheduled before RemoteLoad0 and RemoteLoad1
  //     |            |
  //  RemoteLoad0     |         } mergeable to MultiExchangeOp
  //               RemoteLoad1  }
  OpsBeforeKey beforeKeys;
  for (auto &exchangeOp : exchangeOps) {
    beforeKeys.insert({exchangeOp.second, initOps});
  }
  return beforeKeys;
}

bool isMergeableOp(Op *const op) {
  return op->isConvertibleTo<RemoteLoadOp>() ||
         op->isConvertibleTo<RemoteStoreOp>() ||
         op->isConvertibleTo<HostLoadOp>() ||
         op->isConvertibleTo<HostStoreOp>() ||
         op->isConvertibleTo<MultiExchangeOp>();
}
} // namespace

std::size_t MergeExchange::id() { return typeid(MergeExchange).hash_code(); }

Op *MergeExchange::insertMultiExchange(Graph &graph,
                                       ExchangeOps exchangeOps) const {

  logging::transform::info("[MergeExchange] inserting multi exchange");

  // Strip topocons that would be blocking
  for (auto &op0 : exchangeOps) {
    for (auto &op1 : exchangeOps) {
      if (graph.topoCons->contains(op0.second, op1.second)) {
        graph.topoCons->remove(op0.second, op1.second);
      }
    }
  }

  Op::Settings settings = exchangeOps.back().second->settings;
  settings.name.clear();

  std::vector<ExchangeDescriptor> descriptors;
  descriptors.reserve(exchangeOps.size());

  for (auto &op : exchangeOps) {
    for (int index = 0; index < op.second->getNumExchanges(); ++index) {
      auto descriptor = op.second->getExchangeDescriptor(index);
      descriptors.push_back(descriptor);
    }
  }

  // Insert replacement Op
  auto multiExchangeOpUp = std::make_unique<MultiExchangeOp>(
      Onnx::CustomOperators::MultiExchange, settings, descriptors);
  MultiExchangeOp *multiExchangeOp = multiExchangeOpUp.get();
  graph.moveIntoGraph(std::move(multiExchangeOpUp));

  multiExchangeOp->scheduledPreLoss =
      exchangeOps.back().second->scheduledPreLoss;
  multiExchangeOp->toLoss   = exchangeOps.back().second->toLoss;
  multiExchangeOp->fromLoss = exchangeOps.back().second->fromLoss;

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
  for (auto &op : exchangeOps) {
    for (int index = 0; index < op.second->getNumExchanges(); ++index) {
      auto descriptor = op.second->getExchangeDescriptor(index);
      connectInputsAndOutputs(op.second);
    }
  }

  for (auto &op : exchangeOps) {
    logging::transform::trace("[MergeExchange] Op {} merged into {}.",
                              op.second->debugName(),
                              multiExchangeOp->debugName());
    graph.topoCons->transfer(op.second, multiExchangeOp);
    graph.eraseOp(op.second->id);
  }
  multiExchangeOp->setup();

  logging::transform::debug("[MergeExchange] MultiExchangeOp {} added.",
                            multiExchangeOp->debugName());
  return multiExchangeOp;
}

Op *MergeExchange::conditionallyInsertMultiExchange(
    Graph &graph,
    ExchangeOps exchangeOps,
    const OpsBeforeKey &keys) const {

  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {
    std::stringstream ss;
    ss << std::endl;
    for (auto &opPair : keys) {
      ss << "    ";
      ss << opPair.first->debugName() << ", befores: ";
      std::vector<std::string> opNames(opPair.second.size());
      auto &ops = opPair.second;
      std::transform(ops.begin(), ops.end(), opNames.begin(), [](const Op *op) {
        return op->debugName();
      });
      ss << logging::join(opNames.begin(), opNames.end(), ", ");
      ss << std::endl;
    }
    logging::transform::trace(
        "[MergeExchange::conditionallyInsertMultiExchange] Requiring "
        "topological order for merge: {}",
        ss.str());
  }

  if (exchangeOps.size() > 1 && graph.isSchedulable(keys, true)) {
    auto op = insertMultiExchange(graph, exchangeOps);
    logging::transform::trace(
        "[MergeExchange::conditionallyInsertMultiExchange] Inserted merged "
        "MultiExchangeOp {} for {} candidates.",
        op->debugName(),
        exchangeOps.size());
    return op;
  } else {
    logging::transform::trace(
        "[MergeExchange::conditionallyInsertMultiExchange] Could not insert "
        "merged MultiExchangeOp for {} candidates.",
        exchangeOps.size());
  }
  return nullptr;
}

bool MergeExchange::apply(Graph &graph) const {
  applyToOps(graph, {});
  return true;
}

std::vector<Op *>
MergeExchange::applyToOps(Graph &graph,
                          const std::set<OpId> include_ops) const {
  std::vector<Op *> createdOps;

  auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::Yes);

  OpsBeforeKey beforeKeys;

  // Create opid -> schedule position map
  std::map<OpId, int> opToPosition;
  int i = 0;
  for (Op *op : schedule) {
    opToPosition[op->id] = i++;
  }

  ExchangeOps exchangeOps;
  std::vector<Op *> initOps;

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

    if (!include_ops.empty() && include_ops.find(op->id) == include_ops.end()) {
      // Only merge ops in include_ops
      continue;
    }

    bool isInit      = op->isConvertibleTo<InitOp>();
    bool isMergeable = isMergeableOp(op);

    if (isInit) {
      initOps.push_back(op);
    }

    bool contextChanged = prevOp && op->settings.executionContext !=
                                        prevOp->settings.executionContext;
    bool bspChanged = prevOp && op->hasBatchSerializedPhase() !=
                                    prevOp->hasBatchSerializedPhase();
    bool isMerge = (seenRemoteLoads && op->isConvertibleTo<RemoteStoreOp>()) ||
                   (seenRemoteStores && op->isConvertibleTo<RemoteLoadOp>());
    bool isAof = (op->settings.executionContext ==
                  ExecutionContext::AccumulateOuterFragment);

    bool dataDependency = (isInit || isMergeable) &&
                          hasDataDependency(op, exchangeOps, opToPosition);

    if (contextChanged || dataDependency || !(isInit || isMergeable) ||
        bspChanged || (inhibitMerging && isAof && isMerge)) {
      auto multiOp = conditionallyInsertMultiExchange(
          graph, exchangeOps, initOpConstraints(initOps, exchangeOps));
      if (multiOp != nullptr) {
        createdOps.push_back(multiOp);
      }

      exchangeOps.clear();
      initOps.clear();
      seenRemoteLoads  = false;
      seenRemoteStores = false;
    }

    seenRemoteLoads  = seenRemoteLoads || op->isConvertibleTo<RemoteLoadOp>();
    seenRemoteStores = seenRemoteStores || op->isConvertibleTo<RemoteStoreOp>();

    if (isMergeable) {
      if (ExchangeBaseOp *exchOp = dynamic_cast<ExchangeBaseOp *>(op)) {
        exchangeOps.push_back({i, exchOp});
        logging::transform::trace(
            "[MergeExchange::applyToOps] Adding mergeable Op {}",
            exchOp->debugName());
      }
    }
  }
  auto multiOp = conditionallyInsertMultiExchange(
      graph, exchangeOps, initOpConstraints(initOps, exchangeOps));
  if (multiOp != nullptr) {
    createdOps.push_back(multiOp);
  }

  return createdOps;
}

namespace {
bool init = Transform::registerTransform(new MergeExchange);
}

} // namespace popart
