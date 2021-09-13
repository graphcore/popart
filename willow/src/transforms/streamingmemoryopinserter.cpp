// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/alias/aliasmodel.hpp>
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/op/init.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/lamb.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/tensornames.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/remotesetup.hpp>

#include <transforms/streamingmemoryopinserter.hpp>

namespace popart {

namespace {

static constexpr const double maxCompressedPriority = 9000.0;

// Compress priorities so that nothing is using priorities outside the range
// -9000 to +9000
void compressPriorities(Graph &graph) {
  std::map<double, std::set<Op *, POpCmp>, std::greater<double>> pPriOpsMap;
  std::map<double, std::set<Op *, POpCmp>, std::less<double>> nPriOpsMap;
  for (auto &op : graph.getOps()) {
    if (op.second->settings.schedulePriority > 0.0) {
      pPriOpsMap[op.second->settings.schedulePriority].insert(op.second.get());
    }
    if (op.second->settings.schedulePriority < 0.0) {
      nPriOpsMap[op.second->settings.schedulePriority].insert(op.second.get());
    }
  }

  {
    double pri = maxCompressedPriority;
    for (auto &priOp : pPriOpsMap) {
      for (Op *op : priOp.second) {
        op->settings.schedulePriority = pri;
      }
      pri -= maxCompressedPriority / pPriOpsMap.size();
    }
  }

  {
    double pri = -maxCompressedPriority;
    for (auto &priOp : nPriOpsMap) {
      for (Op *op : priOp.second) {
        op->settings.schedulePriority = pri;
      }
      pri += maxCompressedPriority / nPriOpsMap.size();
    }
  }
}

bool isConstOrCopyOfConst(Tensor *tensor) {
  do {
    if (tensor->tensorType() == TensorType::Const) {
      return true;
    }
    if (tensor->hasProducer()) {
      Op *prod = tensor->getProducer();
      if (prod->isIpuCopyOp()) {
        tensor = prod->input->tensor(0);
      } else {
        return false;
      }
    }
  } while (tensor->hasProducer());
  return false;
}

} // namespace

void StreamingMemoryOpInserter::setPriority(Op *op,
                                            bool isPhased,
                                            bool onDemandOptimizerState,
                                            ExecutionPhaseSchedule schedule) {
  if (isPhased && op->settings.executionContext == ExecutionContext::Normal) {
    switch (schedule) {
    case ExecutionPhaseSchedule::Interleaving: {
      double priority = -(maxCompressedPriority + 1);
      // Init ops must be scheduled first
      if (op->isConvertibleTo<InitOp>()) {
        op->settings.schedulePriority = priority;
      }
      --priority;
      // Remote load & all gather can be scheduled together in a way that
      // minimizes liveness
      if (!onDemandOptimizerState &&
          (op->isConvertibleTo<RemoteLoadOp>() ||
           op->isConvertibleTo<ReplicatedAllGatherOp>())) {
        op->settings.schedulePriority = priority;
      }
      --priority;
      // Cross-phase cross-IPU communication
      if (op->isConvertibleTo<IpuCopyOp>()) {
        op->settings.schedulePriority = priority;
      }
      --priority;
      // Optimizer related ops and remote store can be scheduled together
      // in a way that minimizes liveness
      if ((onDemandOptimizerState &&
           (op->isConvertibleTo<RemoteLoadOp>() ||
            op->isConvertibleTo<ReplicatedAllGatherOp>())) ||
          op->isOptimizerOp() || op->isConvertibleTo<ReplicatedAllReduceOp>() ||
          op->isConvertibleTo<ReplicatedReduceScatterOp>() ||
          op->isConvertibleTo<RemoteStoreOp>()) {
        op->settings.schedulePriority = priority;
      }
      break;
    }
    case ExecutionPhaseSchedule::Batch: {
      double priority = -(maxCompressedPriority + 1);
      // Init ops must be scheduled first
      if (op->isConvertibleTo<InitOp>()) {
        op->settings.schedulePriority = priority;
      }
      --priority;
      // All remote loads must be scheduled before replicated operations
      // to maximize overlap (collectives block overlap)
      if (op->isConvertibleTo<RemoteLoadOp>()) {
        op->settings.schedulePriority = priority;
      }
      --priority;
      // Cross-phase cross-IPU and compute/IO tile communication
      // (overlap blocking)
      if (op->isConvertibleTo<IpuCopyOp>() ||
          op->isConvertibleTo<IoTileCopyOp>()) {
        op->settings.schedulePriority = priority;
      }
      --priority;
      // All collective operations (overlap blocking)
      if (op->isConvertibleTo<ReplicatedReduceScatterOp>() ||
          op->isConvertibleTo<ReplicatedAllReduceOp>() ||
          op->isConvertibleTo<ReplicatedAllGatherOp>()) {
        op->settings.schedulePriority = priority;
      }
      --priority;
      // All optimizer operations
      if (op->isOptimizerOp()) {
        op->settings.schedulePriority = priority;
      }
      --priority;
      // All remote stores must be scheduled at the end, to overlap with the
      // next compute phase
      if (op->isConvertibleTo<RemoteStoreOp>()) {
        op->settings.schedulePriority = priority;
      }
      break;
    }
    case ExecutionPhaseSchedule::BatchClusteredIO: {
      double priority = maxCompressedPriority + 2;
      // Init ops must be scheduled first
      if (op->isConvertibleTo<InitOp>()) {
        op->settings.schedulePriority = priority;
      }
      --priority;
      // All remote loads must be scheduled before replicated operations
      // to maximize overlap (collectives block overlap)
      if (op->isConvertibleTo<RemoteLoadOp>()) {
        op->settings.schedulePriority = priority;
      }
      priority = -(maxCompressedPriority + 1);
      // Cross-phase cross-IPU and compute/IO tile communication
      // (overlap blocking)
      if (op->isConvertibleTo<IpuCopyOp>() ||
          op->isConvertibleTo<IoTileCopyOp>()) {
        op->settings.schedulePriority = priority;
      }
      --priority;
      // All collective operations (overlap blocking)
      if (op->isConvertibleTo<ReplicatedReduceScatterOp>() ||
          op->isConvertibleTo<ReplicatedAllReduceOp>() ||
          op->isConvertibleTo<ReplicatedAllGatherOp>()) {
        op->settings.schedulePriority = priority;
      }
      --priority;
      // All optimizer operations
      if (op->isOptimizerOp()) {
        op->settings.schedulePriority = priority;
      }
      --priority;
      // All remote stores must be scheduled at the end, to overlap with the
      // next compute phase
      if (op->isConvertibleTo<RemoteStoreOp>()) {
        op->settings.schedulePriority = priority;
      }
      break;
    }
    default:
      throw error("Unsupported schedule {}", static_cast<int>(schedule));
    }
  }
}

StreamingMemoryOpInserter::StreamingMemoryOpInserter(Graph &graph_,
                                                     AliasModel &aliasModel_,
                                                     int64_t replicationFactor_,
                                                     int num_stages_,
                                                     int num_phases_)
    : graph{graph_}, aliasModel{aliasModel_},
      replicationFactor{replicationFactor_}, num_stages{num_stages_},
      num_phases{num_phases_}, remoteArgIds{} {}

bool StreamingMemoryOpInserter::isPhasedExecution() const {
  return num_phases > 1;
}

void StreamingMemoryOpInserter::createTensorSchedule() {
  Tensors &tensors = graph.getTensors();
  std::set<TensorId> seenTensors;
  tensorSchedule.clear();
  tensorSchedule.reserve(tensors.getAllTensorIds().size());
  auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::Yes);
  for (int64_t i = 0; i < schedule.size(); ++i) {
    Op *op            = schedule.at(i);
    opScheduleMap[op] = i;
    for (auto &indexAndTensor : op->input->tensorMap()) {
      if (seenTensors.find(indexAndTensor.second->id) == seenTensors.end()) {
        seenTensors.insert(indexAndTensor.second->id);
        tensorSchedule.push_back(indexAndTensor.second);
      }
    }
    for (auto &indexAndTensor : op->output->tensorMap()) {
      if (seenTensors.find(indexAndTensor.second->id) == seenTensors.end()) {
        seenTensors.insert(indexAndTensor.second->id);
        tensorSchedule.push_back(indexAndTensor.second);
      }
    }
  }
}

Tensor *StreamingMemoryOpInserter::findRelatedVarTensor(
    std::vector<Tensor *> front) const {
  while (front.size()) {
    Tensor *t0 = front.back();
    front.pop_back();

    auto config = tensorConfigs.find(t0);

    if (config != tensorConfigs.end() && config->second.rootVarTensor &&
        config->second.rootVarTensor->tensorType() == TensorType::Variable) {
      return config->second.rootVarTensor;
    }

    for (Op *consumer : t0->consumers.getOps()) {
      for (auto iter = consumer->output->tensorMap().rbegin();
           iter != consumer->output->tensorMap().rend();
           ++iter) {
        front.push_back(iter->second);
      }
    }
  }

  return nullptr;
}

void StreamingMemoryOpInserter::updateReplicatedOperations() {
  auto &ops = graph.getOps();

  for (auto &opIdAndOp : ops) {
    if (opIdAndOp.second->isConvertibleTo<ReplicatedAllReduceOp>()) {
      bool reductionForOptimizer = false;
      for (auto tensorAndIndex : opIdAndOp.second->output->indicesMap()) {
        for (auto consumer : tensorAndIndex.first->consumers.getOps()) {
          reductionForOptimizer |= consumer->isOptimizerOp();
        }
      }
      if (reductionForOptimizer) {
        ReplicatedAllReduceOp *replicatedAllReduce =
            dynamic_cast<ReplicatedAllReduceOp *>(opIdAndOp.second.get());

        Tensor *varTensor = findRelatedVarTensor({replicatedAllReduce->inTensor(
            ReplicatedAllReduceOp::getInIndex())});

        if (varTensor) {
          TensorConfig tensorConfig = tensorConfigs.at(varTensor);

          replicatedAllReduce->settings.tileSet =
              tensorConfig.location.storageTileSet;

          setPriority(replicatedAllReduce,
                      isPhasedExecution(),
                      false,
                      tensorConfig.schedule);
        } else {
          logging::transform::warn(
              "[StreamingMemory] {} is an optimizer related "
              "ReplicatedAllReduceOp, but the related variable "
              "has not been found.",
              opIdAndOp.second->debugName());
        }
      }
    }
  }
}

void StreamingMemoryOpInserter::updateOptimizerOperations() {
  OpsSet optimizerOps;

  auto &ops = graph.getOps();

  for (auto &opIdAndOp : ops) {
    if (opIdAndOp.second->isOptimizerOp()) {
      optimizerOps.insert(opIdAndOp.second.get());
    }
  }

  for (Op *op : optimizerOps) {
    std::vector<Tensor *> tensors;
    for (auto iter = op->output->tensorMap().rbegin();
         iter != op->output->tensorMap().rend();
         ++iter) {
      tensors.push_back(iter->second);
    }
    Tensor *varTensor = findRelatedVarTensor(tensors);
    if (varTensor) {
      TensorConfig rootVarConfig = tensorConfigs.at(varTensor);
      op->settings.tileSet       = rootVarConfig.location.storageTileSet;
      setPriority(op, isPhasedExecution(), false, rootVarConfig.schedule);
    } else {
      logging::transform::warn("[StreamingMemory] {} is an optimizer op, "
                               "but the related variable has not been found.",
                               op->debugName());
    }
  }
}

void StreamingMemoryOpInserter::apply() {
  std::set<Op *, POpCmp> opsToSetup;

  if (isPhasedExecution()) {
    // Make sure no priorities outside of the range needed by
    // phased execution are being used.
    compressPriorities(graph);
    sanitizeOps();
  }

  logging::transform::debug(
      "[StreamingMemory] Processing tensors for streaming memory");

  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {
    auto &options = graph.getIr().getSessionOptions();

    std::stringstream ss;
    ss << "\n";
    ss << "    activations: " << options.activationTensorLocationSettings
       << "\n";
    ss << "    weights: " << options.weightTensorLocationSettings << "\n";
    ss << "    optimizer state: "
       << options.optimizerStateTensorLocationSettings << "\n";
    ss << "    accumulator: " << options.accumulatorTensorLocationSettings
       << "\n";
    ss << "    overrides:"
       << "\n";
    for (auto &settingsOverride : options.tensorLocationSettingsOverride) {
      ss << "       " << settingsOverride.first << ": "
         << settingsOverride.second << "\n";
    }

    logging::transform::trace("[StreamingMemory] TensorLocation settings: {}",
                              ss.str());
  }

  createTensorSchedule();

  // Determine tensor configuration in reverse from bottom to top
  // (so that root variable tensors are updated last)
  for (TensorSchedule::reverse_iterator it = tensorSchedule.rbegin();
       it != tensorSchedule.rend();
       ++it) {
    getTensorConfig(*it);
  }

  ReplicationShardedTensors rtsTensors;

  for (auto &tensor : tensorSchedule) {
    // Introduce ops for each tensor.
    applyTensor(tensor, rtsTensors);
  }

  updateReplicatedOperations();
  updateOptimizerOperations();

  // Propagate replicated tensor sharding through the optimizer ops
  applyReplicatedOptimizerSharding(rtsTensors);

  if (isPhasedExecution()) {
    for (const auto &op : graph.getOps()) {
      if (op.second.get()->hasExecutionPhase() &&
          op.second.get()->settings.executionContext !=
              ExecutionContext::Normal) {
        throw error("[StreamingMemory] Op {} inconsistent execution phase {} "
                    "and execution context {}.",
                    op.second.get()->debugName(),
                    op.second.get()->getExecutionPhase(),
                    op.second.get()->settings.executionContext);
      }
    }
    graph.getIr().setExecutionPhasesReady();
  }
}

void StreamingMemoryOpInserter::applyTensor(
    Tensor *tensor,
    ReplicationShardedTensors &rtsTensors) {

  // Get the tensor's configuration.
  const TensorConfig &tensorConfig = tensorConfigs.at(tensor);

  // Nothing to for this tensor if it's not remote.
  if (!tensorConfig.location.isRemote()) {
    logging::transform::trace(
        "[StreamingMemory] Skipping tensor {} (not remote)", tensor->id);
    return;
  }

  // Skip if this is not a root variable tensor or tensor without root
  // variable
  if (tensorConfig.rootVarTensor && tensor != tensorConfig.rootVarTensor) {
    return;
  }

  // Log tensor phase configuration.
  logTensorStreamingConfig(tensorConfig);

  // Some context variables we need to keep track of between contexts.
  TensorId loadedTensorId   = tensor->id;
  TensorId gatheredTensorId = tensor->id;

  RemoteLoadOp *remoteLoad         = nullptr;
  RemoteStoreOp *remoteStore       = nullptr;
  ReplicatedAllGatherOp *allGather = nullptr;

  std::vector<RemoteStoreOp *> precedingRemoteStores;

  for (auto &contextAndConfig : tensorConfig.streamingMap) {
    auto &context = contextAndConfig.first;
    auto &config  = contextAndConfig.second;

    // Reset these pointers:
    remoteLoad  = nullptr;
    remoteStore = nullptr;
    allGather   = nullptr;

    // Create RemoteLoadOp for this phase if we need to.
    if (config.load) {
      remoteLoad = insertRemoteLoadOp(tensorConfig, context, loadedTensorId);
      // If no gather is required, the gathered and loaded tensor ID
      // are identical
      gatheredTensorId = loadedTensorId;
    }

    // Create RemoteStoreOp for this phase if we need to.
    if (config.store) {
      remoteStore = insertRemoteStoreOp(tensorConfig, context, loadedTensorId);
    }

    // Create GatherOp for this phase if we need to.
    if (config.gather) {
      allGather =
          insertReplicatedAllGatherOp(tensorConfig,
                                      context,
                                      loadedTensorId,
                                      gatheredTensorId,
                                      stripAllReservedPrefixes(tensor->id),
                                      tensorConfig.location.shardingDomain);
    }

    // Add constraints to ensure new operations are scheduled in the right
    // order.

    // Load must happen before the all-gather.
    if (remoteLoad && allGather) {
      graph.topoCons->insert(remoteLoad, allGather);
    }

    // Remote load has to take place before associated remote store
    if (remoteLoad && remoteStore) {
      graph.topoCons->insert(remoteLoad, remoteStore);
    }

    if (remoteLoad) {
      // Remote load has to take place after any preceding remote store
      for (auto precedingRemoteStore : precedingRemoteStores) {
        graph.topoCons->insert(precedingRemoteStore, remoteLoad);
      }
    }

    if (remoteStore) {
      // Any modification has to take place before the remote store
      for (Op *modifyingOp : config.modifiers) {
        graph.topoCons->insert(modifyingOp, remoteStore);
      }
      precedingRemoteStores.push_back(remoteStore);
    }

    if (tensorConfig.location.replicatedTensorSharding ==
        ReplicatedTensorSharding::On) {
      rtsTensors.insert(loadedTensorId,
                        gatheredTensorId,
                        tensor->id,
                        stripAllReservedPrefixes(tensor->id));
    }

    std::set<Op *> consumers;
    for (auto consumerOpConfig : config.consumers) {
      consumers.insert(consumerOpConfig.op);
    }

    for (auto consumerOpConfig : config.consumers) {

      bool requiresProducerTopoCons = false;
      bool requiresRewiring         = false;

      if (tensor->getTensorTypeInfo()->type() == TensorType::Variable &&
          tensor->id != consumerOpConfig.tensor->id) {
        // Current tensor (root var) is not the same as the actually
        // consumed tensor (descendant)
        if (consumerOpConfig.tensor->hasProducer()) {
          if (consumers.find(consumerOpConfig.tensor->getProducer()) ==
              consumers.end()) {
            // The producer is in a different context than the consumer,
            // therefore rewire to the root variable tensor
            requiresRewiring         = true;
            requiresProducerTopoCons = true;
          }
        }
      } else {
        requiresRewiring = true;
      }

      if (requiresRewiring) {
        if (remoteLoad) {
          // Loading has to take place before the consumption
          graph.topoCons->insert(remoteLoad, consumerOpConfig.op);
        }
        if (requiresProducerTopoCons) {
          // If we rewire from descendant to the root tensor,
          // preserve topological constraint to the producer of the
          // descendant
          auto producer = consumerOpConfig.tensor->getProducer();
          logging::transform::debug(
              "[StreamingMemory] Inserting topological constraint between "
              "producer {} and consumer {} before connecting the root "
              "variable tensor {}",
              producer->debugName(),
              consumerOpConfig.op->debugName(),
              tensor->id);
          graph.topoCons->insert(producer, consumerOpConfig.op, false);
          if (remoteStore) {
            graph.topoCons->insert(producer, remoteStore, false);
          }
        }

        // Logging graph change
        if (tensorConfig.producerOp) {
          logging::transform::debug(
              "[StreamingMemory] Disconnecting tensor {} between ops {} and {}",
              consumerOpConfig.tensor->id,
              tensorConfig.producerOp->debugName(),
              consumerOpConfig.op->debugName());
        } else {
          logging::transform::debug("[StreamingMemory] Disconnecting tensor {} "
                                    "at op {} (modified: {})",
                                    consumerOpConfig.tensor->id,
                                    consumerOpConfig.op->debugName(),
                                    !config.modifiers.empty());
        }

        // Disconnect original tensor and wire up loaded tensor
        auto indices = consumerOpConfig.inIndices;
        for (auto i : indices) {
          auto *copyOp = dynamic_cast<IpuCopyOp *>(consumerOpConfig.op);

          if (copyOp) {
            auto sourceIpu = copyOp->getSourceIpus().at(tensor->id);
            copyOp->disconnectInTensor(i, consumerOpConfig.tensor);
            copyOp->connectInTensor(i, gatheredTensorId, sourceIpu);
          } else if (consumerOpConfig.op->isOptimizerOp()) {
            consumerOpConfig.op->disconnectInTensor(i, consumerOpConfig.tensor);
            consumerOpConfig.op->connectInTensor(i, loadedTensorId);
          } else {
            consumerOpConfig.op->disconnectInTensor(i, consumerOpConfig.tensor);
            consumerOpConfig.op->connectInTensor(i, gatheredTensorId);
          }
        }
      }
    }

    // Transfer tensor configurations
    if (!loadedTensorId.empty()) {
      tensorConfigs.insert(
          {graph.getTensors().get(loadedTensorId), tensorConfigs.at(tensor)});
    }
    if (!gatheredTensorId.empty()) {
      tensorConfigs.insert(
          {graph.getTensors().get(gatheredTensorId), tensorConfigs.at(tensor)});
    }
  }
}

const StreamingMemoryOpInserter::ReplicatedTensorShardingProposal
StreamingMemoryOpInserter::getReplicatedTensorShardingProposal(
    const ReplicationShardedTensors &rtsTensors) const {
  ReplicatedTensorShardingProposal proposedRTS;

  // Operations that are affected by replicated tensor sharding
  OpsSet opsToProcess;

  // Map to resolve the reference tensor id for replicated weight sharding for
  // a specific Op

  auto addOpsToProcess = [this, &opsToProcess, &proposedRTS](TensorId id,
                                                             TensorId refId) {
    Tensor *t = graph.getTensors().get(id);
    for (Op *consumer : t->consumers.getOps()) {
      logging::transform::trace("[StreamingMemory] Op {} consumes at least "
                                "one replication sharded tensor",
                                consumer->debugName());
      opsToProcess.insert(consumer);
      proposedRTS.setRefTensorId(consumer->id, refId);
    }
  };

  // Mapping from each RemoteArg to it's final consumers
  RemoteArgOpMap argOpMap;
  RemoteOpArgMap opArgMap;
  RemoteArgBufferMap argBufferMap;

  RemoteSetup::getRemoteArgMapping(graph, argOpMap, opArgMap, argBufferMap);

  // Checks if the root tensor, created by InitOp, is RTS
  auto checkRemoteBufferRTS = [&rtsTensors, &proposedRTS, &argOpMap, &opArgMap](
                                  Op *op) {
    if (op->isConvertibleTo<RemoteLoadOp>() ||
        op->isConvertibleTo<RemoteStoreOp>()) {
      auto &args =
          opArgMap.at({op, RemoteLoadOp::getRemoteBufferOffsetInIndex()});
      for (auto arg : args) {
        auto &ops = argOpMap.at(arg);
        for (auto opAndIdx : ops) {
          auto inputs  = opAndIdx.first->input->tensors();
          auto outputs = opAndIdx.first->output->tensors();
          if (std::any_of(inputs.begin(),
                          inputs.end(),
                          [&rtsTensors, &proposedRTS, &opAndIdx](Tensor *t) {
                            return rtsTensors.hasShard(t->id) ||
                                   (proposedRTS.isProposed(t->id) &&
                                    proposedRTS.isProposed(opAndIdx.first->id));
                          }) ||
              std::any_of(outputs.begin(),
                          outputs.end(),
                          [&rtsTensors, &proposedRTS, &opAndIdx](Tensor *t) {
                            return rtsTensors.hasShard(t->id) ||
                                   (proposedRTS.isProposed(t->id) &&
                                    proposedRTS.isProposed(opAndIdx.first->id));
                          })) {
            logging::transform::trace(
                "[StreamingMemory] Op {} implies {} should also be RTS",
                opAndIdx.first->debugName(),
                op->debugName());
            return true;
          }
        }
      }
      return false;
    }
    return true;
  };

  for (TensorId shardId : rtsTensors.getShardTensorIds()) {

    TensorId tensorId = rtsTensors.getTensor(shardId);
    TensorId refId    = rtsTensors.getReference(shardId);
    // Process initial shard consumers
    if (!shardId.empty()) {
      proposedRTS.insert(shardId, ReplicatedTensorShardingMethod::Native);

      logging::transform::trace(
          "[StreamingMemory] Processing replicated tensor "
          "sharding for tensor {} (shard {})",
          tensorId,
          shardId);
      addOpsToProcess(shardId, refId);
    }
  }

  // Propagate replicated tensor sharding as far as possible
  bool rtsChanged = true;
  while (rtsChanged) {
    OpsSet currentOpsToProcess = opsToProcess;
    rtsChanged                 = false;
    for (Op *op : currentOpsToProcess) {
      logging::transform::trace(
          "[StreamingMemory] Analyzing {} for replicated tensor sharding",
          op->debugName());
      TensorId refId = proposedRTS.getRefTensorId(op->id);

      if (!checkRemoteBufferRTS(op)) {
        // Encountered RemoteLoad/RemoteStore that can't be RTS because the
        // remote buffer itself won't be RTS
        logging::transform::trace("[StreamingMemory] {} cannot be RTS because "
                                  "the associated remote buffer is not RTS",
                                  op->debugName());
        continue;
      }

      auto rtsIndices = op->getReplicatedTensorShardingIndices();

      // Loop over one set of indices
      for (auto indices : rtsIndices) {
        bool allIndiciesSharded = true;
        for (InIndex inIdx : indices.first) {
          bool indexSharded = false;
          Tensor *inTensor  = op->input->tensor(inIdx);
          if (proposedRTS.isProposed(inTensor->id)) {
            // Already sharded or shard tensor will be available
            indexSharded = true;

            if (proposedRTS.getMethod(inTensor->id) ==
                    ReplicatedTensorShardingMethod::LocalScatter &&
                (op->isConvertibleTo<VarUpdateOp>() &&
                 inIdx == VarUpdateOp::getVarToUpdateInIndex() &&
                 op->modifiesIndex(inIdx))) {
              proposedRTS.insertVarUpdateWithCopies(op->id);
            }

          } else if (inTensor->hasProducer() &&
                     inTensor->getProducer()
                         ->isConvertibleTo<ReplicatedAllReduceOp>()) {
            // Can try to shard index by changing an AllReduce into a
            // ReduceScatter
            Tensor *varTensor = findRelatedVarTensor({inTensor});

            TensorConfig tensorConfig = tensorConfigs.at(inTensor);
            if (varTensor) {
              tensorConfig = tensorConfigs.at(varTensor);
            } else {
              logging::transform::warn(
                  "[StreamingMemory] {} is an optimizer related "
                  "ReplicatedAllReduce, but the related variable "
                  "has not been found.",
                  inTensor->getProducer()->debugName());
            }

            // Add all consumers to the set of ops to process
            proposedRTS.insert(
                inTensor->id,
                ReplicatedTensorShardingMethod::AllReduceToScatter);
            addOpsToProcess(inTensor->id, refId);
            indexSharded = true;
            rtsChanged   = true;
          } else if (inTensor->anyAlias(
                         [](const Tensor *t) { return t->isWeightTensor(); }) &&
                     op->isConvertibleTo<VarUpdateOp>() &&
                     inIdx == VarUpdateOp::getVarToUpdateInIndex() &&
                     op->modifiesIndex(inIdx)) {
            // If the input is a variable or an alias of a variable, we can
            // use a local reduce scatter to get an RTS copy of the variable
            // Tensor outId is now replicated tensor sharded
            // Add all consumers to the set of ops to process
            proposedRTS.insert(inTensor->id,
                               ReplicatedTensorShardingMethod::LocalScatter);
            proposedRTS.insertVarUpdateWithCopies(op->id);
            proposedRTS.insert(op->id, refId);
            addOpsToProcess(inTensor->id, refId);
            indexSharded = true;
            rtsChanged   = true;
          } else if (inTensor->anyAlias(
                         [](const Tensor *t) { return t->isWeightTensor(); }) &&
                     !op->modifiesIndex(inIdx)) {
            proposedRTS.insert(inTensor->id,
                               ReplicatedTensorShardingMethod::LocalScatter);
            proposedRTS.insert(op->id, refId);
            addOpsToProcess(inTensor->id, refId);
            indexSharded = true;
            rtsChanged   = true;
          } else {
            // Try to backpropagate RTS to the producer
            if (inTensor->hasProducer()) {
              Op *producer = inTensor->getProducer();
              auto producerRtsIndices =
                  producer->getReplicatedTensorShardingIndices();
              if (!producerRtsIndices.empty() &&
                  opsToProcess.find(producer) == opsToProcess.end()) {
                opsToProcess.insert(producer);
                proposedRTS.setRefTensorId(producer->id, refId);
                rtsChanged = true;
              }
            }
          }
          allIndiciesSharded &= indexSharded;
        }
        if (allIndiciesSharded) {
          // Configure the op with all sharded inputs added
          logging::transform::trace("[StreamingMemory] Configuring {} for "
                                    "replicated tensor sharding "
                                    "(indices: {}->{})",
                                    op->debugName(),
                                    indices.first,
                                    indices.second);
          // Add output tensors as RTS tensors
          for (auto outIndex : indices.second) {
            TensorId outId = op->outId(outIndex);
            // Tensor outId is now replicated tensor sharded
            proposedRTS.insert(outId, ReplicatedTensorShardingMethod::Forward);
            // Add all consumers to the set of ops to process
            addOpsToProcess(outId, refId);
          }
          opsToProcess.erase(op);
          proposedRTS.insert(op->id, refId);
          rtsChanged = true;
        }
      }
    }
  }

  // Where replicated tensor sharding couldn't be propagated further
  for (Op *op : opsToProcess) {
    logging::transform::trace(
        "[StreamingMemory] Resolving where RTS could not be propagated for {}",
        op->debugName());
    auto rtsIndices = op->getReplicatedTensorShardingIndices();

    // Loop over one set of indices
    for (auto indices : rtsIndices) {
      bool modifiesSharded    = false;
      bool allIndiciesSharded = true;
      for (InIndex inIdx : indices.first) {
        bool indexSharded = false;
        Tensor *inTensor  = op->input->tensor(inIdx);
        if (rtsTensors.hasShard(inTensor->id)) {
          // Already sharded
          indexSharded = true;
          // If the optimizer Op modifies the input, we say that the optimizer
          // updates this variable. If this variable is an RTS tensor,
          // then we know the update Op must be sharded for that input.
          // We most definitely can't use a ReplicatedAllGather here,
          // because that would no longer update the root var tensor
          modifiesSharded |= op->modifiesIndex(inIdx);
        }
        allIndiciesSharded &= indexSharded;
      }
      if (!allIndiciesSharded) {
        // Not all required indices for this op are RTS tensors:
        // We can try to gather the complete tensor and let the Op operate
        // on the whole tensors rather than shards or we need to bail
        // because the Op updates a sharded tensor
        logging::transform::trace(
            "[StreamingMemory] {} has not all indices sharded",
            op->debugName());

        if (modifiesSharded) {
          for (InIndex inIdx : indices.first) {
            Tensor *inTensor = op->input->tensor(inIdx);
            if (inTensor->anyAlias(
                    [](const Tensor *t) { return t->isWeightTensor(); }) &&
                op->isConvertibleTo<VarUpdateOp>() &&
                inIdx == VarUpdateOp::getVarToUpdateInIndex() &&
                op->modifiesIndex(inIdx)) {
              proposedRTS.insert(
                  op->outId(VarUpdateOp::getUpdatedVarOutIndex()),
                  ReplicatedTensorShardingMethod::LocalScatter);
              proposedRTS.insertVarUpdateWithCopies(op->id);
            } else {
              // Bail with error if op must consume sharded tensor but can't
              throw error(
                  "[StreamingMemory] Op {} modifies a sharded tensor, but "
                  "not all required sharded inputs could be connected. Adjust "
                  "the "
                  "replicated tensor sharding settings. Bailing.",
                  op->debugName());
            }
          }
        }
      }
    }
  }
  return proposedRTS;
}

void StreamingMemoryOpInserter::RTSAllReduceToScatter(
    ReplicationShardedTensors &rtsTensors,
    Tensor *inTensor,
    TensorId refId) {
  if (inTensor->hasProducer() &&
      inTensor->getProducer()->isConvertibleTo<ReplicatedAllReduceOp>()) {

    // Try to shard index by changing an AllReduce into a
    // ReduceScatter
    ReplicatedAllReduceOp *replicatedAllReduce =
        dynamic_cast<ReplicatedAllReduceOp *>(inTensor->getProducer());
    TensorId inId =
        replicatedAllReduce->input->tensor(ReplicatedAllReduceOp::getInIndex())
            ->id;
    TensorId outId = replicatedAllReduce->output
                         ->tensor(ReplicatedAllReduceOp::getOutIndex())
                         ->id;

    replicatedAllReduce->disconnectAllInputs();
    replicatedAllReduce->disconnectAllOutputs();

    TensorStreamingContext context;
    context.context = replicatedAllReduce->settings.executionContext;
    if (context.context == ExecutionContext::Normal) {
      if (isPhasedExecution()) {
        context.phase = replicatedAllReduce->getOptionalExecutionPhase();
      } else {
        context.preLoss = replicatedAllReduce->scheduledPreLoss;
      }
    }

    Tensor *outTensor         = graph.getTensors().get(outId);
    TensorConfig tensorConfig = tensorConfigs.at(outTensor);

    Tensor *varTensor = findRelatedVarTensor({outTensor});

    if (varTensor) {
      tensorConfig = tensorConfigs.at(varTensor);
    } else {
      logging::transform::warn("[StreamingMemory] {} is an optimizer related "
                               "ReplicatedAllReduce, but the related variable "
                               "has not been found.",
                               replicatedAllReduce->debugName());
    }

    // Keep settings that the ReplicatedAllReduce, that is being
    // replaced, had.
    tensorConfig.settings = replicatedAllReduce->settings;

    CommGroup complementaryGroup = getComplementCommGroup(
        graph.getIr(), tensorConfig.location.shardingDomain);

    bool needsAllReduce =
        complementaryGroup.replicaGroupSize > 1 &&
        (complementaryGroup.type == CommGroupType::Orthogonal ||
         complementaryGroup.type == CommGroupType::Consecutive);

    TensorId scatterOutId =
        needsAllReduce ? graph.getIr().createIntermediateTensorId(outId)
                       : outId;

    ReplicatedReduceScatterOp *replicatedReduceScatter =
        insertReplicatedReduceScatterOp(tensorConfig,
                                        context,
                                        inId,
                                        scatterOutId,
                                        stripAllReservedPrefixes(refId),
                                        replicatedAllReduce->getCollectiveOp(),
                                        tensorConfig.location.shardingDomain);

    graph.topoCons->transfer(
        replicatedAllReduce, replicatedReduceScatter, !needsAllReduce);

    if (needsAllReduce) {
      replicatedAllReduce->connectInTensor(ReplicatedAllReduceOp::getInIndex(),
                                           scatterOutId);
      replicatedAllReduce->connectOutTensor(
          ReplicatedAllReduceOp::getOutIndex(), outId);
      replicatedAllReduce->setGCLCommGroup(complementaryGroup);
      replicatedAllReduce->setup();
    } else {
      replicatedAllReduce->getGraph().eraseOp(replicatedAllReduce->id);
    }

    // Tensor outId is now replicated tensor sharded
    rtsTensors.insert(outId, "", outId, stripAllReservedPrefixes(refId));
  } else {
    throw error("[StreamingMemory] Could not apply method {} on tensor {}",
                ReplicatedTensorShardingMethod::AllReduceToScatter,
                inTensor->id);
  }
}

void StreamingMemoryOpInserter::RTSLocalScatter(
    TensorStreamingContext context,
    ReplicationShardedTensors &rtsTensors,
    Tensor *inTensor,
    TensorId refId) {

  // If the input is a variable or an alias of a variable, we can
  // use a local reduce scatter to get an RTS copy of the variable

  TensorId outId = graph.getIr().createIntermediateTensorId(inTensor->id);

  TensorConfig tensorConfig = tensorConfigs.at(inTensor);

  Tensor *varTensor = findRelatedVarTensor({inTensor});

  if (varTensor) {
    tensorConfig = tensorConfigs.at(varTensor);
  }

  insertReplicatedReduceScatterOp(tensorConfig,
                                  context,
                                  inTensor->id,
                                  outId,
                                  stripAllReservedPrefixes(refId),
                                  CollectiveOperator::Local,
                                  tensorConfig.location.shardingDomain);

  // Tensor outId is now replicated tensor sharded
  rtsTensors.insert(
      outId, inTensor->id, inTensor->id, stripAllReservedPrefixes(refId));
}

void StreamingMemoryOpInserter::applyReplicatedOptimizerSharding(
    ReplicationShardedTensors &rtsTensors) {

  // Propagates replicated tensor sharding through the optimizers, e.g.:

  // Rules apply only for operation inputs marked in
  // getReplicatedTensorShardingIndices by the consuming operator, which
  // leaves non-RTS tensors such as optimizer parameters (learning rate,
  // weight decay) alone.

  // 1.) Op has both RTS and non-RTS inputs, set Op and output to be non-RTS
  //    a.) Reuse an existing AllGather if possible
  //    b.) Insert a new one if not
  //  A*  B                   A*    B
  //   \ /                    |     |
  //    Op           ---> AllGather /
  //    |                     |    /
  //    C                     A   /
  //                          |  /
  //                          Op
  //                          |
  //                          C
  //
  // 2.) Op has one RTS and one AllReduce input, change Op and output to be
  // RTS
  //  A*  B                   A*  B
  //  |   |                   |   |
  //  |  AllReduce   --->     |  ReduceScatter
  //  |   |                   |   |
  //  |   B'                  |   B'*
  //   \ /                     \ /
  //    Op                      Op*
  //    |                       |
  //    C                       C*
  //
  //
  // 3.) Op has only RTS inputs, change Op and output to be RTS
  //  A*  B*                  A*  B*
  //   \ /                     \ /
  //    Op          --->        Op*
  //    |                       |
  //    C                       C*
  //
  // 4.) a.) Turn RTS into non-RTS weight for the update if the Op has RTS and
  //         non-RTS inputs
  //  A*  B                                    A*        B
  //  |   |                                   /  \       |
  //  |   |                                  | AllGather |
  //  |   |                                  |     |     |
  //  |   |                                  |     A''   |
  //   \ /                                   |     |     |
  //  VarUpdate <- modifies A*  --->         |    VarUpdate
  //    |                                    |     |
  //    |                                    |     A'''
  //    |                                    |     |
  //    |                                    |   ReduceScatter(Local)
  //    |                                     \    |
  //    |                                      \   A'''*
  //    |                                       \  |
  //    |                                     CopyVarUpdate
  //    |                                          |
  //    A'*                                        A'*
  // where A*->A'* is a weight
  //
  //     b.) Bailing (error) if the Op has RTS and non-RTS inputs, but can't
  //         gather
  //
  //  A*  B
  //   \ /
  //    Op <- modifies A* and not a VarUpdate -> bailing with error
  //    |
  //    C
  //
  // 5.) Try to backpropagate RTS to producer operations
  //
  //  A*  B                   A*  B*
  //  |   |                   |   |
  //  |  Producer   --->      |  Producer*
  //  |   |                   |   |
  //  |   B'                  |   B'*
  //   \ /                     \ /
  //    Op                      Op
  //    |                       |
  //    C                       C
  //
  //
  // 6.) Weights can be turned RTS just for the update
  //
  //  A   B*                  A                    B*
  //  |   |                   |                    |
  //  |   |         --->     /|                    |
  //  |   |                 / ReduceScatter(Local) |
  //  |   |                |  |         .----------'
  //  |   |                |  A*        |
  //   \ /                 |   \       /
  //   VarUpdate           |   VarUpdate* (updates a copied slice of the weight)
  //    |                  |    |
  //    |                  |    A''*
  //    |                  |    |
  //    |                  |  AllGather
  //    |                  |    |
  //    |                   \   A''
  //    |                    \  |
  //    |                  CopyVarUpdate  (updates the actual weight)
  //    |                       |
  //    A'                      A'
  // where A->A' is a weight
  //
  //
  // * = replica sharded tensor or operation
  auto proposedRTS = getReplicatedTensorShardingProposal(rtsTensors);

  // Insert CopyVarUpdates where required
  for (auto opId : proposedRTS.getVarUpdatesWithCopies()) {
    Op *op = graph.getOp(opId);

    if (VarUpdateOp *varUpdateOp = dynamic_cast<VarUpdateOp *>(op)) {
      TensorId inId  = varUpdateOp->inId(VarUpdateOp::getVarToUpdateInIndex());
      TensorId outId = varUpdateOp->outId(VarUpdateOp::getUpdatedVarOutIndex());
      TensorId tmpOutId = graph.getIr().createIntermediateTensorId(outId);

      Tensor *out = graph.getTensors().get(outId);
      varUpdateOp->disconnectOutTensor(out);
      varUpdateOp->createAndConnectOutTensor(
          VarUpdateOp::getUpdatedVarOutIndex(), tmpOutId);

      // At this point, it is unsafe to call setup() on the VarUpdate or
      // CopyVarUpdate due to RTS not having been propagated properly yet.
      // We can inherit the tensor info and setup later.
      Tensor *in     = graph.getTensors().get(inId);
      Tensor *tmpOut = graph.getTensors().get(tmpOutId);
      tmpOut->info   = out->info;

      logging::transform::trace(
          "[StreamingMemory] VarUpdate {} temporary tensors: [{}, {}, {}]",
          varUpdateOp->debugName(),
          in->info,
          tmpOut->info,
          out->info);

      CopyVarUpdateOp *copyVarUpdateOp =
          insertCopyVarUpdateOp(varUpdateOp, inId, tmpOutId, outId);

      if (!proposedRTS.isProposed(varUpdateOp->id)) {
        proposedRTS.insert(tmpOutId,
                           ReplicatedTensorShardingMethod::LocalScatter);
        proposedRTS.insert(outId, ReplicatedTensorShardingMethod::Forward);
        proposedRTS.insert(copyVarUpdateOp->id,
                           proposedRTS.getRefTensorId(varUpdateOp->id));
      }

      proposedRTS.setRefTensorId(copyVarUpdateOp->id,
                                 proposedRTS.getRefTensorId(varUpdateOp->id));

    } else {
      throw error("[StreamingMemory] {} is not a VarUpdateOp", op->debugName());
    }
  }

  // Update schedule
  auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::No);
  for (Op *op : schedule) {
    TensorId refId = proposedRTS.getRefTensorId(op->id);

    if (proposedRTS.isProposed(op->id)) {
      CommGroup shardingDomain;

      // RTS operator
      auto rtsIndices = op->getReplicatedTensorShardingIndices();
      for (auto indices : rtsIndices) {
        for (InIndex inIdx : indices.first) {
          Tensor *inTensor = op->input->tensor(inIdx);

          Tensor *varTensor = findRelatedVarTensor({inTensor});

          if (varTensor) {
            shardingDomain =
                tensorConfigs.at(varTensor).location.shardingDomain;
          }

          if (!rtsTensors.hasShard(inTensor->id)) {
            auto method = proposedRTS.getMethod(inTensor->id);

            switch (method) {
            case ReplicatedTensorShardingMethod::AllReduceToScatter: {
              RTSAllReduceToScatter(rtsTensors, inTensor, refId);
              break;
            }

            case ReplicatedTensorShardingMethod::LocalScatter: {
              TensorStreamingContext context;
              context.context = op->settings.executionContext;
              if (context.context == ExecutionContext::Normal) {
                if (isPhasedExecution()) {
                  context.phase = op->getOptionalExecutionPhase();
                } else {
                  context.preLoss = op->scheduledPreLoss;
                }
              }

              RTSLocalScatter(context, rtsTensors, inTensor, refId);
              break;
            }

            case ReplicatedTensorShardingMethod::Native:
            case ReplicatedTensorShardingMethod::Forward:
            default:
              throw error("[StreamingMemory] Tensor {} should have had a "
                          "sharded version available.",
                          inTensor->id);
            }
          }

          op->disconnectInTensor(inIdx, inTensor);
          op->connectInTensor(inIdx, rtsTensors.getShard(inTensor->id));
        }

        // Configure the op with all sharded inputs added
        logging::transform::trace("[StreamingMemory] Configuring {} for "
                                  "replicated tensor sharding "
                                  "(indices: {}->{}, shardingDomain: {})",
                                  op->debugName(),
                                  indices.first,
                                  indices.second,
                                  shardingDomain);

        op->configureForReplicatedTensorSharding({indices}, shardingDomain);
        // Add output tensors as RTS tensors
        for (auto outIndex : indices.second) {
          TensorId outId = op->outId(outIndex);
          // Tensor outId is now replicated tensor sharded
          rtsTensors.insert(outId, "", outId, stripAllReservedPrefixes(refId));
        }
      }

    } else {
      // Non-RTS operator
      auto inputs = op->input->tensorMap();

      for (auto &input : inputs) {
        InIndex inIdx    = input.first;
        Tensor *inTensor = input.second;
        if (rtsTensors.hasShard(inTensor->id)) {
          // The input tensor is RTS, but this Op is not RTS

          TensorId gatheredId = rtsTensors.getGathered(inTensor->id);
          TensorId tensorId   = rtsTensors.getTensor(inTensor->id);

          if (gatheredId.empty()) {
            auto ref = findRelatedVarTensor({inTensor});
            if (!ref) {
              throw internal_error("[StreamingMemory] Could not find related "
                                   "var tensor for sharded tensor {}. "
                                   "Related var tensor is required to ensure "
                                   "correct allGather output info.",
                                   inTensor->id);
            }
            TensorConfig tensorConfig = tensorConfigs.at(ref);
            TensorStreamingContext context;
            context.context = op->settings.executionContext;
            if (context.context == ExecutionContext::Normal) {
              if (isPhasedExecution()) {
                context.phase = op->getOptionalExecutionPhase();
              } else {
                context.preLoss = op->scheduledPreLoss;
              }
            }
            insertReplicatedAllGatherOp(tensorConfig,
                                        context,
                                        inTensor->id,
                                        gatheredId,
                                        stripAllReservedPrefixes(refId),
                                        tensorConfig.location.shardingDomain);

            rtsTensors.insert(inTensor->id,
                              gatheredId,
                              tensorId,
                              stripAllReservedPrefixes(refId));
          }
          // Disconnect sharded input tensor. Important to be done after
          // findRelatedVarTensor.
          op->disconnectInTensor(inIdx, inTensor);
          Tensor *gatheredTensor = graph.getTensors().get(gatheredId);
          if (gatheredTensor->hasProducer()) {
            auto consBefore = graph.topoCons->getBefores(op);
            for (auto before : consBefore) {
              // Also move the allGather behind the producer
              // Note that if both the optimizer & normal ops are interested
              // in this gathered tensor, then we would have to insert
              // multiple allGathers, if it ever comes up (T26236)
              graph.topoCons->insert(before, gatheredTensor->getProducer());
            }
          }
          op->connectInTensor(inIdx, gatheredId);
        }
      }
    }
    // Re-setup every Op
    std::stringstream ss;
    logging::transform::trace("[StreamingMemory] Setting up {}",
                              op->debugName());
    op->setup();
  }
}

void StreamingMemoryOpInserter::getTensorOpSchedule(
    Tensor *tensor,
    TensorConfig &tensorConfig) const {
  auto &sessionOptions = graph.getIr().getSessionOptions();
  if (tensor->getTensorTypeInfo()->type() == TensorType::Variable) {
    tensorConfig.ioSchedule =
        sessionOptions.executionPhaseSettings.weightIOSchedule;
    // Test optimizer state & accumulator separately, because a tensor can be
    // considered as being both (e.g. SGD1 momentum)
    if (tensor->isOptimizerStateTensor()) {
      tensorConfig.ioSchedule =
          sessionOptions.executionPhaseSettings.optimizerStateIOSchedule;
    }
    if (tensor->isAccumulatorTensor()) {
      tensorConfig.ioSchedule =
          sessionOptions.executionPhaseSettings.accumulatorIOSchedule;
    }
  } else {
    tensorConfig.ioSchedule =
        sessionOptions.executionPhaseSettings.activationIOSchedule;
  }
  tensorConfig.schedule = sessionOptions.executionPhaseSettings.schedule;
}

void StreamingMemoryOpInserter::getTensorConfig(Tensor *tensor) {

  auto producerOp = tensor->getProducerUnsafe();

  {
    auto it = tensorConfigs.find(tensor);
    if (it == tensorConfigs.end()) {
      tensorConfigs.insert({tensor, TensorConfig(graph)});
    }
  }

  TensorConfig &tensorConfig = tensorConfigs.at(tensor);

  tensorConfig.tensor     = tensor;
  tensorConfig.producerOp = producerOp;

  // Get the root variable tensor, if it exists
  getRootVarTensor(tensor, tensorConfig.rootVarTensor);

  // Add this tensor to the root tensor's descendants
  if (tensorConfig.rootVarTensor) {
    auto it = tensorConfigs.find(tensorConfig.rootVarTensor);
    if (it == tensorConfigs.end()) {
      tensorConfigs.insert({tensorConfig.rootVarTensor, TensorConfig(graph)});
    }
    tensorConfigs.at(tensorConfig.rootVarTensor)
        .descendantTensors.insert(tensor);
  }

  // Get all direct consumers
  getConsumerOps(tensor, tensorConfig.consumerOps);
  // Get all indirect (descendant) consumers
  for (Tensor *descendantTensor : tensorConfig.descendantTensors) {
    getConsumerOps(descendantTensor, tensorConfig.consumerOps);
  }
  // Sort all consumers in schedule order
  filterAndSortConsumerOps(tensorConfig.consumerOps);

  // Determine the storage location of the tensor
  // If this tensor has a root variable tensor, the location of that tensor
  // is used instead
  getTensorLocation(tensorConfig.rootVarTensor ? tensorConfig.rootVarTensor
                                               : tensor,
                    tensorConfig.location);

  // Determine the streaming configuration of the tensor
  getTensorStreamingConfig(tensor);

  // Determine the settings that new ops should inherit
  getTensorSettings(
      tensor, producerOp, tensorConfig.consumerOps, tensorConfig.settings);

  // Determine how the new ops should be scheduled
  getTensorOpSchedule(tensor, tensorConfig);

  if (tensorConfig.location.replicatedTensorSharding ==
      ReplicatedTensorSharding::On) {
    tensor->tensorLocationInfo.setSharded(true);
    logging::transform::debug(
        "[StreamingMemory] Enabling replica-sharded loading of tensor {}",
        tensor->id);
  }
}

void StreamingMemoryOpInserter::getRootVarTensor(Tensor *tensor,
                                                 Tensor *&rootTensor) const {
  // Check if the alias is an identity chain
  for (auto alias : aliasModel.allAliases(*tensor)) {
    if (alias->getTensorTypeInfo()->type() == TensorType::Variable) {
      rootTensor = alias;
    }
  }
}

void StreamingMemoryOpInserter::getConsumerOps(
    Tensor *tensor,
    ConsumerOpConfigs &consumerOps) const {

  // Get consuming ops.
  auto consumers = tensor->consumers.getOps();
  for (Op *consumer : consumers) {
    consumerOps.emplace_back(
        tensor, consumer, consumer->input->indices(tensor));
  }
}

void StreamingMemoryOpInserter::filterAndSortConsumerOps(
    ConsumerOpConfigs &consumerOps) const {
  // Process consumers in ascending order of phases
  std::sort(consumerOps.begin(),
            consumerOps.end(),
            [this](const ConsumerOpConfig &lhs, const ConsumerOpConfig &rhs) {
              return opScheduleMap.at(lhs.op) < opScheduleMap.at(rhs.op);
            });

  consumerOps.erase(std::unique(consumerOps.begin(), consumerOps.end()),
                    consumerOps.end());
}

void StreamingMemoryOpInserter::getTensorLocation(
    Tensor *tensor,
    TensorLocation &location) const {
  location             = determineTensorLocation(tensor);
  auto &sessionOptions = graph.getIr().getSessionOptions();
  if (sessionOptions.numIOTiles == 0) {
    location.loadTileSet    = TileSet::Compute;
    location.storageTileSet = TileSet::Compute;
  }
}

void StreamingMemoryOpInserter::getTensorOptionalVGraphId(
    Tensor *tensor,
    const Op *producerOp,
    const ConsumerOpConfigs &consumerOps,
    OptionalVGraphId &streamingVGID) const {

  // If we have a producer op, use the VGID of the producer.
  if (producerOp) {
    if (const IpuCopyOp *copy = dynamic_cast<const IpuCopyOp *>(producerOp)) {
      streamingVGID = copy->getDestIpu();
    } else {
      streamingVGID = producerOp->getOptionalVGraphId();
    }
  }

  // If we don't have a producer op, use the smallest consumer op VGID.
  for (auto consumerOp : consumerOps) {
    OptionalVGraphId consumerVGID = consumerOp.op->getOptionalVGraphId();
    if (IpuCopyOp *copyOp = dynamic_cast<IpuCopyOp *>(consumerOp.op)) {
      // consumerOps can contain non-direct consumers of 'tensor'
      if (copyOp->input->contains(tensor)) {
        auto sourceIpus = copyOp->getSourceIpus();
        if (sourceIpus.find(tensor->id) != sourceIpus.end()) {
          consumerVGID = copyOp->getSourceIpu(tensor->id);
        } else {
          throw error("[StreamingMemory] Unable to get source Ipu of {} for "
                      "IpuCopyOp {}",
                      tensor->id,
                      copyOp->debugName());
        }
      }
    }

    // Pick correct VGID for loading/storing the tensor,
    // if no producer exists
    if (!streamingVGID) {
      streamingVGID = consumerVGID;
    } else if (!tensor->hasProducer() && consumerVGID) {
      streamingVGID = std::min(*streamingVGID, *consumerVGID);
    }
  }
}

void StreamingMemoryOpInserter::getTensorProducerStreamingConfig(
    Tensor *tensor,
    const TensorLocation &location,
    const Op *producerOp,
    TensorStreamingMap &streamingMap) const {
  TensorStreamingContext context;
  if (producerOp) {
    context.context = producerOp->settings.executionContext;
    if (context.context == ExecutionContext::Normal) {
      if (isPhasedExecution()) {
        context.phase = producerOp->getOptionalExecutionPhase();
      } else {
        context.preLoss = producerOp->scheduledPreLoss;
      }
    }
    if (context.context == ExecutionContext::Normal && isPhasedExecution()) {
      // Special case for IpuCopyOp.
      if (const IpuCopyOp *copy = dynamic_cast<const IpuCopyOp *>(producerOp)) {
        if (copy->getMinSourceIpu() % num_stages !=
            copy->getDestIpu() % num_stages) {
          // Inter-phase copy, special case where the producer
          // phase is moved
          context.phase = *context.phase + 1;
        }
      }
    }
    // The tensor is produced & live in the producing context
    streamingMap[context].producer = true;
    streamingMap[context].live     = true;
    if (location.storage == TensorStorage::OffChip) {
      // The tensor is stored in the producing context
      streamingMap[context].store = true;
    }
  }
}

void StreamingMemoryOpInserter::getTensorStreamingConfig(Tensor *tensor) {

  auto &tensorConfig   = tensorConfigs.at(tensor);
  auto &streamingMap   = tensorConfig.streamingMap;
  auto const &location = tensorConfig.location;

  TensorStreamingConfig defaultStreamingConfig;
  getTensorOptionalVGraphId(tensor,
                            tensorConfig.producerOp,
                            tensorConfig.consumerOps,
                            defaultStreamingConfig.streamingVGID);

  auto fromHostContext =
      TensorStreamingContext(ExecutionContext::WeightsFromHostFragment,
                             OptionalExecutionPhase(),
                             ScheduledPreLoss::Undefined);
  auto accumulateContext =
      TensorStreamingContext(ExecutionContext::AccumulateOuterFragment,
                             OptionalExecutionPhase(),
                             ScheduledPreLoss::Undefined);
  auto toHostContext =
      TensorStreamingContext(ExecutionContext::WeightsToHostFragment,
                             OptionalExecutionPhase(),
                             ScheduledPreLoss::Undefined);

  // Create entries for every context and set everything to the default to
  // begin with
  streamingMap[fromHostContext]   = defaultStreamingConfig;
  streamingMap[accumulateContext] = defaultStreamingConfig;
  streamingMap[toHostContext]     = defaultStreamingConfig;
  if (isPhasedExecution()) {
    for (ExecutionPhase p = 0; p < num_phases * 2 - 1; ++p) {
      streamingMap[TensorStreamingContext(
          ExecutionContext::Normal, p, ScheduledPreLoss::Undefined)] =
          defaultStreamingConfig;
    }
  } else {
    streamingMap[TensorStreamingContext(ExecutionContext::Normal,
                                        OptionalExecutionPhase(),
                                        ScheduledPreLoss::Yes)] =
        defaultStreamingConfig;
    streamingMap[TensorStreamingContext(ExecutionContext::Normal,
                                        OptionalExecutionPhase(),
                                        ScheduledPreLoss::No)] =
        defaultStreamingConfig;
  }

  // If the user requests to keep the tensor OnChip but also wants to shard
  // the tensor between replica sets then we will load/store the tensor out of
  // the training loop (e.g. only once) to keep the tensor shard always live
  if (location.storage == TensorStorage::OnChip &&
      location.replicatedTensorSharding == ReplicatedTensorSharding::On) {
    streamingMap[fromHostContext].load = true;
    streamingMap[fromHostContext].live = true;
    streamingMap[toHostContext].store  = true;
    streamingMap[toHostContext].live   = true;
  }

  // Inherit load, store, gather settings from descendants
  for (Tensor *descendant : tensorConfig.descendantTensors) {
    if (descendant != tensor) {
      logging::transform::trace(
          "[StreamingMemory] Transferring streaming settings from {} to {}",
          descendant->id,
          tensor->id);
      for (auto &contextAndConfig : tensorConfigs.at(descendant).streamingMap) {
        streamingMap[contextAndConfig.first].load |=
            contextAndConfig.second.load;
        streamingMap[contextAndConfig.first].store |=
            contextAndConfig.second.store;
        streamingMap[contextAndConfig.first].gather |=
            contextAndConfig.second.gather;
        contextAndConfig.second.load   = false;
        contextAndConfig.second.store  = false;
        contextAndConfig.second.gather = false;
      }
    }
  }

  // Set producer & liveness config on relevant context if possible.
  getTensorProducerStreamingConfig(
      tensor, location, tensorConfig.producerOp, streamingMap);

  // TODO T25043: The remaining code in this function is non-trivial and would
  // benefit from unit testing.

  // Set load, store and gather for each relevant streaming context

  // We first set load/store, assuming we are loading/storing
  // everything in-phase. This is because we may need to insert an
  // replicationAllGather in these phases. Note that we do this by means of an
  // incremental algorithm which processes phases in order by iterating over
  // consumer ops in execution phase order.

  for (auto consumerOp : tensorConfig.consumerOps) {
    // Context of this consumer
    TensorStreamingContext consumerContext;
    consumerContext.context = consumerOp.op->settings.executionContext;
    if (consumerContext.context == ExecutionContext::Normal) {
      if (isPhasedExecution()) {
        consumerContext.phase = consumerOp.op->getOptionalExecutionPhase();
      } else {
        consumerContext.preLoss = consumerOp.op->scheduledPreLoss;
      }
    }

    std::vector<TensorStreamingConfig *> previousConfigs;
    TensorStreamingConfig *currentConfig = &streamingMap[consumerContext];

    currentConfig->consumers.push_back(consumerOp);
    // Live in consumer phases
    currentConfig->live = true;

    if (isPhasedExecution()) {
      // Treat last 2 phases as adjacent previous phases
      for (ExecutionPhase phaseDiff = 2; phaseDiff > 0; --phaseDiff) {
        auto previousContext = consumerContext;
        if (previousContext.phase) {
          // With the current implementation of phased execution,
          // IPUs always are associated with alternate
          // phases, regardless of num_stages.
          // Last phase relevant to this IPU was one or two phases ago.
          previousContext.phase = *(previousContext.phase) - phaseDiff;
          auto it               = streamingMap.find(previousContext);
          if (it != streamingMap.end()) {
            previousConfigs.push_back(&it->second);
          }
        }
      }
    }

    if (currentConfig->producer) {
      // Current phase is producer context -- there is no need to load in the
      // current context
    } else {
      // Not live, load in current phase
      currentConfig->load = true;
    }

    for (auto previousConfig : previousConfigs) {
      if (previousConfig->live) {
        // The tensor is live in the previous context.
        // Instead of storing it in a previous context and loading it in again
        // in this context we are going to keep it live.
        currentConfig->load = false;
        // If tensor was stored in a previous context, defer storing until the
        // current context.
        if (previousConfig->store) {
          currentConfig->store  = true;
          previousConfig->store = false;
        }
      }
    }

    // If the modifiers are not checked yet
    if (currentConfig->modifiers.empty()) {
      // Check if the consumer OP modifies the tensor, e.g. for weights
      // if yes, then the tensor requires to be backed-up at the end of
      // the phase
      getAliasedModifiersInContext(
          tensor, consumerContext, currentConfig->modifiers);
    }

    // If there are modifiers in this context
    if (!currentConfig->modifiers.empty()) {
      // Storing in this context
      currentConfig->store = true;
      for (auto previousConfig : previousConfigs) {
        // Not storing in previous contexts
        previousConfig->store = false;
      }
    }
  }

  // Tensors that are sharded must be gathered wherever they are set to be
  // loaded as per the current TensorStreamingConfig settings,
  // and have consumers.
  if (location.replicatedTensorSharding == ReplicatedTensorSharding::On) {
    for (auto &contextAndConfig : streamingMap) {
      auto &config  = contextAndConfig.second;
      config.gather = config.load && config.consumers.size() > 0;
    }
  }

  // If we are loading/storing in the fromHost/toHost fragments,
  // then we don't actually want to remember all the load/store in any other
  // context at all, but we do need the gathers.
  if (streamingMap[fromHostContext].load && streamingMap[toHostContext].store) {
    for (auto &contextAndConfig : streamingMap) {
      auto &context = contextAndConfig.first;
      auto &config  = contextAndConfig.second;
      if (context != fromHostContext && context != toHostContext) {
        // Load/store in no other contexts
        config.load  = false;
        config.store = false;
        // Live in all contexts
        config.live = true;
      }
    }
  }

  // If there are no loads, no stores are required either
  bool hasLoads = false;
  for (auto &contextAndConfig : streamingMap) {
    hasLoads |= contextAndConfig.second.load;
  }
  if (!hasLoads) {
    for (auto &contextAndConfig : streamingMap) {
      contextAndConfig.second.store = false;
    }
  }
}

void StreamingMemoryOpInserter::getTensorSettings(
    Tensor *tensor,
    const Op *producerOp,
    const ConsumerOpConfigs &consumerOps,
    Op::Settings &settings) const {

  // Inherit settings from producer or consumer
  if (producerOp) {
    settings = producerOp->settings;
  } else {
    if (!consumerOps.empty()) {
      settings = consumerOps.front().op->settings;
    }
    settings.batchSerializedPhase.reset();
  }
  settings.name.clear();
  settings.recomputeType  = RecomputeType::Checkpoint;
  settings.tensorLocation = TensorLocation();
}

void StreamingMemoryOpInserter::getModifiersInContext(
    Tensor *t,
    const TensorStreamingContext context,
    Ops &modifyingConsumerOps) const {
  for (Op *consumer : t->consumers.getOps()) {
    for (InIndex in : consumer->input->indices(t)) {
      auto regions = consumer->modifies(in);
      // Context of this consumer
      TensorStreamingContext consumerContext;
      consumerContext.context = consumer->settings.executionContext;
      if (consumerContext.context == ExecutionContext::Normal) {
        if (isPhasedExecution()) {
          consumerContext.phase = consumer->getOptionalExecutionPhase();
        } else {
          consumerContext.preLoss = consumer->scheduledPreLoss;
        }
      }
      if (!std::all_of(regions.begin(),
                       regions.end(),
                       [](const view::Region &r) {
                         return r.isEmpty() ||
                                r.getAccessType() == view::AccessType::Read;
                       }) &&
          consumerContext == context) {
        if (std::find(modifyingConsumerOps.begin(),
                      modifyingConsumerOps.end(),
                      consumer) == modifyingConsumerOps.end()) {
          modifyingConsumerOps.push_back(consumer);
        }
      }
    }
  }
}

void StreamingMemoryOpInserter::getAliasedModifiersInContext(
    Tensor *tensor,
    TensorStreamingContext context,
    Ops &modifyingConsumerOps) const {
  getModifiersInContext(tensor, context, modifyingConsumerOps);
  for (auto alias : aliasModel.allAliases(*tensor)) {
    getModifiersInContext(alias, context, modifyingConsumerOps);
  }
}

void StreamingMemoryOpInserter::setLoadingOpPhaseAndPriority(
    Op *op,
    const Tensor *const tensor,
    const TensorConfig &tensorConfig,
    const TensorStreamingContext &context) {
  if (context.context == ExecutionContext::Normal && isPhasedExecution()) {
    if (tensorConfig.ioSchedule == ExecutionPhaseIOSchedule::OnDemand) {
      if (context.phase) {
        // RemoteLoad in current phase
        op->setExecutionPhase(context.phase);
        op->settings.schedulePriority = 0.0f;
      }
      // Optimizer states OnDemand are given a priority that delays the load
      // until the optimizer needs it
      if (tensorConfig.tensor->getTensorTypeInfo()->type() ==
              TensorType::Variable &&
          tensorConfig.tensor->isOptimizerStateTensor() &&
          !tensorConfig.tensor->isAccumulatorTensor()) {
        setPriority(op, isPhasedExecution(), true, tensorConfig.schedule);
      }
    } else if (tensorConfig.ioSchedule == ExecutionPhaseIOSchedule::Preload) {
      // RemoteLoad at the end of the previous phase
      if (context.phase) {
        op->setExecutionPhase(*(context.phase) - 1);
      }
      setPriority(op, isPhasedExecution(), false, tensorConfig.schedule);
    }
  } else {
    op->setExecutionPhase({});
    op->settings.schedulePriority = 0.0f;
  }
}

RemoteLoadOp *StreamingMemoryOpInserter::insertRemoteLoadOp(
    const TensorConfig &tensorConfig,
    const TensorStreamingContext context,
    TensorId &loadedTensorId) {

  RemoteLoadOp *remoteLoad = nullptr;
  InitOp *init             = nullptr;

  if (isPhasedExecution() && context.context == ExecutionContext::Normal) {
    // Phase must be set when storing in phase.
    if (!context.phase) {
      throw internal_error(
          "Expected phase to be set when inserting RemoteLoadOp in-phase.");
    }
  }

  // Log this.
  logging::transform::trace("[StreamingMemory] Adding remote load of {} ({}) "
                            "in streaming context {}",
                            loadedTensorId,
                            tensorConfig.tensor->id,
                            context);

  auto remoteLoadOp = std::make_unique<RemoteLoadOp>(
      Onnx::CustomOperators::RemoteLoad, tensorConfig.settings);
  remoteLoad = remoteLoadOp.get();

  // Setting the execution context ensures it's scheduled in the correct
  // fragment
  remoteLoad->settings.executionContext = context.context;
  remoteLoad->scheduledPreLoss          = context.preLoss;

  remoteLoad->settings.optimizerOp   = false;
  remoteLoad->settings.recomputeType = RecomputeType::Checkpoint;
  remoteLoad->setVirtualGraphId(
      tensorConfig.streamingMap.at(context).streamingVGID);
  graph.moveIntoGraph(std::move(remoteLoadOp));

  remoteLoad->connectInTensor(RemoteStoreOp::getRemoteBufferOffsetInIndex(),
                              getRemoteArg(tensorConfig.tensor->id));

  TensorId initTensorId = generateInitTensorId(tensorConfig.tensor->id);
  TensorId prevLoadId   = getPreviousLoadedTensorId(tensorConfig.tensor->id);
  loadedTensorId        = generateLoadedTensorId(tensorConfig.tensor->id);
  TensorId inTensorId;

  // If this tensor is loaded here, it may be loaded in multiple contexts. In
  // this case getPreviousLoadedTensorId is used to 'chain' the tensor inputs.
  // This is necessary to allow be able to outline the RemoteLoadOp code
  if (!prevLoadId.empty()) {
    // Tensor might not have a true producer op, but was previously
    // loaded by a RemoteLoad
    inTensorId = prevLoadId;
  } else if (tensorConfig.producerOp) {
    // Tensor has a true producer op
    inTensorId = tensorConfig.tensor->id;
  } else {
    TensorInfo initInfo = tensorConfig.tensor->info;

    if (tensorConfig.location.replicatedTensorSharding ==
        ReplicatedTensorSharding::On) {
      // The original shape becomes the RTS shape
      // The actual tensor shape is now:
      // (initInfo.nelms() - 1) / replicationFactor + 1

      auto rf = replicationFactor;
      if (tensorConfig.location.shardingDomain.replicaGroupSize > 0 &&
          (tensorConfig.location.shardingDomain.type ==
               CommGroupType::Consecutive ||
           tensorConfig.location.shardingDomain.type ==
               CommGroupType::Orthogonal)) {
        rf = tensorConfig.location.shardingDomain.replicaGroupSize;
      }

      Shape oldShape = initInfo.shape();
      Shape newShape = {(initInfo.nelms() - 1) / rf + 1};

      logging::transform::trace(
          "[StreamingMemory] RTS tensor {} shapes: {} -> {}",
          tensorConfig.tensor->id,
          oldShape,
          newShape);

      initInfo.set(initInfo.dataType(), newShape, oldShape);
    }

    // InitOp as a "producer" op
    auto initOp = std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                           initInfo,
                                           tensorConfig.tensor->tensorType(),
                                           InitType::NoInit,
                                           tensorConfig.settings);
    init        = initOp.get();

    init->settings.executionContext = context.context;
    init->scheduledPreLoss          = context.preLoss;

    init->setVirtualGraphId(
        tensorConfig.streamingMap.at(context).streamingVGID);
    graph.moveIntoGraph(std::move(initOp));
    init->createAndConnectOutTensor(InitOp::getOutIndex(), initTensorId);
    init->setup();
    inTensorId = initTensorId;

    // Do Init on IO tiles
    init->settings.tileSet = tensorConfig.location.loadTileSet;
  }

  // Set priority and phase on init & load
  if (init) {
    setLoadingOpPhaseAndPriority(
        init, tensorConfig.tensor, tensorConfig, context);
  }
  if (remoteLoad) {
    setLoadingOpPhaseAndPriority(
        remoteLoad, tensorConfig.tensor, tensorConfig, context);
  }

  // RemoteLoad always needs both an input and an output,
  // for outlining and aliasing purposes

  // RemoteLoad updates the inTensorId...
  remoteLoad->connectInTensor(RemoteLoadOp::getLocalTensorInIndex(),
                              inTensorId);
  // ... and aliases it under loadedTensorId
  remoteLoad->createAndConnectOutTensor(RemoteLoadOp::getLocalTensorOutIndex(),
                                        loadedTensorId);

  remoteLoad->setup();

  // Do RemoteLoad on IO tiles
  remoteLoad->settings.tileSet = tensorConfig.location.loadTileSet;

  return remoteLoad;
}

ReplicatedAllGatherOp *StreamingMemoryOpInserter::insertReplicatedAllGatherOp(
    const TensorConfig &tensorConfig,
    const TensorStreamingContext context,
    const TensorId &loadedTensorId,
    TensorId &gatheredTensorId,
    const TensorId &referenceTensorId,
    const CommGroup &group) {
  ReplicatedAllGatherOp *allGather = nullptr;

  auto loadedInfo = graph.getTensors().get(loadedTensorId)->info;

  logging::transform::trace("[StreamingMemory] Adding replicated all gather "
                            "of {} ({}) in context {}. Shapes: {} -> {}",
                            loadedTensorId,
                            tensorConfig.tensor->id,
                            context,
                            loadedInfo.shape(),
                            loadedInfo.metaShape());

  TensorInfo gatherInfo(loadedInfo.dataType(), loadedInfo.metaShape());

  // Execute replicated allgather to collect the full weight
  // tensor from the individual replicas
  auto allGatherOp = std::make_unique<ReplicatedAllGatherOp>(
      Onnx::CustomOperators::ReplicatedAllGather,
      group,
      tensorConfig.settings,
      gatherInfo);
  allGather = allGatherOp.get();

  allGather->settings.executionContext = context.context;
  allGather->scheduledPreLoss          = context.preLoss;

  if (isPhasedExecution() && context.phase) {
    allGather->setExecutionPhase(*(context.phase) - 1);
    setLoadingOpPhaseAndPriority(
        allGather, tensorConfig.tensor, tensorConfig, context);
  } else {
    allGather->setExecutionPhase({});
  }

  allGather->settings.optimizerOp   = false;
  allGather->settings.recomputeType = RecomputeType::Checkpoint;
  // RemoteLoad at the end of the previous phase, so that load
  // is executed before inter-IPU copy
  allGather->setVirtualGraphId(
      tensorConfig.streamingMap.at(context).streamingVGID);
  graph.moveIntoGraph(std::move(allGatherOp));

  allGather->connectInTensor(ReplicatedAllGatherOp::getInIndex(),
                             loadedTensorId);

  allGather->connectInTensor(ReplicatedAllGatherOp::getCollectiveLinkedIndex(),
                             getRemoteArg(referenceTensorId));

  gatheredTensorId = generateGatheredTensorId(tensorConfig.tensor->id);

  allGather->createAndConnectOutTensor(ReplicatedAllGatherOp::getOutIndex(),
                                       gatheredTensorId);

  allGather->setup();

  // Do AllGather on IO tiles
  allGather->settings.tileSet = tensorConfig.location.loadTileSet;

  return allGather;
}

CopyVarUpdateOp *StreamingMemoryOpInserter::insertCopyVarUpdateOp(
    VarUpdateOp *op,
    const TensorId &varTensorId,
    const TensorId &updaterTensorId,
    const TensorId &outTensorId) {

  logging::transform::trace(
      "[StreamingMemory] Adding CopyVarUpdate "
      "for variable {} -> {} (updater {}) for VarUpdate {}.",
      varTensorId,
      outTensorId,
      updaterTensorId,
      op->debugName());

  auto copyVarUpdateOpUp = std::make_unique<CopyVarUpdateOp>(op->settings);
  auto copyVarUpdateOp   = copyVarUpdateOpUp.get();
  graph.moveIntoGraph(std::move(copyVarUpdateOpUp));

  copyVarUpdateOp->connectInTensor(CopyVarUpdateOp::getVarToUpdateInIndex(),
                                   varTensorId);
  copyVarUpdateOp->connectInTensor(CopyVarUpdateOp::getUpdaterInIndex(),
                                   updaterTensorId);
  copyVarUpdateOp->connectOutTensor(CopyVarUpdateOp::getUpdatedVarOutIndex(),
                                    outTensorId);

  graph.topoCons->transfer(op, copyVarUpdateOp, false);

  return copyVarUpdateOp;
}

ReplicatedReduceScatterOp *
StreamingMemoryOpInserter::insertReplicatedReduceScatterOp(
    const TensorConfig &tensorConfig,
    const TensorStreamingContext context,
    const TensorId &inTensorId,
    const TensorId &outTensorId,
    const TensorId &weightTensorId,
    const CollectiveOperator &collectiveOp,
    const CommGroup &group) {

  logging::transform::trace(
      "[StreamingMemory] Adding replicated reduce scatter "
      "of {} -> {} in context {}. CollectiveOperator: {}",
      inTensorId,
      outTensorId,
      context,
      collectiveOp);

  auto replicatedReduceScatterOp = std::make_unique<ReplicatedReduceScatterOp>(
      Onnx::CustomOperators::ReplicatedReduceScatter,
      collectiveOp,
      group,
      tensorConfig.settings);
  auto replicatedReduceScatter = replicatedReduceScatterOp.get();
  graph.moveIntoGraph(std::move(replicatedReduceScatterOp));

  replicatedReduceScatter->connectInTensor(
      ReplicatedReduceScatterOp::getInIndex(), inTensorId);

  replicatedReduceScatter->connectInTensor(
      ReplicatedAllGatherOp::getCollectiveLinkedIndex(),
      getRemoteArg(weightTensorId));

  if (graph.getTensors().contains(outTensorId)) {
    replicatedReduceScatter->connectOutTensor(
        ReplicatedReduceScatterOp::getOutIndex(), outTensorId);
  } else {
    replicatedReduceScatter->createAndConnectOutTensor(
        ReplicatedReduceScatterOp::getOutIndex(), outTensorId);
  }

  replicatedReduceScatter->settings.tileSet =
      tensorConfig.location.storageTileSet;
  replicatedReduceScatter->settings.optimizerOp = false;

  if (context.context == ExecutionContext::Normal && isPhasedExecution()) {
    if (context.phase) {
      replicatedReduceScatter->setExecutionPhase(context.phase);
    }
    setPriority(replicatedReduceScatter,
                isPhasedExecution(),
                false,
                tensorConfig.schedule);
  } else {
    replicatedReduceScatter->setExecutionPhase({});
    replicatedReduceScatter->settings.schedulePriority = 0.0f;
  }

  replicatedReduceScatter->settings.executionContext = context.context;
  replicatedReduceScatter->scheduledPreLoss          = context.preLoss;

  replicatedReduceScatter->setup();

  return replicatedReduceScatter;
}

RemoteStoreOp *StreamingMemoryOpInserter::insertRemoteStoreOp(
    const TensorConfig &tensorConfig,
    const TensorStreamingContext context,
    const TensorId &loadedTensorId) {
  RemoteStoreOp *remoteStore = nullptr;

  if (isPhasedExecution() && context.context == ExecutionContext::Normal) {
    // Phase must be set when storing in phase.
    if (!context.phase) {
      throw internal_error(
          "Expected phase to be set when inserting RemoteStoreOp in-phase.");
    }
  }

  // Log this.
  logging::transform::trace(
      "[StreamingMemory] Adding remote store of {} ({}) in context {}",
      loadedTensorId,
      tensorConfig.tensor->id,
      context);

  auto remoteStoreOp = std::make_unique<RemoteStoreOp>(
      Onnx::CustomOperators::RemoteStore, tensorConfig.settings);
  remoteStore = remoteStoreOp.get();

  if (context.context == ExecutionContext::Normal && isPhasedExecution()) {
    if (context.phase) {
      remoteStore->setExecutionPhase(context.phase);
    }
    if (tensorConfig.ioSchedule == ExecutionPhaseIOSchedule::Preload) {
      setPriority(
          remoteStore, isPhasedExecution(), false, tensorConfig.schedule);
    } else if (tensorConfig.tensor->getTensorTypeInfo()->type() ==
                   TensorType::Variable &&
               tensorConfig.tensor->isOptimizerStateTensor() &&
               !tensorConfig.tensor->isAccumulatorTensor()) {
      setPriority(
          remoteStore, isPhasedExecution(), true, tensorConfig.schedule);
    }
  } else {
    remoteStore->setExecutionPhase({});
    remoteStore->settings.schedulePriority = 0.0f;
  }

  remoteStore->settings.optimizerOp   = false;
  remoteStore->settings.recomputeType = RecomputeType::Checkpoint;
  remoteStore->setVirtualGraphId(
      tensorConfig.streamingMap.at(context).streamingVGID);
  graph.moveIntoGraph(std::move(remoteStoreOp));

  remoteStore->connectInTensor(RemoteStoreOp::getRemoteBufferOffsetInIndex(),
                               getRemoteArg(tensorConfig.tensor->id));
  remoteStore->connectInTensor(RemoteStoreOp::getLocalTensorInIndex(),
                               loadedTensorId);

  // Setting the execution context ensures it's scheduled in the correct
  // fragment
  remoteStore->settings.executionContext = context.context;
  remoteStore->scheduledPreLoss          = context.preLoss;

  // Do store on IO tiles
  remoteStore->settings.tileSet = tensorConfig.location.loadTileSet;

  remoteStore->setup();

  return remoteStore;
}

void StreamingMemoryOpInserter::sanitizeOps() const {

  auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::No);

  for (Op *op : schedule) {
    if (op->hasExecutionPhase()) {
      sanitizePlacementAnnotation(op, op->getExecutionPhase());
      // Assign correct schedulePriority to all inter-IPU copies
      if (op->isIpuCopyOp()) {
        // Special type of IPUCopy between execution phases
        // schedulePriority before RemoteStore but after RemoteLoad
        IpuCopyOp *copy = dynamic_cast<IpuCopyOp *>(op);
        if (!op->copiesOptimizerTensors() &&
            copy->getMinSourceIpu() % num_stages !=
                copy->getDestIpu() % num_stages) {
          // Inter-phase copy
          setPriority(copy,
                      isPhasedExecution(),
                      false,
                      graph.getIr()
                          .getSessionOptions()
                          .executionPhaseSettings.schedule);
        } else {
          op->settings.schedulePriority = 0.0;
        }
        // Always keep IPU copy checkpointed
        if (op->settings.recomputeType == RecomputeType::Undefined) {
          op->settings.recomputeType = RecomputeType::Checkpoint;
          logging::transform::trace("[StreamingMemory] {} set to Checkpoint",
                                    op->debugName());
        }
      }
      // OnChip random seed operator if not set by the user
      if (!op->settings.tensorLocation) {
        if (op->opid == Onnx::CustomOperators::GetRandomSeed) {
          op->settings.tensorLocation = TensorLocation();
          logging::transform::trace("[StreamingMemory] {} set to OnChip",
                                    op->debugName());
        }
      }
    }
  }
}

void StreamingMemoryOpInserter::verifyPlacementConsistency(const Op *op) const {
  if (op->hasVirtualGraphId() && op->hasExecutionPhase()) {
    if (op->getExecutionPhase() % num_stages !=
        op->getVirtualGraphId() % num_stages) {
      throw error("[StreamingMemory] Op {} inconsistent execution phase {} "
                  "and virtual graph ID {}",
                  op->debugName(),
                  op->getExecutionPhase(),
                  op->getVirtualGraphId());
    }
  }
}

TensorLocation
StreamingMemoryOpInserter::determineTensorLocation(Tensor *tensor) const {
  auto &ir             = graph.getIr();
  auto &sessionOptions = ir.getSessionOptions();
  auto id              = tensor->id;
  auto type            = tensor->tensorType();
  auto producerOp      = tensor->getProducerUnsafe();
  auto isActivation    = type == TensorType::ActGrad;
  auto isOptimizerState =
      type == TensorType::Variable && tensor->isOptimizerStateTensor();
  auto isAccumulator =
      type == TensorType::Variable && tensor->isAccumulatorTensor();
  auto isWeight =
      type == TensorType::Variable && !isOptimizerState && !isAccumulator;

  auto isSharedOptimizerAndAccumulator = isOptimizerState && isAccumulator;
  auto accumulationDisabled = !(sessionOptions.enableGradientAccumulation &&
                                sessionOptions.accumulationFactor > 1);

  // Result variable.
  OptionalTensorLocation result;
  const char *logReason = "";

  const auto overrideIt =
      sessionOptions.tensorLocationSettingsOverride.find(id);
  bool haveTensorLocationSettingOverride =
      (overrideIt != sessionOptions.tensorLocationSettingsOverride.end());

  if (haveTensorLocationSettingOverride) {

    // If we have a valid tensorLocationSettingsOverride then the user is
    // telling us explicitly where to put this tensor, so use that.

    result    = overrideIt->second;
    logReason = "tensorLocationSettingsOverride in SessionOptions";

  } else {

    // If we don't have an entry for the tensor in
    // tensorLocationSettingsOverride then check to see if the operator that
    // produces this tensor (if known) was created with an
    // 'outputTensorLocation' attribute. If so, use that. If not, see if the
    // tensor's type has associated tensor location settings and use that. If
    // all else fails, use default on-chip.

    if (!result && producerOp) {

      // If a producing operator is known and it is set to have a tensor
      // location (and is not set to recompute), use that.

      if (producerOp->settings.recomputeType == RecomputeType::Recompute) {
        logging::transform::warn(
            "[StreamingMemory] Ignoring output tensor location "
            "attribute on tensor {} because the "
            "tensor is set to recompute",
            id);
      } else {
        result    = producerOp->settings.tensorLocation;
        logReason = "builder attribute";
      }
    }

    if (!result) {

      // If we still don't have a non-default tensor location setting and the
      // tensor belongs to a group we have tensor location settings for, use
      // those tensor location settings.

      if (isActivation) {
        // Use activation tensor location settings.
        result = sessionOptions.activationTensorLocationSettings.location;
        logReason =
            "activationTensorLocationSettings.location in SessionOptions";
      }

      if (isWeight) {
        // Use weight tensor location settings.
        result    = sessionOptions.weightTensorLocationSettings.location;
        logReason = "weightTensorLocationSettings.location in SessionOptions";
      }

      if (isOptimizerState) {
        // Use optimizer state tensor location settings.
        result = sessionOptions.optimizerStateTensorLocationSettings.location;
        logReason = "weightTensorLocationSettings.location in SessionOptions";
      }

      if (isAccumulator &&
          !(isSharedOptimizerAndAccumulator && accumulationDisabled)) {
        // Use optimizer state tensor location settings.
        result = sessionOptions.accumulatorTensorLocationSettings.location;
        logReason =
            "accumulatorTensorLocationSettings.location in SessionOptions";
      }
    }

    if (result && (*result).storage == TensorStorage::OffChip) {

      // If we are planning to offload the tensor to off-chip memory but the
      // tensor belongs to a group of tensors for which we have a tensor
      // location setting that specifies a minimum size for offloading and
      // this tensor is smaller than this minimum size, revert back to
      // on-chip.

      if (isActivation &&
          tooSmallForOffChip(sessionOptions.activationTensorLocationSettings,
                             tensor)) {
        result    = TensorStorage::OnChip;
        logReason = "activationTensorLocationSettings.minElementsForOffChip in "
                    "SessionOptions";
      }

      if (isWeight &&
          tooSmallForOffChip(sessionOptions.weightTensorLocationSettings,
                             tensor)) {
        result    = TensorStorage::OnChip;
        logReason = "weightTensorLocationSettings.minElementsForOffChip in "
                    "SessionOptions";
      }

      if (isOptimizerState &&
          tooSmallForOffChip(
              sessionOptions.optimizerStateTensorLocationSettings, tensor)) {
        result = TensorStorage::OnChip;
        logReason =
            "optimizerStateTensorLocationSettings.minElementsForOffChip in "
            "SessionOptions";
      }
    }

    if (result) {
      if ((*result).replicatedTensorSharding == ReplicatedTensorSharding::On) {
        if (isActivation) {
          (*result).replicatedTensorSharding = ReplicatedTensorSharding::Off;
        }
        if (isWeight &&
            tooSmallForReplicatedTensorSharding(
                sessionOptions.weightTensorLocationSettings, tensor)) {
          (*result).replicatedTensorSharding = ReplicatedTensorSharding::Off;
        }
        if (isOptimizerState &&
            tooSmallForReplicatedTensorSharding(
                sessionOptions.optimizerStateTensorLocationSettings, tensor)) {
          (*result).replicatedTensorSharding = ReplicatedTensorSharding::Off;
        }
      }

      if ((*result).replicatedTensorSharding == ReplicatedTensorSharding::On) {
        if ((!sessionOptions.enableReplicatedGraphs) ||
            (sessionOptions.replicatedGraphCount <= 1)) {
          logging::transform::warn(
              "[StreamingMemory] Unable to shard tensor {} "
              "due to lack of replication",
              id);
          (*result).replicatedTensorSharding = ReplicatedTensorSharding::Off;
        }
      }

      if ((*result).storage != TensorStorage::OnChip &&
          tensor->isOptimizerTensor()) {

        // Don't offload optimizer tensors to off-chip memory.
        (*result).storage = TensorStorage::OnChip;
        logReason         = "it being an optimizer tensor";
      }

      if ((*result).storage != TensorStorage::OnChip &&
          isConstOrCopyOfConst(tensor)) {

        // Don't offload constant (or copy of constant) tensors to off-chip
        // memory.
        (*result).storage = TensorStorage::OnChip;
        logReason         = "it being an constant or copy-of-constant tensor";
      }
    }
  }

  // Finally, it's possible at this point nothing has set the result
  // yet, in which case we default to TensorStorage::OnChip.

  if (!result) {
    result            = TensorLocation();
    (*result).storage = TensorStorage::OnChip;
    logReason         = "absence of tensor location settings";
  }

  // Log the result.
  logging::transform::debug(
      "[StreamingMemory] Determined tensor {} should use tensor "
      "location {} (due to {})",
      id,
      *result,
      logReason);

  return *result;
}

// Make sure the op has a valid placement annotation
void StreamingMemoryOpInserter::sanitizePlacementAnnotation(
    Op *op,
    ExecutionPhase phase) const {
  for (Tensor *input : op->input->tensors()) {
    if (input->hasProducer() && input->getProducer()->hasExecutionPhase()) {
      phase = std::max(phase, input->getProducer()->getExecutionPhase());
    }
  }
  for (Op *before : graph.topoCons->getBefores(op)) {
    if (before->hasExecutionPhase()) {
      phase = std::max(phase, before->getExecutionPhase());
    }
  }

  bool remapPhase = !op->hasExecutionPhase() || op->getExecutionPhase() < phase;

  IpuCopyOp *copy = dynamic_cast<IpuCopyOp *>(op);
  if (copy &&
      copy->getMinSourceIpu() % num_stages != copy->getDestIpu() % num_stages) {
    // Inter-phase copy, needs to be associated with the
    // source execution phase and virtual graph
    if (copy->getMinSourceIpu() % num_stages != phase % num_stages) {
      phase -= 1;
      remapPhase = true;
    }
  }

  if (remapPhase) {
    logging::transform::debug(
        "[StreamingMemory] Mapping operator {} to phase {} -> {}",
        op->debugName(),
        op->hasExecutionPhase() ? op->getExecutionPhase()
                                : unusedExecutionPhase,
        phase);
    op->setExecutionPhase(phase);
  }
  if (!op->hasVirtualGraphId() && !op->isIpuCopyOp()) {
    VGraphId vgid = phase % num_stages;
    logging::transform::debug(
        "[StreamingMemory] Mapping operator {} to VGID {} -> {}",
        op->debugName(),
        op->hasVirtualGraphId() ? op->getVirtualGraphId() : unusedVGraphId,
        vgid);
    op->setVirtualGraphId(vgid);
  }

  if (op->settings.executionContext != ExecutionContext::Normal) {
    op->setExecutionPhase({});
  }

  verifyPlacementConsistency(op);
}

void StreamingMemoryOpInserter::logTensorStreamingConfig(
    const TensorConfig &tensorConfig) const {

  const auto &tensor       = tensorConfig.tensor;
  const auto &streamingMap = tensorConfig.streamingMap;

  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {

    std::stringstream ss;
    ss << "[StreamingMemory] ";
    ss << tensor->id << " -- streaming status: ";

    for (auto &contextAndConfig : streamingMap) {
      auto &context = contextAndConfig.first;
      auto &config  = contextAndConfig.second;
      ss << "| ";
      ss << context << " ";
      if (config.producer) {
        // Produced in this phase
        ss << "P";
      }
      // Using std::map::find as it can be used on a const map.
      if (config.load) {
        // Loaded in this phase
        ss << "L";
      }
      if (config.gather) {
        // Gathered in this phase
        ss << "G";
      }
      if (config.store) {
        // Stored in this phase
        ss << "S";
      }
      if (config.live) {
        // Live in this phase
        ss << "A ";
      } else {
        ss << "_ ";
      }
    }
    ss << "|";

    logging::transform::trace(ss.str());
  }
}

TensorId StreamingMemoryOpInserter::getRemoteArg(TensorId tensorId) {

  auto arg_tensor_id = getRemoteArgTensorId(tensorId);
  TensorInfo argTensorInfo(DataType::INT32, {1});
  std::vector<int> idData(1, 0);
  if (std::find(remoteArgIds.begin(), remoteArgIds.end(), arg_tensor_id) ==
      remoteArgIds.end()) {
    graph.getTensors().addConstInit(
        arg_tensor_id, argTensorInfo, reinterpret_cast<void *>(idData.data()));
    remoteArgIds.push_back(arg_tensor_id);
  }
  return arg_tensor_id;
}

TensorId StreamingMemoryOpInserter::getPreviousLoadedTensorId(TensorId id) {
  TensorId loadedTensorId;
  auto &counter = loadedTensorCounter[id];
  if (counter > 0) {
    loadedTensorId = id + "__l" + std::to_string(counter - 1);
  }
  return loadedTensorId;
}

TensorId StreamingMemoryOpInserter::generateInitTensorId(TensorId id) {
  // There will be at most one init tensor ID for each tensor
  TensorId initTensorId = id + "__init";
  return initTensorId;
}

TensorId StreamingMemoryOpInserter::generateLoadedTensorId(TensorId id) {
  // The loadedTensorId id needs to be unique as the same tensor may be
  // loaded in multiple streaming contexts
  auto &counter           = loadedTensorCounter[id];
  TensorId loadedTensorId = id + "__l" + std::to_string(counter);
  ++counter;
  return loadedTensorId;
}

TensorId StreamingMemoryOpInserter::generateGatheredTensorId(TensorId id) {
  // The gatheredTensorId id needs to be unique as the same tensor may be
  // gathered in multiple streaming contexts
  auto &counter             = gatheredTensorCounter[id];
  TensorId gatheredTensorId = id + "__g" + std::to_string(counter);
  ++counter;
  return gatheredTensorId;
}

bool StreamingMemoryOpInserter::tooSmallForOffChip(
    const TensorLocationSettings &tensorLocationSettings,
    Tensor *tensor) {
  return (tensor->info.nelms() < tensorLocationSettings.minElementsForOffChip);
}

bool StreamingMemoryOpInserter::tooSmallForReplicatedTensorSharding(
    const TensorLocationSettings &tensorLocationSettings,
    Tensor *tensor) {
  return (tensor->info.nelms() <
          tensorLocationSettings.minElementsForReplicatedTensorSharding);
}

StreamingMemoryOpInserter::TensorStreamingContext::TensorStreamingContext()
    : context(ExecutionContext::Normal), phase(OptionalExecutionPhase()),
      preLoss(ScheduledPreLoss::Undefined) {}

StreamingMemoryOpInserter::TensorStreamingContext::TensorStreamingContext(
    ExecutionContext context_,
    OptionalExecutionPhase phase_,
    ScheduledPreLoss preLoss_)
    : context(context_), phase(phase_), preLoss(preLoss_) {}

bool StreamingMemoryOpInserter::TensorStreamingContext::operator<(
    const TensorStreamingContext &rhs) const {
  std::vector<int> lhsVec;
  lhsVec.reserve(3);
  std::vector<int> rhsVec;
  rhsVec.reserve(3);

  auto contextToValue = [](const TensorStreamingContext &c) {
    switch (c.context) {
    case ExecutionContext::WeightsFromHostFragment:
      return 0;
    case ExecutionContext::Normal:
      return 1;
    case ExecutionContext::AccumulateOuterFragment:
      return 2;
    case ExecutionContext::WeightsToHostFragment:
      return 3;
    case ExecutionContext::OptimizerFromHostFragment:
      return 4;
    default:
      throw error("Unexpected ExecutionContext {}", c.context);
    }
  };

  auto preLossToValue = [](const TensorStreamingContext &c) {
    switch (c.preLoss) {
    case ScheduledPreLoss::Undefined:
      return 0;
    case ScheduledPreLoss::Yes:
      return 1;
    case ScheduledPreLoss::No:
      return 2;
    default:
      throw error("Unexpected ScheduledPreLoss {}", c.context);
    }
  };

  lhsVec.push_back(contextToValue(*this));
  lhsVec.push_back(phase ? *phase : unusedExecutionPhase);
  lhsVec.push_back(preLossToValue(*this));

  rhsVec.push_back(contextToValue(rhs));
  rhsVec.push_back(rhs.phase ? *(rhs.phase) : unusedExecutionPhase);
  rhsVec.push_back(preLossToValue(rhs));

  return lhsVec < rhsVec;
}

bool StreamingMemoryOpInserter::TensorStreamingContext::operator==(
    const TensorStreamingContext &rhs) const {
  return context == rhs.context && phase == rhs.phase && preLoss == rhs.preLoss;
}

bool StreamingMemoryOpInserter::TensorStreamingContext::operator!=(
    const TensorStreamingContext &rhs) const {
  return context != rhs.context || phase != rhs.phase || preLoss != rhs.preLoss;
}

std::ostream &
operator<<(std::ostream &output,
           const StreamingMemoryOpInserter::TensorStreamingContext &c) {
  output << "[";
  output << c.context;
  if (c.phase) {
    output << " " << *(c.phase);
  }
  if (c.preLoss != ScheduledPreLoss::Undefined) {
    output << " " << c.preLoss;
  }
  output << "]";
  return output;
}

std::ostream &
operator<<(std::ostream &output,
           const StreamingMemoryOpInserter::ReplicatedTensorShardingMethod &m) {
  switch (m) {
  case StreamingMemoryOpInserter::ReplicatedTensorShardingMethod::Native: {
    output << "Native";
    break;
  }
  case StreamingMemoryOpInserter::ReplicatedTensorShardingMethod::Forward: {
    output << "Forward";
    break;
  }
  case StreamingMemoryOpInserter::ReplicatedTensorShardingMethod::
      AllReduceToScatter: {
    output << "AllReduceToScatter";
    break;
  }
  case StreamingMemoryOpInserter::ReplicatedTensorShardingMethod::
      LocalScatter: {
    output << "LocalScatter";
    break;
  }
  }
  return output;
}

bool StreamingMemoryOpInserter::ConsumerOpConfig::operator==(
    const ConsumerOpConfig &rhs) const {
  return tensor == rhs.tensor && op == rhs.op;
}

void StreamingMemoryOpInserter::ReplicationShardedTensors::insert(
    TensorId shardId,
    TensorId gatheredId,
    TensorId tensorId,
    TensorId refId) {
  if (!shardId.empty()) {
    shardToTensors.insert({shardId, {gatheredId, tensorId, refId}});
  }
  if (!gatheredId.empty()) {
    gatheredToTensors.insert({gatheredId, {shardId, tensorId, refId}});
  }
}

std::set<TensorId>
StreamingMemoryOpInserter::ReplicationShardedTensors::getShardTensorIds()
    const {
  std::set<TensorId> set;
  for (auto &kv : shardToTensors) {
    set.insert(kv.first);
  }
  return set;
}

std::set<TensorId>
StreamingMemoryOpInserter::ReplicationShardedTensors::getGatheredTensorIds()
    const {
  std::set<TensorId> set;
  for (auto &kv : gatheredToTensors) {
    set.insert(kv.first);
  }
  return set;
}

TensorId StreamingMemoryOpInserter::ReplicationShardedTensors::getShard(
    TensorId tensorId) const {
  {
    auto it = shardToTensors.find(tensorId);
    if (it != shardToTensors.end()) {
      return tensorId;
    }
  }
  {
    auto it = gatheredToTensors.find(tensorId);
    if (it != gatheredToTensors.end()) {
      return std::get<0>(it->second);
    }
  }
  return "";
}

TensorId StreamingMemoryOpInserter::ReplicationShardedTensors::getGathered(
    TensorId tensorId) const {
  {
    auto it = gatheredToTensors.find(tensorId);
    if (it != gatheredToTensors.end()) {
      return tensorId;
    }
  }
  {
    auto it = shardToTensors.find(tensorId);
    if (it != shardToTensors.end()) {
      return std::get<0>(it->second);
    }
  }
  return "";
}

TensorId StreamingMemoryOpInserter::ReplicationShardedTensors::getTensor(
    TensorId tensorId) const {
  {
    auto it = shardToTensors.find(tensorId);
    if (it != shardToTensors.end()) {
      return std::get<1>(it->second);
    }
  }
  {
    auto it = gatheredToTensors.find(tensorId);
    if (it != gatheredToTensors.end()) {
      return std::get<1>(it->second);
    }
  }
  return "";
}

TensorId StreamingMemoryOpInserter::ReplicationShardedTensors::getReference(
    TensorId tensorId) const {
  {
    auto it = shardToTensors.find(tensorId);
    if (it != shardToTensors.end()) {
      return std::get<2>(it->second);
    }
  }
  {
    auto it = gatheredToTensors.find(tensorId);
    if (it != gatheredToTensors.end()) {
      return std::get<2>(it->second);
    }
  }
  return "";
}

} // namespace popart
