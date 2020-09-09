#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/op/init.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/lamb.hpp>
#include <popart/op/remote.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/tensornames.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/streamingmemoryopinserter.hpp>

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
  double priority = -(maxCompressedPriority + 1);
  if (isPhased && op->settings.executionContext == ExecutionContext::Normal) {
    switch (schedule) {
    case ExecutionPhaseSchedule::Interleaving: {
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
    default:
      throw error("Unsupported schedule {}", static_cast<int>(schedule));
    }
  }
}

StreamingMemoryOpInserter::StreamingMemoryOpInserter(Graph &graph_,
                                                     int64_t replicationFactor_,
                                                     int num_stages_,
                                                     int num_phases_)
    : graph{graph_}, replicationFactor{replicationFactor_},
      num_stages{num_stages_}, num_phases{num_phases_}, remoteArgIds{} {}

bool StreamingMemoryOpInserter::isPhasedExecution() const {
  return num_phases > 1;
}

void StreamingMemoryOpInserter::createTensorSchedule() {
  Tensors &tensors = graph.getTensors();
  std::set<TensorId> seenTensors;
  tensorSchedule.clear();
  tensorSchedule.reserve(tensors.getAllTensorIds().size());
  auto schedule = graph.getOpSchedule({});
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

Tensor *
StreamingMemoryOpInserter::findRelatedVarTensor(std::vector<Tensor *> front) {
  while (front.size()) {
    Tensor *t0 = front.back();
    front.pop_back();

    auto config = tensorConfigs.at(t0);

    if (config.rootVarTensor &&
        config.rootVarTensor->tensorType() == TensorType::Variable) {
      return config.rootVarTensor;
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
      allGather = insertReplicatedAllGatherOp(
          tensorConfig, context, loadedTensorId, gatheredTensorId);
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

    if (remoteStore) {
      // Any modification has to take place before the remote store
      for (Op *modifyingOp : config.modifiers) {
        graph.topoCons->insert(modifyingOp, remoteStore);
      }
    }

    if (tensorConfig.location.replicatedTensorSharding ==
        ReplicatedTensorSharding::On) {
      rtsTensors.insert(loadedTensorId,
                        gatheredTensorId,
                        tensor->id,
                        stripAllReservedPrefixes(tensor->id));
    }

    for (auto consumerOpConfig : config.consumers) {
      if (remoteLoad) {
        // Loading has to take place before the consumption
        graph.topoCons->insert(remoteLoad, consumerOpConfig.op);
      }
      if (tensor->getTensorTypeInfo()->type() == TensorType::Variable) {
        if (tensor->id != consumerOpConfig.tensor->id) {
          // Current tensor (root var) is not the same as the actually
          // consumed tensor (descendant)
          if (consumerOpConfig.tensor->hasProducer()) {
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
  // 4.) Bailing (error) if the Op has RTS and non-RTS inputs, but can't
  // gather
  //
  //  A*  B
  //   \ /
  //    Op <- modifies A* -> bailing with error
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
  // * = replica sharded tensor or operation

  // Operations that are affected by replicated tensor sharding
  OpsSet opsToProcess;

  // Map to resolve the reference tensor id for replicated weight sharding for
  // a specific Op
  std::map<Op *, TensorId> opToRefTensorId;

  auto addOpsToProcess =
      [this, &opsToProcess, &opToRefTensorId](TensorId id, TensorId refId) {
        Tensor *t = graph.getTensors().get(id);
        for (Op *consumer : t->consumers.getOps()) {
          logging::transform::trace("[StreamingMemory] Op {} consumes at least "
                                    "one replication sharded tensor",
                                    consumer->debugName());
          opsToProcess.insert(consumer);
          opToRefTensorId.insert({consumer, refId});
        }
      };

  for (TensorId shardId : rtsTensors.getShardTensorIds()) {
    TensorId tensorId = rtsTensors.getTensor(shardId);
    TensorId refId    = rtsTensors.getReference(shardId);
    // Process initial shard consumers
    if (!shardId.empty()) {
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
          "[StreamingMemory] Processing {} for replicated tensor sharding",
          op->debugName());
      TensorId refId = opToRefTensorId.at(op);

      auto rtsIndices = op->getReplicatedTensorShardingIndices();

      // Loop over one set of indices
      for (auto indices : rtsIndices) {
        bool allIndiciesSharded = true;
        for (InIndex inIdx : indices.first) {
          bool indexSharded = false;
          Tensor *inTensor  = op->input->tensor(inIdx);
          if (rtsTensors.hasShard(inTensor->id)) {
            // Already sharded
            indexSharded = true;
          } else if (!rtsTensors.getShard(inTensor->id).empty()) {
            // Sharded tensor available
            op->disconnectInTensor(inTensor);
            op->connectInTensor(inIdx, rtsTensors.getShard(inTensor->id));
            indexSharded = true;
            rtsChanged   = true;
          } else if (inTensor->hasProducer() &&
                     inTensor->getProducer()
                         ->isConvertibleTo<ReplicatedAllReduceOp>()) {
            // Try to shard index by changing an AllReduce into a
            // ReduceScatter
            ReplicatedAllReduceOp *replicatedAllReduce =
                dynamic_cast<ReplicatedAllReduceOp *>(inTensor->getProducer());
            TensorId inId = replicatedAllReduce->input
                                ->tensor(ReplicatedAllReduceOp::getInIndex())
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
                context.phase =
                    replicatedAllReduce->getOptionalExecutionPhase();
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
              logging::transform::warn(
                  "[StreamingMemory] {} is an optimizer related "
                  "ReplicatedAllReduce, but the related variable "
                  "has not been found.",
                  replicatedAllReduce->debugName());
            }

            // Keep settings that the ReplicatedAllReduce, that is being
            // replaced, had.
            tensorConfig.settings = replicatedAllReduce->settings;

            ReplicatedReduceScatterOp *replicatedReduceScatter =
                insertReplicatedReduceScatterOp(
                    tensorConfig, context, inId, outId, refId);

            graph.topoCons->transfer(replicatedAllReduce,
                                     replicatedReduceScatter);

            replicatedAllReduce->getGraph().eraseOp(replicatedAllReduce->id);

            // Tensor outId is now replicated tensor sharded
            rtsTensors.insert(
                outId, "", outId, stripAllReservedPrefixes(refId));
            // Add all consumers to the set of ops to process
            addOpsToProcess(outId, refId);
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
                opToRefTensorId.insert({producer, refId});
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
          op->configureForReplicatedTensorSharding({indices});
          // Add output tensors as RTS tensors
          for (auto outIndex : indices.second) {
            TensorId outId = op->outId(outIndex);
            // Tensor outId is now replicated tensor sharded
            rtsTensors.insert(
                outId, "", outId, stripAllReservedPrefixes(refId));
            // Add all consumers to the set of ops to process
            addOpsToProcess(outId, refId);
          }
          opsToProcess.erase(op);
          rtsChanged = true;
        }
      }
    }
  }

  // Where replicated tensor sharding couldn't be propagated further,
  // consume the gathered tensor instead
  for (Op *op : opsToProcess) {
    TensorId refId = opToRefTensorId.at(op);

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
          indexSharded         = true;
          auto modifiedRegions = op->modifies(inIdx);
          // If the optimizer Op modifies the input, we say that the optimizer
          // updates this variable. If this variable is an RTS tensor,
          // then we know the update Op must be sharded for that input.
          // We most definitely can't use a ReplicatedAllGather here,
          // because that would no longer update the root var tensor
          modifiesSharded |=
              std::any_of(modifiedRegions.begin(),
                          modifiedRegions.end(),
                          [](view::Region &r) { return !r.isEmpty(); });
        }
        allIndiciesSharded &= indexSharded;
      }
      if (!allIndiciesSharded) {
        // Not all required indices for this op are RTS tensors:
        // We can try to gather the complete tensor and let the Op operate
        // on the whole tensors rather than shards (2) or we need to bail
        // because the Op updates a sharded tensor (1)

        // 1.) Bail with error if op must consume sharded tensor but can't
        if (modifiesSharded) {
          throw error(
              "[StreamingMemory] Op {} modifies a sharded tensor, but "
              "not all required sharded inputs could be connected. Adjust the "
              "replicated tensor sharding settings. Bailing.",
              op->debugName());
        }

        // 2.) Op cannot do RTS, connect gathered tensors instead
        for (InIndex inIdx : indices.first) {
          Tensor *inTensor = op->input->tensor(inIdx);
          if (rtsTensors.hasShard(inTensor->id)) {
            // Disconnect sharded input tensor
            op->disconnectInTensor(inIdx, inTensor);

            TensorId gatheredId = rtsTensors.getGathered(inTensor->id);
            TensorId tensorId   = rtsTensors.getTensor(inTensor->id);

            if (gatheredId.empty()) {
              TensorConfig tensorConfig = tensorConfigs.at(inTensor);
              TensorStreamingContext context;
              context.context = op->settings.executionContext;
              if (context.context == ExecutionContext::Normal) {
                if (isPhasedExecution()) {
                  context.phase = op->getOptionalExecutionPhase();
                } else {
                  context.preLoss = op->scheduledPreLoss;
                }
              }

              insertReplicatedAllGatherOp(
                  tensorConfig, context, inTensor->id, gatheredId);

              rtsTensors.insert(inTensor->id,
                                gatheredId,
                                tensorId,
                                stripAllReservedPrefixes(refId));
            }

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
    }
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
  auto chains = graph.getTensors().getAliases().aliasChainsFrom(tensor);
  auto shape  = tensor->info.shape();

  // Check if the alias is an identity chain
  for (auto tensorAndChain : chains) {
    if (tensorAndChain.first->getTensorTypeInfo()->type() ==
            TensorType::Variable &&
        !tensorAndChain.second.isEmpty()) {
      rootTensor = tensorAndChain.first;
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
      consumerVGID = copyOp->getSourceIpu(tensor->id);
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
  settings.recomputeType          = RecomputeType::Checkpoint;
  settings.tensorLocation.storage = TensorStorage::Undefined;
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
  auto aliasedTensorMap = graph.getTensors().aliasChainsFrom(tensor);
  auto fullRegion       = view::Region::getFull(tensor->info.shape());
  for (auto &chain : aliasedTensorMap) {
    auto regions = chain.second.apply(fullRegion);
    bool nonEmptyAlias =
        std::any_of(regions.begin(), regions.end(), [](view::Region &r) {
          return !r.isEmpty();
        });
    if (nonEmptyAlias) {
      getModifiersInContext(chain.first, context, modifyingConsumerOps);
    }
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
      initInfo.set(initInfo.dataType(),
                   {(initInfo.nelms() - 1) / replicationFactor + 1});
    }

    // InitOp as a "producer" op
    auto initOp = std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                           initInfo,
                                           TensorType::Cache,
                                           InitType::NoInit,
                                           tensorConfig.settings);
    init        = initOp.get();

    init->settings.executionContext = context.context;
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

  if (tensorConfig.location.replicatedTensorSharding ==
      ReplicatedTensorSharding::On) {
    // Avoid outlining RemoteLoad Ops that result in a
    // different final gathered tensor shape together,
    // because it can have adverse effects due to copying tensors
    // with different final tile mapping using the same host
    // exchange
    std::string gatheredShapeString =
        "[" +
        logging::join(tensorConfig.tensor->info.shape().begin(),
                      tensorConfig.tensor->info.shape().end(),
                      ",") +
        "]";
    remoteLoad->settings.extraOutlineAttributes.insert(
        {"gatheredSize", gatheredShapeString});
  }

  return remoteLoad;
}

ReplicatedAllGatherOp *StreamingMemoryOpInserter::insertReplicatedAllGatherOp(
    const TensorConfig &tensorConfig,
    const TensorStreamingContext context,
    const TensorId &loadedTensorId,
    TensorId &gatheredTensorId) {
  ReplicatedAllGatherOp *allGather = nullptr;

  logging::transform::trace("[StreamingMemory] Adding replicated all gather "
                            "of {} ({}) in context {}",
                            loadedTensorId,
                            tensorConfig.tensor->id,
                            context);

  // Execute replicated allgather to collect the full weight
  // tensor from the individual replicas
  auto allGatherOp = std::make_unique<ReplicatedAllGatherOp>(
      Onnx::CustomOperators::ReplicatedAllGather,
      tensorConfig.settings,
      tensorConfig.tensor->info);
  allGather = allGatherOp.get();

  allGather->settings.executionContext = context.context;

  if (isPhasedExecution() && context.phase) {
    allGather->setExecutionPhase(*(context.phase) - 1);
    setLoadingOpPhaseAndPriority(
        allGather, tensorConfig.tensor, tensorConfig, context);
  }

  allGather->settings.recomputeType = RecomputeType::Checkpoint;
  // RemoteLoad at the end of the previous phase, so that load
  // is executed before inter-IPU copy
  allGather->setVirtualGraphId(
      tensorConfig.streamingMap.at(context).streamingVGID);
  graph.moveIntoGraph(std::move(allGatherOp));

  allGather->connectInTensor(ReplicatedAllGatherOp::getInIndex(),
                             loadedTensorId);

  allGather->connectInTensor(
      ReplicatedAllGatherOp::getCollectiveLinkedIndex(),
      getRemoteArg(stripAllReservedPrefixes(tensorConfig.tensor->id)));

  gatheredTensorId = generateGatheredTensorId(tensorConfig.tensor->id);

  allGather->createAndConnectOutTensor(ReplicatedAllGatherOp::getOutIndex(),
                                       gatheredTensorId);

  allGather->setup();

  // Do AllGather on IO tiles
  allGather->settings.tileSet = tensorConfig.location.loadTileSet;

  return allGather;
}

ReplicatedReduceScatterOp *
StreamingMemoryOpInserter::insertReplicatedReduceScatterOp(
    const TensorConfig &tensorConfig,
    const TensorStreamingContext context,
    const TensorId &inTensorId,
    const TensorId &outTensorId,
    const TensorId &weightTensorId) {

  auto replicatedReduceScatterOp = std::make_unique<ReplicatedReduceScatterOp>(
      Onnx::CustomOperators::ReplicatedReduceScatter, tensorConfig.settings);
  auto replicatedReduceScatter = replicatedReduceScatterOp.get();
  graph.moveIntoGraph(std::move(replicatedReduceScatterOp));

  replicatedReduceScatter->connectInTensor(ReplicatedAllReduceOp::getInIndex(),
                                           inTensorId);

  replicatedReduceScatter->connectInTensor(
      ReplicatedAllGatherOp::getCollectiveLinkedIndex(),
      getRemoteArg(weightTensorId));

  replicatedReduceScatter->connectOutTensor(
      ReplicatedAllReduceOp::getOutIndex(), outTensorId);

  replicatedReduceScatter->setup();

  replicatedReduceScatter->settings.tileSet =
      tensorConfig.location.storageTileSet;

  setPriority(replicatedReduceScatter,
              isPhasedExecution(),
              false,
              tensorConfig.schedule);

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
    }
  } else {
    remoteStore->setExecutionPhase({});
    remoteStore->settings.schedulePriority = 0.0f;
  }

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

  remoteStore->setup();

  // Do store on IO tiles
  remoteStore->settings.tileSet = tensorConfig.location.loadTileSet;

  return remoteStore;
}

void StreamingMemoryOpInserter::sanitizeOps() const {

  auto schedule = graph.getOpSchedule({});

  for (Op *op : schedule) {
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
        setPriority(
            copy,
            isPhasedExecution(),
            false,
            graph.getIr().getSessionOptions().executionPhaseSettings.schedule);
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
    if (op->settings.tensorLocation.storage == TensorStorage::Undefined) {
      if (op->opid == Onnx::CustomOperators::GetRandomSeed) {
        op->settings.tensorLocation.storage = TensorStorage::OnChip;
        logging::transform::trace("[StreamingMemory] {} set to OnChip",
                                  op->debugName());
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

  // Result variable.
  TensorLocation result = TensorLocation();
  const char *logReason = "";

  const auto overrideIt =
      sessionOptions.tensorLocationSettingsOverride.find(id);
  bool haveTensorLocationSettingOverride =
      (overrideIt != sessionOptions.tensorLocationSettingsOverride.end());

  if (haveTensorLocationSettingOverride &&
      isValidTensorLocation(overrideIt->second)) {

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
    // all else fails, offload.

    if (result.storage == TensorStorage::Undefined && producerOp) {

      // If a producing operator is known and it is set to have a tensor
      // location (and is not set to recompute), use that.

      if (isValidTensorLocation(producerOp->settings.tensorLocation)) {
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
    }

    if (result.storage == TensorStorage::Undefined) {

      // If we still don't have a tensor location setting and the tensor
      // belongs to a group we have tensor location settings for, use those
      // tensor location settings.

      if (isActivation &&
          isValidTensorLocation(
              sessionOptions.activationTensorLocationSettings.location)) {
        // Use activation tensor location settings.
        result = sessionOptions.activationTensorLocationSettings.location;
        logReason =
            "activationTensorLocationSettings.location in SessionOptions";
      }

      if (isWeight &&
          isValidTensorLocation(
              sessionOptions.weightTensorLocationSettings.location)) {
        // Use weight tensor location settings.
        result    = sessionOptions.weightTensorLocationSettings.location;
        logReason = "weightTensorLocationSettings.location in SessionOptions";
      }

      if (isOptimizerState &&
          isValidTensorLocation(
              sessionOptions.optimizerStateTensorLocationSettings.location)) {
        // Use optimizer state tensor location settings.
        result = sessionOptions.optimizerStateTensorLocationSettings.location;
        logReason = "weightTensorLocationSettings.location in SessionOptions";
      }

      if (isAccumulator &&
          isValidTensorLocation(
              sessionOptions.accumulatorTensorLocationSettings.location)) {
        // Use optimizer state tensor location settings.
        result = sessionOptions.accumulatorTensorLocationSettings.location;
        logReason =
            "accumulatorTensorLocationSettings.location in SessionOptions";
      }
    }

    if (result.storage == TensorStorage::OffChip) {

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

    if (result.replicatedTensorSharding == ReplicatedTensorSharding::On) {
      if (isActivation) {
        result.replicatedTensorSharding = ReplicatedTensorSharding::Off;
      }
      if (isWeight &&
          tooSmallForReplicatedTensorSharding(
              sessionOptions.weightTensorLocationSettings, tensor)) {
        result.replicatedTensorSharding = ReplicatedTensorSharding::Off;
      }
      if (isOptimizerState &&
          tooSmallForReplicatedTensorSharding(
              sessionOptions.optimizerStateTensorLocationSettings, tensor)) {
        result.replicatedTensorSharding = ReplicatedTensorSharding::Off;
      }
    }

    if (result.replicatedTensorSharding == ReplicatedTensorSharding::On) {
      if ((!sessionOptions.enableReplicatedGraphs) ||
          (sessionOptions.replicatedGraphCount <= 1)) {
        logging::transform::warn("[StreamingMemory] Unable to shard tensor {} "
                                 "due to lack of replication",
                                 id);
        result.replicatedTensorSharding = ReplicatedTensorSharding::Off;
      }
    }

    if (result.storage != TensorStorage::OnChip &&
        tensor->isOptimizerTensor()) {

      // Don't offload optimizer tensors to off-chip memory.
      result.storage = TensorStorage::OnChip;
      logReason      = "it being an optimizer tensor";
    }

    if (result.storage != TensorStorage::OnChip &&
        isConstOrCopyOfConst(tensor)) {

      // Don't offload constant (or copy of constant) tensors to off-chip
      // memory.
      result.storage = TensorStorage::OnChip;
      logReason      = "it being an constant or copy-of-constant tensor";
    }
  }

  // Finally, it's possible at this point nothing has set the result
  // yet, in which case we default to TensorStorage::OnChip.

  if (result.storage == TensorStorage::Undefined) {
    result.storage = TensorStorage::OnChip;
    logReason      = "absence of tensor location settings";
  }

  // Log the result.
  logging::transform::debug(
      "[StreamingMemory] Determined tensor {} should use tensor "
      "location {} (due to {})",
      id,
      result,
      logReason);

  return result;
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
