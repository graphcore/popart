#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/op/init.hpp>
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
static constexpr const double initPriority          = -9996.0;
static constexpr const double remoteLoadPriority    = -9997.0;
static constexpr const double allGatherPriority     = -9997.0;
static constexpr const double ipuCopyPriority       = -9998.0;

static constexpr const double allReducePriorityInterleave          = -9999.0;
static constexpr const double optimizerStateLoadPriorityInterleave = -9999.0;
static constexpr const double varUpdatePriorityInterleave          = -9999.0;
static constexpr const double remoteStorePriorityInterleave        = -9999.0;

static constexpr const double allReducePriorityBatch          = -9999.0;
static constexpr const double optimizerStateLoadPriorityBatch = -10000.0;
static constexpr const double varUpdatePriorityBatch          = -10001.0;
static constexpr const double remoteStorePriorityBatch        = -10002.0;

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
    if (tensor->tensorType() == TensorType::Const)
      return true;
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

StreamingMemoryOpInserter::StreamingMemoryOpInserter(Graph &graph_,
                                                     int64_t replicationFactor_,
                                                     int num_stages_,
                                                     int num_phases_)
    : graph{graph_}, replicationFactor{replicationFactor_},
      num_stages{num_stages_}, num_phases{num_phases_}, remoteArgIds{},
      tensorStoreMap{}, tensorLoadMap{} {}

void StreamingMemoryOpInserter::apply() {
  std::set<Op *, POpCmp> opsToSetup;

  // Make sure no priorities outside of the range needed by
  // phased execution are being used.
  compressPriorities(graph);
  sanitizeOps();

  // Remove tensors and Cached from memory as needed
  // If two ops are on the same virtual graph,
  // and their phase is non-adjacently different,
  // then the tensor should be disconnected and backed up / restored

  logging::transform::debug(
      "[StreamingMemory] Processing tensors across execution phases");
  Tensors &tensors = graph.getTensors();

  for (TensorId id : tensors.getAllTensorIds()) {
    // Introduce ops for each tensor.
    auto tensor = tensors.get(id);
    applyTensor(tensor, opsToSetup);
  }

  // Get new schedule
  auto schedule = graph.getOpSchedule({});

  // Re-setup all ops that require it (update tensor shapes)
  for (Op *op : schedule) {
    if (opsToSetup.find(op) != opsToSetup.end()) {
      op->setup();
    }
  }

  graph.getIr().setExecutionPhasesReady();
}

void StreamingMemoryOpInserter::applyTensor(Tensor *tensor,
                                            SetupOps &opsToSetup) {

  // Work out the tensor's configuration.
  TensorConfig tensorConfig{graph};
  getTensorConfig(tensor, tensorConfig);

  // Nothing to for this tensor if it's not remote.
  if (!tensorConfig.location.isRemote()) {
    logging::transform::trace(
        "[StreamingMemory] Skipping tensor {} (not remote)", tensor->id);
    return;
  }

  // Nothing to do for this tensor if there are no loads.
  if (!tensorConfig.phaseConfig.loadStoreOutOfPhase &&
      !isLoadRequired(tensorConfig.phaseConfig)) {
    logging::transform::trace(
        "[StreamingMemory] Skipping tensor {} (no load required)", tensor->id);
    return;
  }

  // Log tensor phase configuration.
  logTensorPhaseConfig(tensorConfig);

  // Some context variables we need to keep track of between phases.
  TensorId loadedTensorId   = tensor->id;
  TensorId gatheredTensorId = tensor->id;

  RemoteLoadOp *remoteLoad   = nullptr;
  RemoteStoreOp *remoteStore = nullptr;

  auto &phaseConfig = tensorConfig.phaseConfig;

  if (phaseConfig.loadStoreOutOfPhase) {
    // Insert RemoteLoadOp/RemoteStoreOp out of phase.
    remoteLoad = insertRemoteLoadOp(
        tensorConfig, OptionalExecutionPhase(), loadedTensorId);
    remoteStore = insertRemoteStoreOp(
        tensorConfig, OptionalExecutionPhase(), loadedTensorId);
  }

  for (ExecutionPhase phase = 0; phase < num_phases * 2 - 1; ++phase) {

    if (!phaseConfig.loadStoreOutOfPhase) {
      // Reset these pointers.
      remoteLoad  = nullptr;
      remoteStore = nullptr;

      // Create RemoteLoadOp for this phase if we need to.
      if (phaseConfig.loadInPhase[phase]) {
        remoteLoad = insertRemoteLoadOp(tensorConfig, phase, loadedTensorId);
        gatheredTensorId = loadedTensorId;
      }

      // Create RemoteStoreOp for this phase if we need to.
      if (phaseConfig.storeInPhase[phase]) {
        remoteStore = insertRemoteStoreOp(tensorConfig, phase, loadedTensorId);
      }
    }

    // Gathers will happen in-phase regardless of whether loads/stores are
    // in-phase. After this part of the code the gatheredTensorId should point
    // the ID of the complete tensor.
    ReplicatedAllGatherOp *allGather = nullptr;

    if (phaseConfig.gatherInPhase[phase]) {
      allGather = insertReplicatedAllGatherOp(
          tensorConfig, phase, loadedTensorId, gatheredTensorId);
    }

    // Add constraints to ensure new operations are scheduled in the right
    // order.

    // Load must happen before the all-gather.
    if (remoteLoad && allGather) {
      graph.topoCons->insert(remoteLoad, allGather, true);
    }

    // Remote load has to take place before associated remote store
    if (remoteLoad && remoteStore) {
      graph.topoCons->insert(remoteLoad, remoteStore);
    }

    if (remoteStore) {
      // Any modification has to take place before the remote store
      for (Op *modifyingOp : tensorConfig.phaseConfig.modifiersInPhase[phase]) {
        graph.topoCons->insert(modifyingOp, remoteStore);
      }
    }

    for (Op *consumerOp : tensorConfig.phaseConfig.consumersInPhase[phase]) {
      if (tensor->isAcclTensor()) {
        if (remoteLoad) {
          graph.topoCons->insert(remoteLoad, consumerOp, true);
        }
        if (allGather) {
          graph.topoCons->insert(allGather, consumerOp, true);
        }
      } else {
        if (remoteLoad) {
          // Loading has to take place before the consumption
          graph.topoCons->insert(remoteLoad, consumerOp);
        }
      }

      // Logging graph change
      if (tensorConfig.producerOp) {
        logging::transform::debug(
            "[StreamingMemory] Disconnecting tensor {} between ops {} and {}",
            tensor->id,
            tensorConfig.producerOp->debugName(),
            consumerOp->debugName());
      } else {
        logging::transform::debug(
            "[StreamingMemory] Disconnecting tensor {} at op {} (modified: {})",
            tensor->id,
            consumerOp->debugName(),
            !tensorConfig.phaseConfig.modifiersInPhase[phase].empty());
      }

      // Disconnect original tensor and wire up loaded tensor
      auto indices = consumerOp->input->indices(tensor);
      for (auto i : indices) {
        auto *copyOp = dynamic_cast<IpuCopyOp *>(consumerOp);

        if (copyOp) {
          auto sourceIpu = copyOp->getSourceIpus().at(tensor->id);
          copyOp->disconnectInTensor(i, tensor);
          copyOp->connectInTensor(i, gatheredTensorId, sourceIpu);
        } else if (consumerOp->isOptimizerOp()) {
          consumerOp->disconnectInTensor(i, tensor);
          consumerOp->connectInTensor(i, loadedTensorId);

          // Check all optimizer ops connected of the current optimizer op

          std::vector<Op *> optimizerFrontOps(1, consumerOp);
          std::set<Op *, POpCmp> optimizerOps;

          while (!optimizerFrontOps.empty()) {
            Op *optFrontOp = optimizerFrontOps.back();
            optimizerFrontOps.pop_back();
            if (optFrontOp->isOptimizerOp() &&
                optimizerOps.find(optFrontOp) == optimizerOps.end()) {
              optimizerOps.insert(optFrontOp);
              for (auto tensorAndIndex : optFrontOp->input->tensorMap()) {
                if (tensorAndIndex.second->hasProducer()) {
                  optimizerFrontOps.push_back(
                      tensorAndIndex.second->getProducer());
                }
              }
              for (auto tensorAndIndex : optFrontOp->output->tensorMap()) {
                for (Op *op : tensorAndIndex.second->consumers.getOps()) {
                  optimizerFrontOps.push_back(op);
                }
              }
            }
          }

          for (Op *optimizerOp : optimizerOps) {
            opsToSetup.insert(optimizerOp);

            if (tensorConfig.optimizerSchedule ==
                ExecutionPhaseOptimizerSchedule::Interleaving) {
              optimizerOp->settings.schedulePriority =
                  varUpdatePriorityInterleave;
            } else if (tensorConfig.optimizerSchedule ==
                       ExecutionPhaseOptimizerSchedule::Batch) {
              optimizerOp->settings.schedulePriority = varUpdatePriorityBatch;
            }
            optimizerOp->settings.useIoTiles =
                tensorConfig.location.storeOnIOTiles;

            // If the current tensor is the weight tensor being updated,
            // update the AllReduce / ReduceScatter operation
            if (tensor->getTensorTypeInfo()->type() == TensorType::Variable &&
                !tensor->isAcclTensor() &&
                optimizerOp->input->hasIndex(
                    VarUpdateWithUpdaterOp::getUpdaterInIndex())) {
              Tensor *grad = optimizerOp->input->tensor(
                  VarUpdateWithUpdaterOp::getUpdaterInIndex());

              ReplicatedAllReduceOp *replicatedAllReduce = nullptr;

              if (grad->hasProducer()) {
                Op *producer = grad->getProducer();
                replicatedAllReduce =
                    dynamic_cast<ReplicatedAllReduceOp *>(producer);
              }

              if (replicatedAllReduce) {

                if (tensorConfig.optimizerSchedule ==
                    ExecutionPhaseOptimizerSchedule::Interleaving) {
                  replicatedAllReduce->settings.schedulePriority =
                      allReducePriorityInterleave;
                } else if (tensorConfig.optimizerSchedule ==
                           ExecutionPhaseOptimizerSchedule::Batch) {
                  replicatedAllReduce->settings.schedulePriority =
                      allReducePriorityBatch;
                }
                replicatedAllReduce->settings.useIoTiles =
                    tensorConfig.location.storeOnIOTiles;

                if (tensorConfig.location.replicatedTensorSharding) {
                  auto replicatedReduceScatterOp =
                      std::make_unique<ReplicatedReduceScatterOp>(
                          Onnx::CustomOperators::ReplicatedReduceScatter,
                          replicatedAllReduce->settings);
                  auto replicatedReduceScatter =
                      replicatedReduceScatterOp.get();
                  replicatedReduceScatter->fromLoss =
                      replicatedAllReduce->fromLoss;
                  replicatedReduceScatter->toLoss = replicatedAllReduce->toLoss;
                  graph.moveIntoGraph(std::move(replicatedReduceScatterOp));

                  TensorId inId =
                      replicatedAllReduce->input
                          ->tensor(ReplicatedAllReduceOp::getInIndex())
                          ->id;
                  TensorId outId =
                      replicatedAllReduce->output
                          ->tensor(ReplicatedAllReduceOp::getOutIndex())
                          ->id;
                  replicatedAllReduce->disconnectAllInputs();
                  replicatedAllReduce->disconnectAllOutputs();

                  graph.topoCons->transfer(replicatedAllReduce,
                                           replicatedReduceScatter);

                  replicatedAllReduce->getGraph().eraseOp(
                      replicatedAllReduce->id);

                  replicatedReduceScatter->connectInTensor(
                      ReplicatedAllReduceOp::getInIndex(), inId);

                  replicatedReduceScatter->connectInTensor(
                      ReplicatedAllGatherOp::getCollectiveLinkedIndex(),
                      getRemoteArg(tensor->id));

                  replicatedReduceScatter->connectOutTensor(
                      ReplicatedAllReduceOp::getOutIndex(), outId);

                  replicatedReduceScatter->setup();

                  if (tensorConfig.optimizerSchedule ==
                      ExecutionPhaseOptimizerSchedule::Interleaving) {
                    replicatedReduceScatter->settings.schedulePriority =
                        allReducePriorityInterleave;
                  } else if (tensorConfig.optimizerSchedule ==
                             ExecutionPhaseOptimizerSchedule::Batch) {
                    replicatedReduceScatter->settings.schedulePriority =
                        allReducePriorityBatch;
                  }
                  replicatedReduceScatter->settings.useIoTiles =
                      tensorConfig.location.storeOnIOTiles;

                  // Tie the reduction operation to the SGD0VarUpdate to get
                  // the same schedule behaviour as if the reduction was
                  // still integrated into SGD0VarUpdate
                  graph.topoCons->insert(
                      replicatedReduceScatter, optimizerOp, true);
                }
              }
            }

            // Lamb + replicated weight sharding:
            // Distributed L2 norm of the weight and updater tensor
            if (tensor->getTensorTypeInfo()->type() == TensorType::Variable &&
                !tensor->isAcclTensor() &&
                tensorConfig.location.replicatedTensorSharding &&
                dynamic_cast<LambSquareOp *>(optimizerOp)) {

              Tensor *out =
                  optimizerOp->output->tensor(LambSquareOp::getOutIndex());

              // Make sure reduction is only added once
              auto lambSqConsumers = out->consumers.getOps();
              if (!std::any_of(lambSqConsumers.begin(),
                               lambSqConsumers.end(),
                               [](Op *op) {
                                 return dynamic_cast<ReplicatedAllReduceOp *>(
                                     op);
                               })) {

                TensorId lambIntoReduceId =
                    graph.getIr().createIntermediateTensorId(out->id);

                optimizerOp->disconnectOutTensor(out);
                optimizerOp->createAndConnectOutTensor(
                    ReplicatedAllReduceOp::getOutIndex(), lambIntoReduceId);
                optimizerOp->setup();

                auto reduceOpUp =
                    std::make_unique<ReplicatedAllReduceInplaceOp>(
                        Onnx::CustomOperators::ReplicatedAllReduceInplace,
                        optimizerOp->settings);
                auto reduceOp = reduceOpUp.get();
                graph.moveIntoGraph(std::move(reduceOpUp));

                reduceOp->connectInTensor(
                    ReplicatedAllReduceInplaceOp::getInIndex(),
                    lambIntoReduceId);
                reduceOp->connectOutTensor(
                    ReplicatedAllReduceInplaceOp::getOutIndex(), out->id);

                reduceOp->setup();
              }
            }
          }
        } else {
          consumerOp->disconnectInTensor(i, tensor);
          consumerOp->connectInTensor(i, gatheredTensorId);
        }
      }
    }
  }
}

void StreamingMemoryOpInserter::getTensorSchedule(
    Tensor *tensor,
    TensorConfig &tensorConfig) const {
  auto &sessionOptions = graph.getIr().getSessionOptions();
  if (tensor->getTensorTypeInfo()->type() == TensorType::Variable) {
    if (tensor->isAcclTensor()) {
      tensorConfig.ioSchedule =
          sessionOptions.executionPhaseSettings.optimizerIOSchedule;
    } else {
      tensorConfig.ioSchedule =
          sessionOptions.executionPhaseSettings.weightIOSchedule;
    }
  } else {
    tensorConfig.ioSchedule =
        sessionOptions.executionPhaseSettings.activationIOSchedule;
  }
  tensorConfig.optimizerSchedule =
      sessionOptions.executionPhaseSettings.optimizerSchedule;
}

void StreamingMemoryOpInserter::getTensorConfig(
    Tensor *tensor,
    TensorConfig &tensorConfig) const {

  auto producerOp = tensor->getProducerUnsafe();

  tensorConfig.tensor     = tensor;
  tensorConfig.producerOp = producerOp;

  getOrderedConsumerOps(tensor, tensorConfig.consumerOps);
  getTensorLocation(tensor, tensorConfig.location);
  getTensorPhaseConfig(tensor,
                       producerOp,
                       tensorConfig.consumerOps,
                       tensorConfig.location,
                       tensorConfig.phaseConfig);
  getTensorSettings(
      tensor, producerOp, tensorConfig.consumerOps, tensorConfig.settings);
  getTensorSchedule(tensor, tensorConfig);

  if (tensorConfig.location.replicatedTensorSharding) {
    tensor->tensorLocationInfo.setSharded(true);
    logging::transform::debug(
        "[StreamingMemory] Enabling replica-sharded loading of tensor {}",
        tensor->id);
  }
}

void StreamingMemoryOpInserter::getOrderedConsumerOps(Tensor *tensor,
                                                      Ops &consumerOps) const {

  // Get consuming ops.
  consumerOps = tensor->consumers.getOps();

  // Process consumers in ascending order of phases
  std::sort(
      consumerOps.begin(), consumerOps.end(), [](const Op *lhs, const Op *rhs) {
        return (lhs->hasExecutionPhase() ? lhs->getExecutionPhase() : -1) <
               (rhs->hasExecutionPhase() ? rhs->getExecutionPhase() : -1);
      });
}

void StreamingMemoryOpInserter::getTensorLocation(
    Tensor *tensor,
    TensorLocation &location) const {
  location             = determineTensorLocation(tensor);
  auto &sessionOptions = graph.getIr().getSessionOptions();
  location.loadOnIOTiles &= sessionOptions.numIOTiles > 0;
  location.storeOnIOTiles &= sessionOptions.numIOTiles > 0;
}

void StreamingMemoryOpInserter::getTensorOptionalVGraphId(
    Tensor *tensor,
    const Op *producerOp,
    const Ops &consumerOps,
    OptionalVGraphId &loadStoreVGID) const {

  // If we have a producer op, use the VGID of the producer.
  if (producerOp) {
    if (const IpuCopyOp *copy = dynamic_cast<const IpuCopyOp *>(producerOp)) {
      loadStoreVGID = copy->getDestIpu();
    } else {
      loadStoreVGID = producerOp->getVirtualGraphId();
    }
  }

  // If we don't have a producer op, use the smallest consumer op VGID.
  for (auto consumerOp : consumerOps) {
    OptionalVGraphId consumerVGID = consumerOp->getOptionalVGraphId();
    if (IpuCopyOp *copyOp = dynamic_cast<IpuCopyOp *>(consumerOp)) {
      consumerVGID = copyOp->getSourceIpu();
    }

    // Pick correct VGID for loading/storing the tensor,
    // if no producer exists
    if (!loadStoreVGID) {
      loadStoreVGID = consumerVGID;
    } else if (!tensor->hasProducer() && consumerVGID) {
      loadStoreVGID = std::min(*loadStoreVGID, *consumerVGID);
    }
  }
}

void StreamingMemoryOpInserter::getTensorProducerExecutionPhase(
    Tensor *tensor,
    const Op *producerOp,
    OptionalExecutionPhase &producerExecutionPhase) const {
  if (producerOp) {
    if (producerOp->getOptionalExecutionPhase()) {
      producerExecutionPhase = producerOp->getOptionalExecutionPhase();
    }
    // Special case for IpuCopyOp.
    if (const IpuCopyOp *copy = dynamic_cast<const IpuCopyOp *>(producerOp)) {
      if (copy->getSourceIpu() % num_stages !=
          copy->getDestIpu() % num_stages) {
        // Inter-phase copy, special case where the producer
        // phase is moved
        producerExecutionPhase = *producerExecutionPhase + 1;
      }
    }
  }
}

void StreamingMemoryOpInserter::getTensorPhaseConfig(
    Tensor *tensor,
    const Op *producerOp,
    const Ops &consumerOps,
    const TensorLocation &location,
    TensorPhaseConfig &phaseConfig) const {

  // Set phaseConfig.producerExecutionPhase if possible.
  getTensorProducerExecutionPhase(
      tensor, producerOp, phaseConfig.producerExecutionPhase);
  // Set phaseConfig.loadStoreVGID.
  getTensorOptionalVGraphId(
      tensor, producerOp, consumerOps, phaseConfig.loadStoreVGID);

  // Set phaseConfig.consumersInPhase.
  for (auto consumerOp : consumerOps) {
    const auto consumerExecutionPhase =
        *consumerOp->getOptionalExecutionPhase();
    phaseConfig.consumersInPhase[consumerExecutionPhase].push_back(consumerOp);
  }

  // If the user requests to keep the tensor OnChip but also wants to shard the
  // tensor between replica sets then we will load/store the tensor out of the
  // training loop (e.g. only once) to keep the tensor in memory as much as
  // possible.
  phaseConfig.loadStoreOutOfPhase =
      (location.storage == TensorStorage::OnChip &&
       location.replicatedTensorSharding);

  // Set phaseConfig.livePhases (only used for logging, really).
  if (phaseConfig.producerExecutionPhase) {
    // Live in producer phase.
    phaseConfig.livePhases.insert(*phaseConfig.producerExecutionPhase);
  }
  for (auto consumerOp : consumerOps) {
    // Live in consumer phases.
    const auto consumerExecutionPhase =
        *consumerOp->getOptionalExecutionPhase();
    phaseConfig.livePhases.insert(consumerExecutionPhase);
  }

  // TODO T25043: The remaining code in this function is non-trivial and would
  // benefit from unit testing.

  // Set phaseConfig.{load,store,reduce}InPhase.

  // Set everything to false to begin with.
  for (ExecutionPhase p = 0; p < num_phases * 2 - 1; ++p) {
    phaseConfig.loadInPhase[p]   = false;
    phaseConfig.storeInPhase[p]  = false;
    phaseConfig.gatherInPhase[p] = false;
  }

  // We first set's set {load,store}InPhase assuming we are loading/storing
  // everything in-phase. This is because we may need to insert an
  // replicationAllGather in these phases. Note that we do this by means of an
  // incremental algorithm which processes phases in order by iterating over
  // consumer ops in execution phase order.

  if (phaseConfig.producerExecutionPhase) {
    // The tensor is stored in the producing phase.
    phaseConfig.storeInPhase[*phaseConfig.producerExecutionPhase] = true;
  }

  for (auto consumerOp : consumerOps) {
    // Phase of this consumer.
    const auto phase           = *consumerOp->getOptionalExecutionPhase();
    const auto isProducerPhase = phaseConfig.producerExecutionPhase &&
                                 (phaseConfig.producerExecutionPhase == phase);

    // With the current implementation IPUs always are associated with alternate
    // phases, irregardless of num_stages. Last phase relevant to this IPU was
    // two phases ago.
    const auto previousPhase = phase - 2;

    if (phaseConfig.livePhases.find(previousPhase) !=
        phaseConfig.livePhases.end()) {
      // The tensor is live in the previous phase. Instead of storing it in the
      // previous phase and loading it in again in this phase we are going to
      // keep it live.
      phaseConfig.loadInPhase[phase] = false;
      // If tensor was stored in the previous phase, defer storing until the
      // current phase.
      if (phaseConfig.storeInPhase[previousPhase]) {
        phaseConfig.storeInPhase[phase]         = true;
        phaseConfig.storeInPhase[previousPhase] = false;
      }
    } else if (isProducerPhase) {
      // Current phase is producer phase -- there is no need to load in current
      // phase
    } else {
      // Not live, load in current phase
      phaseConfig.loadInPhase[phase] = true;
    }

    if (phaseConfig.modifiersInPhase.find(phase) ==
        phaseConfig.modifiersInPhase.end()) {
      // Check if the consumer OP modifies the tensor, e.g. for weights
      // if yes, then the tensor requires to be backed-up at the end of
      // the phase
      getAliasedModifiersInPhase(
          tensor, phase, phaseConfig.modifiersInPhase[phase]);
      if (!phaseConfig.modifiersInPhase[phase].empty()) {
        // Storing in this phase
        phaseConfig.storeInPhase[phase] = true;
        // Not storing in previous phase
        phaseConfig.storeInPhase[previousPhase] = false;
      }
    }
  }

  // Tensors that are sharded must be gathered wherever they are set to be
  // loaded as per the current loadInPhase settings. This is true even if
  // we load out of phase. Accumulated tensors are never gathered.
  if (location.replicatedTensorSharding && !tensor->isAcclTensor()) {
    for (ExecutionPhase p = 0; p < num_phases * 2 - 1; ++p) {
      phaseConfig.gatherInPhase[p] = phaseConfig.loadInPhase[p];
    }
  }

  // If we are loading/storing out of phase then we don't actually want to
  // remember all the load/store in phase flags at all, but we do need the
  // gathers.
  if (phaseConfig.loadStoreOutOfPhase) {
    for (ExecutionPhase p = 0; p < num_phases * 2 - 1; ++p) {
      phaseConfig.loadInPhase[p]  = false;
      phaseConfig.storeInPhase[p] = false;
    }
  }
}

void StreamingMemoryOpInserter::getTensorSettings(
    Tensor *tensor,
    const Op *producerOp,
    const Ops &consumerOps,
    Op::Settings &settings) const {

  // Inherit settings from producer or consumer
  if (producerOp) {
    settings = producerOp->settings;
  } else {
    if (!consumerOps.empty()) {
      settings = consumerOps.front()->settings;
    }
    settings.batchSerializedPhase.reset();
  }
  settings.name.clear();
  settings.recomputeType          = RecomputeType::Checkpoint;
  settings.tensorLocation.storage = TensorStorage::Undefined;
}

void StreamingMemoryOpInserter::getModifiersInPhase(
    Tensor *t,
    const ExecutionPhase phase,
    Ops &modifyingConsumerOps) const {
  for (Op *consumer : t->consumers.getOps()) {
    for (InIndex in : consumer->input->indices(t)) {
      auto regions = consumer->modifies(in);
      if (!std::all_of(regions.begin(),
                       regions.end(),
                       [](const view::Region &r) {
                         return r.isEmpty() ||
                                r.getAccessType() == view::AccessType::Read;
                       }) &&
          consumer->hasExecutionPhase() &&
          consumer->getExecutionPhase() == phase) {
        if (std::find(modifyingConsumerOps.begin(),
                      modifyingConsumerOps.end(),
                      consumer) == modifyingConsumerOps.end()) {
          modifyingConsumerOps.push_back(consumer);
        }
      }
    }
  }
}

void StreamingMemoryOpInserter::getAliasedModifiersInPhase(
    Tensor *tensor,
    const ExecutionPhase phase,
    Ops &modifyingConsumerOps) const {
  getModifiersInPhase(tensor, phase, modifyingConsumerOps);
  auto aliasedTensorMap = graph.getTensors().aliasChainsFrom(tensor);
  auto fullRegion       = view::Region::getFull(tensor->info.shape());
  for (auto &chain : aliasedTensorMap) {
    auto regions = chain.second.apply(fullRegion);
    bool nonEmptyAlias =
        std::any_of(regions.begin(), regions.end(), [](view::Region &r) {
          return !r.isEmpty();
        });
    if (nonEmptyAlias) {
      getModifiersInPhase(chain.first, phase, modifyingConsumerOps);
    }
  }
}

RemoteLoadOp *StreamingMemoryOpInserter::insertRemoteLoadOp(
    const TensorConfig &tensorConfig,
    const OptionalExecutionPhase phase,
    TensorId &loadedTensorId) {

  RemoteLoadOp *remoteLoad = nullptr;

  if (!tensorConfig.phaseConfig.loadStoreOutOfPhase) {
    // Phase must be set when storing in phase.
    if (!phase) {
      throw internal_error(
          "Expected phase to be set when inserting RemoteLoadOp in-phase.");
    }
  }

  // Log this.
  if (tensorConfig.phaseConfig.loadStoreOutOfPhase) {
    logging::transform::trace(
        "[StreamingMemory] Adding remote load of {} ({}) out of phase",
        loadedTensorId,
        tensorConfig.tensor->id);
  } else {
    logging::transform::trace(
        "[StreamingMemory] Adding remote load of {} ({}) in phase {}",
        loadedTensorId,
        tensorConfig.tensor->id,
        *phase);
  }

  if (!tensorConfig.phaseConfig.loadStoreOutOfPhase) {
    auto tensorLoadMapEntry =
        tensorLoadMap.find({tensorConfig.tensor->id, *phase});

    if (tensorLoadMapEntry != tensorLoadMap.end()) {
      loadedTensorId = tensorLoadMapEntry->second.first;
      remoteLoad     = tensorLoadMapEntry->second.second;

      return remoteLoad;
    }
  }

  auto remoteLoadOp = std::make_unique<RemoteLoadOp>(
      Onnx::CustomOperators::RemoteLoad, tensorConfig.settings);
  remoteLoad = remoteLoadOp.get();

  if (!tensorConfig.phaseConfig.loadStoreOutOfPhase) {
    if (tensorConfig.ioSchedule == ExecutionPhaseIOSchedule::OnDemand) {
      // RemoteLoad in current phase
      if (phase) {
        remoteLoad->setExecutionPhase(phase);
      }
      if (tensorConfig.tensor->isAcclTensor()) {
        // Optimizer state: Load as late as possible
        if (tensorConfig.optimizerSchedule ==
            ExecutionPhaseOptimizerSchedule::Interleaving) {
          remoteLoad->settings.schedulePriority =
              optimizerStateLoadPriorityInterleave;
        } else if (tensorConfig.optimizerSchedule ==
                   ExecutionPhaseOptimizerSchedule::Batch) {
          remoteLoad->settings.schedulePriority =
              optimizerStateLoadPriorityBatch;
        }
      }
    } else if (tensorConfig.ioSchedule == ExecutionPhaseIOSchedule::Preload) {
      // Very low schedulePriority on loads (at the end of the previous
      // phase)
      remoteLoad->settings.schedulePriority = remoteLoadPriority;
      // RemoteLoad at the end of the previous phase, so that load
      // is executed before inter-IPU copy
      if (phase) {
        remoteLoad->setExecutionPhase(*phase - 1);
      }
    }
  } else {
    remoteLoad->setExecutionPhase({});
    remoteLoad->settings.schedulePriority = 0.0f;
  }

  remoteLoad->settings.recomputeType = RecomputeType::Checkpoint;
  remoteLoad->setVirtualGraphId(tensorConfig.phaseConfig.loadStoreVGID);
  graph.moveIntoGraph(std::move(remoteLoadOp));

  remoteLoad->connectInTensor(RemoteStoreOp::getRemoteBufferOffsetInIndex(),
                              getRemoteArg(tensorConfig.tensor->id));

  TensorId initTensorId = generateInitTensorId(tensorConfig.tensor);
  loadedTensorId        = generateLoadedTensorId(tensorConfig.tensor, phase);
  TensorId inTensorId;

  if (tensorConfig.phaseConfig.loadStoreOutOfPhase) {
    // Setting the execution context ensures it's scheduled out of the main
    // loop.
    remoteLoad->settings.executionContext =
        ExecutionContext::WeightsFromHostFragment;
  }

  if (auto prevLoadId = getPreviousLoadedTensorId(tensorConfig.tensor->id)) {
    // Tensor might not have a true producer op, but was previously
    // loaded by a RemoteLoad
    inTensorId = prevLoadId.value();
  } else if (tensorConfig.producerOp) {
    // Tensor has a true producer op
    inTensorId = tensorConfig.tensor->id;
  } else {
    TensorInfo initInfo = tensorConfig.tensor->info;

    if (tensorConfig.location.replicatedTensorSharding) {
      initInfo.set(initInfo.dataType(),
                   {(initInfo.nelms() - 1) / replicationFactor + 1});
    }

    // InitOp as a "producer" op
    auto initOp = std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                           initInfo,
                                           TensorType::Cache,
                                           InitType::NoInit,
                                           tensorConfig.settings);
    Op *init    = initOp.get();

    if (!tensorConfig.phaseConfig.loadStoreOutOfPhase) {
      if (tensorConfig.tensor->isAcclTensor()) {
        if (tensorConfig.optimizerSchedule ==
            ExecutionPhaseOptimizerSchedule::Interleaving) {
          remoteLoad->settings.schedulePriority =
              optimizerStateLoadPriorityInterleave;
        } else if (tensorConfig.optimizerSchedule ==
                   ExecutionPhaseOptimizerSchedule::Batch) {
          remoteLoad->settings.schedulePriority =
              optimizerStateLoadPriorityBatch;
        }
        init->setExecutionPhase(phase);
      } else {
        init->settings.schedulePriority = initPriority;
        init->setExecutionPhase(*phase - 1);
      }
    } else {
      init->setExecutionPhase({});
      init->settings.schedulePriority = 0.0f;
    }

    if (tensorConfig.phaseConfig.loadStoreOutOfPhase) {
      // Setting the execution context ensures it's scheduled out of the main
      // loop.
      init->settings.executionContext =
          ExecutionContext::WeightsFromHostFragment;
    }

    init->setVirtualGraphId(tensorConfig.phaseConfig.loadStoreVGID);
    graph.moveIntoGraph(std::move(initOp));
    init->createAndConnectOutTensor(InitOp::getOutIndex(), initTensorId);
    init->setup();
    inTensorId = initTensorId;

    // Do Init on IO tiles
    init->settings.useIoTiles = tensorConfig.location.loadOnIOTiles;
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
  remoteLoad->settings.useIoTiles = tensorConfig.location.loadOnIOTiles;

  if (tensorConfig.location.replicatedTensorSharding) {
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

  // If this tensor is loaded in-phase it may be loaded in multiple phases. In
  // this case getPreviousLoadedTensorId is used to 'chain' the tensor inputs.
  // This is necessary to allow be able to outline the RemoteLoadOp code. For
  // this to work we need to populate the tensorLoadMap here.
  if (!tensorConfig.phaseConfig.loadStoreOutOfPhase) {
    tensorLoadMap.emplace(TensorPhase(tensorConfig.tensor->id, *phase),
                          RemoteLoadOpData(loadedTensorId, remoteLoad));
  }

  return remoteLoad;
}

ReplicatedAllGatherOp *StreamingMemoryOpInserter::insertReplicatedAllGatherOp(
    const TensorConfig &tensorConfig,
    const ExecutionPhase phase,
    const TensorId &loadedTensorId,
    TensorId &gatheredTensorId) {
  ReplicatedAllGatherOp *allGather = nullptr;

  logging::transform::trace(
      "[StreamingMemory] Adding replicated all gather of {} ({}) in phase {}",
      loadedTensorId,
      tensorConfig.tensor->id,
      phase);

  // Execute replicated allgather to collect the full weight
  // tensor from the individual replicas
  auto allGatherOp = std::make_unique<ReplicatedAllGatherOp>(
      Onnx::CustomOperators::ReplicatedAllGather,
      tensorConfig.settings,
      tensorConfig.tensor->info);
  allGather                            = allGatherOp.get();
  allGather->settings.schedulePriority = allGatherPriority;
  allGather->settings.recomputeType    = RecomputeType::Checkpoint;
  // RemoteLoad at the end of the previous phase, so that load
  // is executed before inter-IPU copy
  allGather->setExecutionPhase(phase - 1);
  allGather->setVirtualGraphId(tensorConfig.phaseConfig.loadStoreVGID);
  graph.moveIntoGraph(std::move(allGatherOp));

  allGather->connectInTensor(ReplicatedAllGatherOp::getInIndex(),
                             loadedTensorId);

  allGather->connectInTensor(ReplicatedAllGatherOp::getCollectiveLinkedIndex(),
                             getRemoteArg(tensorConfig.tensor->id));

  gatheredTensorId = generateGatheredTensorId(tensorConfig.tensor, phase);

  allGather->createAndConnectOutTensor(ReplicatedAllGatherOp::getOutIndex(),
                                       gatheredTensorId);

  allGather->setup();

  // Do AllGather on IO tiles
  allGather->settings.useIoTiles = tensorConfig.location.loadOnIOTiles;

  return allGather;
}

RemoteStoreOp *StreamingMemoryOpInserter::insertRemoteStoreOp(
    const TensorConfig &tensorConfig,
    const OptionalExecutionPhase phase,
    const TensorId &loadedTensorId) {
  RemoteStoreOp *remoteStore = nullptr;

  if (!tensorConfig.phaseConfig.loadStoreOutOfPhase) {
    // Phase must be set when storing in phase.
    if (!phase) {
      throw internal_error(
          "Expected phase to be set when inserting RemoteLoadOp in-phase.");
    }
  }

  // Log this.
  if (tensorConfig.phaseConfig.loadStoreOutOfPhase) {
    logging::transform::trace(
        "[StreamingMemory] Adding remote store of {} ({}) out of phase",
        loadedTensorId,
        tensorConfig.tensor->id);
  } else {
    logging::transform::trace(
        "[StreamingMemory] Adding remote store of {} ({}) in phase {}",
        loadedTensorId,
        tensorConfig.tensor->id,
        phase);
  }

  if (!tensorConfig.phaseConfig.loadStoreOutOfPhase) {
    auto tensorStoreMapEntry =
        tensorStoreMap.find({tensorConfig.tensor->id, *phase});

    if (tensorStoreMapEntry != tensorStoreMap.end()) {
      remoteStore = tensorStoreMapEntry->second.second;
      return remoteStore;
    }
  }

  auto remoteStoreOp = std::make_unique<RemoteStoreOp>(
      Onnx::CustomOperators::RemoteStore, tensorConfig.settings);
  remoteStore = remoteStoreOp.get();

  if (!tensorConfig.phaseConfig.loadStoreOutOfPhase) {
    remoteStore->setExecutionPhase(phase);
    if (tensorConfig.ioSchedule == ExecutionPhaseIOSchedule::Preload ||
        (tensorConfig.ioSchedule == ExecutionPhaseIOSchedule::OnDemand &&
         tensorConfig.tensor->isAcclTensor())) {
      // Very low schedulePriority on stores (at the end of a phase)
      if (tensorConfig.optimizerSchedule ==
          ExecutionPhaseOptimizerSchedule::Interleaving) {
        remoteStore->settings.schedulePriority = remoteStorePriorityInterleave;
      } else {
        remoteStore->settings.schedulePriority = remoteStorePriorityBatch;
      }
    }
  } else {
    remoteStore->setExecutionPhase({});
    remoteStore->settings.schedulePriority = 0.0f;
  }

  remoteStore->settings.recomputeType = RecomputeType::Checkpoint;
  remoteStore->setVirtualGraphId(tensorConfig.phaseConfig.loadStoreVGID);
  graph.moveIntoGraph(std::move(remoteStoreOp));

  remoteStore->connectInTensor(RemoteStoreOp::getRemoteBufferOffsetInIndex(),
                               getRemoteArg(tensorConfig.tensor->id));
  remoteStore->connectInTensor(RemoteStoreOp::getLocalTensorInIndex(),
                               loadedTensorId);

  if (tensorConfig.phaseConfig.loadStoreOutOfPhase) {
    // Setting the execution context ensures it's scheduled out of the main
    // loop.
    remoteStore->settings.executionContext =
        ExecutionContext::WeightsToHostFragment;
  }

  remoteStore->setup();

  // Do allgather on IO tiles
  remoteStore->settings.useIoTiles = tensorConfig.location.loadOnIOTiles;

  if (!tensorConfig.phaseConfig.loadStoreOutOfPhase) {
    tensorStoreMap.emplace(TensorPhase(tensorConfig.tensor->id, *phase),
                           RemoteStoreOpData(loadedTensorId, remoteStore));
  }

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
          copy->getSourceIpu() % num_stages !=
              copy->getDestIpu() % num_stages) {
        // Inter-phase copy
        op->settings.schedulePriority = ipuCopyPriority;
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
  auto isWeight = type == TensorType::Variable && !tensor->isAcclTensor();
  auto isOptimizerState =
      type == TensorType::Variable && tensor->isAcclTensor();

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

      // If we still don't have a tensor location setting and the tensor belongs
      // to a group we have tensor location settings for, use those tensor
      // location settings.

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
    }

    if (result.storage == TensorStorage::OffChip) {

      // If we are planning to offload the tensor to off-chip memory but the
      // tensor belongs to a group of tensors for which we have a tensor
      // location setting that specifies a minimum size for offloading and this
      // tensor is smaller than this minimum size, revert back to on-chip.

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

    if (result.replicatedTensorSharding) {
      if (isActivation) {
        result.replicatedTensorSharding = false;
      }
      if (isWeight &&
          tooSmallForReplicatedTensorSharding(
              sessionOptions.weightTensorLocationSettings, tensor)) {
        result.replicatedTensorSharding = false;
      }
      if (isOptimizerState &&
          tooSmallForReplicatedTensorSharding(
              sessionOptions.optimizerStateTensorLocationSettings, tensor)) {
        result.replicatedTensorSharding = false;
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
      tensorLocationToStr(result),
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
      copy->getSourceIpu() % num_stages != copy->getDestIpu() % num_stages) {
    // Inter-phase copy, needs to be associated with the
    // source execution phase and virtual graph
    if (copy->getSourceIpu() % num_stages != phase % num_stages) {
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

void StreamingMemoryOpInserter::logTensorPhaseConfig(
    const TensorConfig &tensorConfig) const {

  const auto &tensor      = tensorConfig.tensor;
  const auto &phaseConfig = tensorConfig.phaseConfig;

  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {

    std::stringstream ss;
    ss << tensor->id << " phase liveness status: ";
    for (ExecutionPhase p = 0; p < num_phases * 2 - 1; ++p) {
      ss << "| ";
      ss << "(" << p << ") ";
      if (p == phaseConfig.producerExecutionPhase) {
        // Produced in this phase
        ss << "P";
      }
      // Using std::map::find as it can be used on a const map.
      auto loadIt = phaseConfig.loadInPhase.find(p);
      if (loadIt != phaseConfig.loadInPhase.end()) {
        if (loadIt->second) {
          // Loaded in this phase
          ss << "L";
        }
      }
      auto storeIt = phaseConfig.storeInPhase.find(p);
      if (storeIt != phaseConfig.storeInPhase.end()) {
        if (storeIt->second) {
          // Stored in this phase
          ss << "S";
        }
      }
      if (phaseConfig.livePhases.find(p) != phaseConfig.livePhases.end()) {
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

nonstd::optional<std::string>
StreamingMemoryOpInserter::getPreviousLoadedTensorId(TensorId &id) {
  nonstd::optional<std::string> prev_id;
  int64_t last_phase = -1;
  for (auto &elem : tensorLoadMap) {
    if (elem.first.first == id && elem.first.second > last_phase) {
      prev_id    = std::get<0>(elem.second);
      last_phase = elem.first.second;
    }
  }
  return prev_id;
}

TensorId StreamingMemoryOpInserter::generateInitTensorId(Tensor *tensor) {
  // The copiedTensor id needs to be unique as the same tensor may be copied to
  // multiple phases
  TensorId initTensorId = tensor->id + "_init";
  return initTensorId;
}

TensorId StreamingMemoryOpInserter::generateLoadedTensorId(
    Tensor *tensor,
    OptionalExecutionPhase phase) {
  // The copiedTensor id needs to be unique as the same tensor may be copied to
  // multiple phases
  TensorId loadedTensorId = tensor->id + "_l";
  if (phase) {
    loadedTensorId += std::to_string(*phase);
  }
  return loadedTensorId;
}

TensorId
StreamingMemoryOpInserter::generateGatheredTensorId(Tensor *tensor,
                                                    ExecutionPhase phase) {
  // The copiedTensor id needs to be unique as the same tensor may be copied to
  // multiple phases
  TensorId gatheredTensorId = tensor->id + "_g" + std::to_string(phase);
  return gatheredTensorId;
}

bool StreamingMemoryOpInserter::isLoadRequired(
    const TensorPhaseConfig &phaseConfig) {
  bool loadRequired = false;
  for (auto &loadStore : phaseConfig.loadInPhase) {
    loadRequired |= loadStore.second;
  }
  return loadRequired;
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

} // namespace popart
