// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/boundary.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/lamb.hpp>
#include <popart/op/remote.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/pingpong.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

static constexpr const double maxCompressedPriority = 9000.0;
static constexpr const double initPriority          = -9996.0;
static constexpr const double remoteLoadPriority    = -9997.0;
static constexpr const double allGatherPriority     = -9997.0;
static constexpr const double ipuCopyPriority       = -9998.0;
static constexpr const double allReducePriority     = -9999.0;
static constexpr const double varUpdatePriority     = -9999.0;
static constexpr const double remoteStorePriority   = -9999.0;
static constexpr const double acclStateLoadPriority = -9999.0;

std::size_t PingPong::id(int pass) {
  return typeid(PingPong).hash_code() + pass;
}

TensorId PingPong::generateInitTensorId(Tensor *tensor) const {
  // The copiedTensor id needs to be unique as the same tensor may be copied to
  // multiple phases
  TensorId initTensorId = tensor->id + "_init";
  return initTensorId;
}

TensorId PingPong::generateLoadedTensorId(Tensor *tensor,
                                          int64_t load_index) const {
  // The copiedTensor id needs to be unique as the same tensor may be copied to
  // multiple phases
  TensorId loadedTensorId = tensor->id + "_l" + std::to_string(load_index);
  return loadedTensorId;
}

TensorId PingPong::generateGatheredTensorId(Tensor *tensor,
                                            int64_t load_index) const {
  // The copiedTensor id needs to be unique as the same tensor may be copied to
  // multiple phases
  TensorId gatheredTensorId = tensor->id + "_g" + std::to_string(load_index);
  return gatheredTensorId;
}

// The cost of an Op, simplified to only account for weights
float PingPong::costFn(Op *op) const {
  float w_weights = 1.f;
  float total     = 0;

  for (auto map : op->input->indicesMap()) {
    if (map.first->tensorType() == TensorType::Variable ||
        map.first->tensorType() == TensorType::Const) {
      total += w_weights * static_cast<float>(map.first->info.nbytes());
    }
  }

  logging::trace("Op {} cost {}", op->opid, total);

  return total;
}

namespace {

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

void PingPong::verifyPlacementConsistency(const Op *op,
                                          const unsigned num_stages) const {
  if (op->hasVirtualGraphId() && op->hasPingPongPhase()) {
    if (op->getPingPongPhase() % num_stages !=
        op->getVirtualGraphId() % num_stages) {
      throw error("[PingPong] Op {} inconsistent PingPong phase {} "
                  "and virtual graph ID {}",
                  op->debugName(),
                  op->getPingPongPhase(),
                  op->getVirtualGraphId());
    }
  }
}

void PingPong::verifyPingPongPhases(Graph &graph) const {
  // Verify pingpong phase annotations
  for (auto &op : graph.getOps()) {
    Op *op0 = op.second.get();
    if (op0->hasPingPongPhase()) {
      auto phase0 = op0->getPingPongPhase();
      for (Tensor *input : op0->input->tensors()) {
        if (input->hasProducer() && input->getProducer()->hasPingPongPhase()) {
          Op *op1     = input->getProducer();
          auto phase1 = op1->getPingPongPhase();
          if (phase1 > phase0) {
            throw error("[PingPong] Op {} {} (I/O) before op {} {}, "
                        "but pingpong phases disagree ({} vs. {})",
                        op1->id,
                        op1->debugName(),
                        op0->id,
                        op0->debugName(),
                        phase1,
                        phase0);
          }
        }
      }
      for (Op *op1 : graph.topoCons->getBefores(op0)) {
        auto phase1 = op1->getPingPongPhase();
        if (phase1 > phase0) {
          logging::transform::warn(
              "[PingPong] Op {} {} (topologically) before op {} {}, "
              "but pingpong phases disagree ({} vs. {}). "
              "Removing constraint.",
              op1->id,
              op1->debugName(),
              op0->id,
              op0->debugName(),
              phase1,
              phase0);
          graph.topoCons->remove(op1, op0);
        }
      }
    }
  }
}

// Make sure the op has a valid placement annotation
void PingPong::sanitizePlacementAnnotation(const Graph &graph,
                                           Op *op,
                                           PingPongPhase phase,
                                           unsigned num_stages) const {
  for (Tensor *input : op->input->tensors()) {
    if (input->hasProducer() && input->getProducer()->hasPingPongPhase()) {
      phase = std::max(phase, input->getProducer()->getPingPongPhase());
    }
  }
  for (Op *before : graph.topoCons->getBefores(op)) {
    if (before->hasPingPongPhase()) {
      phase = std::max(phase, before->getPingPongPhase());
    }
  }

  bool remapPhase = !op->hasPingPongPhase() || op->getPingPongPhase() < phase;

  IpuCopyOp *copy = dynamic_cast<IpuCopyOp *>(op);
  if (copy &&
      copy->getSourceIpu() % num_stages != copy->getDestIpu() % num_stages) {
    // Inter-phase copy, needs to be associated with the
    // source pingpong phase and virtual graph
    if (copy->getSourceIpu() % num_stages != phase % num_stages) {
      phase -= 1;
      remapPhase = true;
    }
  }

  if (remapPhase) {
    logging::transform::debug(
        "[PingPong] Mapping operator {} to phase {} -> {}",
        op->debugName(),
        op->hasPingPongPhase() ? op->getPingPongPhase() : unusedPingPongPhase,
        phase);
    op->setPingPongPhase(phase);
  }
  if (!op->hasVirtualGraphId() && !op->isIpuCopyOp()) {
    VGraphId vgid = phase % num_stages;
    logging::transform::debug("[PingPong] Mapping operator {} to VGID {} -> {}",
                              op->debugName(),
                              op->hasVirtualGraphId() ? op->getVirtualGraphId()
                                                      : unusedVGraphId,
                              vgid);
    op->setVirtualGraphId(vgid);
  }
  verifyPlacementConsistency(op, num_stages);
}

void PingPong::getModifiersInPhase(
    PingPongPhase phase,
    Tensor *t,
    std::vector<Op *> &modifyingConsumerOps) const {
  for (Op *consumer : t->consumers.getOps()) {
    for (InIndex in : consumer->input->indices(t)) {
      auto regions = consumer->modifies(in);
      if (!std::all_of(regions.begin(),
                       regions.end(),
                       [](const view::Region &r) {
                         return r.isEmpty() ||
                                r.getAccessType() == view::AccessType::Read;
                       }) &&
          consumer->hasPingPongPhase() &&
          consumer->getPingPongPhase() == phase) {
        if (std::find(modifyingConsumerOps.begin(),
                      modifyingConsumerOps.end(),
                      consumer) == modifyingConsumerOps.end()) {
          modifyingConsumerOps.push_back(consumer);
        }
      }
    }
  }
}

bool PingPong::isValidCacheType(const CacheType cacheType) const {
  return (cacheType == CacheType::OffChip) || (cacheType == CacheType::OnChip);
}

bool PingPong::tooSmallForOffChip(const CacheSettings &cacheSettings,
                                  Tensor *tensor) const {
  return (tensor->info.nelms() < cacheSettings.minElementsForOffChip);
}

const char *PingPong::cacheTypeToStr(const CacheType cacheType) const {
  const char *result = "";
  switch (cacheType) {
  case CacheType::Undefined: {
    result = "Undefined";
    break;
  }
  case CacheType::OffChip: {
    result = "OffChip";
    break;
  }
  case CacheType::OnChip: {
    result = "OnChip";
    break;
  }
  default: {
    throw error(
        "[PingPong] Unexpected value for cacheType in cacheTypeToStr ({})",
        static_cast<int>(cacheType));
  }
  }

  return result;
}

CacheType PingPong::determineCacheType(Graph &graph, Tensor *tensor) const {
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
  CacheType result      = CacheType::Undefined;
  const char *logReason = "";

  const auto overrideIt = sessionOptions.cacheSettingOverride.find(id);
  bool haveCacheSettingOverride =
      (overrideIt != sessionOptions.cacheSettingOverride.end());

  if (haveCacheSettingOverride && isValidCacheType(overrideIt->second)) {

    // If we have a valid cacheSettingOverride then the user is telling us
    // explicitly where to put this tensor, so use that.

    result    = overrideIt->second;
    logReason = "cacheSettingOverride in SessionOptions";

  } else {

    // If we don't have an entry for the tensor in cacheSettingOverride then
    // check to see if the operator that produces this tensor (if known) was
    // created with an 'cacheOutput' attribute. If so, ue that. If not, see
    // if the tensor's type has associated cache settings and use that. If
    // all else fails, offload.

    if (result == CacheType::Undefined && producerOp) {

      // If a producing operator is known and it is set to have a cache type
      // (and is not set to recompute), use that.

      if (isValidCacheType(producerOp->settings.cacheType)) {
        if (producerOp->settings.recomputeType == RecomputeType::Recompute) {
          logging::transform::warn(
              "[PingPong] Ignoring cache attribute on tensor {} because the "
              "tensor is set to recompute",
              id);
        } else {
          result    = producerOp->settings.cacheType;
          logReason = "builder attribute";
        }
      }
    }

    if (result == CacheType::Undefined) {

      // If we still don't have a cache setting and the tensor belongs to a
      // group we have cache settings for, use those cache settings.

      if (isActivation &&
          isValidCacheType(sessionOptions.activationCacheSettings.cacheType)) {
        // Use activation cache settings.
        result    = sessionOptions.activationCacheSettings.cacheType;
        logReason = "activationCacheSettings.cacheType in SessionOptions";
      }

      if (isWeight &&
          isValidCacheType(sessionOptions.weightCacheSettings.cacheType)) {
        // Use weight cache settings.
        result    = sessionOptions.weightCacheSettings.cacheType;
        logReason = "weightCacheSettings.cacheType in SessionOptions";
      }

      if (isOptimizerState &&
          isValidCacheType(
              sessionOptions.optimizerStateCacheSettings.cacheType)) {
        // Use optimizer state cache settings.
        result    = sessionOptions.optimizerStateCacheSettings.cacheType;
        logReason = "weightCacheSettings.cacheType in SessionOptions";
      }
    }

    if (result == CacheType::OffChip) {

      // If we are planning to offload the tensor to off-chip memory but the
      // tensor belongs to a group of tensors for which we have a cache setting
      // that specifies a minimum size for offloading and this tensor is smaller
      // than this minimum size, revert back to on-chip.

      if (isActivation &&
          tooSmallForOffChip(sessionOptions.activationCacheSettings, tensor)) {
        result = CacheType::OnChip;
        logReason =
            "activationCacheSettings.minElementsForOffChip in SessionOptions";
      }

      if (isWeight &&
          tooSmallForOffChip(sessionOptions.weightCacheSettings, tensor)) {
        result = CacheType::OnChip;
        logReason =
            "weightCacheSettings.minElementsForOffChip in SessionOptions";
      }

      if (isOptimizerState &&
          tooSmallForOffChip(sessionOptions.optimizerStateCacheSettings,
                             tensor)) {
        result    = CacheType::OnChip;
        logReason = "optimizerStateCacheSettings.minElementsForOffChip in "
                    "SessionOptions";
      }
    }

    if (result != CacheType::OnChip && tensor->isOptimizerTensor()) {

      // Don't offload optimizer tensors to off-chip memory.
      result    = CacheType::OnChip;
      logReason = "it being an optimizer tensor";
    }

    if (result != CacheType::OnChip && isConstOrCopyOfConst(tensor)) {

      // Don't offload constant (or copy of constant) tensors to off-chip
      // memory.
      result    = CacheType::OnChip;
      logReason = "it being an constant or copy-of-constant tensor";
    }
  }

  // Finally, it's possible at this point nothing has set the result
  // yet, in which case we default to CacheType::OffChip.

  if (result == CacheType::Undefined) {
    result    = CacheType::OnChip;
    logReason = "absence of cache settings";
  }

  // Log the result.
  logging::transform::debug(
      "[PingPong] Determined tensor {} should use cache type {} (due to {})",
      id,
      cacheTypeToStr(result),
      logReason);

  return result;
}

void PingPong::getAliasedModifiersInPhase(
    Graph &graph,
    PingPongPhase phase,
    Tensor *t,
    std::vector<Op *> &modifyingConsumerOps) const {
  getModifiersInPhase(phase, t, modifyingConsumerOps);
  auto aliasedTensorMap = graph.getTensors().aliasChainsFrom(t);
  auto fullRegion       = view::Region::getFull(t->info.shape());
  for (auto &chain : aliasedTensorMap) {
    auto regions = chain.second.apply(fullRegion);
    bool nonEmptyAlias =
        std::any_of(regions.begin(), regions.end(), [](view::Region &r) {
          return !r.isEmpty();
        });
    if (nonEmptyAlias) {
      getModifiersInPhase(phase, chain.first, modifyingConsumerOps);
    }
  }
}

bool PingPong::apply(Graph &graph) const {
  auto &ir               = graph.getIr();
  auto &sessionOptions   = ir.getSessionOptions();
  auto replicationFactor = sessionOptions.enableReplicatedGraphs
                               ? sessionOptions.replicatedGraphCount
                               : 1;
  bool replicatedWeightSharding =
      ir.getSessionOptions().replicatedWeightSharding;
  size_t rwsMinNumElems =
      ir.getSessionOptions().replicatedWeightShardingMinNumElements;
  auto numIOTiles           = sessionOptions.numIOTiles;
  const auto total_num_ipus = ir.getDeviceInfo()->getNumIpus();
  const auto num_ipus       = total_num_ipus / replicationFactor;

  // const auto training = ir.canTrain();
  const auto num_phases = sessionOptions.pingPongPhases;

  // Left and right pingpong stage
  const auto num_stages = 2;

  if (graph.getOps().size() == 0 || num_phases < 2) {
    return false;
  }

  logging::transform::debug(
      "[PingPong] pingpong scheme with {} phases, {} ({}) ipus",
      num_phases,
      num_ipus,
      total_num_ipus);

  auto schedule = graph.getOpSchedule({});

  if (pass == 1 || pass == 2) {
    for (Tensor *tensor : graph.getTensors().getOfType(TensorType::Variable)) {
      // The mechanism by which we handle offloaded (off-chip) tensors of type
      // TensorType::Variable is setting a flag in cacheInfo.
      CacheType cacheType = determineCacheType(graph, tensor);
      tensor->cacheInfo.setCached(cacheType == CacheType::OffChip);
      logging::transform::debug("[PingPong] Set Variable {} to {}.",
                                tensor->id,
                                cacheTypeToStr(cacheType));
    }
  }

  if (pass == 1) {
    float cumulative_cost = 0.f;

    for (Op *op : schedule) {
      cumulative_cost += costFn(op);
    }

    float cost_per_phase = cumulative_cost / static_cast<float>(num_phases);

    // Greedy claiming of ops per phase according to execution schedule
    // TODO T10602: Find better graph-cut algorithm for phase splitting
    PingPongPhase phase = 0;
    float phase_cost    = 0;
    for (Op *op : schedule) {

      auto cost = costFn(op);
      // Every phase should handle at least some cost, but not too much
      if (phase_cost > 0 && phase_cost + cost > cost_per_phase &&
          phase < (num_phases - 1)) {
        ++phase;
        phase_cost = 0;
      }
      phase_cost += cost;

      bool has_phase = op->hasPingPongPhase();

      if (has_phase && op->getPingPongPhase() >= num_phases) {
        throw error("[PingPong] Maximum phase is {}, but op {} has phase {}.",
                    num_phases - 1,
                    op->debugName(),
                    op->getPingPongPhase());
      }

      sanitizePlacementAnnotation(
          graph, op, has_phase ? op->getPingPongPhase() : phase, num_stages);
    }

    // Recomputation annotation
    logging::transform::debug("[PingPong] Recomputation & Cache annotation");
    for (auto &op : graph.getOps()) {
      // Cached everything not set by the user by default
      if (op.second->settings.cacheType == CacheType::Undefined) {
        if (op.second->opid == Onnx::CustomOperators::GetRandomSeed) {
          op.second->settings.cacheType = CacheType::Uncached;
          logging::transform::trace("[PingPong] {} set to Uncached",
                                    op.second->debugName());
        }
      }
      // Recompute everything else (fwd) not set by the user by default
      if (op.second->settings.recomputeType == RecomputeType::Undefined) {
        if (ir.autoRecomputationEnabled() && !op.second->isIpuCopyOp() &&
            !dynamic_cast<GetRandomSeedOp *>(op.second.get())) {
          op.second->settings.recomputeType = RecomputeType::Recompute;
          logging::transform::trace("[PingPong] {} set to Recompute",
                                    op.second->debugName());
        } else {
          op.second->settings.recomputeType = RecomputeType::Checkpoint;
        }
      }
    }
  }

  // Figure out the right phase for ops that did not get a phase yet
  while (true) {
    int num_ops_without_phase = 0;
    // Need to get the schedule every time,
    // because setting phases can change schedule order
    for (Op *op : schedule) {
      if (!op->hasPingPongPhase()) {
        // Check which phase the consumers of the output are in
        op->inheritPlacementAttributes(true);

        if (op->hasPingPongPhase()) {
          // Make sure phase adheres to producer/consumer and topological
          // constraints
          sanitizePlacementAnnotation(
              graph, op, op->getPingPongPhase(), num_stages);
        } else {
          ++num_ops_without_phase;
        }
      }
    }
    if (num_ops_without_phase == 0) {
      break;
    }
  }

  // Tensor cache store/load inserted in the third ping-pong pass only
  if (pass == 2) {
    std::set<Op *, POpCmp> opsToSetup;

    // Make sure no priorities outside of the range needed by
    // pingpong are being used.
    compressPriorities(graph);

    for (Op *op : schedule) {
      sanitizePlacementAnnotation(
          graph, op, op->getPingPongPhase(), num_stages);
      // Assign correct schedulePriority to all inter-IPU copies
      if (op->isIpuCopyOp()) {
        // Special type of IPUCopy between PingPong phases
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
          logging::transform::trace("[PingPong] {} set to Checkpoint",
                                    op->debugName());
        }
      }
      // Cached everything not set by the user by default
      if (op->settings.cacheType == CacheType::Undefined) {
        if (op->opid == Onnx::CustomOperators::GetRandomSeed) {
          op->settings.cacheType = CacheType::OnChip;
          logging::transform::trace("[PingPong] {} set to OnChip",
                                    op->debugName());
        }
      }
    }

    // Remove tensors and Cached from memory as needed
    // If two ops are on the same virtual graph,
    // and their phase is non-adjacently different,
    // then the tensor should be disconnected and backed up / restored
    Op::Settings settings(graph, "");

    std::vector<TensorId> cacheArgIds;

    std::map<std::pair<TensorId, int64_t>, std::pair<TensorId, RemoteStoreOp *>>
        tensorStoreMap;
    std::map<std::pair<TensorId, int64_t>,
             std::tuple<TensorId, TensorId, RemoteLoadOp *>>
        tensorLoadMap;

    auto getCacheArg = [&graph, &cacheArgIds](TensorId tensorId) -> TensorId {
      auto arg_tensor_id = getCacheArgTensorId(tensorId);
      TensorInfo argTensorInfo(DataType::INT32, {1});
      std::vector<int> idData(1, 0);
      if (std::find(cacheArgIds.begin(), cacheArgIds.end(), arg_tensor_id) ==
          cacheArgIds.end()) {
        graph.getTensors().addConstInit(
            arg_tensor_id,
            argTensorInfo,
            reinterpret_cast<void *>(idData.data()));
        cacheArgIds.push_back(arg_tensor_id);
      }
      return arg_tensor_id;
    };

    auto getPreviousLoadedTensorId =
        [&tensorLoadMap](TensorId id) -> nonstd::optional<std::string> {
      nonstd::optional<std::string> prev_id;
      int64_t last_phase = -1;
      for (auto &elem : tensorLoadMap) {
        if (elem.first.first == id && elem.first.second > last_phase) {
          prev_id    = std::get<0>(elem.second);
          last_phase = elem.first.second;
        }
      }
      return prev_id;
    };

    logging::transform::debug(
        "[PingPong] Processing tensors across PingPong phases");
    Tensors &tensors = graph.getTensors();
    for (TensorId id : tensors.getAllTensorIds()) {
      Tensor *tensor  = tensors.get(id);
      auto producerOp = tensor->getProducerUnsafe();

      // Determine cache type for this tensor.
      CacheType cacheType = determineCacheType(graph, tensor);

      if (cacheType == CacheType::OnChip) {
        // Do not process this tensor further
        continue;
      }

      OptionalPingPongPhase producerPingPongPhase;
      OptionalVGraphId loadStoreVGID;

      if (producerOp) {
        if (producerOp->getOptionalPingPongPhase()) {
          producerPingPongPhase = producerOp->getOptionalPingPongPhase();
        }
        if (IpuCopyOp *copy = dynamic_cast<IpuCopyOp *>(producerOp)) {
          if (copy->getSourceIpu() % num_stages !=
              copy->getDestIpu() % num_stages) {
            // Inter-phase copy, special case where the producer
            // phase is moved
            producerPingPongPhase = *producerPingPongPhase + 1;
          }
          loadStoreVGID = copy->getDestIpu();
        } else {
          loadStoreVGID = producerOp->getVirtualGraphId();
        }
      }

      auto consumerOps = tensor->consumers.getOps();

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
      settings.recomputeType = RecomputeType::Checkpoint;
      settings.cacheType     = CacheType::Undefined;

      // Process consumers in ascending order of phases
      std::sort(
          consumerOps.begin(),
          consumerOps.end(),
          [](const Op *lhs, const Op *rhs) {
            return (lhs->hasPingPongPhase() ? lhs->getPingPongPhase() : -1) <
                   (rhs->hasPingPongPhase() ? rhs->getPingPongPhase() : -1);
          });

      // Set if the tensor is live in the current phase
      std::set<PingPongPhase> livePhases;
      // Set of modifying ops per phase
      std::map<PingPongPhase, std::vector<Op *>> modifiersInPhase;
      // Set of consuming ops per phase
      std::map<PingPongPhase, std::vector<Op *>> consumersInPhase;
      // Set of phases that load or store
      std::map<PingPongPhase, std::pair<bool, bool>> loadStoreInPhase;

      if (producerPingPongPhase) {
        livePhases.insert(*producerPingPongPhase);
        // Do not load in producer phase
        loadStoreInPhase[*producerPingPongPhase].first = false;
        // Store in producer phase
        loadStoreInPhase[*producerPingPongPhase].second = true;
      }

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

        auto consumerPingPongPhase = *consumerOp->getOptionalPingPongPhase();

        if (livePhases.find(consumerPingPongPhase - num_stages) !=
            livePhases.end()) {
          // Live in adjacent previous phase, do not load in current phase
          loadStoreInPhase[consumerPingPongPhase].first = false;
          if (loadStoreInPhase[consumerPingPongPhase - num_stages].second) {
            // Move store from adjacent previous phase to current phase
            loadStoreInPhase[consumerPingPongPhase].second              = true;
            loadStoreInPhase[consumerPingPongPhase - num_stages].second = false;
          }
        } else if (producerPingPongPhase &&
                   producerPingPongPhase == consumerPingPongPhase) {
          // Current phase is producer phase, do not load in current phase
          loadStoreInPhase[consumerPingPongPhase].first = false;
        } else {
          // Not live, load in current phase
          loadStoreInPhase[consumerPingPongPhase].first = true;
        }

        if (modifiersInPhase.find(consumerPingPongPhase) ==
            modifiersInPhase.end()) {
          // Check if the consumer OP modifies the tensor, e.g. for weights
          // if yes, then the tensor requires to be backed-up at the end of
          // the phase
          getAliasedModifiersInPhase(graph,
                                     consumerPingPongPhase,
                                     tensor,
                                     modifiersInPhase[consumerPingPongPhase]);
          if (!modifiersInPhase[consumerPingPongPhase].empty()) {
            // Storing in this phase
            loadStoreInPhase[consumerPingPongPhase].second = true;
            // Not storing in previous phase
            loadStoreInPhase[consumerPingPongPhase - num_stages].second = false;
          }
        }
        livePhases.insert(consumerPingPongPhase);
        consumersInPhase[consumerPingPongPhase].push_back(consumerOp);
      }

      if (logging::shouldLog(logging::Module::transform,
                             logging::Level::Trace)) {
        std::stringstream ss;
        ss << tensor->id << " phase liveness status: ";
        for (PingPongPhase p = 0; p < num_phases * 2 - 1; ++p) {
          ss << "| ";
          ss << "(" << p << ") ";
          if (p == producerPingPongPhase) {
            // Produced in this phase
            ss << "P";
          }
          if (loadStoreInPhase[p].first) {
            // Loaded in this phase
            ss << "L";
          }
          if (loadStoreInPhase[p].second) {
            // Stored in this phase
            ss << "S";
          }
          if (livePhases.find(p) != livePhases.end()) {
            // Live in this phase
            ss << "A ";
          } else {
            ss << "_ ";
          }
        }
        ss << "|";
        logging::transform::trace(ss.str());
      }

      bool loadRequired = false;
      for (auto &phaseLoadStore : loadStoreInPhase) {
        loadRequired |= phaseLoadStore.second.first;
      }
      if (!loadRequired) {
        // Tensor never loaded, so caching is disabled
        continue;
      }

      bool tensorUseIoTiles = replicatedWeightSharding &&
                              tensor->tensorType() == TensorType::Variable &&
                              numIOTiles > 0;
      bool tensorReplicatedWeightSharding = false;
      if (replicatedWeightSharding && tensor->info.nelms() > rwsMinNumElems &&
          tensor->tensorType() == TensorType::Variable) {
        tensorReplicatedWeightSharding = true;
        tensor->cacheInfo.setSharded(tensorReplicatedWeightSharding);
        logging::transform::debug(
            "[PingPong] Enabling replica-sharded loading of tensor {}",
            tensor->id);
      }

      TensorId loadedTensorId   = tensor->id;
      TensorId gatheredTensorId = tensor->id;

      for (auto &phaseLoadStore : loadStoreInPhase) {
        PingPongPhase currentPingPongPhase         = phaseLoadStore.first;
        RemoteLoadOp *remoteLoad                   = nullptr;
        RemoteStoreOp *remoteStore                 = nullptr;
        ReplicatedAllGatherOp *replicatedAllGather = nullptr;

        // Load
        if (phaseLoadStore.second.first) {
          logging::transform::trace(
              "[PingPong] Adding cache load of {} ({}) in phase {}",
              loadedTensorId,
              tensor->id,
              currentPingPongPhase);
          auto tensorLoadMapEntry =
              tensorLoadMap.find({tensor->id, currentPingPongPhase});
          if (tensorLoadMapEntry == tensorLoadMap.end()) {
            auto remoteLoadOp = std::make_unique<RemoteLoadOp>(
                Onnx::CustomOperators::RemoteLoad, settings);
            remoteLoad = remoteLoadOp.get();

            if (tensor->isAcclTensor()) {
              // Optimizer state: Load as late as possible
              remoteLoad->settings.schedulePriority = acclStateLoadPriority;
              // RemoteLoad in current phase
              remoteLoad->setPingPongPhase(currentPingPongPhase);
            } else {
              // Very low schedulePriority on loads (at the end of the previous
              // phase)
              remoteLoad->settings.schedulePriority = remoteLoadPriority;
              // RemoteLoad at the end of the previous phase, so that load
              // is executed before inter-IPU copy
              remoteLoad->setPingPongPhase(currentPingPongPhase - 1);
            }

            remoteLoad->settings.recomputeType = RecomputeType::Checkpoint;
            remoteLoad->setVirtualGraphId(loadStoreVGID);
            graph.moveIntoGraph(std::move(remoteLoadOp));

            remoteLoad->connectInTensor(
                RemoteStoreOp::getRemoteBufferOffsetInIndex(),
                getCacheArg(tensor->id));

            TensorId initTensorId = generateInitTensorId(tensor);
            loadedTensorId =
                generateLoadedTensorId(tensor, currentPingPongPhase);
            TensorId inTensorId;

            if (auto prevLoadId = getPreviousLoadedTensorId(tensor->id)) {
              // Tensor might not have a true producer op, but was previously
              // loaded by a RemoteLoad
              inTensorId = prevLoadId.value();
            } else if (producerOp) {
              // Tensor has a true producer op
              inTensorId = tensor->id;
            } else {
              TensorInfo initInfo = tensor->info;

              if (tensorReplicatedWeightSharding) {
                initInfo.set(initInfo.dataType(),
                             {(initInfo.nelms() - 1) / replicationFactor + 1});
              }

              // InitOp as a "producer" op
              auto initOp =
                  std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                           initInfo,
                                           TensorType::Cache,
                                           InitType::NoInit,
                                           settings);
              Op *init = initOp.get();

              if (tensor->isAcclTensor()) {
                init->settings.schedulePriority = acclStateLoadPriority;
                init->setPingPongPhase(currentPingPongPhase);
              } else {
                init->settings.schedulePriority = initPriority;
                init->setPingPongPhase(currentPingPongPhase - 1);
              }

              init->setVirtualGraphId(loadStoreVGID);
              graph.moveIntoGraph(std::move(initOp));
              init->createAndConnectOutTensor(InitOp::getOutIndex(),
                                              initTensorId);
              init->setup();
              inTensorId = initTensorId;

              // Do Init on IO tiles
              init->settings.useIoTiles = tensorUseIoTiles;
            }
            // RemoteLoad always needs both an input and an output,
            // for outlining and aliasing purposes

            // RemoteLoad updates the inTensorId...
            remoteLoad->connectInTensor(RemoteLoadOp::getCachedTensorInIndex(),
                                        inTensorId);
            // ... and aliases it under loadedTensorId
            remoteLoad->createAndConnectOutTensor(
                RemoteLoadOp::getCachedTensorOutIndex(), loadedTensorId);

            remoteLoad->setup();

            // Do RemoteLoad on IO tiles
            remoteLoad->settings.useIoTiles = tensorUseIoTiles;

            if (tensorReplicatedWeightSharding && !tensor->isAcclTensor()) {
              // Execute replicated allgather to collect the full weight
              // tensor from the individual replicas
              auto replicatedAllGatherOp =
                  std::make_unique<ReplicatedAllGatherOp>(
                      Onnx::CustomOperators::ReplicatedAllGather,
                      settings,
                      tensor->info);
              replicatedAllGather = replicatedAllGatherOp.get();
              replicatedAllGather->settings.schedulePriority =
                  allGatherPriority;
              replicatedAllGather->settings.recomputeType =
                  RecomputeType::Checkpoint;
              // RemoteLoad at the end of the previous phase, so that load
              // is executed before inter-IPU copy
              replicatedAllGather->setPingPongPhase(currentPingPongPhase - 1);
              replicatedAllGather->setVirtualGraphId(loadStoreVGID);
              graph.moveIntoGraph(std::move(replicatedAllGatherOp));

              replicatedAllGather->connectInTensor(
                  ReplicatedAllGatherOp::getInIndex(), loadedTensorId);

              replicatedAllGather->connectInTensor(
                  ReplicatedAllGatherOp::getCollectiveLinkedIndex(),
                  getCacheArg(tensor->id));

              gatheredTensorId =
                  generateGatheredTensorId(tensor, currentPingPongPhase);

              replicatedAllGather->createAndConnectOutTensor(
                  ReplicatedAllGatherOp::getOutIndex(), gatheredTensorId);

              replicatedAllGather->setup();
              graph.topoCons->insert(remoteLoad, replicatedAllGather, true);

              // Avoid outlining RemoteLoad Ops that result in a
              // different final gathered tensor shape together,
              // because it can have adverse effects due to copying tensors
              // with different final tile mapping using the same host
              // exchange
              std::string gatheredShapeString =
                  "[" +
                  logging::join(tensor->info.shape().begin(),
                                tensor->info.shape().end(),
                                ",") +
                  "]";
              remoteLoad->settings.extraOutlineAttributes.insert(
                  {"gatheredSize", gatheredShapeString});

              // Do AllGather on IO tiles
              replicatedAllGather->settings.useIoTiles = tensorUseIoTiles;

            } else {
              // Gathered and loaded tensor are the same
              gatheredTensorId = loadedTensorId;
            }

            tensorLoadMap.emplace(
                std::pair<TensorId, int64_t>(tensor->id, currentPingPongPhase),
                std::tuple<TensorId, TensorId, RemoteLoadOp *>(
                    loadedTensorId, gatheredTensorId, remoteLoad));
          } else {
            loadedTensorId   = std::get<0>(tensorLoadMapEntry->second);
            gatheredTensorId = std::get<1>(tensorLoadMapEntry->second);
            remoteLoad       = std::get<2>(tensorLoadMapEntry->second);
          }
        }

        // Store
        if (phaseLoadStore.second.second) {
          logging::transform::trace(
              "[PingPong] Adding cache store of {} ({}) in phase {}",
              loadedTensorId,
              tensor->id,
              currentPingPongPhase);
          auto tensorStoreMapEntry =
              tensorStoreMap.find({tensor->id, currentPingPongPhase});
          if (tensorStoreMapEntry == tensorStoreMap.end()) {
            auto remoteStoreOp = std::make_unique<RemoteStoreOp>(
                Onnx::CustomOperators::RemoteStore, settings);
            remoteStore = remoteStoreOp.get();
            // Very low schedulePriority on stores (at the end of a phase)
            remoteStore->settings.schedulePriority = remoteStorePriority;
            remoteStore->settings.recomputeType    = RecomputeType::Checkpoint;
            remoteStore->setPingPongPhase(currentPingPongPhase);
            remoteStore->setVirtualGraphId(loadStoreVGID);
            graph.moveIntoGraph(std::move(remoteStoreOp));

            remoteStore->connectInTensor(
                RemoteStoreOp::getRemoteBufferOffsetInIndex(),
                getCacheArg(tensor->id));
            remoteStore->connectInTensor(
                RemoteStoreOp::getCachedTensorInIndex(), loadedTensorId);
            remoteStore->setup();

            // Do allgather on IO tiles
            remoteStore->settings.useIoTiles = tensorUseIoTiles;

            tensorStoreMap.emplace(
                std::pair<TensorId, int64_t>(tensor->id, currentPingPongPhase),
                std::pair<TensorId, RemoteStoreOp *>(loadedTensorId,
                                                     remoteStore));
          } else {
            remoteStore = tensorStoreMapEntry->second.second;
          }
        }

        // Cache load has to take place before associated cache store
        if (remoteLoad && remoteStore) {
          graph.topoCons->insert(remoteLoad, remoteStore);
        }

        if (remoteStore) {
          // Any modification has to take place before the cache store
          for (Op *modifyingOp : modifiersInPhase[currentPingPongPhase]) {
            graph.topoCons->insert(modifyingOp, remoteStore);
          }
        }

        for (Op *consumerOp : consumersInPhase[currentPingPongPhase]) {
          if (tensor->isAcclTensor()) {
            if (remoteLoad) {
              graph.topoCons->insert(remoteLoad, consumerOp, true);
            }
            if (replicatedAllGather) {
              graph.topoCons->insert(replicatedAllGather, consumerOp, true);
            }
          } else {
            if (remoteLoad) {
              // Loading has to take place before the consumption
              graph.topoCons->insert(remoteLoad, consumerOp);
            }
          }

          // Logging graph change
          if (producerOp) {
            logging::transform::debug(
                "[PingPong] Disconnecting tensor {} between ops {} and {}",
                tensor->id,
                producerOp->debugName(),
                consumerOp->debugName());
          } else {
            logging::transform::debug(
                "[PingPong] Disconnecting tensor {} at op {} (modified: {})",
                tensor->id,
                consumerOp->debugName(),
                !modifiersInPhase[currentPingPongPhase].empty());
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
                optimizerOp->settings.schedulePriority = varUpdatePriority;
                optimizerOp->settings.useIoTiles       = tensorUseIoTiles;

                // If the current tensor is the weight tensor being updated,
                // update the AllReduce / ReduceScatter operation
                if (tensor->getTensorTypeInfo()->type() ==
                        TensorType::Variable &&
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

                    replicatedAllReduce->settings.schedulePriority =
                        allReducePriority;
                    replicatedAllReduce->settings.useIoTiles = tensorUseIoTiles;

                    if (tensorReplicatedWeightSharding) {
                      auto replicatedReduceScatterOp =
                          std::make_unique<ReplicatedReduceScatterOp>(
                              Onnx::CustomOperators::ReplicatedReduceScatter,
                              replicatedAllReduce->settings);
                      auto replicatedReduceScatter =
                          replicatedReduceScatterOp.get();
                      replicatedReduceScatter->fromLoss =
                          replicatedAllReduce->fromLoss;
                      replicatedReduceScatter->toLoss =
                          replicatedAllReduce->toLoss;
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
                          getCacheArg(tensor->id));

                      replicatedReduceScatter->connectOutTensor(
                          ReplicatedAllReduceOp::getOutIndex(), outId);

                      replicatedReduceScatter->setup();

                      replicatedReduceScatter->settings.schedulePriority =
                          allReducePriority;
                      replicatedReduceScatter->settings.useIoTiles =
                          tensorUseIoTiles;

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
                if (tensor->getTensorTypeInfo()->type() ==
                        TensorType::Variable &&
                    !tensor->isAcclTensor() && tensorReplicatedWeightSharding &&
                    dynamic_cast<LambSquareOp *>(optimizerOp)) {

                  Tensor *out =
                      optimizerOp->output->tensor(LambSquareOp::getOutIndex());

                  // Make sure reduction is only added once
                  auto lambSqConsumers = out->consumers.getOps();
                  if (!std::any_of(
                          lambSqConsumers.begin(),
                          lambSqConsumers.end(),
                          [](Op *op) {
                            return dynamic_cast<ReplicatedAllReduceOp *>(op);
                          })) {

                    TensorId lambIntoReduceId =
                        ir.createIntermediateTensorId(out->id);

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

    // Get new schedule
    schedule = graph.getOpSchedule({});

    // Re-setup all ops that require it (update tensor shapes)
    for (Op *op : schedule) {
      if (opsToSetup.find(op) != opsToSetup.end()) {
        op->setup();
      }
    }

    ir.setPingPongPhasesReady();
  }
  verifyPingPongPhases(graph);
  return true;
}

namespace {
// PingPong 1: Map ops to phases, enable caching on variables
bool init1 = Transform::registerTransform(new PingPong(1));
// PingPong 2: Enable caching on variables, map remaining ops to phases,
// cut graph and insert cache ops.
bool init2 = Transform::registerTransform(new PingPong(2));
} // namespace

} // namespace popart
