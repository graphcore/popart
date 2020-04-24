// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/boundary.hpp>
#include <popart/op/cache.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/loss.hpp>
#include <popart/op/recomputeprereq.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/pingpong.hpp>

namespace popart {

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

// Make sure the op has no phase higher than any succeeding ops
PingPongPhase
getSanitizedPingPongPhase(const Graph &graph, Op *op, PingPongPhase phase) {
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
  return phase;
}

// Compress priorities so that nothing is using priorities outside the range
// -9000 to +9000
void compressPriorities(Graph &graph) {
  double max_pri = 9000;

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
    double pri = max_pri;
    for (auto &priOp : pPriOpsMap) {
      for (Op *op : priOp.second) {
        op->settings.schedulePriority = pri;
      }
      pri -= max_pri / pPriOpsMap.size();
    }
  }

  {
    double pri = -max_pri;
    for (auto &priOp : nPriOpsMap) {
      for (Op *op : priOp.second) {
        op->settings.schedulePriority = pri;
      }
      pri += max_pri / nPriOpsMap.size();
    }
  }
}

} // namespace

bool PingPong::apply(Graph &graph) const {
  auto &ir                = graph.getIr();
  auto &sessionOptions    = ir.getSessionOptions();
  auto replicationDivisor = sessionOptions.enableReplicatedGraphs
                                ? sessionOptions.replicatedGraphCount
                                : 1;
  const auto num_ipus = ir.getDeviceInfo()->getNumIpus() / replicationDivisor;
  // const auto training = ir.canTrain();
  const auto num_phases = sessionOptions.pingPongPhases;

  if (graph.getOps().size() == 0 || num_phases < 2) {
    return true;
  }

  logging::transform::debug(
      "[PingPong] pingpong scheme with {} phases, {} ipus",
      num_phases,
      num_ipus);

  if (pass == 1 || pass == 2) {
    auto schedule = graph.getOpSchedule({});

    // Set all variable tensors to cached.
    // TODO: Offer more refined scheme
    for (TensorId id : graph.getTensors().getIds(TensorType::Variable)) {
      auto tensor = graph.getTensors().get(id);
      tensor->setCached(true);
    }

    float cumulative_cost = 0.f;

    for (Op *op : schedule) {
      cumulative_cost += costFn(op);
    }

    float cost_per_phase = cumulative_cost / static_cast<float>(num_phases);

    // Eager claiming of ops per phase according to execution schedule
    // TODO: Find better graph-cut algorithm for phase splitting
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

      bool has_phase    = op->hasPingPongPhase();
      auto actual_phase = getSanitizedPingPongPhase(
          graph, op, has_phase ? op->getPingPongPhase() : phase);

      logging::transform::debug(
          "[PingPong] (FWD) mapping operator {} to {} phase {}",
          op->opid,
          has_phase ? "(existing)" : "",
          actual_phase);
      op->setPingPongPhase(actual_phase);
      if (!op->hasVirtualGraphId() && !dynamic_cast<IpuCopyOp *>(op))
        op->setVirtualGraphId(actual_phase % num_ipus);
      logging::transform::debug(
          "[PingPong] (FWD) mapping operator {} to VGID {}",
          op->opid,
          op->hasVirtualGraphId() ? op->getVirtualGraphId() : -1);
    }

    // Put the user defined losses on the final virtual graph.
    // Losses should occur on the same virtual graph as the last FWD operators.
    if (pass == 1) {
      for (auto &loss : graph.getLosses()) {
        loss->virtualGraph((num_phases - 1) % num_ipus);
      }
    }

    // Recomputation annotation
    logging::transform::debug("[PingPong] Recomputation & Cache annotation");
    for (auto &op : graph.getOps()) {
      // Cached everything not set by the user by default
      if (op.second->settings.cacheType == CacheType::UNDEFINED) {
        if (op.second->opid == Onnx::CustomOperators::GetRandomSeed) {
          op.second->settings.cacheType = CacheType::UNCACHED;
          logging::transform::trace("[PingPong] {} set to UNCACHED",
                                    op.second->debugName());
        } else {
          op.second->settings.cacheType = CacheType::CACHED;
          logging::transform::trace("[PingPong] {} set to CACHED",
                                    op.second->debugName());
        }
      }
      // Recompute everything else (fwd) not set by the user by default
      if (op.second->settings.recomputeType == RecomputeType::UNDEFINED) {
        if (!op.second->isIpuCopyOp() &&
            !dynamic_cast<GetRandomSeedOp *>(op.second.get())) {
          op.second->settings.recomputeType = RecomputeType::RECOMPUTE;
          logging::transform::trace("[PingPong] {} set to RECOMPUTE",
                                    op.second->debugName());
        } else {
          op.second->settings.recomputeType = RecomputeType::CHECKPOINT;
        }
      }
    }
  }

  if (pass == 3) {
    // Remap backward pass to the right phase
    for (Op *op : graph.getOpSchedule({})) {
      if (op->fromLoss == PathFromLoss::Yes) {
        if (op->hasPingPongPhase()) {
          // By virtue of creating backward ops (copying fwd ops settings),
          // the initial phase will be the same as the forward phase
          auto producer_phase = op->getOptionalPingPongPhase().get();

          // Remap to correct phase
          PingPongPhase phase = 2 * num_phases - 2 - producer_phase;

          // Make sure phase adheres to producer/consumer and topological
          // constraints
          phase = getSanitizedPingPongPhase(graph, op, phase);

          logging::transform::debug(
              "[PingPong] (BWD) mapping operator {} to phase {}",
              op->opid,
              phase);
          op->setPingPongPhase(phase);
          if (!op->hasVirtualGraphId() && !dynamic_cast<IpuCopyOp *>(op))
            op->setVirtualGraphId(phase % num_ipus);
          logging::transform::debug(
              "[PingPong] (BWD) mapping operator {} to VGID {}",
              op->opid,
              op->hasVirtualGraphId() ? op->getVirtualGraphId() : -1);
        }
      }
    }
  }

  // Figure out the right phase for ops that did not get a phase yet
  while (true) {
    int num_ops_without_phase = 0;
    for (Op *op : graph.getOpSchedule({})) {
      if (!static_cast<bool>(op->getOptionalPingPongPhase())) {
        // Check which phase the consumers of the output are in
        bool has_phase      = false;
        PingPongPhase phase = 0;

        for (auto input : op->input->tensorMap()) {
          Tensor *tensor = input.second;
          Op *producerOp = tensor->getProducerUnsafe();
          if (producerOp &&
              static_cast<bool>(producerOp->getOptionalPingPongPhase())) {
            auto op_phase = producerOp->getOptionalPingPongPhase().get();
            phase         = has_phase ? std::max(op_phase, phase) : op_phase;
            has_phase     = true;
          }
        }

        for (auto output : op->output->tensorMap()) {
          Tensor *tensor = output.second;
          for (Op *consumerOp : tensor->consumers.getOps()) {
            if (static_cast<bool>(consumerOp->getOptionalPingPongPhase())) {
              PingPongPhase op_phase =
                  consumerOp->getOptionalPingPongPhase().get();
              if (op->isIpuCopyOp()) {
                // Ensure the IPU copy is enrolled in the phase
                // before the first consumer
                op_phase--;
              }
              phase     = has_phase ? std::min(op_phase, phase) : op_phase;
              has_phase = true;
            }
          }
        }

        if (has_phase) {

          // Make sure phase adheres to producer/consumer and topological
          // constraints
          phase = getSanitizedPingPongPhase(graph, op, phase);

          logging::transform::debug(
              "[PingPong] (REM) mapping operator {} to phase {}",
              op->opid,
              phase);
          op->setPingPongPhase(phase);
          if (!op->hasVirtualGraphId() && !dynamic_cast<IpuCopyOp *>(op))
            op->setVirtualGraphId(phase % num_ipus);
          logging::transform::debug(
              "[PingPong] (REM) mapping operator {} to VGID {}",
              op->opid,
              op->hasVirtualGraphId() ? op->getVirtualGraphId() : -1);
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
  if (pass == 4) {
    ir.setPingPongPhasesReady();

    // Make sure no priorities outside of the range needed by
    // pingpong are being used.
    compressPriorities(graph);

    for (auto &op : graph.getOps()) {
      // Assign correct priority to all Inter-IPU copies
      if (op.second->isIpuCopyOp()) {
        // TODO: T13885 differentiate between IPU copies between same-phase IPUs
        //       and different-phase IPUs (8 vs. 8)
        // Special type of IPUCopy between PingPong phases
        // Priority before CacheStore but after CacheLoad
        IpuCopyOp *copy = dynamic_cast<IpuCopyOp *>(op.second.get());
        if (!op.second->copiesOptimizerTensors() &&
            copy->getDestIpu() % 2 != copy->getSourceIpu() % 2) {
          op.second->settings.schedulePriority = -9999.0;
        }
        // Always keep IPU copy checkpointed
        if (op.second->settings.recomputeType == RecomputeType::UNDEFINED) {
          op.second->settings.recomputeType = RecomputeType::CHECKPOINT;
          logging::transform::trace("[PingPong] {} set to CHECKPOINT",
                                    op.second->debugName());
        }
      }
    }

    // Insert, for every bwd phase, dummy consumers for recomputation
    logging::transform::debug(
        "[PingPong] Recomputation prerequirement dummy consumers");
    std::map<int64_t, std::set<TensorId>> bwdRecomputePrereqs;
    for (TensorId tensor_id : graph.getTensors().getAllTensorIds()) {
      Tensor *tensor = graph.getTensors().get(tensor_id);
      Op *producerOp = tensor->hasProducer() ? tensor->getProducer() : nullptr;
      bool cached    = false;
      // Enabling factors
      cached |=
          ((producerOp &&
            producerOp->settings.recomputeType != RecomputeType::RECOMPUTE &&
            producerOp->settings.cacheType == CacheType::CACHED));
      cached |= (!producerOp && tensor->isCached());

      // Disabling factors
      cached &= !tensor->isOptimizerTensor();

      // If we have a cached tensor ...
      if (cached) {
        for (Op *consumer : tensor->consumers.getOps()) {
          // ... and a consumer of that tensor that is set to recompute ...
          if (consumer->settings.recomputeType == RecomputeType::RECOMPUTE &&
              (consumer->toLoss == PathToLoss::Yes ||
               consumer->scheduledPreLoss == ScheduledPreLoss::Yes) &&
              consumer->fromLoss != PathFromLoss::Yes) {
            // ... then we need to ensure that tensor is stored and loaded
            // by inserting operations as backward pass recompute
            // prerequisites
            bwdRecomputePrereqs[2 * num_phases - 2 -
                                consumer->getPingPongPhase()]
                .insert(tensor->id);
          }
        }
      }
    }

    for (auto &entry : bwdRecomputePrereqs) {
      auto recomputePrereqOp =
          std::make_unique<RecomputePrereqOp>(Op::Settings(graph, ""));
      auto recomputePrereq = recomputePrereqOp.get();
      // Very high priority on prerequisites (at the start of a phase)
      recomputePrereq->settings.schedulePriority = 10000.0;
      recomputePrereq->fromLoss                  = PathFromLoss::Yes;
      recomputePrereq->toLoss                    = PathToLoss::No;
      recomputePrereq->setPingPongPhase(entry.first);
      VGraphId vgid = entry.first % num_ipus;
      recomputePrereq->setVirtualGraphId(vgid);
      graph.moveIntoGraph(std::move(recomputePrereqOp));
      int i = 0;
      for (TensorId id : entry.second) {
        recomputePrereq->connectInTensor(i, id);
        i++;
      }
    }

    // Remove tensors and CACHED from memory as needed
    // If two ops are on the same virtual graph,
    // and their phase is non-adjacently different,
    // then the tensor should be disconnected and backed up / restored
    Op::Settings settings(graph, "");

    std::vector<TensorId> cacheArgIds;

    std::map<std::pair<TensorId, int64_t>, std::pair<TensorId, CacheStoreOp *>>
        tensorStoreMap;
    std::map<std::pair<TensorId, int64_t>, std::pair<TensorId, CacheLoadOp *>>
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
        [&tensorLoadMap](TensorId id) -> boost::optional<std::string> {
      boost::optional<std::string> prev_id;
      int64_t last_phase = -1;
      for (auto &elem : tensorLoadMap) {
        if (elem.first.first == id && elem.first.second > last_phase) {
          prev_id    = elem.second.first;
          last_phase = elem.first.second;
        }
      }
      return prev_id;
    };

    logging::transform::debug(
        "[PingPong] Processing tensors across PingPong phases");
    Tensors &tensors = graph.getTensors();
    for (TensorId id : tensors.getAllTensorIds()) {
      Tensor *tensor = tensors.get(id);

      auto producerOp = tensor->getProducerUnsafe();

      PathFromLoss producerPathFromLoss = PathFromLoss::Undefined;
      PathToLoss producerPathToLoss     = PathToLoss::Undefined;
      int64_t producerPingPongPhase     = -1;
      if (producerOp && producerOp->getOptionalPingPongPhase()) {
        producerPathFromLoss  = producerOp->fromLoss;
        producerPathToLoss    = producerOp->toLoss;
        producerPingPongPhase = producerOp->getOptionalPingPongPhase().get();
      }

      auto consumerOps = tensor->consumers.getOps();

      // Process consumers in ascending order of phases
      std::sort(
          consumerOps.begin(),
          consumerOps.end(),
          [](const Op *lhs, const Op *rhs) {
            return (lhs->hasPingPongPhase() ? lhs->getPingPongPhase() : -1) <
                   (rhs->hasPingPongPhase() ? rhs->getPingPongPhase() : -1);
          });

      for (auto consumerOp : consumerOps) {
        if (consumerOp->getOptionalPingPongPhase()) {
          PathFromLoss consumerPathFromLoss = consumerOp->fromLoss;
          PathToLoss consumerPathToLoss     = consumerOp->toLoss;
          auto consumerPingPongPhase =
              consumerOp->getOptionalPingPongPhase().get();

          // Check if the consumer OP modifies the tensor, e.g. for weights
          // if yes, then the tensor requires to be backed-up at the end of the
          // phase
          bool consumerModifiesTensor = false;
          for (InIndex index : consumerOp->input->indices(tensor)) {
            auto modified = consumerOp->modifies(index);
            if (!std::all_of(
                    modified.begin(),
                    modified.end(),
                    [](const view::Region &r) { return r.isEmpty(); })) {
              consumerModifiesTensor = true;
            }
          }

          // Phases are not adjacent or the tensor does not have a producer
          // and is explicitly cached
          // Recomputation is not enabled on the producer
          // CACHED is enabled on the producer
          bool cached = false;

          // Enabling factors
          cached |= ((producerOp &&
                      producerOp->settings.recomputeType !=
                          RecomputeType::RECOMPUTE &&
                      producerOp->settings.cacheType == CacheType::CACHED) &&
                     (abs(producerPingPongPhase - consumerPingPongPhase) > 2));
          cached |= (!producerOp && tensor->isCached());

          // Disabling factors
          cached &= !tensor->isOptimizerTensor();

          if (cached) {
            // Do we need to store the tensor in the producer phase?
            // (Only if it wasn't stored in the producer phase yet)
            CacheStoreOp *cacheStoreProducer = nullptr;
            if (producerOp) {
              auto storePhase = producerPingPongPhase;
              // Special case when the producer is an IPUCopy from
              // the previous phase
              if (dynamic_cast<IpuCopyOp *>(producerOp))
                storePhase += 1;

              auto tensorStoreMapEntry =
                  tensorStoreMap.find({tensor->id, storePhase});
              if (tensorStoreMapEntry == tensorStoreMap.end()) {
                auto cacheStoreOp = std::make_unique<CacheStoreOp>(
                    Onnx::CustomOperators::CacheStore, settings);
                cacheStoreProducer = cacheStoreOp.get();
                // Very low priority on stores (at the end of a phase)
                cacheStoreProducer->settings.schedulePriority = -10000.0;
                cacheStoreProducer->fromLoss = producerPathFromLoss;
                cacheStoreProducer->toLoss   = producerPathToLoss;
                cacheStoreProducer->settings.recomputeType =
                    RecomputeType::CHECKPOINT;
                cacheStoreProducer->setPingPongPhase(storePhase);
                VGraphId vgid = consumerPingPongPhase % num_ipus;
                cacheStoreProducer->setVirtualGraphId(vgid);
                graph.moveIntoGraph(std::move(cacheStoreOp));

                cacheStoreProducer->connectInTensor(
                    CacheStoreOp::getRemoteBufferOffsetInIndex(),
                    getCacheArg(tensor->id));
                cacheStoreProducer->connectInTensor(
                    CacheStoreOp::getCachedTensorInIndex(), tensor->id);
                cacheStoreProducer->setup();
                tensorStoreMap.emplace(
                    std::pair<TensorId, int64_t>(tensor->id, storePhase),
                    std::pair<TensorId, CacheStoreOp *>(tensor->id,
                                                        cacheStoreProducer));
              } else {
                cacheStoreProducer = tensorStoreMapEntry->second.second;
              }
            }

            // Do we need to load the tensor in the consumer phase?
            // (Only if it wasn't loaded in the consumer phase yet)
            CacheLoadOp *cacheLoad = nullptr;
            TensorId loadedTensorId;
            auto tensorLoadMapEntry =
                tensorLoadMap.find({tensor->id, consumerPingPongPhase});
            if (tensorLoadMapEntry == tensorLoadMap.end()) {
              auto cacheLoadOp = std::make_unique<CacheLoadOp>(
                  Onnx::CustomOperators::CacheLoad, settings);
              cacheLoad = cacheLoadOp.get();
              // Very low priority on loads (at the end of the previous phase)
              cacheLoad->settings.schedulePriority = -9998.0;
              cacheLoad->fromLoss                  = consumerPathFromLoss;
              cacheLoad->toLoss                    = consumerPathToLoss;
              cacheLoad->settings.recomputeType    = RecomputeType::CHECKPOINT;
              // CacheLoad at the end of the previous phase, so that load
              // is executed before inter-IPU copy
              cacheLoad->setPingPongPhase(consumerPingPongPhase - 1);
              VGraphId cLoadVgid = consumerPingPongPhase % num_ipus;
              cacheLoad->setVirtualGraphId(cLoadVgid);
              graph.moveIntoGraph(std::move(cacheLoadOp));

              cacheLoad->connectInTensor(
                  CacheStoreOp::getRemoteBufferOffsetInIndex(),
                  getCacheArg(tensor->id));

              TensorId initTensorId = generateInitTensorId(tensor);
              loadedTensorId =
                  generateLoadedTensorId(tensor, consumerPingPongPhase);
              TensorId inTensorId;

              if (auto prevLoadId = getPreviousLoadedTensorId(tensor->id)) {
                // Tensor might not have a true producer op, but was previously
                // loaded by a CacheLoad
                inTensorId = prevLoadId.get();
              } else if (producerOp) {
                // Tensor has a true producer op
                inTensorId = tensor->id;
              } else {
                // InitOp as a "producer" op
                auto initOp =
                    std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                             tensor->info,
                                             TensorType::Cache,
                                             InitType::NONE,
                                             settings);
                Op *init                        = initOp.get();
                init->settings.schedulePriority = -9997.0;
                init->fromLoss                  = consumerPathFromLoss;
                init->toLoss                    = consumerPathToLoss;
                init->setPingPongPhase(consumerPingPongPhase - 1);
                VGraphId cAllVgid = consumerPingPongPhase % num_ipus;
                init->setVirtualGraphId(cAllVgid);
                graph.moveIntoGraph(std::move(initOp));
                init->createAndConnectOutTensor(InitOp::getOutIndex(),
                                                initTensorId);
                init->setup();
                inTensorId = initTensorId;
              }
              // CacheLoad always needs both an input and an output,
              // for outlining and aliasing purposes

              // CacheLoad updates the inTensorId...
              cacheLoad->connectInTensor(CacheLoadOp::getCachedTensorInIndex(),
                                         inTensorId);
              // ... and aliases it under loadedTensorId
              cacheLoad->createAndConnectOutTensor(
                  CacheLoadOp::getCachedTensorOutIndex(), loadedTensorId);

              cacheLoad->setup();

              tensorLoadMap.emplace(std::pair<TensorId, int64_t>(
                                        tensor->id, consumerPingPongPhase),
                                    std::pair<TensorId, CacheLoadOp *>(
                                        loadedTensorId, cacheLoad));
            } else {
              loadedTensorId = tensorLoadMapEntry->second.first;
              cacheLoad      = tensorLoadMapEntry->second.second;

              // Cache loads always belong to the lowest consumer phase
              if (cacheLoad->toLoss != PathToLoss::Yes &&
                  consumerPathToLoss == PathToLoss::Yes) {
                cacheLoad->toLoss   = PathToLoss::Yes;
                cacheLoad->fromLoss = PathFromLoss::No;
              }
            }

            // Do we need to store the tensor in the consumer phase?
            // (Only if it wasn't stored in the consumer phase yet,
            // and has been modified by the consumer)
            CacheStoreOp *cacheStoreConsumer = nullptr;
            if (consumerModifiesTensor) {
              auto tensorStoreMapEntry =
                  tensorStoreMap.find({tensor->id, consumerPingPongPhase});
              if (tensorStoreMapEntry == tensorStoreMap.end()) {
                auto cacheStoreOp = std::make_unique<CacheStoreOp>(
                    Onnx::CustomOperators::CacheStore, settings);
                cacheStoreConsumer = cacheStoreOp.get();
                // Very low priority on stores (at the end of a phase)
                cacheStoreConsumer->settings.schedulePriority = -10000.0;
                cacheStoreConsumer->fromLoss = consumerPathFromLoss;
                cacheStoreConsumer->toLoss   = consumerPathToLoss;
                cacheStoreConsumer->settings.recomputeType =
                    RecomputeType::CHECKPOINT;
                cacheStoreConsumer->setPingPongPhase(consumerPingPongPhase);
                VGraphId cStoreVgid = consumerPingPongPhase % num_ipus;
                cacheStoreConsumer->setVirtualGraphId(cStoreVgid);
                graph.moveIntoGraph(std::move(cacheStoreOp));

                cacheStoreConsumer->connectInTensor(
                    CacheStoreOp::getRemoteBufferOffsetInIndex(),
                    getCacheArg(tensor->id));
                cacheStoreConsumer->connectInTensor(
                    CacheStoreOp::getCachedTensorInIndex(), loadedTensorId);
                cacheStoreConsumer->setup();
                tensorStoreMap.emplace(std::pair<TensorId, int64_t>(
                                           tensor->id, consumerPingPongPhase),
                                       std::pair<TensorId, CacheStoreOp *>(
                                           loadedTensorId, cacheStoreConsumer));
              } else {
                cacheStoreConsumer = tensorStoreMapEntry->second.second;

                // Cache stores always belong to the highest consumer phase
                if (cacheStoreConsumer->fromLoss != PathFromLoss::Yes &&
                    consumerPathFromLoss == PathFromLoss::Yes) {
                  cacheStoreConsumer->toLoss   = PathToLoss::No;
                  cacheStoreConsumer->fromLoss = PathFromLoss::Yes;
                }
              }
            }

            // Set up the correct topology
            if (producerOp) {
              logging::transform::debug(
                  "[PingPong] Disconnecting tensor {} between ops {} and {}",
                  tensor->id,
                  producerOp->opid,
                  consumerOp->opid);
              // Load has to be scheduled after the associated store,
              // if the store precedes the load in the DAG
              if (cacheStoreProducer != nullptr && cacheLoad != nullptr &&
                  producerPingPongPhase < consumerPingPongPhase)
                graph.topoCons.get()->insert(cacheStoreProducer, cacheLoad);
            } else {
              logging::transform::debug(
                  "[PingPong] Disconnecting tensor {} at op {} (modified: {})",
                  tensor->id,
                  consumerOp->opid,
                  consumerModifiesTensor);
            }
            if (consumerModifiesTensor) {
              // Consumer-modified tensor; in this case we have to store also
              // if the consumer has modified the tensor, and in that case the
              // load precedes the store
              if (cacheLoad != nullptr && cacheStoreConsumer != nullptr)
                graph.topoCons.get()->insert(cacheLoad, cacheStoreConsumer);

              if (consumerOp != nullptr && cacheStoreConsumer != nullptr)
                graph.topoCons.get()->insert(consumerOp, cacheStoreConsumer);
            }
            graph.topoCons.get()->insert(cacheLoad, consumerOp);
            // Disconnect original tensor and wire up loaded tensor
            auto indices = consumerOp->input->indices(tensor);
            for (auto i : indices) {
              auto *copyOp = dynamic_cast<IpuCopyOp *>(consumerOp);
              if (copyOp) {
                auto sourceIpu = copyOp->getSourceIpus().at(tensor->id);
                copyOp->disconnectInTensor(i, tensor);
                copyOp->connectInTensor(i, loadedTensorId, sourceIpu);
              } else {
                consumerOp->disconnectInTensor(i, tensor);
                consumerOp->connectInTensor(i, loadedTensorId);
              }
            }
          }
        }
      }
    }

    // Insert boundaries to stop the subgraph outlining algorithm
    // 2x at the end of each phase
    if (sessionOptions.enableOutlining) {
      for (PingPongPhase phase = 0; phase < 2 * num_phases - 2; ++phase) {
        {
          auto boundaryOp = std::make_unique<BoundaryOp>(
              Op::Settings(graph, "PhaseBoundary"));
          auto boundary = boundaryOp.get();
          boundary->setPingPongPhase(phase);
          // Before CacheAlloc/CacheStore/CacheLoad
          boundary->settings.schedulePriority = -9996.0;
          VGraphId bdVgid                     = phase % num_ipus;
          boundary->setVirtualGraphId(bdVgid);
          graph.moveIntoGraph(std::move(boundaryOp));
        }
        {
          auto boundaryOp = std::make_unique<BoundaryOp>(
              Op::Settings(graph, "PhaseBoundary"));
          auto boundary = boundaryOp.get();
          boundary->setPingPongPhase(phase);
          // After CacheAlloc/CacheStore/CacheLoad
          boundary->settings.schedulePriority = -10001.0;
          VGraphId bdVgid                     = phase % num_ipus;
          boundary->setVirtualGraphId(bdVgid);
          graph.moveIntoGraph(std::move(boundaryOp));
        }
      }
    }
  }
  return true;
}

namespace {
// PingPong 1: Map forward pass to phases
bool init1 = Transform::registerTransform(new PingPong(1));
// PingPong 2: Map forward + loss pass to phases
bool init2 = Transform::registerTransform(new PingPong(2));
// PingPong 3: Map backward pass to phases
bool init3 = Transform::registerTransform(new PingPong(3));
// PingPong 3: Map remaining ops to phases, cut graph and insert cache ops.
bool init4 = Transform::registerTransform(new PingPong(4));
} // namespace

} // namespace popart
