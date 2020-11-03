// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/init.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/remote.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/slice.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/batchserialscheduler.hpp>

namespace popart {

BatchSerialScheduler::BatchSerialScheduler(Graph &graph_) : graph(graph_) {}

void BatchSerialScheduler::apply() {
  auto &ir               = graph.getIr();
  auto settings          = ir.getSessionOptions().batchSerializationSettings;
  int64_t batchSerFactor = settings.factor;
  auto schedule          = graph.getOpSchedule({}, RequireOptimalSchedule::Yes);

  // Crystallize schedule within batch serialized phase by inserting topo
  // cons

  for (size_t i = 0; i < schedule.size(); ++i) {
    opScheduleIndex[schedule.at(i)]   = i;
    opSubgraphEquivId[schedule.at(i)] = schedule.at(i)->getSubgraphEquivId();
  }

  // Find equivalence classes, derive positions
  Section section   = -1;
  Position position = 0;
  bool nextSection  = true;
  for (Op *op : schedule) {
    logging::transform::trace(
        "[BatchSerialize] BSP: {} S: {} P: {} prio: {} OP: {}, SID: {}",
        op->hasBatchSerializedPhase()
            ? std::to_string(op->getBatchSerializedPhase())
            : "*",
        section,
        position,
        op->settings.schedulePriority,
        op->debugName(),
        op->getSubgraphEquivId());
    if (op->hasBatchSerializedPhase()) {
      auto bsp = op->getBatchSerializedPhase();
      if (bsp == 0) {
        if (nextSection) {
          ++section;
          nextSection = false;
        }
        opToPosition[{section, bsp}][op]       = position;
        positionToOp[{section, bsp}][position] = op;
        opToSection[op]                        = section;

        // First batch defines schedule order
        position++;
      } else if (bsp > 0) {
        nextSection = true;
      }
    } else {
      // Ops with no annotated bsp that occur after a section
      opsBehindSection[section].push_back(op);
    }
  }

  // Find seed fronts
  auto parallelTraceFront = findParallelTraceFronts(schedule, batchSerFactor);

  logging::trace("[BatchSerialize] Parallel trace fronts: {}",
                 parallelTraceFront.size());

  std::set<std::pair<std::vector<Tensor *>, TraceDirection>> visited;

  while (!parallelTraceFront.empty()) {
    std::map<std::tuple<OpId, TraceDirection, int>, std::vector<Tensor *>>
        nextFronts;
    auto traceFront = parallelTraceFront.top();
    auto &tensors   = traceFront.tensors;
    auto &direction = traceFront.direction;
    parallelTraceFront.pop();

    std::vector<TensorId> ids;
    for (Tensor *t : tensors) {
      ids.push_back(t->id);
    }

    logging::transform::trace(
        "[BatchSerialize] Current ({}) front: {} (score: {}, remaining: "
        "{})",
        direction == TraceDirection::Forward ? "forward" : "backward",
        ids,
        traceFront.score(),
        parallelTraceFront.size());

    std::vector<Tensor *> frontTensors;
    std::vector<std::vector<Op *>> frontOps;
    visited.insert({tensors, direction});
    for (Tensor *t : tensors) {
      std::vector<Op *> fops;
      if (direction == TraceDirection::Forward) {
        fops = t->consumers.getOps();
        frontOps.push_back(fops);
      } else {
        if (t->hasProducer()) {
          fops.push_back(t->getProducer());
          frontOps.push_back(fops);
        } else {
          // Change direction on tensors without producers
          frontTensors.push_back(t);
        }
      }
    }
    if (frontTensors.size() != 0) {
      nextFronts[{-1, TraceDirection::Forward, -1}] = frontTensors;
    }

    // Skip tracing of certain tensors that can lead to false
    // positive isomporphism results
    if (std::any_of(tensors.begin(), tensors.end(), [](Tensor *t) {
          TensorId id = t->id;
          return id.find(reservedIndexPrefix()) != std::string::npos ||
                 id.find(reservedRandomSeedPrefix()) != std::string::npos;
          ;
        })) {
      continue;
    }

    if (!frontOps.empty() && !std::any_of(frontOps.begin(),
                                          frontOps.end(),
                                          [](const std::vector<Op *> &ops) {
                                            return ops.empty();
                                          })) {
      for (Op *op0 : frontOps.front()) {
        if (!op0->hasBatchSerializedPhase() ||
            op0->getBatchSerializedPhase() != 0 ||
            equivProcessedOps.find(op0) != equivProcessedOps.end()) {
          continue;
        }
        equivProcessedOps.insert(op0);

        section = opToSection.at(op0);
        std::vector<Op *> equivalentOps;
        std::set<BatchSerializedPhase> foundBSPs;
        foundBSPs.insert(op0->getBatchSerializedPhase());

        for (auto tensorAndIndex : op0->output->indicesMap()) {
          for (InIndex index : tensorAndIndex.second) {
            nextFronts[{op0->id, TraceDirection::Forward, index}].push_back(
                tensorAndIndex.first);
          }
        }
        for (auto tensorAndIndex : op0->input->indicesMap()) {
          for (InIndex index : tensorAndIndex.second) {
            nextFronts[{op0->id, TraceDirection::Backward, index}].push_back(
                tensorAndIndex.first);
          }
        }

        std::map<BatchSerializedPhase, std::vector<Op *>> binnedOps;

        for (auto ops : frontOps) {
          // Iterate through potentially isomorphic ops
          for (Op *op1 : ops) {
            if (op1->id != op0->id && op1->toLoss == op0->toLoss &&
                op1->fromLoss == op0->fromLoss &&
                opSubgraphEquivId[op1] == opSubgraphEquivId[op0] &&
                op1->hasBatchSerializedPhase() &&
                foundBSPs.find(op1->getBatchSerializedPhase()) ==
                    foundBSPs.end() &&
                equivProcessedOps.find(op1) == equivProcessedOps.end()) {
              binnedOps[op1->getBatchSerializedPhase()].push_back(op1);
            }
          }
        }

        for (auto &phaseAndOps : binnedOps) {
          // Pick the top Op from each batch serialized phase bin
          auto &binOps = phaseAndOps.second;
          // Get element with highest local isomorphism score against op0
          Op *op1 = *std::max_element(
              binOps.begin(), binOps.end(), [this, &op0](Op *lhs, Op *rhs) {
                std::set<std::pair<Op *, Op *>> visitedOpsLhs;
                std::set<std::pair<Op *, Op *>> visitedOpsRhs;
                if (lhs->id == rhs->id) {
                  return false;
                }
                int depth            = 1;
                int64_t lhsScore     = 0;
                int64_t rhsScore     = 0;
                int64_t lastLhsScore = 0;
                int64_t lastRhsScore = 0;
                do {
                  visitedOpsLhs.clear();
                  visitedOpsRhs.clear();
                  lastLhsScore = lhsScore;
                  lastRhsScore = rhsScore;
                  lhsScore =
                      getLocalIsoScore({op0, lhs}, visitedOpsLhs, depth, true);
                  rhsScore =
                      getLocalIsoScore({op0, rhs}, visitedOpsRhs, depth, true);
                  ++depth;
                } while (std::abs(lhsScore - rhsScore) < 1 &&
                         lastLhsScore != lhsScore && lastRhsScore != rhsScore);
                return lhsScore < rhsScore;
              });

          BatchSerializedPhase bsp = op1->getBatchSerializedPhase();
          foundBSPs.insert(bsp);

          for (auto tensorAndIndex : op1->output->indicesMap()) {
            for (InIndex index : tensorAndIndex.second) {
              nextFronts[{op0->id, TraceDirection::Forward, index}].push_back(
                  tensorAndIndex.first);
            }
          }
          for (auto tensorAndIndex : op1->input->indicesMap()) {
            for (InIndex index : tensorAndIndex.second) {
              nextFronts[{op0->id, TraceDirection::Backward, index}].push_back(
                  tensorAndIndex.first);
            }
          }

          auto pos                          = opToPosition[{section, 0}][op0];
          opToPosition[{section, bsp}][op1] = pos;
          positionToOp[{section, bsp}][pos] = op1;
          opToSection[op1]                  = section;
          equivProcessedOps.insert(op1);
        }
      }
    }
    for (auto nextFront : nextFronts) {
      bool alreadyVisited =
          visited.find({nextFront.second, std::get<1>(nextFront.first)}) !=
          visited.end();
      if (alreadyVisited || nextFront.second.size() != batchSerFactor) {
        std::vector<TensorId> idsLocal;
        for (Tensor *tx : nextFront.second) {
          idsLocal.push_back(tx->id);
        }
        logging::transform::trace(
            "[BatchSerialize] Front {}{} size {} is a deadend",
            idsLocal,
            alreadyVisited ? " (already visited)" : "",
            idsLocal.size());
      } else {
        // All front tensors for the different BSPs have been found
        parallelTraceFront.push(
            TraceFront(nextFront.second, std::get<1>(nextFront.first)));
      }
    }
  }

  // Check if for any BSP > 0, the isomorphic Op in BSP 0 could not be found
  // and clean up BSP settings on non-isomorphic Ops
  section = -1;
  for (Op *op : schedule) {
    if (op->hasBatchSerializedPhase() && op->getBatchSerializedPhase() >= 0) {
      if (opToSection.find(op) == opToSection.end()) {
        logging::warn("[BatchSerialize] Could not find isomorphic "
                      "position for {} (id: {})",
                      op->debugName(),
                      opSubgraphEquivId.at(op));
        op->setBatchSerializedPhase(OptionalBatchSerializedPhase());
        opsBehindSection[section].push_back(op);
      } else {
        section   = opToSection.at(op);
        auto bsp0 = op->getBatchSerializedPhase();
        if (bsp0 == 0) {
          auto pos       = opToPosition.at({section, bsp0}).at(op);
          bool hasIsoOps = false;
          // Check if the Op with BSP == 0 has any isomorphic operations
          for (auto bsp1 = 1; bsp1 < batchSerFactor; ++bsp1) {
            auto posToOp = positionToOp.find({section, bsp1});
            hasIsoOps |= (posToOp != positionToOp.end() &&
                          posToOp->second.find(pos) != posToOp->second.end());
          }
          if (!hasIsoOps) {
            logging::warn("[BatchSerialize] Could not find isomorphic "
                          "position for {} (id: {})",
                          op->debugName(),
                          opSubgraphEquivId.at(op));
            opToSection.erase(op);
            op->setBatchSerializedPhase(OptionalBatchSerializedPhase());
          }
        }
      }
    }
  }

  const bool isOverlapSchedule =
      (settings.batchSchedule ==
       BatchSerializationBatchSchedule::OverlapOnIo) ||
      (settings.batchSchedule ==
       BatchSerializationBatchSchedule::OverlapOnCompute);

  if (isOverlapSchedule) {
    // Swap positions of operations inside the bins
    tryToMakeAmenableToParallelization();
  }

  // Crystallize schedule within each batch serialized phase
  for (auto &positions : positionToOp) {
    Op *prev = nullptr;
    for (auto &pos : positions.second) {
      logging::transform::trace("[BatchSerialize] Fixed: {} {} {} {}",
                                positions.first.first,
                                positions.first.second,
                                pos.first,
                                pos.second->debugName());
      Op *op = pos.second;
      if (prev) {
        graph.topoCons->insert(prev, op, true);
      }
      prev = op;
    }
    if (prev) {
      for (Op *op : opsBehindSection[positions.first.first]) {
        // Stop Ops behind the batch serialized phases to slide up
        // in the schedule
        graph.topoCons->insert(prev, op);
      }
    }
  }

  if (isOverlapSchedule) {
    // Insert constraints to interleave loading/computing
    // between batch serialized phases
    addParallelizationConstraints(graph);
  }
}

BatchSerialScheduler::TraceFront::TraceFront(std::vector<Tensor *> tensors_,
                                             TraceDirection direction_)
    : tensors(tensors_), direction(direction_) {}

int64_t BatchSerialScheduler::TraceFront::score() const {
  int64_t score = 0;
  if (direction == TraceDirection::Forward) {
    for (Tensor *t : tensors) {
      score += t->consumers.getOps().size();
    }
  } else {
    for (Tensor *t : tensors) {
      score += t->hasProducer() ? 1 : 0;
    }
  }
  return score;
}

// Trace fronts with fewer producers/consumers first
// (smaller chance of matching wrong isomorphic ops)
bool BatchSerialScheduler::TraceFront::operator<(const TraceFront &rhs) const {
  return score() > rhs.score();
}

BatchSerialTensorContext getBatchSerialTensorContext(const Op *op) {
  const auto &ir = op->getIr();
  auto settings  = ir.getSessionOptions();
  VGraphId vgid  = op->hasVirtualGraphId() ? op->getVirtualGraphId() : -1;
  ExecutionPhase executionPhase =
      (ir.getSessionOptions().executionPhaseSettings.phases > 1 &&
       op->hasExecutionPhase())
          ? op->getExecutionPhase()
          : -1;
  PipelineStage pipelineStage =
      (ir.getSessionOptions().enablePipelining && op->hasPipelineStage())
          ? op->getPipelineStage()
          : -1;
  return BatchSerialTensorContext(
      settings.batchSerializationSettings.concatOnVirtualGraphChange
          ? vgid
          : unusedVGraphId,
      settings.batchSerializationSettings.concatOnExecutionPhaseChange
          ? executionPhase
          : unusedExecutionPhase,
      settings.batchSerializationSettings.concatOnPipelineStageChange
          ? pipelineStage
          : unusedPipelineStage);
}

std::priority_queue<BatchSerialScheduler::TraceFront>
BatchSerialScheduler::findParallelTraceFronts(std::vector<Op *> schedule,
                                              int64_t batchSerFactor) const {
  std::priority_queue<BatchSerialScheduler::TraceFront> queue;

  for (size_t schedule_pos = 0; schedule_pos < schedule.size();
       ++schedule_pos) {
    Op *op = schedule.at(schedule_pos);

    logging::trace(
        "[BatchSerialScheduler] Find trace fronts from {} (pos: {}/{})",
        op->debugName(),
        schedule_pos,
        schedule.size() - 1);

    if (!op->hasBatchSerializedPhase()) {

      // Breadth first search, find patterns like:
      // 1.)
      //
      // |           |           |
      // Op (BSP 0)  Op (BSP 1)  Op (BSP 2)
      //    \        |         /
      //     '------ Op ------'
      //             |
      //
      // 2.)
      //
      // |           |           |
      // Op (BSP 0)  |           |
      // '---------- Op (BSP 1)  |
      //             '---------- Op (BSP 2)
      //                         '------------ Op

      std::map<std::pair<SubgraphEquivId, BatchSerialTensorContext>,
               std::vector<Tensor *>>
          equivTraceTensors;
      std::vector<Tensor *> traceTensors(batchSerFactor);
      std::set<Op *> visited;
      std::deque<std::pair<Op *, int>> opQueue;
      opQueue.push_back({op, 2 * batchSerFactor});
      while (!opQueue.empty()) {
        auto opAndDepth = opQueue.front();
        Op *op0         = opAndDepth.first;
        opQueue.pop_front();
        visited.insert(op0);

        if (opAndDepth.second == 0) {
          continue;
        }

        for (auto &idxAndTensor : op0->input->tensorMap()) {
          if (idxAndTensor.second->hasProducer()) {
            Op *op1 = idxAndTensor.second->getProducer();
            if (op1->hasBatchSerializedPhase()) {
              BatchSerializedPhase bsp1 = op1->getBatchSerializedPhase();
              SubgraphEquivId id        = opSubgraphEquivId.at(op1);
              BatchSerialTensorContext context =
                  getBatchSerialTensorContext(op1);
              if (equivTraceTensors.find({id, context}) ==
                  equivTraceTensors.end()) {
                equivTraceTensors.insert(
                    {{id, context}, std::vector<Tensor *>(batchSerFactor)});
              }
              if (!equivTraceTensors.at({id, context}).at(bsp1)) {
                equivTraceTensors.at({id, context})[bsp1] = idxAndTensor.second;
              }
              if (std::all_of(
                      equivTraceTensors.at({id, context}).begin(),
                      equivTraceTensors.at({id, context}).end(),
                      [](const Tensor *t) { return static_cast<bool>(t); })) {
                traceTensors = equivTraceTensors.at({id, context});
                opQueue      = {};
                break;
              }
            }
            if (op0->hasBatchSerializedPhase() !=
                    op1->hasBatchSerializedPhase() ||
                (op0->hasBatchSerializedPhase() &&
                 op1->hasBatchSerializedPhase() &&
                 op0->getBatchSerializedPhase() !=
                     op1->getBatchSerializedPhase())) {
              opQueue.push_front({op1, opAndDepth.second - 1});
            } else {
              opQueue.push_back({op1, opAndDepth.second - 1});
            }
          }
        }
      }

      logging::trace("[BatchSerialScheduler] Visited {} ops.", visited.size());

      if (std::all_of(traceTensors.begin(),
                      traceTensors.end(),
                      [](const Tensor *t) { return static_cast<bool>(t); })) {
        queue.push(BatchSerialScheduler::TraceFront(
            traceTensors, BatchSerialScheduler::TraceDirection::Backward));
      }
    }
  }
  return queue;
}

int64_t BatchSerialScheduler::getLocalIsoScore(
    std::pair<Op *, Op *> ops,
    std::set<std::pair<Op *, Op *>> &visitedOps,
    int maxDepth,
    bool cached) {
  if (cached) {
    auto it = cachedIsoScores.find({ops.first, ops.second, maxDepth});
    if (it != cachedIsoScores.end()) {
      return it->second;
    }
  }

  int64_t score = 0;
  if (visitedOps.find(ops) != visitedOps.end() || maxDepth == 0 ||
      ops.first->settings.recomputeType != ops.second->settings.recomputeType ||
      ops.first->scheduledPreLoss != ops.second->scheduledPreLoss ||
      (ops.first->getOptionalExecutionPhase() !=
       ops.second->getOptionalExecutionPhase()) ||
      (ops.first->getOptionalPipelineStage() !=
       ops.second->getOptionalPipelineStage())) {
    return score;
  }

  auto sit0 = opToSection.find(ops.first);
  auto sit1 = opToSection.find(ops.second);

  if (sit0 != opToSection.end() && sit1 != opToSection.end()) {
    if (sit0->second != sit1->second) {
      // Ops already annotated in different sections: Not isomorphic
      return score;
    } else {
      auto pos0 =
          opToPosition.at({sit0->second, ops.first->getBatchSerializedPhase()})
              .at(ops.first);
      auto pos1 =
          opToPosition.at({sit1->second, ops.second->getBatchSerializedPhase()})
              .at(ops.second);
      if (pos0 != pos1) {
        // Ops already annotated in same section but different positions: Not
        // isomorphic
      }
    }
  }

  visitedOps.insert(ops);

  // Check if the ops have the same subgraph equivalent ID
  if (opSubgraphEquivId[ops.first] == opSubgraphEquivId[ops.second]) {
    // Possibly isomorphic
    ++score;

    for (auto &input : ops.first->input->tensorMap()) {
      Tensor *tfirst  = ops.first->input->tensor(input.first);
      Tensor *tsecond = ops.second->input->tensor(input.first);
      if (tfirst->hasProducer() && tsecond->hasProducer()) {
        Op *pfirst  = tfirst->getProducer();
        Op *psecond = tsecond->getProducer();
        if (opSubgraphEquivId[pfirst] == opSubgraphEquivId[psecond]) {
          score += getLocalIsoScore(
              {pfirst, psecond}, visitedOps, maxDepth - 1, false);
        }
      }
    }

    for (auto &output : ops.first->output->tensorMap()) {
      if (!ops.first->output->hasIndex(output.first) ||
          !ops.second->output->hasIndex(output.first)) {
        continue;
      }
      Tensor *tfirst  = ops.first->output->tensor(output.first);
      Tensor *tsecond = ops.second->output->tensor(output.first);

      auto csfirst  = tfirst->consumers.getOps();
      auto cssecond = tsecond->consumers.getOps();

      std::set<Op *, POpCmp> cssecondMatched;

      for (Op *cfirst : csfirst) {
        Op *maxOp         = nullptr;
        int64_t max_score = 0;
        std::set<std::pair<Op *, Op *>> maxVisitedOps;
        for (Op *csecond : cssecond) {
          // Find csecond that matches cfirst best
          if (opSubgraphEquivId[cfirst] == opSubgraphEquivId[csecond] &&
              cssecondMatched.find(csecond) == cssecondMatched.end()) {
            std::set<std::pair<Op *, Op *>> &localVisitedOps = visitedOps;
            int64_t local_score                              = getLocalIsoScore(
                {cfirst, csecond}, localVisitedOps, maxDepth - 1, false);
            if (local_score > max_score) {
              max_score     = local_score;
              maxVisitedOps = localVisitedOps;
              maxOp         = csecond;
            }
          }
        }
        if (maxOp) {
          cssecondMatched.insert(maxOp);
          visitedOps.insert(maxVisitedOps.begin(), maxVisitedOps.end());
          score += max_score;
        }
      }
    }
  }

  if (cached) {
    cachedIsoScores[{ops.first, ops.second, maxDepth}] = score;
  }
  return score;
}

void BatchSerialScheduler::addParallelizationConstraints(Graph &graph) const {
  // For maximum overlap we want the IoTileCopy (IO to compute) and the
  // preceding RemoteLoadOps for phase N+1 to happen during the compute
  // stage of batch N. To achieve this, let's schedule the IoTileCopy (IO
  // to compute) for N+1 to happen IoTileCopy (compute to IO) for N.

  auto &ir      = graph.getIr();
  auto settings = ir.getSessionOptions().batchSerializationSettings;

  for (auto &positions : positionToOp) {
    auto section = positions.first.first;
    auto phase   = positions.first.second;

    auto addConstraint = [&](Op *batchNPlus1Op, Op *batchNOp) {
      std::stringstream ss;
      ss << "[BatchSerialScheduler] Added parallelization constraint "
         << batchNPlus1Op->str() << " -> " << batchNOp->str()
         << " (section {}, BSP {}-{}).";
      logging::transform::debug(ss.str(), section, phase - 1, phase);
      graph.topoCons->insert(batchNPlus1Op, batchNOp, true);
    };

    if (phase > 0) {

      // If we're overlapping, lift our batch N+1's loading IoTileCopy before
      // batch N's saving tile copy. This will force RemoteLoadOps for batch
      // N+1 to happen prior to the saving of the results of N's compute phase.

      Op *batchNPlus1Op = getLastIoTileCopyToCompute(section, phase);
      Op *batchNOp      = getFirstIoTileCopyToIo(section, phase - 1);

      if (batchNPlus1Op && batchNOp) {
        addConstraint(batchNPlus1Op, batchNOp);
      }

      // If we're in OverlapOnCompute mode, try and lift batch N+1's
      // RemoteLoadOps to the front batch N's compute phase.

      if (settings.batchSchedule ==
          BatchSerializationBatchSchedule::OverlapOnCompute) {
        batchNPlus1Op = getLastRemoteLoad(section, phase);
        batchNOp      = getFirstComputeOp(section, phase - 1);

        if (batchNPlus1Op && batchNOp) {
          addConstraint(batchNPlus1Op, batchNOp);
        }
      }
    }
  }
}

Op *BatchSerialScheduler::getLastRemoteLoad(
    const Section section,
    const BatchSerializedPhase phase) const {

  auto findIt = positionToOp.find(std::make_pair(section, phase));

  if (findIt == positionToOp.end()) {
    return nullptr;
  }

  for (auto it = findIt->second.rbegin(); it != findIt->second.rend(); ++it) {
    auto op = it->second;
    if (op->isConvertibleTo<RemoteLoadOp>()) {
      return op;
    }
  }

  return nullptr;
}

Op *BatchSerialScheduler::getLastIoTileCopyToCompute(
    const Section section,
    const BatchSerializedPhase phase) const {

  auto findIt = positionToOp.find(std::make_pair(section, phase));

  if (findIt == positionToOp.end()) {
    return nullptr;
  }

  for (auto it = findIt->second.rbegin(); it != findIt->second.rend(); ++it) {
    auto op = it->second;
    if ((op->isConvertibleTo<IoTileCopyOp>()) &&
        (op->settings.tileSet == TileSet::Compute)) {
      return op;
    }
  }

  return nullptr;
}

Op *BatchSerialScheduler::getFirstIoTileCopyToIo(
    const Section section,
    const BatchSerializedPhase phase) const {

  const auto findIt = positionToOp.find(std::make_pair(section, phase));

  if (findIt == positionToOp.end()) {
    return nullptr;
  }

  for (auto it = findIt->second.begin(); it != findIt->second.end(); ++it) {
    auto op = it->second;
    if ((op->isConvertibleTo<IoTileCopyOp>()) &&
        (op->settings.tileSet == TileSet::IO)) {
      return op;
    }
  }

  return nullptr;
}

Op *BatchSerialScheduler::getFirstComputeOp(
    const Section section,
    const BatchSerializedPhase phase) const {

  const auto findIt = positionToOp.find(std::make_pair(section, phase));

  if (findIt == positionToOp.end()) {
    return nullptr;
  }

  for (auto it = findIt->second.begin(); it != findIt->second.end(); ++it) {
    auto op = it->second;
    if ((!op->isConvertibleTo<RemoteLoadOp>()) &&
        (!op->isConvertibleTo<IoTileCopyOp>()) &&
        (!op->isConvertibleTo<RemoteStoreOp>())) {
      return op;
    }
  }

  return nullptr;
}

bool BatchSerialScheduler::areSwappable(Graph &graph,
                                        Op *earlierOp,
                                        Op *laterOp) const {

  if (graph.topoCons->contains(earlierOp, laterOp)) {
    // Don't go against topological constraints.
    return false;
  }

  if (earlierOp->isParentOf(laterOp)) {
    // Don't reverse direct parent/child relationships.
    return false;
  }

  return true;
}

void BatchSerialScheduler::pushEarlier(
    PositionsToOpVector &vec,
    std::function<bool(Op *)> isPushOp,
    std::function<bool(Op *)> considerSwappingWith,
    std::function<bool(Op *, Op *)> areSwappable) const {
  // Move select to the front where this is possible.
  for (auto it1 = vec.begin(); it1 != vec.end(); ++it1) {

    if (isPushOp(it1->second)) {
      // Push selected op the furthest forward that we can.
      auto opIt = it1;

      while (true) {
        if (opIt == vec.begin()) {
          std::stringstream ss;
          ss << "[BatchSerialScheduler] Considered moving "
             << opIt->second->str() << " earlier in the schedule "
             << "but no ops are available at earlier positions.";
          logging::transform::trace(ss.str());
          break;
        }

        // Get the op previous to ours.
        auto prevOpIt = opIt;
        prevOpIt--;

        // Check if we should consider swapping.
        if (!considerSwappingWith(prevOpIt->second)) {
          std::stringstream ss;
          ss << "[BatchSerialScheduler] Not considering swapping "
             << opIt->second->str() << " with " << prevOpIt->second->str()
             << " (which is earlier in the schedule).";
          logging::transform::trace(ss.str());
          break;
        }

        // Check if we can swap.
        if (!areSwappable(prevOpIt->second, opIt->second)) {
          std::stringstream ss;
          ss << "[BatchSerialScheduler] Not able to swap "
             << opIt->second->str() << " with " << prevOpIt->second->str()
             << " (which is earlier in the schedule) "
             << "as the ops are not swappable.";
          logging::transform::trace(ss.str());
          break;
        }

        std::stringstream ss;
        ss << "[BatchSerialScheduler] Moving " << opIt->second->str()
           << " earlier in the schedule by "
           << "swapping it with " << prevOpIt->second->str()
           << " (in an attempt to make the "
           << "schedule more amenable to parallelization)";
        logging::transform::debug(ss.str());

        // OK, we can swap it!
        auto prevOp      = prevOpIt->second;
        prevOpIt->second = opIt->second;
        opIt->second     = prevOp;

        // We're now at the location of prevOpIt.
        opIt = prevOpIt;
      }
    }
  }
}

void BatchSerialScheduler::pushLater(
    PositionsToOpVector &vec,
    std::function<bool(Op *)> isPushOp,
    std::function<bool(Op *)> considerSwappingWith,
    std::function<bool(Op *, Op *)> areSwappable) const {
  // Move select to the front where this is possible.
  for (auto it1 = vec.rbegin(); it1 != vec.rend(); ++it1) {

    if (isPushOp(it1->second)) {
      // Push selected op the furthest forward that we can.
      auto opIt = it1;

      while (true) {
        if (opIt == vec.rbegin()) {
          std::stringstream ss;
          ss << "[BatchSerialScheduler] Considered moving "
             << opIt->second->str() << " later in the schedule "
             << "but no ops are available at later positions.";
          logging::transform::trace(ss.str());
          break;
        }

        // Get the op previous to ours.
        auto nextOpIt = opIt;
        nextOpIt--;

        // Check if we should consider swapping.
        if (!considerSwappingWith(nextOpIt->second)) {
          std::stringstream ss;
          ss << "[BatchSerialScheduler] Not considering swapping "
             << opIt->second->str() << " with " << nextOpIt->second->str()
             << " (which is later in the schedule).";
          logging::transform::trace(ss.str());
          break;
        }

        // Check if we can swap.
        if (!areSwappable(opIt->second, nextOpIt->second)) {
          std::stringstream ss;
          ss << "[BatchSerialScheduler] Not able to swap "
             << opIt->second->str() << " with " << nextOpIt->second->str()
             << " (which is later in the schedule) "
             << "as the ops are not swappable.";
          logging::transform::trace(ss.str());
          break;
        }

        std::stringstream ss;
        ss << "[BatchSerialScheduler] Moving " << opIt->second->str()
           << " later in the schedule by "
           << "swapping it with " << nextOpIt->second->str()
           << " (in an attempt to make the "
           << "schedule more amenable to parallelization)";
        logging::transform::debug(ss.str());

        // OK, we can swap it!
        auto nextOp      = nextOpIt->second;
        nextOpIt->second = opIt->second;
        opIt->second     = nextOp;

        // We're now at the location of prevOpIt.
        opIt = nextOpIt;
      }
    }
  }
}

void BatchSerialScheduler::tryToMakeAmenableToParallelization() {

  // Predicate function to test if an op is a RemoteLoadOp.
  auto isRemoteLoad = [](Op *op) -> bool {
    return (op->isConvertibleTo<RemoteLoadOp>());
  };
  // Predicate function to test if an op is a RemoteLoadStore.
  auto isRemoteStore = [](Op *op) -> bool {
    return (op->isConvertibleTo<RemoteStoreOp>());
  };
  // Predicate function to test if an op is an IoTileCopyOp (from IO to
  // compute).
  auto isIoTileCopyToCompute = [](Op *op) -> bool {
    return (op->isConvertibleTo<IoTileCopyOp>()) &&
           (op->settings.tileSet == TileSet::Compute);
  };
  // Predicate function to test if an op is an IoTileCopyOp (from compute to
  // IO).
  auto isIoTileCopyToIo = [](Op *op) -> bool {
    return (op->isConvertibleTo<IoTileCopyOp>()) &&
           (op->settings.tileSet == TileSet::IO);
  };
  auto areSwappableL = [&](Op *earlierOp, Op *laterOp) {
    return areSwappable(graph, earlierOp, laterOp);
  };

  for (auto &entry : positionToOp) {
    auto &posOpMap = entry.second;

    // Turn the map into a vector (it's easier to work with).
    PositionsToOpVector posOpVec;
    posOpVec.reserve(posOpMap.size());
    for (auto &posOpMapEntry : posOpMap) {
      posOpVec.push_back(
          std::make_pair(posOpMapEntry.first, posOpMapEntry.second));
    }

    // Move any RemoteLoadOp to the start of the schedule if possible, but don't
    // move past other RemoteLoadOps.
    pushEarlier(
        posOpVec,
        isRemoteLoad,
        [&](Op *op) { return !isRemoteLoad(op); },
        areSwappableL);

    // Move any IoTileCopyOp (from IO to compute) to the start, too. Don't move
    // past other IoTileCopyOp (from IO) ops or any RemoteLoadOps.
    pushEarlier(
        posOpVec,
        isIoTileCopyToCompute,
        [&](Op *op) { return !isRemoteLoad(op) && !isIoTileCopyToCompute(op); },
        areSwappableL);

    // Move any RemoteStoreOp back (later) to the back of the schedule if
    // possible, but don't move past other RemoteStoreOp.
    pushLater(
        posOpVec,
        isRemoteStore,
        [&](Op *op) { return !isRemoteStore(op); },
        areSwappableL);

    // Move any IoTileCopyOp (from IO to compute) to the back of the schedule if
    // possible. Don't move past other IoTileCopyOp (from IO) op or
    // RemoteStoreOps.
    pushLater(
        posOpVec,
        isIoTileCopyToIo,
        [&](Op *op) { return !isRemoteStore(op) && !isIoTileCopyToIo(op); },
        areSwappableL);

    // Turn the vector back into a map.
    posOpMap.clear();
    for (auto &posOpVecEntry : posOpVec) {
      posOpMap[posOpVecEntry.first] = posOpVecEntry.second;
    }
  }
}

} // namespace popart
