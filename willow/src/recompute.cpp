// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/intervals.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/recompute.hpp>
#include <popart/scheduler_requireoptimal.hpp>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/util.hpp"
#include "popart/vertex.hpp"

namespace popart {

namespace recompute {

namespace {

void allCheckpointAfterLastCheckpoint(std::vector<Op *> lossOps) {
  // Change to checkpoint mode after the last recompute checkpoints op.
  // This reduces the compute time as no need to recompute twice the last
  // recompute segment.
  logging::transform::info("Turn recompute ops to checkpoint between the last "
                           "checkpoints and the loss.");
  OpSearchHelper opSearch;
  for (auto op : lossOps) {
    opSearch.pushInputProducers(op);
  }
  while (!opSearch.empty()) {
    // Get an op from the search list
    Op *x = opSearch.pop();
    // If it is a recompute op, make it to a checkpoint and
    // keep searching the ops before it.
    if (x->settings.recomputeType == RecomputeType::Recompute) {
      opSearch.pushInputProducers(x);
      x->settings.recomputeType = RecomputeType::Checkpoint;
    }
  }
}

void annotateNormOnly(Graph &graph) {
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (op->toLoss == PathToLoss::Yes && op->isNorm() && op->canRecompute()) {
      // don't checkpoint Norms as their outputs are large and
      // relatively cheap to recompute
      op->settings.recomputeType = RecomputeType::Recompute;
    } else {
      op->settings.recomputeType = RecomputeType::Checkpoint;
    }
  }
}

void annotateStandard(const Graph &graph) {
  std::vector<Op *> fwdOps;
  for (auto op : graph.getOpSchedule({}, RequireOptimalSchedule::Yes)) {
    if (op->toLoss == PathToLoss::Yes) {
      fwdOps.push_back(op);
    }
  }

  if (fwdOps.size() == 0) {
    return;
  }

  logging::ir::debug("Ops which are fwdToBwd determined (non-empty set)");

  // liveSets[i] : set of ops whose outputs have not all
  // been consumed by their (non-grad) consumers just after
  // linearised[i] has run. By this defn,
  // linearised[i] \in live[i]
  auto liveSets = graph.getLiveSets(fwdOps);

  // The memory (bytes) which will be needed to
  // store all the output tensors in a liveness set.
  std::vector<int64_t> memoryOfLives;
  for (auto &liveSet : liveSets) {
    int64_t mem = 0;
    for (auto op : liveSet) {
      mem += op->memOfOutputs();
    }
    memoryOfLives.push_back(mem);
  }

  logging::ir::debug("Memory of outputs of live sets determined");

  int nFwdOps = static_cast<int>(fwdOps.size());
  if (nFwdOps != liveSets.size() || memoryOfLives.size() != nFwdOps) {
    throw internal_error("sizes of vectors do not match");
  }

  // resnet-50 has more activation memory for early layers, see
  // https://github.com/albanie/convnet-burden/blob/master/reports/resnet18.md

  std::vector<int64_t> memOfFwds;
  memOfFwds.reserve(fwdOps.size());
  for (auto &op : fwdOps) {
    memOfFwds.push_back(op->memOfOutputs());
  }

  // std::vector<std::array<int, 2>> intervals =
  // getDecreasingIntervals(memOfFwds);
  //

  std::vector<std::array<int, 2>> intervals =
      getDecreasingIntervals(static_cast<int>(fwdOps.size()));

  logging::ir::debug("Decreasing intervals obtained");

  //   defn, checkpoints: Ops whose
  //   outputs we guarantee will be available
  //   at any time
  OpSet checkpoints;

  // we choose the lowest memory set from each interval,
  // and add its members to checkpoints.
  for (auto interval : intervals) {
    int begin            = interval[0];
    int end              = interval[1];
    int64_t lowestMemory = std::numeric_limits<int64_t>::max();
    OpSet bestSet{};
    for (int i = begin; i < end; ++i) {
      if (memoryOfLives[i] < lowestMemory) {
        lowestMemory = memoryOfLives[i];
        bestSet      = liveSets[i];
      }
    }
    for (Op *op : bestSet) {
      if (checkpoints.count(op) == 0) {
        checkpoints.insert(op);
      }
    }
  }

  for (auto op : fwdOps) {
    if (checkpoints.count(op) == 0 && op->canRecompute()) {
      op->settings.recomputeType = RecomputeType::Recompute;
    }
  }
}

void logAnnotations(Graph &graph) {
  std::stringstream ss;
  for (auto op : graph.getOpSchedule({}, RequireOptimalSchedule::No)) {
    ss << op->settings.recomputeType << "    " << op->toLoss << "    "
       << op->fromLoss << "    " << op->debugName() << "\n";
  }
  logging::trace("[autoAnnotate] Resulting annotations: \n{}", ss.str());
}

} // namespace

void annotateRecomputeAll(Graph &graph) {
  std::vector<Op *> lossOps;
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (op->toLoss == PathToLoss::Yes &&
        op->settings.recomputeType == RecomputeType::Undefined &&
        op->canRecompute()) {
      op->settings.recomputeType = RecomputeType::Recompute;
    }
    if (op->toLoss == PathToLoss::Yes && op->fromLoss == PathFromLoss::Yes) {
      lossOps.push_back(op);
    }
  }
  // Do not use allCheckpointAfterLastCheckpoint with explicit recomputation
  // TODO: T61001
  if (!graph.getIr().getSessionOptions().explicitRecomputation) {
    allCheckpointAfterLastCheckpoint(lossOps);
  }
}

void annotateRecomputePipeline(Graph &graph) {
  std::vector<Op *> lossOps;
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (op->toLoss == PathToLoss::Yes &&
        op->settings.recomputeType == RecomputeType::Undefined &&
        op->canRecompute()) {
      op->settings.recomputeType = RecomputeType::Recompute;
    }
    if (op->toLoss == PathToLoss::Yes && op->fromLoss == PathFromLoss::Yes) {
      lossOps.push_back(op);
    }
  }
  // Do not use allCheckpointAfterLastCheckpoint with explicit recomputation
  // TODO: T61001
  if (!graph.getIr().getSessionOptions().explicitRecomputation) {
    allCheckpointAfterLastCheckpoint(lossOps);
  }
}

void autoAnnotate(Graph &graph, RecomputationType rctype) {

  switch (rctype) {

  case RecomputationType::None: {
    logging::transform::info("Using 'None' auto-recompute method");
    break;
  }
  case RecomputationType::Standard: {
    logging::transform::info("Using 'Standard' auto-recompute method");
    annotateStandard(graph);
    break;
  }
  case RecomputationType::NormOnly: {
    logging::transform::info("Using 'NormOnly' auto-recompute method");
    annotateNormOnly(graph);
    break;
  }
  case RecomputationType::RecomputeAll: {
    logging::transform::info("Using 'RecomputeAll' auto-recompute method");
    annotateRecomputeAll(graph);
    break;
  }
  case RecomputationType::Pipeline:
    if (graph.getIr().getSessionOptions().explicitPipeliningEnabled()) {
      annotateRecomputePipeline(graph);
    } else {
      throw error("Invalid RecomputationType::Pipeline in autoAnnotate when "
                  "using implicit pipelining.");
    }
    break;
  case RecomputationType::N:
  default: {
    throw error("Invalid RecomputationType in autoAnnotate");
  }
  }

  logAnnotations(graph);
}

} // namespace recompute

} // namespace popart
