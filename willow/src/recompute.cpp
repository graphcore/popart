// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <queue>
#include <popart/graph.hpp>
#include <popart/intervals.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/batchnorm.hpp>
#include <popart/op/call.hpp>
#include <popart/op/conv.hpp>
#include <popart/op/groupnorm.hpp>
#include <popart/pbwrap.hpp>
#include <popart/recompute.hpp>
#include <popart/scheduler_requireoptimal.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

namespace popart {

namespace recompute {

namespace {

void allCheckpointAfterLastCheckpoint(Op *lossOp) {
  // Change to checkpoint mode after the last recompute checkpoints op.
  // This reduces the compute time as no need to recompute twice the last
  // recomute segment.
  logging::transform::info("Turn recompute ops to checkpoint between the last "
                           "checkpoints and the loss.");
  OpSearchHelper opSearch;
  opSearch.pushInputProducers(lossOp);
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
    if (op->toLoss == PathToLoss::Yes && op->isNorm()) {
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
  std::vector<std::set<Op *>> liveSets = graph.getLiveSets(fwdOps);

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
  std::set<Op *> checkpoints;

  // we choose the lowest memory set from each interval,
  // and add its members to checkpoints.
  for (auto interval : intervals) {
    int begin            = interval[0];
    int end              = interval[1];
    int64_t lowestMemory = std::numeric_limits<int64_t>::max();
    std::set<Op *> bestSet{};
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
    if (checkpoints.count(op) == 0) {
      op->settings.recomputeType = RecomputeType::Recompute;
    }
  }
}

} // namespace

void annotateRecomputeAll(Graph &graph) {
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (op->toLoss == PathToLoss::Yes &&
        op->settings.recomputeType == RecomputeType::Undefined) {
      op->settings.recomputeType = RecomputeType::Recompute;
    }
  }
  Op *loss_op = graph.getOp(graph.getIr().getFinalLossOpId());
  allCheckpointAfterLastCheckpoint(loss_op);
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

  case RecomputationType::N:
  case RecomputationType::Pipeline:
  default: {
    throw error("Invalid RecomputationType in autoAnnotate");
  }
  }
}

} // namespace recompute

} // namespace popart
