#include <poponnx/graph.hpp>
#include <poponnx/intervals.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/op/batchnorm.hpp>
#include <poponnx/op/call.hpp>
#include <poponnx/op/conv.hpp>
#include <poponnx/op/groupnorm.hpp>
#include <poponnx/pbwrap.hpp>
#include <poponnx/recompute.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/tensors.hpp>

namespace poponnx {

namespace recompute {

namespace {

void annotateNormOnly(Graph &graph) {
  for (auto op : graph.getOpSchedule({})) {
    if (op->toLoss == PathToLoss::Yes && op->isNorm()) {
      // don't checkpoint Norms as their outputs are large and
      // relatively cheap to recompute
      op->settings.recomputeType = RecomputeType::RECOMPUTE;
    } else {
      op->settings.recomputeType = RecomputeType::CHECKPOINT;
    }
  }
}

void annotateStandard(const Graph &graph) {
  std::vector<Op *> fwdOps;
  for (auto op : graph.getOpSchedule({})) {
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
    throw error("ILE : sizes of vectors do not match");
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
      op->settings.recomputeType = RecomputeType::RECOMPUTE;
    }
  }
}

} // namespace

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

  case RecomputationType::N:
  default: {
    throw error("Invalid RecomputationType in autoAnnotate");
  }
  }
}

} // namespace recompute

} // namespace poponnx
