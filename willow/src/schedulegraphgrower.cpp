// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <schedulegraphgrower.hpp>
#include <tuple>
#include <utility>
#include <poprithms/logging/timepartitionlogger.hpp>
#include <poprithms/schedule/shift/allocweight.hpp>
#include <poprithms/schedule/shift/fromcache.hpp>
#include <poprithms/schedule/shift/kahndecider.hpp>
#include <poprithms/schedule/shift/rotationtermination.hpp>
#include <poprithms/schedule/shift/schedulecache.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/schedule/shift/summarywriter.hpp>
#include <poprithms/schedule/shift/transitiveclosureoptimizations.hpp>
#include <poprithms/schedule/vanilla/types.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/tensorindex.hpp>
#include <popart/topocons.hpp>
#include <poparttracepoint.hpp>

#include "popart/basicoptionals.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensor.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensors.hpp"
#include "popart/vertex.hpp"

namespace {

using namespace poprithms::schedule;

using shift::AllocAddress;
using shift::AllocWeight;
using shift::OpAddress;
using shift::ScheduleIndex;
using KahnTieBreaker = shift::KahnTieBreaker;
} // namespace

namespace popart {

namespace {

std::string ioNames(Op *op) {
  std::ostringstream oss;
  for (auto elem : op->input->tensorIdMap()) {
    oss << elem.second << '_';
  }
  for (auto elem : op->output->tensorIdMap()) {
    oss << elem.second << '_';
  }
  return oss.str();
}
} // namespace

// Get the schedule from the shift::Graph as a vector of Op pointers.
// The shift::Graph must have already been initialised through a call to
// `ShiftGraphGrower::initialize`.
std::vector<Op *> ShiftGraphGrower::getSchedule() const {

  if (scheduledShiftGraph.nOps() < g.nOps()) {
    std::ostringstream oss;
    oss << "Logical error in ShiftGraphGrower::getSchedule(). "
        << "The solution ScheduledGraph has " << scheduledShiftGraph.nOps()
        << ", whereas the shift::Graph has " << g.nOps()
        << ". This suggests that the Solution has not been generated for this "
           "shift::Graph. ";
    throw error(oss.str());
  }

  // 1. Get vector of all ops.
  // We know all op addresses are 0..nOps
  std::vector<OpAddress> opAddrs(nOps);
  std::iota(opAddrs.begin(), opAddrs.end(), 0);

  // 2.
  const auto schToOpAddr = scheduledShiftGraph.getSubSchedule(opAddrs);

  // 3. Convert schedule on OpAddress to schedule on popart::Op.
  std::vector<Op *> schToOp;
  schToOp.reserve(nOps);

  std::transform(schToOpAddr.cbegin(),
                 schToOpAddr.cend(),
                 std::back_inserter(schToOp),
                 [this](const auto &opAddr) { return addressToOp[opAddr]; });

  return schToOp;
}

void ShiftGraphGrower::initialize(const shift::Settings &settings,
                                  shift::ScheduleCache &cache) {

  POPART_TRACEPOINT();

  const auto hitType =
      shift::probeCache(g, settings.rotationTermination(), &cache);

  std::stringstream logNameStream;
  logNameStream << "Scheduling. TCOs="
                << (settings.tcos() ==
                    shift::TransitiveClosureOptimizations::allOn())
                << " Refine="
                << settings.rotationTermination().longerThan({10., 100})
                << " CacheHit=" << std::get<0>(hitType);
  const auto logName = logNameStream.str();
  const auto scopedStopwatch =
      pg.getIr().timePartitionLogger().scopedStopwatch(logName);

  logging::ir::debug("{}. Graph has {} Ops. ", logName, g.nOps());

  auto gCopy = g;

  scheduledShiftGraph = shift::fromCache(
      std::move(gCopy), settings, shift::FileWriter::Default(), &cache, &cache);
}

bool ShiftGraphGrower::isSchedulable() const {
  return vanilla::Query<uint64_t>::isSchedulable(
      g.getFwdEdges_u64(), g.getFwdLinks(), vanilla::VerifyEdges::No);
}

std::string ShiftGraphGrower::getSerializationString() const {
  return g.getSerializationString();
}

Op *ShiftGraphGrower::toOp(OpAddress a) const { return addressToOp.at(a); }

void ShiftGraphGrower::setBasic() {
  addressToOp.reserve(nOps);
  for (const auto &popartTensorId : allPopartTensorIds) {
    auto t = pg.getTensors().get(popartTensorId);
    // We ignore Variable Tensors contribution, as they are always live.
    // TODO(jn) confirm that ping-pong and host reduction agree with this.
    if (t->tensorType() != TensorType::Variable) {
      auto w            = static_cast<AllocWeight>(t->info.nbytes());
      allocAddresses[t] = g.insertAlloc(w);
    }
  }
  for (const auto &x : pg.getOps()) {
    auto op         = x.second.get();
    opAddresses[op] = g.insertOp({}, {}, op->str());
    addressToOp.push_back(op);
  }
  for (const auto &x : pg.getOps()) {
    auto op        = x.second.get();
    auto opAddress = opAddresses[op];
    for (const auto t : op->input->tensors()) {
      if (auto producer = t->getProducerUnsafe()) {
        g.insertConstraint(opAddresses[producer], opAddress);
      }
      // Don't consider allocs for variables and overwritten tensors
      if (t->tensorType() != TensorType::Variable && !op->overwritesTensor(t)) {
        g.insertOpAlloc(opAddress, allocAddresses[t]);
      }
    }
    for (const auto popartTensor : op->output->tensors()) {
      g.insertOpAlloc(opAddress, allocAddresses[popartTensor]);
    }
    for (const auto before : pg.topoCons->getBefores(op)) {
      g.insertConstraint(opAddresses[before], opAddress);
    }
  }
}

void ShiftGraphGrower::annotateExecutionPhase() {

  const auto sw = pg.getIr().timePartitionLogger().scopedStopwatch(
      "[Scheduler] annotateExecutionPhase");

  // Insert bin constraints to ensure ops are sorted by execution phase.
  std::vector<std::vector<OpAddress>> bins;
  for (const auto &x : pg.getOps()) {
    auto op = x.second.get();
    if (op->getOptionalExecutionPhase()) {
      auto opAddress = opAddresses[op];
      auto phase     = *op->getOptionalExecutionPhase();
      if (phase < -1) {
        throw internal_error(
            "phase < -1 unexpected. This function needs adjustment");
      }
      uint64_t binIndex = static_cast<uint64_t>(1LL + phase);
      if (binIndex >= bins.size()) {
        bins.resize(binIndex + 1);
      }
      bins[binIndex].push_back(opAddress);
    }
  }
  g.insertBinConstraints(bins, "executionPhaseStart_");
}

void ShiftGraphGrower::annotateExecutionContext() {

  const auto sw = pg.getIr().timePartitionLogger().scopedStopwatch(
      "[Scheduler] annotateExecutionContext");

  std::vector<OpAddress> weightsToOps;
  std::vector<OpAddress> normalOps;
  std::vector<OpAddress> accumulateOuter;
  std::vector<OpAddress> weightsFromOps;
  for (const auto &x : pg.getOps()) {
    auto op        = x.second.get();
    auto opAddress = opAddresses[op];
    switch (op->settings.executionContext) {
    case (ExecutionContext::WeightsFromHostFragment): {
      weightsFromOps.push_back(opAddress);
      break;
    }
    case (ExecutionContext::Normal): {
      normalOps.push_back(opAddress);
      break;
    }
    case (ExecutionContext::AccumulateOuterFragment): {
      accumulateOuter.push_back(opAddress);
      break;
    }
    case (ExecutionContext::WeightsToHostFragment): {
      weightsToOps.push_back(opAddress);
      break;
    }
    case (ExecutionContext::OptimizerFromHostFragment): {
      // do nothing.
      break;
    }
    case (ExecutionContext::Subgraph): {
      // do nothing.
      break;
    }
    default: {
      throw error("Unsupported ExecutionContext ({})",
                  op->settings.executionContext);
    }
    }
  }
  std::vector<std::vector<OpAddress>> bins;
  if (!weightsFromOps.empty()) {
    bins.push_back(weightsFromOps);
  }
  if (!normalOps.empty()) {
    bins.push_back(normalOps);
  }
  if (!accumulateOuter.empty()) {
    bins.push_back(accumulateOuter);
  }
  if (!weightsToOps.empty()) {
    bins.push_back(weightsToOps);
  }
  if (bins.size() > 1) {

    g.insertBinConstraints(bins, "executionContext_");
  }
}

void ShiftGraphGrower::annotatePipelineStages() {

  const auto sw = pg.getIr().timePartitionLogger().scopedStopwatch(
      "[Scheduler] annotatePipelineStages");

  // Adding pipelineStage bins is not required for correctness.
  // Constraining the Ops to be within their pipelineStage improves
  // scheduling runtime as swaps with no effect are invalid.
  std::vector<std::vector<OpAddress>> bins;
  for (const auto &x : pg.getOps()) {
    auto op = x.second.get();
    if (op->hasPipelineStage() &&
        op->settings.executionContext == ExecutionContext::Normal) {
      auto opAddress = opAddresses[op];
      auto stage     = *op->getOptionalPipelineStage();
      if (stage < -1) {
        throw internal_error(
            "stage < -1 unexpected. This function needs adjustment");
      }
      uint64_t binIndex = static_cast<uint64_t>(1LL + stage);
      if (binIndex >= bins.size()) {
        bins.resize(binIndex + 1);
      }
      bins[binIndex].push_back(opAddress);
    }
  }

  g.insertBinConstraints(bins, "PipelineStageStart_");
}

// The general setting of an op's scheduledPreLoss setting may look like:
//
//         scheduledPreLoss?
// Op0     Yes
// Op1     Yes
//     ...
// Loss    No
// Loss'   No
//     ...
// OpN-1   No
// OpN     No
//
// However, the loss final loss can be computed arbitrarily, and therefore
// gradient operations can be grown in the auto-diff transform that do not
// have a dependency of any operations with a path to the loss. For example,
// if:
//   loss = Mul(ReduceSum(Reshape(probs)), const)
// the ReshapeGrad, ReduceSumGrad and MulGrad operations that produce the
// gradient of 'loss' tensor do not depend on operations with a path to the
// 'loss' tensor. Therefore they can be scheduled early, leading to corrupted
// scheduledPreLoss settings, such as:
//
//         scheduledPreLoss?
// Op0     Yes
// Loss'   No
// Op1     No
//     ...
// Loss    No
//     ...
// OpN-1   No
// OpN     No.
//
// The implicit recomputation transform depends on this setting
// correctly indicating whether an op is in the forward or backward
// pass, so insert scheduler constraints to prevent this from happening.
void ShiftGraphGrower::annotateToLossFromLoss() {
  // All ops that have:
  //   op.toLoss == PathToLoss::Yes, and
  //   op.fromLoss == PathFromLoss::No
  // are scheduled before all ops that have:
  //   op.fromLoss == PathFromLoss::Yes
  // The final loss Op, where PathToLoss::Yes && PathFromLoss::Yes,
  // can be set to SchedulePreLoss::No so must be placed in the second bin.
  std::vector<OpAddress> toLoss;
  std::vector<OpAddress> fromLoss;

  for (const auto &x : pg.getOps()) {
    auto op        = x.second.get();
    auto opAddress = opAddresses[op];

    if (op->fromLoss == PathFromLoss::No && op->toLoss == PathToLoss::Yes) {
      toLoss.push_back(opAddress);
    } else if (op->fromLoss == PathFromLoss::Yes) {
      fromLoss.push_back(opAddress);
    }
  }
  g.insertBinConstraints({toLoss, fromLoss}, "PreOrPostLoss_");
}

void ShiftGraphGrower::annotateAccumulateOuterFragmentOps() {

  const auto sw = pg.getIr().timePartitionLogger().scopedStopwatch(
      "[Scheduler] annotateAccumulateOuterFragmentOps");
  // The scheduler can be slow when there are a lot of ops unconstrained in
  // the accumulate outer fragment. To battle this, we sometimes add
  // constraints here. Depending on where we are in the IR pipeline and
  // session options this may be done in different ways. If the user set
  // 'enableAccumulateOuterFragmentParallelization' we will use bin
  // constraints this transform.
  if (pg.getIr().getSessionOptions().enablePipelining) {
    auto aufSettings =
        pg.getIr().getSessionOptions().accumulateOuterFragmentSettings;
    auto schedule = aufSettings.schedule;
    if (schedule == AccumulateOuterFragmentSchedule::OverlapCycleOptimized ||
        schedule == AccumulateOuterFragmentSchedule::OverlapMemoryOptimized) {
      // Use cluster grouping from AccumulateOuterFragmentParallelizer
      // transform as bins, this should allow parallelization accross IPUs.
      std::vector<std::vector<OpAddress>> bins;
      auto opBins = pg.getIr().getAccumulateOuterFragmentBinConstraints(pg);
      for (auto opBin : opBins) {
        std::vector<OpAddress> bin;
        for (auto op : opBin) {
          bin.push_back(opAddresses[op]);
        }
        bins.push_back(bin);
      }
      g.insertBinConstraints(bins, "AccumulateOuterFragmentCluster_");
    } else if (schedule == AccumulateOuterFragmentSchedule::Serial) {
      // Revert back to default behaviour for pipelining models where we
      // serialize the ops based on virtual graph id.
      std::vector<std::vector<OpAddress>> outer_bins;
      for (const auto &x : pg.getOps()) {
        auto op         = x.second.get();
        bool should_bin = op->hasPipelineStage() && op->hasVirtualGraphId() &&
                          (op->settings.executionContext ==
                           ExecutionContext::AccumulateOuterFragment);

        if (should_bin) {
          auto opAddress = opAddresses[op];
          auto vgraph    = *op->getOptionalVGraphId();
          if (vgraph < -1) {
            throw internal_error("vgraph < -1 unexpected. This function "
                                 "needs adjustment");
          }
          uint64_t binIndex = static_cast<uint64_t>(1LL + vgraph);
          if (binIndex >= outer_bins.size()) {
            outer_bins.resize(binIndex + 1);
          }
          outer_bins[binIndex].push_back(opAddress);
        }
      }
      g.insertBinConstraints(outer_bins, "OuterPipelineStageStart_");
    }
  }
}

void ShiftGraphGrower::annotatePriorities() {

  const auto sw = pg.getIr().timePartitionLogger().scopedStopwatch(
      "[Scheduler] annotatePriorities");
  std::vector<std::array<OpAddress, 2>> ties;
  for (const auto &x : pg.getOps()) {
    auto op        = x.second.get();
    auto tiedAfter = opAddresses[op];
    for (auto op2 : pg.topoCons->getTiedBefores(op)) {
      ties.push_back({opAddresses[op2], tiedAfter});
    }
  }
  // more important than actual memory (use +1 otherwise)
  g.insertAttractions(ties, AllocWeight(1.0, -1));

  std::vector<OpAddress> opIotas(nOps);
  std::iota(opIotas.begin(), opIotas.end(), 0);

  // A priority which takes precedence over memory liveness:
  using OpPriority = double;
  std::vector<
      std::tuple<ExecutionPhase, OpPriority, BatchSerializedPhase, OpPriority>>
      super;

  // A priority which is secondary to memory liveness:
  using OpTypeStr = std::string;
  using IoNames   = std::string;
  using UniqueId  = int;
  std::vector<std::tuple<OpTypeStr, IoNames, UniqueId>> sub;

  for (const auto &x : pg.getOps()) {
    auto op             = x.second.get();
    auto op_batchserial = op->getOptionalBatchSerializedPhase();
    auto op_phase       = op->getOptionalExecutionPhase();
    auto op_priority    = op->settings.schedulePriority;

    // Executuion phase -1 to N are reserved
    // -2 : No execution phase set (unusedExecutionPhase)
    // -1 : Load weights of phase 0
    // 0 - N: Compute phase n, load weights of phase n+1
    auto op_phase_or =
        op_phase &&
                pg.getIr().getSessionOptions().executionPhaseSettings.phases > 1
            ? *op_phase
            : unusedExecutionPhase;

    // Batchserial -1 to N are reserved
    // -2 : No batchserial phase set (unusedBatchSerializedPhase)
    // -1 : Init accumulator and updatee tensors
    // 0 - N : Compute batch element n
    auto op_batchserial_or =
        op_batchserial && pg.getIr()
                                  .getSessionOptions()
                                  .batchSerializationSettings.factor > 1
            ? *op_batchserial
            : unusedBatchSerializedPhase;

    auto op_priority_pre_or  = op_batchserial ? 0.0 : op_priority;
    auto op_priority_post_or = op_batchserial ? op_priority : 0.0;

    // to strongly encourage Ops to be appear in
    // 1) ascending execution phases
    // 2) descending priority for ops without batch-serial phase
    // 3) ascending batch-serial phase
    // 4) descending priority within batch-serial phase
    // 5) ScheduledPreLoss::Yes before ScheduledPreLoss::No
    super.push_back({-op_phase_or,
                     op_priority_pre_or,
                     -op_batchserial_or,
                     op_priority_post_or});
    sub.push_back({op->opid.type, ioNames(op), op->id});
  }
  g.insertStartAttractors(opIotas, super, -2);
  g.insertStartAttractors(opIotas, sub, +2);
}

void ShiftGraphGrower::appendGCons(const OpsBeforeKey &gCons) {
  for (const auto &x : gCons) {
    auto after        = x.first;
    auto befores      = x.second;
    auto addressAfter = opAddresses[after];
    for (auto b : befores) {
      auto addressBefore = opAddresses[b];
      g.insertConstraint(addressBefore, addressAfter);
    }
  }
}

ShiftGraphGrower::ShiftGraphGrower(const Graph &_pg_)
    : pg(_pg_), nOps(pg.getOps().size()),
      allPopartTensorIds(pg.getTensors().getAllTensorIds()),
      scheduledShiftGraph({}) {}

} // namespace popart
