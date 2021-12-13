// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <set>
#include <thread>
#include <tuple>
#include <utility>
#include <popart/popx/creatorx.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/range/algorithm/find.hpp>
#include <boost/range/algorithm_ext.hpp>

#include <filereader.hpp>
#include <gcl/TileAllocation.hpp>
#include <poplar/CSRFunctions.hpp>
#include <poplar/CycleCount.hpp>
#include <poplar/RandomSeed.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>
#include <poprithms/logging/timepartitionlogger.hpp>
#include <poputil/exceptions.hpp>
#include <popart/boollogic.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>
#include <popart/logging.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>
#include <popart/op/convbase.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/op/if.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/subgraph.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/popx/debugcontextx.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/executablexserialization.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/callx.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>
#include <popart/popx/op/convbasex.hpp>
#include <popart/popx/op/matmulx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/poplaroptionsx.hpp>
#include <popart/popx/popopx.hpp>
#include <popart/popx/pritask.hpp>
#include <popart/recompute.hpp>
#include <popart/stepio.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/randomsetup.hpp>
#include <popart/variablesettings.hpp>

#include <popart/op/varupdate.hpp>
#include <popart/popx/op/ipucopyx.hpp>
#include <popart/tensornames.hpp>

#include <popx/rng/rngstatelowering.hpp>

#include <poparttracepoint.hpp>

#include <stepiosplitter.hpp>
#include <popart/subgraphpartitioner.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

namespace {

void defaultLogPrinter(int progress, int total) {

  // To avoid overlogging such as
  //   [info] Compilation 39% complete
  //   [info] Compilation 39% complete
  //   [info] Compilation 39% complete
  //   [info] Compilation 39% complete
  //
  // We keep track of the last percentage logged, and only log if the percentage
  // is different.

  static float lastPercentage{-1.};
  if (total != 0) {
    float percentage = std::floor(100.0f * static_cast<float>(progress) /
                                  static_cast<float>(total));
    if (percentage - lastPercentage != 0.0f) {
      lastPercentage = percentage;
      logging::devicex::info("Compilation {}% complete", percentage);
    }
  }
}

// Walk the producers of an ops inputs, applying function f to every producer.
// The producers are walked in a top down fashion. If f returns false of an op,
// then further producers below it are not traversed.
template <typename Predicate> void walkProducers(Op *op, Predicate f) {
  std::vector<Op *> toCheck;
  std::set<Op *> seen;

  auto addProducers = [&toCheck, &seen](Op *x) {
    for (auto t : x->input->tensors()) {
      if (t->hasProducer()) {
        auto p = t->getProducer();
        if (seen.find(p) == seen.end()) {
          toCheck.push_back(p);
          seen.insert(p);
        }
      }
    }
  };

  addProducers(op);
  while (!toCheck.empty()) {
    auto x = toCheck.back();
    toCheck.pop_back();
    if (f(x)) {
      addProducers(x);
    }
  }
}

// This is a helper class used to find the recompute ops required before running
// an op. The constructor takes a vector of ops to use as a schedule. Calls to
// `getRequiredRecomputeOps` will return ops in the order in which they are
// found in this schedule. If an op is not in the schedule, it will not be
// returned by `getRequiredRecomputeOps`.
class FindRequiredRecomputes {
public:
  FindRequiredRecomputes(const std::vector<Op *> &schedule)
      : opSchedule(schedule) {}

  // This will return the ops that need to be recomputed before running the
  // parameter `op`. Ops will be returned in the order in which they are found
  // in schedule.
  //
  // An op will only ever be returned once from this function.
  // For example if Op A requires C,
  // and            Op B requires C and D.
  // Calling with A before B will return:
  //   finder.getRequiredRecomputeOps(OpA) -> [C]
  //   finder.getRequiredRecomputeOps(OpB) -> [D]
  // but calling with B before A will return:
  //   finder.getRequiredRecomputeOps(OpB) -> [C, D]
  //   finder.getRequiredRecomputeOps(OpA) -> []
  std::vector<Op *> getRequiredRecomputeOps(Op *op) {

    const auto addGraphTask = op->getIr().timePartitionLogger().scopedStopwatch(
        "FindRequiredRecomputes::getRequiredRecomputeOps");

    ExecutionPhase opPhase = -1;

    std::set<Op *> toRerun;
    if (op->getIr().getSessionOptions().enablePipelining) {
      toRerun = getPipelineRecomputeOps(op, opPhase);
    } else {
      toRerun = getRecomputeOps(op, opPhase);
    }

    if (!toRerun.empty()) {
      std::vector<Op *> rerunSchedule;
      for (auto x : opSchedule) {
        if (toRerun.find(x) != toRerun.end()) {
          rerunSchedule.push_back(x);
          alreadySeen.insert({x, opPhase});
        }
      }
      return rerunSchedule;
    } else {
      return {};
    }
  }

private:
  std::set<Op *> getRecomputeOps(Op *op, ExecutionPhase &opPhase) {
    std::set<Op *> toRerun;
    auto isSpecialCaseGradOp = [](Op *x) {
      return x->getIr().getSessionOptions().enableGradientAccumulation &&
             x->settings.executionContext ==
                 ExecutionContext::AccumulateOuterFragment;
    };

    // Ensure op is post loss and not a special case grad op.
    if (op->scheduledPreLoss == ScheduledPreLoss::No &&
        !isSpecialCaseGradOp(op)) {
      if (op->hasExecutionPhase() &&
          op->getIr().getSessionOptions().executionPhaseSettings.phases >= 2) {
        opPhase = op->getExecutionPhase();
      }

      walkProducers(op, [&toRerun, &op, opPhase, this](Op *x) {
        bool sameExecutionPhase =
            (op->getIr().getSessionOptions().executionPhaseSettings.phases >=
                 2 &&
             op->hasExecutionPhase() && x->hasExecutionPhase() &&
             op->getExecutionPhase() == x->getExecutionPhase());
        if (x->settings.recomputeType == RecomputeType::Recompute &&
            alreadySeen.find({x, opPhase}) == alreadySeen.end() &&
            !sameExecutionPhase) {
          toRerun.insert(x);
          return true;
        } else {
          return false;
        }
      });
    }
    return toRerun;
  }

  std::set<Op *> getPipelineRecomputeOps(Op *op, ExecutionPhase opPhase) {
    std::set<Op *> toRerun;
    walkProducers(op, [&toRerun, &op, opPhase, this](Op *x) {
      if (x->settings.recomputeType == RecomputeType::Recompute &&
          alreadySeen.find({x, opPhase}) == alreadySeen.end() &&
          (x->getPipelineStage() != op->getPipelineStage() ||
           op->scheduledPreLoss == ScheduledPreLoss::No)) {
        toRerun.insert(x);
        return true;
      } else {
        return false;
      }
    });
    return toRerun;
  }

  std::set<std::pair<Op *, ExecutionPhase>> alreadySeen;
  const std::vector<Op *> &opSchedule;
};

void validateGclOptions(const std::map<std::string, std::string> &gclOptions) {
  std::set<std::string> validOptions = {"useSynclessCollectives",
                                        "maxBytesPerTile"};
  for (auto &key_value : gclOptions) {
    const std::string &key = key_value.first;
    if (validOptions.find(key) == validOptions.end()) {
      throw error("'{}' is not a valid gcl option. Valid options are {}",
                  key,
                  validOptions);
    }
  }
}

} // namespace

int ProgressLogger::current(int start, int end, int progress, int total) {
  float ratio = static_cast<float>(progress) / static_cast<float>(total);
  return start + std::floor(static_cast<float>(end - start) * ratio);
}

ProgressLogger::ProgressLogger(const SessionOptions &options) {
  if (options.compilationProgressLogger) {
    callback_ = options.compilationProgressLogger;
  } else {
    callback_ = defaultLogPrinter;
  }

  progressTotal = options.compilationProgressTotal;
}

void ProgressLogger::compilationStart() { callback_(3, progressTotal); }

void ProgressLogger::preplanningStart() { callback_(4, progressTotal); }

void ProgressLogger::preplanningEnd() { callback_(6, progressTotal); }

void ProgressLogger::creatingSequence(int task, int numTasks) {
  callback_(current(7, 40, task, numTasks), progressTotal);
}

void ProgressLogger::operator()(int progress, int total) {
  callback_(current(41, progressTotal, progress, total), progressTotal);
}

void ProgressLogger::complete() { callback_(progressTotal, progressTotal); }

devicex_memory_allocation_err::devicex_memory_allocation_err(
    const devicex_memory_allocation_err &rhs)
    : popart::memory_allocation_err(rhs.what()),
      exception(std::move(rhs.exception)), reportOptions(rhs.reportOptions) {}

devicex_memory_allocation_err::devicex_memory_allocation_err(
    const poplar::graph_memory_allocation_error &e,
    const poplar::OptionFlags &_reportOptions)
    : popart::memory_allocation_err(e.what()), exception(std::move(e)),
      reportOptions(_reportOptions) {}

std::unique_ptr<memory_allocation_err>
devicex_memory_allocation_err::clone() const {
  return std::make_unique<devicex_memory_allocation_err>(*this);
}

std::string devicex_memory_allocation_err::getSummaryReport() const {

  if (exception.profilePath.size() != 0) {

    std::stringstream ss;
    poplar::printProfileSummary(ss, exception.profilePath, reportOptions);
    return ss.str();
  } else {
    throw error("Need to set the 'debug.allowOutOfMemory' engine option to "
                "true to get the graph report");
  }
}

std::string devicex_memory_allocation_err::getProfilePath() const {
  return exception.profilePath;
}

IrLowering::IrLowering(const Ir &ir,
                       std::shared_ptr<DeviceInfo> deviceInfo_,
                       bool prepareGraphHasBeenCalled)
    : _ir(ir), deviceInfo(deviceInfo_), progressLogger(ir.getSessionOptions()),
      prepareGraphHasBeenCalled_(prepareGraphHasBeenCalled),
      tileCounterGraphConstVar(0), tileCounterGraphScalarVar(-1), tensors_(ir),
      progs(PopPrograms(this)), rngStateLowering() {
  POPART_TRACEPOINT();

  // Set the opxTrace flag based on the environment variable
  auto POPART_OPX_TRACE = getPopartEnvVar("OPX_TRACE");
  opxTrace              = POPART_OPX_TRACE ? (*POPART_OPX_TRACE == "1") : false;

  if (ir.getExecutionMode() == Ir::ExecutionMode::Training) {
    lstmOptions.set("inferenceOnly", "false");
  } else {
    lstmOptions.set("inferenceOnly", "true");
  }

  for (auto it : ir.getSessionOptions().lstmOptions) {
    logging::devicex::info("Setting LSTM option {} = {}", it.first, it.second);
    lstmOptions.set(it.first, it.second);
  }

  for (auto it : ir.getSessionOptions().matmulOptions) {
    logging::devicex::info(
        "Setting MatMul option {} = {}", it.first, it.second);
    matmulOptions.set(it.first, it.second);
  }

  const auto &userGclOptions = ir.getSessionOptions().gclOptions;
  validateGclOptions(userGclOptions);

  if (userGclOptions.find("useSynclessCollectives") != userGclOptions.end()) {
    gclOptions.set("useSynclessCollectives",
                   userGclOptions.at("useSynclessCollectives"));
  }

  if (userGclOptions.find("maxBytesPerTile") != userGclOptions.end()) {
    auto &val = userGclOptions.at("maxBytesPerTile");
    gclOptions.set("maxBytesPerTile", val);
  }
}

IrLowering::~IrLowering() = default;

std::map<Op *, int, POpCmp> IrLowering::getMainGraphOpSeriesNums() const {
  std::map<Op *, int, POpCmp> nums;
  int num = 0;
  for (auto entry : contextOpRegistry) {
    if (entry.first.first == ExecutionContext::Normal) {
      for (auto op : entry.second) {
        auto found = nums.find(op);
        if (found == nums.end()) {
          nums.insert({op, num});
          ++num;
        }
      }
    }
  }
  return nums;
}

void IrLowering::setMetadataFromIr() {
  // Maximum input of any Op (at least 1 to avoid division by 0)
  maxOpInputs = 1;
  for (Op *op : ir().getAllOps()) {
    maxOpInputs = std::max(maxOpInputs, op->input->n());
  }
}

void IrLowering::verifyTaskOrder(const std::vector<TaskId> &taskOrder) const {
  logging::debug("Verifying task order");
  int errors = 0;
  std::set<Op *> seen;
  std::set<Op *> recomputeSeen;

  for (auto taskId : taskOrder) {
    std::vector<Op *> taskOps;
    for (ExecutionContext context :
         {ExecutionContext::WeightsFromHostFragment,
          ExecutionContext::WeightsToHostFragment,
          ExecutionContext::Normal,
          ExecutionContext::AccumulateOuterFragment,
          ExecutionContext::OptimizerFromHostFragment,
          ExecutionContext::Subgraph}) {
      auto id_ops = contextOpRegistry.find({context, taskId});
      if (id_ops != contextOpRegistry.end()) {
        taskOps = id_ops->second;
        break;
      }
    }

    for (auto op : taskOps) {
      // If this is the first time we are seeing this op, it is not a recompute
      // grow.
      if (seen.find(op) == seen.end()) {
        // Check all the ops dependencies have already been run
        for (auto before : op->getGraph().topoCons->getBefores(op)) {
          if (seen.find(before) == seen.end()) {
            logging::devicex::warn("Op {} required op {} to be run before it.",
                                   op->id,
                                   before->id);
            errors++;
          }
        }
        seen.insert(op);
      }
      // This is a recompute op, to be recomputed before op with TaskId 'taskId'
      // is run
      else {
        std::vector<Op *> recomputes = requiredRecomputes.at(taskId);
        for (auto before : op->getGraph().topoCons->getBefores(op)) {
          if (std::find(recomputes.begin(), recomputes.end(), before) !=
                  recomputes.end() &&
              recomputeSeen.find(before) == recomputeSeen.end()) {
            logging::devicex::warn(
                "Recompute of op {} required op {} to be run before it.",
                op->id,
                before->id);
            errors++;
          }
        }
        recomputeSeen.insert(op);
      }
    }
  }

  if (errors > 0) {
    throw error("Encountered {} errors when verifying task order", errors);
  }
}

std::string
IrLowering::getContextOpString(ExecutionContext context,
                               const std::vector<TaskId> &taskOrder) const {
  std::stringstream ss;
  ss << "Context: " << context << std::endl;
  auto seriesNums = getMainGraphOpSeriesNums();
  std::set<Op *> seen;
  for (auto taskId : taskOrder) {
    auto task_ops = contextOpRegistry.find({context, taskId});
    if (task_ops == contextOpRegistry.end())
      continue;
    for (auto op : task_ops->second) {
      auto found = seen.count(op);
      seen.insert(op);
      std::string type;
      if (op->scheduledPreLoss == ScheduledPreLoss::Yes) {
        type = "preL";
      } else if (found != 0) {
        type = "re.1";
      } else if (op->settings.recomputeType == RecomputeType::Recompute) {
        type = "re.0";
      } else {
        std::ostringstream ss2;
        ss2 << ((op->toLoss == PathToLoss::Yes) ? "tY" : "tN");
        ss2 << ((op->fromLoss == PathFromLoss::Yes) ? "fY" : "fN");
        type = ss2.str();
      }
      type +=
          op->settings.recomputeType == RecomputeType::Recomputed ? "-R" : "-F";
      if (context == ExecutionContext::Normal) {
        ss << type << "  " << seriesNums[op] << "  ";
      }
      if (logging::shouldLog(logging::Module::devicex, logging::Level::Trace)) {
        ss << op->debugName() << "  ExecutionPhase: "
           << (op->hasExecutionPhase() ? op->getExecutionPhase()
                                       : unusedExecutionPhase)
           << "  Pipeline: "
           << (op->hasPipelineStage() ? op->getPipelineStage()
                                      : unusedPipelineStage)
           << "  VGID: "
           << (op->hasVirtualGraphId() ? op->getVirtualGraphId()
                                       : unusedVGraphId)
           << "  priority: " << op->settings.schedulePriority << std::endl;
      } else {
        ss << op->str() << std::endl;
      }
    }
  }
  return ss.str();
}

std::map<Op *, int, POpCmp> IrLowering::getMainGraphOpCounts() const {
  std::map<Op *, int, POpCmp> counts;
  for (auto entry : contextOpRegistry) {
    for (auto op : entry.second) {
      auto found = counts.find(op);
      if (found == counts.end()) {
        counts.insert({op, 1});
      } else {
        ++found->second;
      }
    }
  }
  return counts;
}

void IrLowering::instrumentWithHardwareCycleCounter(snap::program::Sequence &sq,
                                                    int64_t tileId,
                                                    std::string id) {
  poplar::Tensor cycleCountTensor =
      poplar::cycleCount(graph().getPoplarGraph(),
                         sq.getPoplarSequence(),
                         static_cast<unsigned int>(tileId),
                         poplar::SyncType::INTERNAL,
                         cycleCountPrefix());

  // Create stream
  auto st = graph().getPoplarGraph().addDeviceToHostFIFO(
      cycleCountStreamId(id),
      cycleCountTensor.elementType(),
      cycleCountTensor.numElements());

  cycleCountIds.push_back(id);

  // Add program fragment to copy to host stream
  auto cyclesToHostStream =
      poplar::program::Copy(cycleCountTensor, st, true, {"copyCycleCounter"});
  progs.cycleCountTensorToHostFragment().getPoplarSequence().add(
      cyclesToHostStream);
}

snap::Tensor IrLowering::getConst(snap::Graph &graph,
                                  const poplar::Type &type,
                                  const std::vector<size_t> &shape,
                                  double val,
                                  const poplar::DebugContext &debugContext) {
  const auto tensor =
      graph.getPoplarGraph().addConstant(type, shape, val, debugContext);
  const auto tilesTotal = graph.getPoplarGraph().getTarget().getTilesPerIPU();
  const auto tile       = tileCounterGraphConstVar % tilesTotal;
  tileCounterGraphConstVar++;

  graph.getPoplarGraph().setTileMapping(tensor, tile);
  return snap::Tensor{tensor, graph};
}

snap::Tensor
IrLowering::getScalarVariable(snap::Graph &graph,
                              const poplar::Type &type,
                              const poplar::DebugContext &debugContext) {
  const auto tensor =
      graph.getPoplarGraph().addVariable(type, {}, debugContext);
  const auto tilesTotal = graph.getPoplarGraph().getTarget().getTilesPerIPU();
  const auto tile =
      (tilesTotal + (tileCounterGraphScalarVar % tilesTotal)) % tilesTotal;
  tileCounterGraphScalarVar--;

  graph.getPoplarGraph().setTileMapping(tensor, tile);
  return snap::Tensor{tensor, graph};
}

snap::Graph &IrLowering::getVirtualGraph(VGraphId virtualGraphIndex,
                                         TileSet tileSet) {
  if (virtualGraphIndex < 0 || virtualGraphIndex >= virtualGraphs.size()) {
    throw error("Invalid virtual graph index {} ({} available)",
                virtualGraphIndex,
                virtualGraphs.size());
  }
  VirtualGraph &vg = virtualGraphs.at(virtualGraphIndex);

  if (tileSet == TileSet::IO) {
    if (!vg.hasIoTilesGraph()) {
      throw error("No IO tile graph for index {}", virtualGraphIndex);
    }
    return vg.getIoTilesGraph();
  } else {
    if (!vg.hasComputeTilesGraph()) {
      throw error("No compute tile graph for index {}", virtualGraphIndex);
    }
    return vg.getComputeTilesGraph();
  }
}

namespace {

std::vector<poplar::program::Program>
toPoplarProgs(const std::vector<snap::program::Program> &snapProgs) {
  std::vector<poplar::program::Program> poplarProgs(snapProgs.size());
  for (int i = 0; i < snapProgs.size(); i++) {
    poplarProgs[i] = snapProgs[i].getPoplarProgram();
  }
  return poplarProgs;
}

} // namespace

std::string IrLowering::getSerializedGraph() const {
  std::stringstream ss;
  pGraph->getPoplarGraph().serialize(
      ss, toPoplarProgs(progs.progs()), poplar::SerializationFormat::Binary);
  return ss.str();
}

std::unique_ptr<PopOpx> IrLowering::createOpx(Op *op) {
  if (dv_p == nullptr) {
    throw error("IrLowering::setDevice has not been called.");
  }

  auto opx = OpxManager::createOpx(op, dv_p);

  if (!opx) {
    if (op->opid == Onnx::Operators::Constant_1 ||
        op->opid == Onnx::Operators::Constant_9) {
      throw internal_error("No PopOpx for {}", op->opid);
    } else {
      auto pattern = PreAliasPatternManager::opReplacementPattern(op);
      if (pattern != "") {
        throw error("Could not create opx for '{}'. This op should have been "
                    "removed by pattern {}",
                    op->opid,
                    pattern);
      } else {
        throw error("Could not create opx for '{}' and there were no patterns "
                    "intended to remove it. Please check it is defined and "
                    "registered correctly",
                    op->opid);
      }
    }
  }

  return opx;
}

// The Id of the task which adds a Tensor to a snap::Graph
PriTaskDependency IrLowering::taskWhichCreates(TensorId id) const {
  Tensor *tensor = ir().getTensor(id);
  if (!tensor->hasProducer()) {
    // Tensors without producers are created by special tasks
    return {initTensorTaskId(id), DependencyType::Tensor};
  } else {
    auto producer  = tensor->getProducer();
    OutIndex index = producer->output->indices(tensor).front();
    if (getOpx(producer->id)->outputCreatedExternally(index)) {
      // Tensors with producers but requirements to create a special task
      return {initTensorTaskId(id), DependencyType::Tensor};
    } else {
      // Tensors with producer Ops are created (added to a Graph) by their
      // producer's OpTask
      return {opTensorTaskId(tensor->getProducer(), tensor),
              DependencyType::Tensor};
    }
  }
}

TaskId IrLowering::taskWhichPopulates(TensorId id) const {
  Tensor *tensor = ir().getTensor(id);

  // OpTasks both initialize a Tensor, and generate the code to set its value
  if (tensor->hasProducer()) {
    // For partial grown Ops:
    // The tensor is created with opTensorTaskId, but the code is only
    // inserted with opTaskId
    return opTaskId(tensor->getProducer());
  }

  // if a Tensor is of type Stream, the Copy from host to device populates it
  else if (!ir().useSyntheticData() &&
           tensor->tensorType() == TensorType::Stream) {
    return fromHostTaskId(tensor->id);
  }

  // default:
  else {
    return initTensorTaskId(id);
  }
}

PriTask
IrLowering::getDependencyFreeInitTensorCreatorTask(const TensorId &tensorID) {
  const Tensor *tensor = ir().getTensor(tensorID);
  return initTensorTask(getInitTensorCreators(tensor, true));
}

std::vector<ICreatorCandidatePtr>
IrLowering::getCreatorEndpoints(const Tensor *startTensor,
                                bool excludeEndpointsFromPath,
                                bool includeDeadends) const {

  std::vector<ICreatorCandidatePtr> endpoints;
  std::vector<std::pair<const Tensor *, std::vector<OpxInAndOutIndex>>>
      searchStack;

  std::vector<OpxInAndOutIndex> startPath;
  const Graph *currentGraph = &startTensor->getGraph();
  while (currentGraph->id != ir().getMainGraph().id) {
    // For tensors created within a subgraph:
    // Detect graph outputs and try to propagate out through the first call site
    // until the top-level graph is reached.
    Op *op = currentGraph->getCallSiteOps(1).front();
    // Insert the first call site as delegate element on the path
    // this simulates "entering" a subgraph through a regular SubgraphOp
    // (Theoretically, adding all call sites would be possible too,
    // but should not be necessary)
    startPath.push_back({getOpx(op->id)});
    currentGraph = &op->getGraph();
  }

  // Add search starting point
  searchStack.push_back({startTensor, startPath});

  // Depth-first creator search
  while (!searchStack.empty()) {
    auto tensorAndPath                          = searchStack.back();
    const Tensor *tensor                        = tensorAndPath.first;
    std::vector<OpxInAndOutIndex> pathFromInput = tensorAndPath.second;
    searchStack.pop_back();

    // Check if any of the consumers can extend the path
    for (Op *op : tensor->consumers.getOps()) {
      auto conOpId      = op->id;
      const PopOpx *opx = getOpx(conOpId);

      for (InIndex inIndex : op->input->indices(ir().getTensor(tensor->id))) {
        auto f_create = [&]() {
          auto updatedPath = pathFromInput;
          if (!excludeEndpointsFromPath) {
            // note: no valid outIndex
            updatedPath.push_back({opx, inIndex, -1});
          }

          // Get all ops to deduce global schedule position of the Opx
          std::vector<Op *> ops;
          ops.reserve(pathFromInput.size());
          for (auto &opxOnPath : pathFromInput) {
            Op *opOnPath = &opxOnPath.opx->getOp<Op>();
            ops.push_back(opOnPath);
          }

          endpoints.push_back(std::make_shared<InputCreatorCandidate>(
              inIndex,
              opx,
              updatedPath,
              livenessAnalyzer->getGlobalSchedulePosition(ops)));
        };

        auto f_unwind = [&]() {
          auto updatedPath = pathFromInput;
          for (auto &ind_ten : op->output->tensorMap()) {
            auto nextOutputTensor = ind_ten.second;
            auto outIndex         = ind_ten.first;
            if (opx->canUnwind(inIndex, outIndex)) {
              updatedPath.push_back({opx, inIndex, outIndex});
              searchStack.push_back({nextOutputTensor, updatedPath});
            }
          }
        };

        auto f_deadend = [&]() {
          auto updatedPath = pathFromInput;
          if (includeDeadends) {
            if (!excludeEndpointsFromPath) {
              updatedPath.push_back(
                  {opx, inIndex, -1}); // note: no valid outIndex
            }
            endpoints.push_back(std::make_shared<InputCreatorCandidate>(
                inIndex, opx, updatedPath, 0));
          }
        };

        // TODO: T13654 Generalize for other subgraphing ops (if, loop).
        // Create common base class for Loop, If, Call
        auto f_delegate = [&]() {
          auto updatedPath = pathFromInput;

          // Mark as delegate visited Opx on path
          updatedPath.push_back({opx});

          const SubgraphOpx *sgopx = dynamic_cast<const SubgraphOpx *>(opx);

          // Get delegated endpoints
          SubgraphOp *subgraphOp = &sgopx->getOp<SubgraphOp>();
          auto callgraphs        = subgraphOp->getCalledGraphs();

          for (auto callgraph : callgraphs) {
            InIndex subgraphInIndex =
                subgraphOp->opInToSubgraphInIndex(inIndex);
            if (subgraphInIndex > -1 &&
                subgraphInIndex < callgraph->getInputIds().size()) {
              auto in_tensor_id      = callgraph->getInputId(subgraphInIndex);
              const Tensor *inTensor = ir().getTensor(in_tensor_id);
              searchStack.push_back({inTensor, updatedPath});
            }
          }
        };

        switch (opx->getInputCreatorType(inIndex)) {
        // Opx has poplar call to layout tensor at this
        // inIndex
        case InputCreatorType::CanCreate: {
          logging::devicex::trace("{} can create, path depth {}",
                                  op->debugName(),
                                  pathFromInput.size());
          f_create();
          break;
        }
        case InputCreatorType::CanDelegate: {
          logging::devicex::trace("{} can delegate, path depth {}",
                                  op->debugName(),
                                  pathFromInput.size());
          f_delegate();
          break;
        }
        // Recursively search the DAG downstream of the op until we
        // have set of endpoints that can create the tensor
        case InputCreatorType::CanUnwind: {
          logging::devicex::trace("{} can unwind, path depth {}",
                                  op->debugName(),
                                  pathFromInput.size());
          f_unwind();
          break;
        }
        case InputCreatorType::CanCreateOrUnwind: {
          logging::devicex::trace("{} can create or unwind, path depth {}",
                                  op->debugName(),
                                  pathFromInput.size());
          f_create();
          f_unwind();
          break;
        }
        case InputCreatorType::CanDelegateOrUnwind: {
          logging::devicex::trace("{} can delegate or unwind, path depth {}",
                                  op->debugName(),
                                  pathFromInput.size());
          f_delegate();
          f_unwind();
          break;
        }
        // Consuming op can't create tensor
        case InputCreatorType::Deadend: {
          logging::devicex::trace("{} is a deadend, path depth {}",
                                  op->debugName(),
                                  pathFromInput.size());
          f_deadend();
          break;
        }
        default: {
          throw error("InputCreatorType not implemented for Opx of OpId {}",
                      op->id);
        }
        }
      }
    }

    auto subgraphEscape = [&]() {
      // Example 1: Tensor is created before a subgraph, creator is behind it
      //            D can be created, D escaped from C, C is unwound to B,
      //            B delegates to A
      // A-delegate        escape-D-create
      //          |        |
      //          B-unwind-C
      //
      // Example 2: Tensor is created inside a subgraph, creator is behind it
      //            C can be created, C escaped from B, B is unwound to A
      //                   escape-C-create
      //                   |
      //          A-unwind-B

      // Check if the path can continue behind a subgraph
      auto graphOutputIds = tensor->getGraph().getOutputIds();
      for (OutIndex o = 0; o < graphOutputIds.size(); ++o) {
        if (graphOutputIds[o] == tensor->id) {
          // Current tensor is the graph output at index o
          auto pathToInput = pathFromInput;
          std::reverse(pathToInput.begin(), pathToInput.end());
          // Get the call site by walking back on the path
          for (auto &opxOnPath : pathToInput) {
            if (opxOnPath.isDelegate) {
              // Is a delegate: Get the caller Op

              Op *op = &opxOnPath.opx->getOp<Op>();
              if (!op->isConvertibleTo<SubgraphOp>()) {
                // subgraphEscape only supports cases where the caller Op
                // is of type SubgraphOp.
                // TODO: T13654 Support remaining cases
                // (24/11/2021 currently only IfOp)
                return;
              }

              SubgraphOp *subgraphOp = &opxOnPath.opx->getOp<SubgraphOp>();
              for (auto graph : subgraphOp->getCalledGraphs()) {
                // Loop over all callees
                if (graph->id == tensor->getGraph().id) {
                  // This callee is the graph where the current tensor is in
                  // Get delegate output tensor corresponding to subgraph output

                  // TODO: T13654 Generalize for other subgraphing ops (if,
                  // loop).

                  OutIndex opo = subgraphOp->subgraphOutToOpOutIndex(o);
                  if (opo > -1 && opo < subgraphOp->output->n()) {
                    Tensor *nextOutputTensor = subgraphOp->output->tensor(opo);
                    // Continue search behind the subgraph
                    searchStack.push_back({nextOutputTensor, pathFromInput});
                    return;
                  }
                }
              }
            }
          }
        }
      }
    };
    subgraphEscape();
  }

  return endpoints;
}

void IrLowering::removeNonDependencyFreeCreators(
    std::vector<ICreatorCandidatePtr> &candidates) {
  auto not_dependency_free = [](const ICreatorCandidatePtr &candidate) {
    auto dnfTensorIDs = candidate->mustExistBeforeCreate();

    // No sets at all
    if (dnfTensorIDs.empty()) {
      logging::devicex::trace("Nothing in dnfTensorIDs");
      return false;
    }

    // At least one set is empty
    for (auto &set : dnfTensorIDs) {
      if (set.empty()) {
        logging::devicex::trace("Returning as a set is empty");
        return false;
      }
    }

    return true;
  };

  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(), not_dependency_free),
      candidates.end());
}

std::vector<ICreatorCandidatePtr>
IrLowering::getTensorCreators(const Tensor *tensor, bool dependencyFree) const {
  // Search of the graph to get the candidate Opxs that
  // know how to create this tensor.
  // The pathFromInput argument is an empty vector, as
  // we are starting the search from the root (input)

  logging::devicex::trace("Get tensor creator for {}, {} elements",
                          tensor->id,
                          tensor->info.nelms());

  std::vector<ICreatorCandidatePtr> candidates = getCreatorEndpoints(tensor);

  if (dependencyFree) {
    logging::devicex::trace("Finding a dependency-free creator for {}",
                            tensor->id);
    removeNonDependencyFreeCreators(candidates);
  }

  logging::devicex::trace(
      "{} creator candidate(s) for {}", candidates.size(), tensor->id);

  if (candidates.size() > 0) {
    std::sort(
        candidates.begin(), candidates.end(), ICreatorCandidate::greaterThan);

    if (candidates.front()->getNumElems() == tensor->info.nelms()) {
      logging::devicex::trace("Candidate {} priority {} creates tensor alone.",
                              candidates.front()->str(),
                              candidates.front()->getMaxCreatorPriority());
      // A single top-priority candidate can initialize the tensor fully.
      std::vector<ICreatorCandidatePtr> topCandidates;

      for (auto candidate : candidates) {
        if (candidate->getNumElems() == tensor->info.nelms() &&
            candidate->getMaxCreatorPriority() ==
                candidates.front()->getMaxCreatorPriority()) {
          topCandidates.push_back(candidate);
        }
      }

      return topCandidates;
    } else {
      logging::devicex::trace("Multiple candidates needed.");
      // Multiple creators need to be concatenated to form the full tensor.
      std::shared_ptr<InputMultiCreatorCandidate> multiCandidate =
          std::make_shared<InputMultiCreatorCandidate>();
      for (auto candidate : candidates) {
        // Important to add candidates sorted by priority.
        // Highest first - ICreatorCandidate::greaterThan.
        multiCandidate->addCreatorCandidate(candidate);
      }
      logging::devicex::trace("Using multi-candidate {}.",
                              multiCandidate->str());
      return {multiCandidate};
    }
  } else {
    logging::devicex::trace("No suitable candidate.");
    return {};
  }
}

// Design decision : leave the option for a Tensor to be
// created based on complex global criteria open.
InitTensorPtrs IrLowering::getInitTensorCreators(const Tensor *tensor,
                                                 bool dependencyFree) const {
  auto candidates = getTensorCreators(tensor, dependencyFree);

  // 1. A unique candidate creator will create the tensor
  // 2. The tensor will be unwound (have its layout modified)
  //    by view-changing opxs on the path from the input to
  //    the candidate candidate
  if (candidates.size()) {
    // the inputs of creator which must have snap::Tensors
    // before creator creates input tensor at index inIndex.

    InitTensorPtrs creators;

    for (auto candidate : candidates) {
      for (auto &mustExist : candidate->mustExistBeforeCreate()) {
        // Every candidate can be enrolled multiple times,
        // with a different set of mustExist tensors
        logging::devicex::debug(
            "Creator candidate for snap::Tensor {} is {}, must exist: {}",
            tensor->id,
            candidate->str(),
            mustExist);
        auto creator = std::make_shared<InitTensorCreator>(
            candidate, mustExist, tensor->id, 1.0f);
        creators.insert(creator);
      }
    }

    return creators;
  } else {
    logging::devicex::trace("Reverting to linear creator");
    return {std::make_shared<InitTensorLinear>(tensor->id, 1.0f)};
  }
}

PriTask IrLowering::initRandomSeed() {
  auto streamedSeedId = GetRandomSeedOp::getStreamedSeedTensorId();

  auto initRandomSeedTask = [this, streamedSeedId]() {
    logging::devicex::debug("Initializing random seed.");
    SequenceMap seqs(graph());
    auto &prog = seqs.getSequence(&progs.setRandomSeedFromHostFragment());

    // NOTE: The `streamedSeedId` tensor is a 2xUINT32 tensor streamed from the
    // host to the device and serves as the basis of two mechanisms:
    //
    // - Firstly, it is used as a base value to compute explicit seed tensor
    //   values for random ops in the IR. This mechanism is used instead of
    //   relying on the IPU's random state so that random Ops are deterministic
    //   in the case of recomputation.
    // - Secondly, we use `streamedSeedId` to derive two RNG states on the
    // device
    //   using RngStateLowering.

    auto &seed = tensors_.get(streamedSeedId);

    // Set the RNG state based on the replica-identical seed.
    if (!rngStateLowering) {
      throw internal_error("[IrLowering] Member 'rngStateLowering' unexpected "
                           "not set");
    }

    rngStateLowering->lowerInitRngStatesFromSeed(
        prog, {seed.getPoplarTensor(), graph()}, "initRngStatesFromSeed");

    // Now, change `streamedSeedId` in a way that is distinct for each replica
    // so that 1) the explicit seeds for random ops derived from
    // `streamedSeedId` are now replica distinct and 2) we can set the RNG
    // state to it's natural resting state: replica differing.

    auto offset = graph().getPoplarGraph().addReplicationIndexConstant();
    graph().getPoplarGraph().setTileMapping(offset, 0);
    popops::addInPlace(graph().getPoplarGraph(),
                       seed[0].getPoplarTensor(),
                       offset,
                       prog.getPoplarSequence());

    return seqs;
  };

  std::vector<PriTaskDependency> deps;
  deps.push_back(taskWhichCreates(streamedSeedId));
  // Stream the seed tensor to device before using to set RNGs
  deps.push_back({fromHostTaskId(streamedSeedId), DependencyType::Scheduler});

  return {
      +1e6,                   // high priority
      initRandomSeedTaskId(), // name of this task
      deps,                   // depends on
      initRandomSeedTask      // what to run when the task is executed
  };
}

template <typename T>
void IrLowering::setInitVal(Tensor *tensor, DataType dst) {
  auto src = tensor->info.dataType();
  if (dst == DataType::UNDEFINED) {
    dst = src;
  }

  auto setValue = [this, tensor](const void *ptr) {
    graph().getPoplarGraph().setInitialValue<T>(
        tensors_.get(tensor->id).getPoplarTensor(),
        poplar::ArrayRef<T>(static_cast<const T *>(ptr), tensor->info.nelms()));
  };

  const void *ptr = static_cast<const void *>(tensor->tensorData()->data());

  if (src == dst) {
    setValue(ptr);
  } else {
    std::vector<char> castData = cast(src, dst, ptr, tensor->info.nbytes());
    ptr                        = static_cast<const void *>(castData.data());
    setValue(ptr);
  }
}

// Using specialised poplar function for setting init val for FLOAT16
void IrLowering::setInitValHalf(Tensor *tensor) {

  graph().getPoplarGraph().setInitialValueHalf(
      tensors_.get(tensor->id).getPoplarTensor(),
      poplar::ArrayRef<uint16_t>(
          static_cast<const uint16_t *>(tensor->tensorData()->data()),
          tensor->info.nelms()));
}

PriTask IrLowering::setInitTensorValTask(Tensor *tensor) {
  // See T6254. Currently we just use setInitialValue for all constant tensors
  auto f = [this, tensor]() {
    logging::devicex::debug("Setting initial value for tensor {}",
                            tensor->str());

    auto srcDataType = tensor->info.dataType();
    auto dstDataType = srcDataType;

    if (ir().getSessionOptions().enableSupportedDataTypeCasting) {
      dstDataType = getCompatibleDataType(srcDataType);
    }

    // see T5925 for making a more compact way of matching
    // types than using this switch statement

    switch (dstDataType) {
    case DataType::FLOAT: {
      setInitVal<float>(tensor);
      break;
    }
    case DataType::INT32: {
      setInitVal<int32_t>(tensor);
      break;
    }
    case DataType::FLOAT16: {
      setInitValHalf(tensor);
      break;
    }
    case DataType::BOOL: {
      setInitVal<bool>(tensor);
      break;
    }
    case DataType::UINT32: {
      setInitVal<uint32_t>(tensor);
      break;
    }
    case DataType::INT8: {
      setInitVal<int8_t>(tensor);
      break;
    }
    case DataType::UINT8: {
      setInitVal<uint8_t>(tensor);
      break;
    }
    case DataType::UNDEFINED:
    case DataType::INT64:
    case DataType::UINT16:
    case DataType::INT16:
    case DataType::STRING:
    case DataType::DOUBLE:
    case DataType::UINT64:
    case DataType::COMPLEX64:
    case DataType::COMPLEX128:
    case DataType::BFLOAT16:
    default: {
      throw error(
          "setInitTensorValTask not implemented for Tensor {} of Type {}. ",
          tensor->id,
          tensor->info.data_type());
    }
    }
    return SequenceMap(graph());
  };

  return {// priority unimportant
          0,
          // name of this task
          setInitTensorValTaskId(tensor->id),
          // snap::Tensor must exist. Other that this, this task can be
          // performed any time
          {{initTensorTaskId(tensor->id), DependencyType::Tensor}},
          f};
}

PriTask IrLowering::streamFromHostTask(TensorId streamTensorId,
                                       std::vector<Tensor *> tensors) {
  auto f = [this, streamTensorId, tensors]() {
    std::set<VGraphIdAndTileSet> vgidsAndTileSets;

    std::vector<std::pair<Tensor *, Op *>> consumerOps;
    for (Tensor *t : tensors) {
      auto consumers = t->consumers.getOps();
      for (auto c : consumers) {
        consumerOps.push_back({t, c});
      }
    }
    for (auto &tensorAndOp : consumerOps) {
      Tensor *tensor = tensorAndOp.first;
      Op *op         = tensorAndOp.second;
      // Assume another op will copy the tensor for an ipucopy
      if (op->isConvertibleTo<IpuCopyOp>()) {
        // VirtualGraphId with subgraph call introspection
        // for the current tensor
        auto index          = op->input->indicesMap().at(tensor)[0];
        auto vgidAndTileSet = op->getIntrospectionInVirtualGraphId(index);
        vgidsAndTileSets.insert(vgidAndTileSet);
      }
    }

    for (auto tensor : tensors) {
      vgidsAndTileSets.insert(tensor->getVirtualGraphIdAndTileSetUnsafe());
    }

    // Only stream the tensor once for all op's that consume it on an ipu
    for (auto &vgidAndTileSet : vgidsAndTileSets) {
      auto tensor = ir().getTensor(streamTensorId);
      poplar::OptionFlags options{};

      // Determine stream configuration.
      auto mode           = getReplicatedStreamMode(tensor);
      auto bufferingDepth = getBufferingDepth(tensor);

      if (bufferingDepth > 1) {
        // Configure the buffering depth of the stream.
        options.set("bufferingDepth", std::to_string(bufferingDepth));
      }

      logging::devicex::debug(
          "Creating host-to-device FIFO {} copied to "
          "ipu:{} (mode: {}, memory: {}, buffering depth: {})",
          streamTensorId,
          vgidAndTileSet,
          (mode == poplar::ReplicatedStreamMode::REPLICATE) ? "replicate"
                                                            : "broadcast",
          "hexopt",
          bufferingDepth);

      snap::Graph *graph;

      if (vgidAndTileSet.first != unusedVGraphId) {
        graph = &getVirtualGraph(vgidAndTileSet.first, vgidAndTileSet.second);
      } else {
        graph = pGraph.get();
      }
      fromHostStreams.emplace(
          streamTensorId,
          graph->getPoplarGraph().addHostToDeviceFIFO(h2dId(streamTensorId),
                                                      popType(tensor->info),
                                                      tensor->info.nelms(),
                                                      mode,
                                                      options));
    }
    return SequenceMap(graph());
  };
  return {
      0,                                    // priority unimportant
      streamFromHostTaskId(streamTensorId), // name of this task
      {{}},
      f // what to run when the task is executed
  };
}

PriTask IrLowering::streamToHostTask(TensorId streamTensorId,
                                     std::vector<Tensor *> tensors,
                                     bool isAnchorStream) {
  auto f = [this, tensors, streamTensorId, isAnchorStream]() {
    poplar::OptionFlags options{};
    auto streamTensor   = ir().getTensor(streamTensorId);
    auto bufferingDepth = getBufferingDepth(streamTensor);
    if (bufferingDepth > 1) {
      // Configure the buffering depth of the stream.
      options.set("bufferingDepth", std::to_string(bufferingDepth));
    }

    logging::devicex::debug(
        "Creating device-to-host FIFO for snap::Tensor "
        "{} (isAnchorStream = {}) with {} elements, buffering depth: {}",
        streamTensorId,
        isAnchorStream,
        tensors.front()->info.nelms(),
        bufferingDepth);

    auto pToHostStreams = &toHostAnchorStreams;
    if (!isAnchorStream) {
      pToHostStreams = &toHostWeightStreams;
    }

    pToHostStreams->emplace(streamTensorId,
                            graph().getPoplarGraph().addDeviceToHostFIFO(
                                d2hId(streamTensorId, isAnchorStream),
                                popType(tensors.front()->info),
                                tensors.front()->info.nelms(),
                                options));
    return SequenceMap(graph());
  };

  return {
      0, // priority unimportant
      streamToHostTaskId(streamTensorId, isAnchorStream), // name of this task
      {},
      f // what to run when the task is executed,
  };
}

bool IrLowering::hasRemoteBuffer(RemoteBufferId id) const {
  return remoteBuffers.find(id) != remoteBuffers.end();
}

const std::string IrLowering::getRemoteBufferName(RemoteBufferId id) {
  return "RB_" + std::to_string(id);
}

const std::pair<poplar::RemoteBuffer, nonstd::optional<snap::Tensor>> &
IrLowering::getRemoteBuffer(RemoteBufferId id) const {
  return remoteBuffers.at(id);
}

void IrLowering::createRemoteBuffer(RemoteBufferId id, snap::Tensor tensor) {
  auto info    = ir().getRemoteBufferInfo(id);
  auto name    = getRemoteBufferName(id);
  auto type    = tensor.elementType();
  auto size    = tensor.numElements();
  auto repeats = info.repeats;

  logging::devicex::info(
      "Creating remote buffer {}, type {}, size {}, repeats {}",
      name,
      type,
      size,
      repeats);

  remoteBuffers.insert({id,
                        {graph().getPoplarGraph().addRemoteBuffer(
                             name, type, size, repeats, true),
                         nonstd::optional<snap::Tensor>(tensor)}});
}

bool IrLowering::hasStreamTensor(TensorId tid) const {
  return streamTensors.find(tid) != streamTensors.end();
}
snap::Tensor IrLowering::getStreamTensor(TensorId tid) const {
  return streamTensors.at(tid);
}
void IrLowering::setStreamTensor(TensorId tid, snap::Tensor t) {
  streamTensors[tid] = t;
}

std::shared_ptr<gcl::CollectiveBalancedReorder>
IrLowering::getCollectiveBalancedReorder(TensorId tensor_id) {
  return collectiveReorders[tensor_id];
}

const gcl::CollectiveBalancedHostRearrangement &
IrLowering::getCollectiveBalancedHostRearrangement(
    const TensorId &tensor_id) const {
  return collectiveReorders.at(tensor_id)->getHostRearrangement();
}

void IrLowering::setCollectiveBalancedReorder(
    TensorId tensor_id,
    std::shared_ptr<gcl::CollectiveBalancedReorder> cbr) {
  collectiveReorders[tensor_id] = cbr;
}

int IrLowering::getNumFragments(const Graph &graph) const {
  return progs.getNumFragments(graph);
}

bool IrLowering::containsFragments(const Graph &graph) const {
  return progs.containsFragments(graph);
}

bool IrLowering::containsFragment(const Graph &graph,
                                  SubgraphPartIndex subgraphPart) const {
  return progs.containsFragment(graph, subgraphPart);
}

void IrLowering::createFragment(const Graph &graph_,
                                SubgraphPartIndex subgraphPart) {

  const auto addOpTasksTimer =
      ir().timePartitionLogger().scopedStopwatch("IrLowering::createFragment");
  return progs.createFragment(graph_, subgraphPart);
}

std::vector<poplar::Function> &
IrLowering::getFragmentFunctions(const Graph &_graph) {
  logging::devicex::trace("[getFragmentFunction] Getting function for graph {}",
                          _graph.id.str());
  return progs.getFragmentFunctions(_graph, graph());
}

poplar::Function &
IrLowering::getFragmentFunction(const Graph &_graph,
                                SubgraphPartIndex subgraphPart) {
  logging::devicex::trace(
      "[getFragmentFunction] Getting function for graph {}, part {}",
      _graph.id.str(),
      subgraphPart);
  return progs.getFragmentFunction(_graph, subgraphPart, graph());
}

void IrLowering::addPipelinedCopyTasks(PriTasks &tasks) {
  auto schedule =
      ir().getMainGraph().getOpSchedule({}, RequireOptimalSchedule::Yes);
  TaskId prevTaskId;

  logging::devicex::debug("Adding pipelined copy tasks");
  for (auto iter = schedule.rbegin(); iter != schedule.rend(); iter++) {
    auto &op = *iter;
    if (op->isConvertibleTo<IpuCopyOp>() &&
        op->settings.executionContext == ExecutionContext::Normal) {
      auto task = pipelinedCopyTask(op, prevTaskId);
      tasks.add(task);
      prevTaskId = task.name;
    }
  }
}

PriTask IrLowering::pipelinedCopyTask(Op *op, TaskId prevTaskId) {
  auto copyOp  = dynamic_cast<IpuCopyOp *>(op);
  auto opx     = getOpx(copyOp->id);
  auto copyOpx = dynamic_cast<IpuCopyOpx *>(opx);

  auto f = [this, copyOp, copyOpx]() {
    SequenceMap seqs(graph());
    logging::debug("Adding pipelined copies for op {}", copyOp->debugName());
    auto &prog = progs.pipelineIpuCopyFragment(
        logging::format("{}, {}, PipelineStage({})",
                        copyOp->debugName(),
                        copyOp->getFromToStr(),
                        copyOp->getPipelineStage()));
    copyOpx->growPipelined(seqs.getSequence(&prog),
                           pipelineIpuCopySrcDst[copyOp->id]);
    return seqs;
  };

  std::vector<PriTaskDependency> deps;
  if (!prevTaskId.empty()) {
    // Ensure the ops are scheduled in the order we're iterating through them
    // here.
    deps.push_back({prevTaskId, DependencyType::Scheduler});
  }

  // The ops opTask needs to run first to create the destination tensor.
  deps.push_back({opTaskId(op), DependencyType::Output});

  return {-100, pipelinedCopyTaskId(op), deps, f};
}

void IrLowering::addOpTasks(PriTasks &tasks) {

  const auto addOpTasksTimer = ir().timePartitionLogger().scopedStopwatch(
      "adding Op tasks (Ir Lowering)");

  // Ensure there is a program fragment for every Ir Graph
  logging::devicex::debug("[addOpTasks] for {} Graphs.",
                          ir().getGraphSchedule().size());
  for (auto graph : ir().getGraphSchedule()) {
    int numParts = subgraphPartitioner->getNumSubgraphParts(*graph);
    for (int p = 0; p < numParts; ++p) {
      if (!containsFragment(*graph, p)) {
        createFragment(*graph, p);
      }
    }
  }

  auto mainGraphSchedule =
      ir().getMainGraph().getOpSchedule({}, RequireOptimalSchedule::Yes);

  // repeating logic in Ir::getOpSchedule (can be simplified there?)
  std::vector<Op *> allOps;
  std::set<const Graph *> addedGraphs;
  std::function<void(const Graph *)> addGraph;
  addGraph = [this, &allOps, &addedGraphs, &addGraph](const Graph *graph) {
    const auto addGraphTask = ir().timePartitionLogger().scopedStopwatch(
        "addOpTasks::addGraph (Ir Lowering)");

    if (addedGraphs.find(graph) != addedGraphs.end()) {
      return;
    }
    logging::devicex::trace("[addOpTasks] Adding graph {}", graph->id);
    addedGraphs.insert(graph);

    // Add each op in the graph
    for (auto op : graph->getOpSchedule({}, RequireOptimalSchedule::Yes)) {
      // If the op calls another graph, then
      // the ops in that graph should be scheduled first
      for (auto calledGraph : op->getCalledGraphs()) {
        addGraph(calledGraph);
      }
      if (op->settings.recomputeType == RecomputeType::Recompute) {
        throw internal_error("non-main Graph Op which is Recompute");
      }
      allOps.push_back(op);
    }
  };

  logging::devicex::debug("Adding Graphs for all Ops in main Graph schedule");
  for (auto op : mainGraphSchedule) {
    for (auto calledGraph : op->getCalledGraphs()) {
      addGraph(calledGraph);
    }
    allOps.push_back(op);
  }

  double priority = 0.0;
  TaskId prevOpTaskId;

  std::set<std::pair<Op *, ExecutionPhase>> seenRecomputeOps;
  FindRequiredRecomputes recomputeFinder(allOps);

  // Map of TensorIds to possible initialization methods
  std::map<TensorId, InitTensorPtrs> initTensorMap;

  // Iterate through Ops according to the Ir's schedule
  for (Op *op : allOps) {

    const auto timer0 = ir().timePartitionLogger().scopedStopwatch(
        "adding input/output tasks for " + op->opid.type);

    auto opInputs = getOpx(op->id)->getInputsToPrepare();
    for (int i = 0; i < opInputs.size(); i++) {

      const auto addOpTasksTimer = ir().timePartitionLogger().scopedStopwatch(
          "adding Op input Tensor tasks (Ir Lowering)");

      auto opInput = opInputs[i];
      if (!tasks.contains(initTensorTaskId(std::get<1>(opInput)))) {
        if (std::get<0>(opInput).empty()) {
          // No tensor to clone or alias from
          auto creators =
              getInitTensorCreators(ir().getTensor(std::get<1>(opInput)));
          initTensorMap[std::get<1>(opInput)].insert(creators.begin(),
                                                     creators.end());
        } else {
          // Tensor can be cloned or aliased
          if (std::get<2>(opInput)) {
            initTensorMap[std::get<1>(opInput)].insert(
                std::make_shared<InitTensorAliasing>(
                    std::get<0>(opInput), std::get<1>(opInput), 100.0f));
          }
          initTensorMap[std::get<1>(opInput)].insert(
              std::make_shared<InitTensorPostIrAliasing>(
                  std::get<0>(opInput), std::get<1>(opInput), 50.0f));
          initTensorMap[std::get<1>(opInput)].insert(
              std::make_shared<InitTensorCloning>(
                  std::get<0>(opInput), std::get<1>(opInput), 10.0f));
        }
      }
    }

    auto opOutputs = getOpx(op->id)->getOutputsToPrepare();
    for (int i = 0; i < opOutputs.size(); i++) {

      const auto addOpTasksTimer = ir().timePartitionLogger().scopedStopwatch(
          "adding Op output Tensor tasks (Ir Lowering)");
      auto opOutput = opOutputs[i];
      if (!tasks.contains(initTensorTaskId(std::get<1>(opOutput)))) {
        if (std::get<0>(opOutput).empty()) {
          // No tensor to clone or alias from
          auto creators =
              getInitTensorCreators(ir().getTensor(std::get<1>(opOutput)));
          initTensorMap[std::get<1>(opOutput)].insert(creators.begin(),
                                                      creators.end());
        } else {
          // Tensor can be cloned or aliased
          if (std::get<2>(opOutput)) {
            initTensorMap[std::get<1>(opOutput)].insert(
                std::make_shared<InitTensorAliasing>(
                    std::get<0>(opOutput), std::get<1>(opOutput), 100.0f));
          }
          initTensorMap[std::get<1>(opOutput)].insert(
              std::make_shared<InitTensorPostIrAliasing>(
                  std::get<0>(opOutput), std::get<1>(opOutput), 50.0f));
          initTensorMap[std::get<1>(opOutput)].insert(
              std::make_shared<InitTensorCloning>(
                  std::get<0>(opOutput), std::get<1>(opOutput), 10.0f));
        }
      }
    }

    auto rerunSchedule = recomputeFinder.getRequiredRecomputeOps(op);
    if (!rerunSchedule.empty()) {
      logging::devicex::debug("Adding recompute rerun schedule for op {}",
                              op->debugName());
      requiredRecomputes[opTaskId(op)] = rerunSchedule;
    }

    logging::devicex::debug(
        "Created task for {}, adding dependencies and enqueueing", op->str());
    auto opTaskVec = opTasks(op, priority, prevOpTaskId);

    for (auto &task : opTaskVec) {
      tasks.add(task);
    }
    prevOpTaskId = opTaskVec.back().name;
    priority -= 1.;
  }

  // Add tasks for tensors to be created
  for (auto &tensorIdAndInits : initTensorMap) {
    if (!tasks.contains(initTensorTaskId(tensorIdAndInits.first))) {
      tasks.add(initTensorTask(tensorIdAndInits.second));
    }
  }
}

bool IrLowering::tryInitTensorByPostIRAliasing(
    TensorId dstId,
    const ViewChangers &viewChangers) {

  const auto addOpTasksTimer = ir().timePartitionLogger().scopedStopwatch(
      "Initializing Tensor By Post IR Aliasing (Ir Lowering)");

  for (Tensor *aliased :
       aliasZeroCopy->getPostIRAliases(ir().getTensor(dstId))) {
    if (tensors_.contains(aliased->id)) {

      // Can only alias if the view changers associated with the tensors are
      // also compatible
      bool viewChangerCompatible = true;
      if (tensors_.hasViewChangers(aliased->id) || !viewChangers.empty()) {
        ViewChangers aliasedViewChangers;
        if (tensors_.hasViewChangers(aliased->id)) {
          aliasedViewChangers = tensors_.getViewChangers(aliased->id);
        }
        viewChangerCompatible = aliasedViewChangers == viewChangers;
      }

      if (tensors_.canAlias(aliased->id) && viewChangerCompatible) {
        logging::devicex::debug("Creating snap::Tensor '{}' "
                                "by aliasing from snap::Tensor '{}'",
                                dstId,
                                aliased->id);
        tensors_.insertAliased(dstId, aliased->id);
        efficientlyCreatedInputTensors.insert(dstId);
        aliasZeroCopy->activateAlias(aliased, ir().getTensor(dstId));
        return true;
      } else {
        logging::devicex::trace(
            "[PopTensors] Rejecting aliasing of {} due to {}",
            aliased->id,
            viewChangerCompatible ? "constant or aliased region"
                                  : "differing view changers");
        // Tensor can't be aliased
        aliasZeroCopy->removePostIRAliases(ir().getTensor(aliased->id));
      }
    }
  }
  return false;
}

PriTask IrLowering::initTensorTask(InitTensorPtrs inits) {

  TensorId dstId;
  std::map<TensorId, bool> dependencyTensorValues;
  std::vector<boollogic::Term> terms;
  for (auto &init : inits) {
    dstId = init->getDstId();
    if (init->getDependsOnIds().empty()) {
      // No dependencies
      terms.push_back(true);
    } else {
      // Dependencies
      std::vector<boollogic::Term> subterms;
      for (auto srcId : init->getDependsOnIds()) {
        subterms.push_back(boollogic::Term::varTerm(srcId));
        dependencyTensorValues[srcId] = false;
      }
      terms.push_back(boollogic::Term::andTerm(subterms));
    }
  }

  // Conjunctive normal form of the term that has to be satisfied in order to
  // allocate this tensor
  auto cnfTerm = boollogic::Term::orTerm(terms).getCNF();

  // Translate from conjunctive normal form to dependencies
  std::vector<PriTaskDependency> deps;
  if (!cnfTerm.evaluate(dependencyTensorValues)) {
    // The term does not evaluate to true if all variables are set to false,
    // therefore, we have at least one PriTaskDepenceny

    // Evaluate the CNF
    // Note: We only expect Var/And/Or terms, does not handle True/False/Not
    std::function<void(boollogic::Term)> evaluate =
        [this, &deps, &evaluate](boollogic::Term t) {
          switch (t.getType()) {
          case boollogic::Type::And: {
            // Conjunction of dependencies
            for (auto &st : t.getTerms()) {
              evaluate(st);
            }
            break;
          }
          case boollogic::Type::Or: {
            // Multiple tasks can fulfill the dependency
            std::set<TaskId> orTaskIds;
            for (auto &st : t.getTerms()) {
              auto taskIds = taskWhichCreates(st.getVar()).getTaskIds();
              for (auto &taskId : taskIds) {
                orTaskIds.insert(taskId);
              }
            }
            PriTaskDependency dep(orTaskIds, DependencyType::Tensor);
            deps.push_back(dep);
            break;
          }
          case boollogic::Type::Var: {
            // Only one task can fulfill the dependency
            PriTaskDependency dep(taskWhichCreates(t.getVar()).getTaskIds(),
                                  DependencyType::Tensor);
            deps.push_back(dep);
            break;
          }
          case boollogic::Type::Not:
            break;
          case boollogic::Type::True:
            break;
          case boollogic::Type::False:
            break;
          default:
            break;
          }
        };
    evaluate(cnfTerm);
  }

  Tensor *tensor = ir().getTensor(dstId);

  // Initialising priMod to 0.0 since there is no guarantee that the tensor
  // to be initialised has a consumer in it's graph.
  double priMod = 0.0;

  // Design decision:
  // The order in which these initTensorTasks are run affects tensor
  // layout. This can affect the behaviour of random operations, as well
  // as overall memory consumption.
  // Let's fix the order of execution of a set of initTensorTasks, based
  // on some condition. We do this here by giving it a unique priority
  // based on:
  // - 0th consumer id
  // - the index at which it is consumed by this consumer
  if (tensor->consumers.getOps().size() > 0) {
    // Tensor does have consumers
    auto firstConsumer = tensor->consumers.getOps().front();
    auto priMod0       = firstConsumer->id;
    auto priMod1       = firstConsumer->input->indices(tensor).front();
    priMod             = static_cast<double>(priMod0) +
             static_cast<double>(priMod1) / maxOpInputs;
  }

  logging::devicex::trace(
      "Adding initTensorTask for tensor: {}, dependencies: {}", dstId, deps);

  auto f = [dstId, inits, this]() {
    // Try each init, sorted by priority
    for (auto &init : inits) {
      bool success = init->initTensor(*this);
      logging::devicex::trace("Trying to create {} with {} (success: {})",
                              dstId,
                              init->str(),
                              success ? "yes" : "no");
      if (success) {
        return SequenceMap(graph());
      }
    }
    // None of the inits worked
    throw error("Failed to initialize tensor {}", dstId);
  };

  return {-1e6 + priMod, initTensorTaskId(dstId), deps, f};
}

std::vector<PriTask>
IrLowering::opTasks(Op *op, double priority, TaskId prevOpTaskId) {
  std::vector<PriTask> priTasks;

  // although priority should guarantee that this
  // task is only run after inputs are all created,
  // we add a dependency to the input tensors, just
  // in case someone plays with the priorities.
  // Moreover, we must state the copy-from-host deps
  std::vector<PriTaskDependency> deps;

  // Add initTensorTask dependencies for externally created output tensors
  PopOpx *opx = getOpx(op->id);
  for (auto t_inds : op->output->indicesMap()) {
    if (opx->outputCreatedExternally(t_inds.second.front())) {
      Tensor *tensor = t_inds.first;

      logging::devicex::trace("Operation {} depends on it's output tensor {} "
                              "being externally created.",
                              op->debugName(),
                              tensor->id);

      PriTaskDependency creatorTask = {initTensorTaskId(tensor->id),
                                       DependencyType::Tensor};

      // Make sure we only add the creatorTask once in the dependency list
      if (std::find(deps.begin(), deps.end(), creatorTask) == deps.end()) {
        deps.push_back(creatorTask);
      }
    }
  }

  auto addGraphOpsToDeps = [&](const Graph *graph) {
    const auto schedule = graph->getOpSchedule({}, RequireOptimalSchedule::Yes);

    logging::devicex::debug("Add Graph Ops to dependencies, for {} Ops",
                            schedule.size());

    for (auto graphOp : schedule) {
      PriTaskDependency taskId = {opTaskId(graphOp), DependencyType::SubGraph};
      if (std::find(deps.begin(), deps.end(), taskId) == deps.end()) {
        deps.push_back(taskId);
      }
    }

    // Graph inputs
    for (auto &inputId : graph->getInputIds()) {
      PriTaskDependency creatorTask = taskWhichCreates(inputId);
      if (std::find(deps.begin(), deps.end(), creatorTask) == deps.end()) {
        deps.push_back(creatorTask);
      }
    }

    // Graph outputs
    for (auto &outputId : graph->getOutputIds()) {
      PriTaskDependency creatorTask = taskWhichCreates(outputId);
      if (std::find(deps.begin(), deps.end(), creatorTask) == deps.end()) {
        deps.push_back(creatorTask);
      }
    }
  };

  for (auto &graph : op->getCalledGraphs()) {
    addGraphOpsToDeps(graph);
  }

  logging::devicex::debug("Adding PriTaskDep to the order of the Ir schedule");

  // Depends on previous op task. This preserves op ordering from ir.
  // Note: the first opTask has no previous opTask
  if (!prevOpTaskId.empty()) {
    PriTaskDependency prevTask = {prevOpTaskId, DependencyType::Scheduler};
    // Add dependency only if not already added
    if (std::find(deps.begin(), deps.end(), prevTask) == deps.end()) {
      deps.push_back(prevTask);
    }
  }

  auto taskId = opTaskId(op);
  if (requiredRecomputes.find(taskId) != requiredRecomputes.end()) {
    for (const auto &recompOp : requiredRecomputes[taskId]) {
      PriTaskDependency recompTask = {opTaskId(recompOp),
                                      DependencyType::Output};
      if (std::find(deps.begin(), deps.end(), recompTask) == deps.end()) {
        deps.push_back(recompTask);
      }
    }
  }

  auto opTaskGrowFunc = [op, this]() {
    SequenceMap seqs(graph());
    const auto &containingGraph = op->getGraph();
    const auto &opts            = ir().getSessionOptions();
    // if this Op is not in the main scope
    if (!containingGraph.id.str().empty()) {
      // We need to create a mapping to avoid the index operator creating just
      // a singleton vector of sequence, as opposed to having all parts.
      seqs.addScopeFragments(progs.scopeFragments(containingGraph));
      // Get the right subgraphPart for op to lower in to.
      auto subgraphPart = subgraphPartitioner->getOpSubgraphPartBegin(op);
      PopOpx *opx       = getOpx(op->id);
      logging::devicex::debug(
          "Creating output tensors for non-main {} in {}, part {}",
          opx->op_p->debugName(),
          containingGraph.id.str(),
          subgraphPart);
      // Record each scope task fragment separately first.
      growOpx(opx, seqs[&progs.scopeFragment(containingGraph, subgraphPart)]);

    } else if (opts.implicitPipeliningEnabled()) {
      pipelinedOpTaskFunc(opTaskId(op), op, seqs);
    } else {
      opTaskFunc(opTaskId(op), op, seqs);
    }
    return seqs;
  };

  if (HostLoadOp *hlop = dynamic_cast<HostLoadOp *>(op)) {
    auto hostStreamDepencendy =
        PriTaskDependency(streamFromHostTaskId(hlop->getHostStreamTensorId()),
                          DependencyType::Tensor);
    logging::devicex::trace(
        "Op {} depends on stream {}", op->debugName(), hostStreamDepencendy);
    deps.push_back(hostStreamDepencendy);
  }

  if (HostStoreOp *hsop = dynamic_cast<HostStoreOp *>(op)) {
    auto hostStreamDepencendy = PriTaskDependency(
        streamToHostTaskId(hsop->getHostStreamTensorId(), true),
        DependencyType::Tensor);
    logging::devicex::trace(
        "Op {} depends on stream {}", op->debugName(), hostStreamDepencendy);
    deps.push_back(hostStreamDepencendy);
  }

  if (MultiExchangeOp *exchangeOp = dynamic_cast<MultiExchangeOp *>(op)) {
    for (int index = 0; index < exchangeOp->getNumExchanges(); ++index) {
      auto descriptor = exchangeOp->getExchangeDescriptor(index);
      if (descriptor.isHostExchange()) {
        auto hostStreamDepencendy =
            ((descriptor.getDirection() == ExchangeDirection::Store)
                 ? PriTaskDependency(
                       streamToHostTaskId(descriptor.getHostStreamTensorId(),
                                          true),
                       DependencyType::Tensor)
                 : PriTaskDependency(
                       streamFromHostTaskId(descriptor.getHostStreamTensorId()),
                       DependencyType::Tensor));
        logging::devicex::trace(
            "Op {} descriptor index {} ({}) depends on stream {}",
            op->debugName(),
            index,
            descriptor.getDirection(),
            hostStreamDepencendy);
        deps.push_back(hostStreamDepencendy);
      }
    }
  }

  // Part growing has the same basic dependencies as full growing, except
  // that parts can depend on a subset of input tensors rather than all of them.
  auto basePartDeps = deps;

  std::map<OpxGrowPartId, std::vector<PriTaskDependency>> opGrowPartIds;

  for (auto t_inds : op->input->indicesMap()) {
    Tensor *tensor                = t_inds.first;
    auto partIds                  = opx->getInGrowPartIds(tensor);
    PriTaskDependency creatorTask = taskWhichCreates(tensor->id);

    PriTaskDependency populatorTask = {taskWhichPopulates(tensor->id),
                                       DependencyType::Scheduler};
    for (auto partId : partIds) {
      auto it = opGrowPartIds.find(partId);
      if (it == opGrowPartIds.end()) {
        opGrowPartIds[partId] = basePartDeps;
      }
      auto &partDeps = opGrowPartIds[partId];
      // Make sure we only add the creatorTask once in the dependency list
      if (std::find(partDeps.begin(), partDeps.end(), creatorTask) ==
          partDeps.end()) {
        logging::devicex::trace(
            "Adding Op {} part {} dependency {}", op->id, partId, creatorTask);
        partDeps.push_back(creatorTask);
      }
      if (std::find(partDeps.begin(), partDeps.end(), populatorTask) ==
          partDeps.end()) {
        logging::devicex::trace("Adding Op {} part {} dependency {}",
                                op->id,
                                partId,
                                populatorTask);
        partDeps.push_back(populatorTask);
      }
    }
    // Make sure we only add the creatorTask once in the dependency list
    if (std::find(deps.begin(), deps.end(), creatorTask) == deps.end()) {
      deps.push_back(creatorTask);
    }
    if (std::find(deps.begin(), deps.end(), populatorTask) == deps.end()) {
      deps.push_back(populatorTask);
    }
  }
  for (auto t_inds : op->output->indicesMap()) {
    Tensor *tensor = t_inds.first;
    auto partId    = opx->getOutGrowPartId(tensor);
    auto it        = opGrowPartIds.find(partId);
    if (it == opGrowPartIds.end()) {
      opGrowPartIds[partId] = basePartDeps;
    }
  }

  for (auto &partIdAndDependencies : opGrowPartIds) {
    if (partIdAndDependencies.first != unusedGrowPartId) {
      auto opTaskPartGrowFunc = [partIdAndDependencies, opx, this]() {
        SequenceMap seqs(graph());

        // Grow a part of the Opx
        opx->growPart(partIdAndDependencies.first);

        return seqs;
      };

      priTasks.push_back({priority,
                          opPartTaskId(op, partIdAndDependencies.first),
                          partIdAndDependencies.second,
                          opTaskPartGrowFunc});
      deps.push_back(
          PriTaskDependency(opPartTaskId(op, partIdAndDependencies.first),
                            DependencyType::Scheduler));
      deps.push_back(
          PriTaskDependency(opPartTaskId(op, partIdAndDependencies.first),
                            DependencyType::Output));
    }
  }

  priTasks.push_back({priority, opTaskId(op), deps, opTaskGrowFunc});

  return priTasks;
}

void IrLowering::growOpx(PopOpx *opx,
                         SequenceMap::SequenceInterval seqInterval) {

  if (!rngStateLowering) {
    throw internal_error("[IrLowering] Member 'rngStateLowering' unexpected "
                         "not set");
  }

  logging::devicex::trace("Calling growOpx for Op {} with debugName {}",
                          opx->op_p->str(),
                          opx->op_p->debugName());

  auto seqIt             = seqInterval.first;
  auto printTensorForOpx = [this, opx, &seqIt](TensorId id,
                                               snap::Tensor tensor) {
    if (printTensorIds.find(id) != printTensorIds.end() &&
        printedTensorIds.find(id) == printedTensorIds.end()) {
      auto printProg = poplar::program::PrintTensor(
          id, tensor.getPoplarTensor(), opx->debugContext());
      seqIt->getPoplarSequence().add(printProg);
      printedTensorIds.insert(id);
    }
  };

  if (opxTrace) {
    seqIt->getPoplarSequence().add(
        poplar::program::PrintTensor(opx->op_p->str() + "/enter",
                                     opxTraceTensor.getPoplarTensor(),
                                     opx->debugContext()));
  }

  // Add print tensor for tensors in POPART_PRINT_TENSORS on inputs.
  for (auto in : opx->op_p->input->tensorIdMap()) {
    auto idx = in.first;
    auto id  = in.second;
    if (dv_p->lowering().tensors().contains(id)) {
      auto tensor = opx->getInTensor(idx);
      printTensorForOpx(id, tensor);
    }
  }

  // Build a map of all tensors that are inputs and not
  // supposed to be modified by this operation
  std::map<InIndex, std::pair<snap::Tensor, snap::Tensor>> nonModifiedTensors;
  if (ir().getSessionOptions().opxModifyChecking) {
    if (ir().getSessionOptions().replicatedGraphCount > 1) {
      throw error("Can not do opx modify checking when using replicated "
                  "graphs. Set SessionOptions::opxModifyChecking to false.");
    }

    for (auto &inputMap : opx->op_p->input->tensorMap()) {
      auto regions = opx->op_p->modifies(inputMap.first);
      // Check that no region of the input tensor is marked as modified
      // according to the IR
      if (std::all_of(regions.begin(),
                      regions.end(),
                      [](const view::Region &r) { return r.isEmpty(); })) {
        snap::Tensor inTensor = opx->get(inputMap.second->id);
        // Check that this isn't a phony tensor or a tensor with post-IR aliases
        if (inTensor.numElements() > 0 &&
            aliasZeroCopy->getActiveAliasedTensors({inputMap.second}, false)
                .empty()) {
          // Clone the input tensor with its current values
          // to check if the original input tensor has been modified
          // during opx->grow(seq)
          auto inTensorClone = snap::Tensor{
              graph().getPoplarGraph().clone(inTensor.getPoplarTensor(),
                                             opx->debugContext("orig")),
              graph()};
          seqIt->add(snap::program::Copy(
              inTensor, inTensorClone, false, opx->debugContext("check")));
          nonModifiedTensors[inputMap.first] =
              std::make_pair(inTensor, inTensorClone);
        }
      }
    }
  }

  auto begin = subgraphPartitioner->getOpSubgraphPartBegin(opx->op_p);
  auto end   = subgraphPartitioner->getOpSubgraphPartEnd(opx->op_p);

  // Grow code for the op into a separate vector, because we may decide to
  // now include code for this in the poplar program. But we need to grow it
  // in any case.
  std::vector<snap::program::Sequence> seqVec;

  // Add at least one Seq, so we can grow RNG/SR state changes.
  std::stringstream ss;
  ss << opx->op_p->getGraph().id.str() << "/" << begin;
  seqVec.resize(1,
                snap::program::Sequence(opx->debugContext(ss.str()), graph()));

  // Lower any pre-Op RNG state logic.
  rngStateLowering->lowerSetRngState(seqVec.front(), opx);

  // Lower the Op.
  {
    const auto growTimeTracker = ir().timePartitionLogger().scopedStopwatch(
        "Lowering ops to poplar (\"grow\" methods)");
    opx->grow(seqVec);
  }

  // Lower any post-Op RNG state logic.
  rngStateLowering->lowerGetRngState(seqVec.back(), opx);

  // Sanity check: did the op grow over the correct number of subgraph parts?
  if (seqVec.size() != (end - begin)) {
    throw internal_error("Expected {} to lower into {} subgraph parts (got {})",
                         opx->op_p->debugName(),
                         end - begin,
                         seqVec.size());
  }

  if (aliasZeroCopy->opRequired(opx->op_p)) {
    // Code of an Op can be skipped if the Op is not required,
    // meaning the Op has:
    // 1.) No outputs which are consumed/live downstream
    // 2.) No side effects

    // Move programs into sequence mapper.
    for (auto it = seqVec.cbegin(); it != seqVec.cend(); ++it) {
      if (it != seqVec.cbegin()) {
        ++seqIt;
      }

      if (seqIt == seqInterval.second) {
        // This shouldn't happen. An op lowered into a number of subgraph
        // parts but for whatever reason there aren't that many parts subgraph
        // parts available to lower the op into.
        throw internal_error(
            "Insufficient sequences available to lower op {} with debugName {}",
            opx->op_p->str(),
            opx->op_p->debugName());
      }

      // Add the lowered sequence to the mapped sequence.
      seqIt->add(*it);
    }
  } else {
    for (auto out : opx->op_p->output->tensorMap()) {
      snap::Tensor outTensor = opx->getOutTensor(out.first);
      seqIt->getPoplarSequence().add(poplar::program::WriteUndef(
          outTensor.getPoplarTensor(), opx->debugContext()));
    }
    logging::devicex::trace(
        "Skipping code sequence for Op {} with debugName {}",
        opx->op_p->str(),
        opx->op_p->debugName());
  }

  // Add print tensor for tensors in POPART_PRINT_TENSORS on outputs.
  for (auto out : opx->op_p->output->tensorIdMap()) {
    auto idx = out.first;
    auto id  = out.second;
    if (dv_p->lowering().tensors().contains(id)) {
      auto tensor = opx->getOutTensor(idx);
      printTensorForOpx(id, tensor);
    }
  }

  if (ir().getSessionOptions().opxModifyChecking) {
    for (auto &nonModified : nonModifiedTensors) {
      // Compare input tensor with the input tensor clone before executing the
      // Op, skip checking non-finite numbers
      // (which always evaluate to non-equal)

      popops::expr::Any lhsExpr = popops::expr::_1;
      popops::expr::Any rhsExpr = popops::expr::_2;

      if (nonModified.second.first.elementType() == poplar::FLOAT ||
          nonModified.second.first.elementType() == poplar::HALF) {
        lhsExpr =
            popops::expr::Select(popops::expr::_1,
                                 popops::expr::Const(0),
                                 popops::expr::IsFinite(popops::expr::_1));
        rhsExpr =
            popops::expr::Select(popops::expr::_2,
                                 popops::expr::Const(0),
                                 popops::expr::IsFinite(popops::expr::_2));
      }

      auto check        = popops::map(opx->graph().getPoplarGraph(),
                               popops::expr::NotEqual(lhsExpr, rhsExpr),
                               {nonModified.second.first.getPoplarTensor(),
                                nonModified.second.second.getPoplarTensor()},
                               seqIt->getPoplarSequence(),
                               opx->debugContext("opxModifyChecking"),
                               {});
      auto checkReduced = check.flatten();
      // Convert result to boolean scalar
      if (check.numElements() > 1) {
        checkReduced = popops::reduce(opx->graph().getPoplarGraph(),
                                      checkReduced,
                                      {0},
                                      {popops::Operation::LOGICAL_OR},
                                      seqIt->getPoplarSequence(),
                                      opx->debugContext("opxModifyChecking"));
      } else {
        checkReduced = checkReduced.squeeze({0});
      }
      auto ifProg = poplar::program::ErrorProgram(
          logging::format("Op {} claims input {} is not modified, "
                          "but the Poplar tensors disagree.",
                          opx->op_p->debugName(),
                          nonModified.first),
          check,
          opx->debugContext("if"));
      auto elseProg =
          snap::program::Sequence(opx->debugContext("else"), graph());
      seqIt->getPoplarSequence().add(
          poplar::program::If(checkReduced,
                              ifProg,
                              elseProg.getPoplarSequence(),
                              opx->debugContext("opxModifyCheck")));
    }
  }

  if (opxTrace) {
    seqIt->getPoplarSequence().add(
        poplar::program::PrintTensor(opx->op_p->str() + "/exit",
                                     opxTraceTensor.getPoplarTensor(),
                                     opx->debugContext()));
  }
}

void IrLowering::opTaskFunc(TaskId taskId, Op *op, SequenceMap &seqs) {
  PopOpx *opx              = getOpx(op->id);
  ExecutionContext context = op->settings.executionContext;

  contextOpRegistry[{context, taskId}].push_back(op);

  if (context == ExecutionContext::OptimizerFromHostFragment) {
    growOpx(opx, seqs[&progs.streamOptimizerFromHostFragment()]);
  }

  // Special case for running operators before the main loop.
  else if (context == ExecutionContext::WeightsFromHostFragment) {
    growOpx(opx, seqs[&progs.streamWeightsFromHostFragment()]);
  }

  // Special case for running operators after the main loop.
  else if (context == ExecutionContext::WeightsToHostFragment) {
    growOpx(opx, seqs[&progs.weightsToHostFragment()]);
  }

  // pre-loss : create vertices for all recompute types
  else if (op->scheduledPreLoss == ScheduledPreLoss::Yes) {

    // Pre-loss, not recompute
    if (op->settings.recomputeType == RecomputeType::Checkpoint ||
        op->settings.recomputeType == RecomputeType::Undefined ||
        op->settings.recomputeType == RecomputeType::Recomputed) {
      logging::devicex::debug("Adding checkpoint Op {}", op->debugName());
      growOpx(opx, seqs[&progs.forwardFragment()]);
    }

    // Pre-loss, recompute
    else if (op->settings.recomputeType == RecomputeType::Recompute) {
      logging::devicex::debug("Adding (first) {} Op {}",
                              op->settings.recomputeType,
                              op->debugName());

      growOpx(opx, progs.createRecomputeFragment(op->id));
      seqs.getSequence(&progs.forwardFragment())
          .add(*progs.recomputeFragment(op->id));
    }

    // Pre-loss, not recompute or checkpoint
    else {
      throw internal_error("Unrecognised recompute type");
    }
  }

  // post-loss
  else if (op->scheduledPreLoss == ScheduledPreLoss::No) {
    if (op->settings.recomputeType == RecomputeType::Recompute) {
      std::stringstream oss;
      op->append(oss);
      throw internal_error("Recompute Op which is ScheduledPreLoss::No is "
                           "not permitted: \n{}",
                           oss.str());
    }

    // 2 special case Ops when there is a gradient accumulator / velocity.
    // If we are doing gradient accumulation, we need to ensure the reset
    // and var update aren't run every time. Instead, these fragments sit
    // outside the "main" loop of the forwards and backwards passes.
    // special case Op 1:
    if (ir().getSessionOptions().enableGradientAccumulation &&
        context == ExecutionContext::AccumulateOuterFragment) {
      outerLoopFragEmpty = false;
      growOpx(opx, seqs[&progs.accumulateOuterFragment()]);
    } else {
      auto found = requiredRecomputes.find(taskId);
      if (found != requiredRecomputes.end()) {
        auto &rerunSchedule = found->second;
        for (auto opToRerun : rerunSchedule) {
          logging::devicex::debug("Adding (second) recompute Op {}",
                                  opToRerun->debugName());

          seqs.getSequence(&progs.backwardFragment())
              .add(*progs.recomputeFragment(opToRerun->id));
          contextOpRegistry[{context, taskId}].push_back(opToRerun);
          ExecutionPhase phase =
              op->hasExecutionPhase() &&
                      this->ir()
                              .getSessionOptions()
                              .executionPhaseSettings.phases >= 2
                  ? op->getExecutionPhase()
                  : -1;
          progs.recordRecomputed(opToRerun->id, phase);
        }
      }

      logging::devicex::debug("Adding post-turning check-point Op {}",
                              op->debugName());

      growOpx(opx, seqs[&progs.backwardFragment()]);
    }
  }

  else {
    throw internal_error("Unknown SchedulePreLoss in prepare, should "
                         "updateVertices have been called recently?");
  }
}

void IrLowering::pipelinedOpTaskFunc(TaskId taskId, Op *op, SequenceMap &seqs) {
  PopOpx *opx              = getOpx(op->id);
  ExecutionContext context = op->settings.executionContext;

  contextOpRegistry[{context, taskId}].push_back(op);

  if (context == ExecutionContext::OptimizerFromHostFragment) {
    growOpx(opx, seqs[&progs.streamOptimizerFromHostFragment()]);
  } else if (op->isConvertibleTo<HostLoadOp>()) {
    growOpx(opx,
            seqs[&progs.pipelineToDeviceStreamFragment(op->getPipelineStage(),
                                                       op->debugName())]);
  }

  else if (ir().getSessionOptions().enableGradientAccumulation &&
           context == ExecutionContext::AccumulateOuterFragment) {
    outerLoopFragEmpty = false;
    growOpx(opx, seqs[&progs.accumulateOuterFragment()]);
  }

  // Special case for running operators before the main loop.
  else if (context == ExecutionContext::WeightsFromHostFragment) {
    growOpx(opx, seqs[&progs.streamWeightsFromHostFragment()]);
  }

  // Special case for running operators before the main loop.
  else if (context == ExecutionContext::WeightsToHostFragment) {
    // Special case for running operators before the main loop.
    growOpx(opx, seqs[&progs.weightsToHostFragment()]);
  }

  else {
    auto found = requiredRecomputes.find(taskId);
    if (found != requiredRecomputes.end()) {
      auto &rerunSchedule = found->second;

      // Add the recomputations.
      for (auto opToRerun : rerunSchedule) {
        logging::devicex::debug("Adding (second) recompute Op {}",
                                opToRerun->debugName());
        if (progs.hasBeenRecomputed(opToRerun->id, -1)) {
          throw internal_error("Ops to recompute should only appear in once in "
                               "requiredRecomputes");
        }
        progs.recordRecomputed(opToRerun->id, -1);
        seqs
            .getSequence(&progs.pipelineMainFragment(
                op->getPipelineStage(), "recompute of " + opToRerun->str()))
            .add(*progs.recomputeFragment(opToRerun->id));

        contextOpRegistry[{context, taskId}].push_back(opToRerun);
      }
    }

    if (op->isConvertibleTo<IpuCopyOp>()) {
      // IpuCopyOps are handled as a special case in pipelining. Here,
      // the destination tensor is created using the
      // `createPipelinedOutput` method. Later, for each pipeline cycle
      // the copy appears in, a new copy program is added to the cycles
      // sequence using `IpuCopyOpx::growPipelined`.
      pipelineIpuCopySrcDst[op->id] =
          dynamic_cast<IpuCopyOpx *>(opx)->createPipelinedOutput();
    } else if (op->settings.recomputeType == RecomputeType::Checkpoint ||
               op->settings.recomputeType == RecomputeType::Undefined) {
      logging::devicex::debug(
          "Adding post-turning check-point Op {} {} in pipelinedOpTaskFunc",
          op->str(),
          op->debugName());
      auto seqsKey =
          &progs.pipelineMainFragment(op->getPipelineStage(), op->str());
      logging::devicex::debug("Obtained pipeline forward frag for ",
                              op->debugName());
      logging::devicex::debug(
          "Growing {} {} in pipelinedOpTaskFunc", op->str(), op->debugName());

      growOpx(opx, seqs[seqsKey]);
    } else if (op->settings.recomputeType == RecomputeType::Recompute) {
      logging::devicex::debug("Adding (first) recompute Op {}",
                              op->debugName());

      growOpx(opx, progs.createRecomputeFragment(op->id));
      seqs.getSequence(
              &progs.pipelineMainFragment(op->getPipelineStage(), op->str()))
          .add(*progs.recomputeFragment(op->id));
    }
  }
}

unsigned IrLowering::getReplicationFactor() const {

  unsigned replicationFactor = 1;
  if (ir().getSessionOptions().enableReplicatedGraphs) {
    replicationFactor =
        static_cast<unsigned>(ir().getSessionOptions().replicatedGraphCount);
  }

  else {
    // A check on user input consistency
    if (static_cast<unsigned>(ir().getSessionOptions().replicatedGraphCount) >
        1) {
      throw error(
          "enableReplicatedGraphs is false, but replicatedGraphCount > 1. "
          "Either enable replication, or set the replicated graph count to 1");
    }
  }

  return replicationFactor;
}

unsigned IrLowering::getGlobalReplicationFactor() const {
  if (ir().getSessionOptions().enableDistributedReplicatedGraphs) {

    // This comes from outside this popart process because some procses might
    // have a different local replication factor than this process.
    return ir().getSessionOptions().globalReplicationFactor;
  }

  // When running without distributed graphs enabled, simply use
  // getReplicationFactor()
  return getReplicationFactor();
}

unsigned IrLowering::getGlobalReplicaOffset() const {

  unsigned globalReplicaOffset = 0;
  if (ir().getSessionOptions().enableDistributedReplicatedGraphs) {
    globalReplicaOffset =
        static_cast<unsigned>(ir().getSessionOptions().globalReplicaOffset);
  }

  return globalReplicaOffset;
}

bool IrLowering::isReplicatedGraph() const {
  const unsigned localReplicas   = getReplicationFactor();
  const bool isLocallyReplicated = localReplicas > 1;
  const bool isGloballyReplicated =
      getGlobalReplicationFactor() > localReplicas;
  return isLocallyReplicated || isGloballyReplicated;
}

unsigned IrLowering::getAccumulationFactor() const {
  return ir().getSessionOptions().getAccumulationFactor();
}

// Floating point settings are not supported on CPU
void IrLowering::setFloatingPointBehaviour(snap::Graph &graph) {

  if (ir().getSessionOptions().enableFloatingPointChecks) {
    if (deviceInfo->getType() == DeviceType::Ipu) {
      logging::devicex::info("Enabling all floating point checks");
      // Not enabling stochasitc rounding, that is done in a separate call
      poplar::FloatingPointBehaviour behaviour(true, true, true, false, true);
      poplar::setFloatingPointBehaviour(
          graph.getPoplarGraph(),
          progs.initFragment().getPoplarSequence(),
          behaviour,
          "/init");
    } else {
      logging::devicex::warn(
          "Floating point checks cannot be enabled for non IPU devices");
    }
  }
}

// Stochastic rounding is only supported on the IPU
void IrLowering::setStochasticRoundingBehaviour(snap::Graph &graph) {

  if (ir().getSessionOptions().enableStochasticRounding) {
    if (deviceInfo->getType() == DeviceType::Ipu) {
      logging::devicex::info("Enabling stochastic rounding");
      bool behaviour = true;
      poplar::setStochasticRounding(graph.getPoplarGraph(),
                                    progs.initFragment().getPoplarSequence(),
                                    behaviour,
                                    "/init");
    } else {
      logging::devicex::warn(
          "Stochastic rounding cannot be enabled for non IPU devices");
    }
  }
}

void IrLowering::prePlanConvolutions() {
  // Get a map of conv ops on each poplar graph
  std::vector<Op *> convOps;
  for (Op *op : ir().getAllOps()) {
    if (op->isConvertibleTo<MultiConvBaseOp>()) {
      convOps.push_back(op);
    } else if (op->isConvertibleTo<MultiConvWeightsGradBaseOp>()) {
      convOps.push_back(op);
    }
  }
  std::map<snap::Graph *, std::vector<Op *>> convGraphsOps;
  for (Op *convOp : convOps) {
    snap::Graph &graph = getOpx(convOp->id)->graph();
    auto it            = convGraphsOps.find(&graph);
    if (it == convGraphsOps.end()) {
      convGraphsOps.emplace(&graph, std::vector<Op *>{convOp});
    } else {
      convGraphsOps.at(&graph).push_back(convOp);
    }
  }

  if (convGraphsOps.size()) {
    logging::devicex::debug("Pre-planning convolutions");
  }

  for (const auto &graph_op : convGraphsOps) {
    snap::Graph *graph    = graph_op.first;
    std::vector<Op *> ops = graph_op.second;

    std::vector<poplin::ConvParams> allConvParams;
    std::vector<poplar::OptionFlags> allOptionFlags;

    for (Op *op : ops) {
      if (op->isConvertibleTo<MultiConvBaseOp>()) {
        auto convOp  = dynamic_cast<MultiConvBaseOp *>(op);
        auto convOpx = dynamic_cast<MultiConvBaseOpx *>(getOpx(op->id));
        for (int i = 0; i < convOp->numConvs(); i++) {
          allConvParams.push_back(
              getPoplarConvParams(convOp->getParameters(i)));
          allOptionFlags.push_back(convOpx->getConvOptions(i));
        }
      } else if (op->isConvertibleTo<MultiConvWeightsGradBaseOp>()) {
        auto convOp = dynamic_cast<MultiConvWeightsGradBaseOp *>(op);
        auto convOpx =
            dynamic_cast<MultiConvWeightsGradBaseOpx *>(getOpx(op->id));
        for (int i = 0; i < convOp->numConvs(); i++) {
          auto wuConvParams =
              getConvWeightUpdateParameters(convOp->getParameters(i));
          allConvParams.push_back(getPoplarConvParams(wuConvParams));
          allOptionFlags.push_back(convOpx->getConvOptions(i));
        }
      }
    }
    const poplar::Target target = graph->getPoplarGraph().getTarget();
    std::set<ConvPlanParams> allConvPlanParams;

    for (int i = 0; i < allConvParams.size(); i++) {
      ConvPlanParams convPlanParams =
          std::make_tuple(&target, allConvParams.at(i), &allOptionFlags.at(i));
      allConvPlanParams.insert(convPlanParams);
    }

    poplin::preplanConvolutions(
        graph->getPoplarGraph(), allConvPlanParams, dv_p->convCache);
  }
}

void IrLowering::prePlanMatMuls() {
  std::vector<Op *> matMulOps;

  // Get a map of matmul ops on each poplar graph
  for (Op *op : ir().getAllOps()) {
    if (op->isConvertibleTo<MatMulOp>()) {
      matMulOps.push_back(op);
    }
  }
  std::map<snap::Graph *, std::vector<Op *>> matMulGraphsOps;
  for (Op *matMulOp : matMulOps) {
    snap::Graph &graph = getOpx(matMulOp->id)->graph();
    auto it            = matMulGraphsOps.find(&graph);
    if (it == matMulGraphsOps.end()) {
      matMulGraphsOps.emplace(&graph, std::vector<Op *>{matMulOp});
    } else {
      matMulGraphsOps.at(&graph).push_back(matMulOp);
    }
  }

  if (matMulGraphsOps.size()) {
    logging::devicex::debug("Pre-planning matmuls");
  }

  for (const auto &graph_op : matMulGraphsOps) {
    snap::Graph *graph    = graph_op.first;
    std::vector<Op *> ops = graph_op.second;

    std::vector<poplin::MatMulParams> allMatMulParams;
    std::vector<poplar::OptionFlags> allOptionFlags;

    for (Op *op : matMulOps) {
      auto matMulOp  = dynamic_cast<MatMulOp *>(op);
      auto matMulOpx = dynamic_cast<MatMulOpx *>(getOpx(op->id));

      poplin::MatMulParams matMulParams;
      // The input tensors to the matmul opx are reshaped before being passed
      // to the poplibs matmul call. These tensors don't exist at this point,
      // so we create a dummy graph, and perform these same transformations on
      // dummy input tensors, in order to get the correct input shapes from
      // which to generate the MatMulParams.
      // TODO: T31134 we could avoid this by moving the reshaping of inputs into
      // the IR.
      snap::Graph dummyGraph(graph->getPoplarGraph().getTarget());
      auto inputType = popType(matMulOp->lhsIn()->info.dataType());
      auto dummyLhs =
          snap::Tensor{dummyGraph.getPoplarGraph().addVariable(
                           inputType, matMulOp->lhsIn()->info.shape_szt()),
                       dummyGraph};
      auto dummyRhs =
          snap::Tensor{dummyGraph.getPoplarGraph().addVariable(
                           inputType, matMulOp->rhsIn()->info.shape_szt()),
                       dummyGraph};

      auto inputs = MatMulOpx::groupedMatMulInputsFromOpxInputs(
          *matMulOp, dummyLhs, dummyRhs);

      matMulParams.inputType  = inputType;
      matMulParams.outputType = matMulOpx->getOutputType(inputs.first);
      matMulParams.aShape     = inputs.first.shape();
      matMulParams.bShape     = inputs.second.shape();
      allMatMulParams.push_back(matMulParams);

      auto opts = matmulOptions;
      MatMulOpx::appendPoplarOptionsForOp(*matMulOp, opts);
      allOptionFlags.push_back(opts);
    }

    const poplar::Target *target = &(graph->getPoplarGraph().getTarget());
    std::set<MatMulPlanParams> allMatMulPlanParams;

    for (int i = 0; i < allMatMulParams.size(); i++) {
      MatMulPlanParams matMulPlanParams =
          std::make_tuple(target, allMatMulParams.at(i), &allOptionFlags.at(i));
      allMatMulPlanParams.insert(matMulPlanParams);
    }

    poplin::preplanMatMuls(allMatMulPlanParams, dv_p->matmulCache);
  }
}

void IrLowering::prepareGraph() {
  POPART_TRACEPOINT();
  progressLogger.compilationStart();

  const auto prepareGraphTimer = ir().timePartitionLogger().scopedStopwatch(
      "Preparing poplar Graph (Ir lowering)");

  if (prepareGraphHasBeenCalled_) {
    logging::devicex::info("Poplar graph has already been prepared");
    return;
  }

  logging::devicex::info("Poplar version: {}", poplar::versionString());
  logging::devicex::info("Poplar release githash: {}", poplar::packageHash());

  auto popartPrintTensors = getPopartEnvVar("PRINT_TENSORS");
  if (popartPrintTensors && (*popartPrintTensors != "") != 0) {
    boost::split(
        printTensorIds, *popartPrintTensors, [](char c) { return c == ' '; });
    logging::devicex::debug("Printing tensors {}", printTensorIds);
  }

  initPoplarGraph();
  progs.initWithSnapGraph(graph());
  rngStateLowering = std::make_unique<RngStateLowering>(*this, graph());

  logging::devicex::info("Poplar graph initialised");

  {

    const auto addCodeletsTimer =
        ir().timePartitionLogger().scopedStopwatch("Adding codelets");

    popops::addCodelets(graph().getPoplarGraph());
    poplin::addCodelets(graph().getPoplarGraph());
    popnn::addCodelets(graph().getPoplarGraph());
    poprand::addCodelets(graph().getPoplarGraph());

    // Add custom codelets as per the user provided list of paths. Allow poplar
    // to infer the file type from the extension. Also feed through the compile
    // flags.
    for (auto codelet : ir().getSessionOptions().customCodelets) {
      logging::devicex::info("Adding codelet: {}", codelet);
      graph().getPoplarGraph().addCodelets(
          codelet,
          poplar::CodeletFileType::Auto,
          ir().getSessionOptions().customCodeletCompileFlags);
    }
  }

  setFloatingPointBehaviour(graph());
  setStochasticRoundingBehaviour(graph());

  // Calculate metadata gathered from the IR
  setMetadataFromIr();

  // A quick diagram of how things are linked together (arrows are
  // member associations):
  //
  // +--------------------------+     +------------------+
  // | SubgraphCopyingStrategy  +<----+ LivenessAnalyzer |
  // +--------------------------+     +------------------+
  //                                    ^
  //                                    |  +---------------------+
  //                                    +--+ AliasZeroCopy       |
  //                                    |  +---------------------+
  //                                    |
  //                                    |  +---------------------+
  //                                    +--+ SubgraphPartitioner |
  //                                       +---------------------+
  //
  // Where:
  //
  //  * LivenessAnalyzer uses SubgraphCopyingStrategy to determine where to
  //    place copies in the global schedule.
  //  * AliasZeroCopy uses the global schedule in LivenessAnalyzer to determine
  //    which copies can be avoided when calling subgraphs.
  //  * SubgraphPartitioner uses the global schedule in LivenessAnalyzer (and
  //    in particular the position of the copies in it) to determine how to
  //    partition subgraphs ready for lowering CallOp.

  auto copyingStrategy = ir().getSessionOptions().subgraphCopyingStrategy;
  switch (copyingStrategy) {
  case SubgraphCopyingStrategy::OnEnterAndExit: {
    logging::devicex::debug("Using 'OnEnterAndExit' subgraph IO copying "
                            "strategy");
    subgraphCopyingStrat =
        std::make_unique<liveness::OnEnterAndExitSubgraphCopyingStrategy>();
    break;
  }
  case SubgraphCopyingStrategy::JustInTime: {
    logging::devicex::debug("Using 'JustInTime' subgraph IO copying strategy");
    subgraphCopyingStrat =
        std::make_unique<liveness::JustInTimeSubgraphCopyingStrategy>();
    break;
  }
  default: {
    throw error("Invalid value for SubgraphCopyingStrategy ({})",
                static_cast<int>(copyingStrategy));
  }
  }

  // Initialize the liveness analyzer
  livenessAnalyzer = std::make_unique<liveness::LivenessAnalyzer>(
      &ir(), subgraphCopyingStrat.get());

  subgraphCopyingStrat->setIr(&ir());
  subgraphCopyingStrat->setLivenessAnalyzer(livenessAnalyzer.get());

  aliasZeroCopy =
      std::make_unique<liveness::AliasZeroCopy>(&ir(), livenessAnalyzer.get());

  subgraphCopyingStrat->apply();
  livenessAnalyzer->apply();

  if (ir().getSessionOptions().aliasZeroCopy) {
    aliasZeroCopy->apply();
  }

  subgraphPartitioner = std::make_unique<liveness::SubgraphPartitioner>();
  subgraphPartitioner->setIr(&ir());
  subgraphPartitioner->setLivenessAnalyzer(livenessAnalyzer.get());
  subgraphPartitioner->apply();

  if (ir().virtualGraphsEnabled()) {
    auto numIPUs     = graph().getPoplarGraph().getTarget().getNumIPUs();
    auto tilesPerIPU = graph().getPoplarGraph().getTarget().getTilesPerIPU();

    int numIOTiles = ir().getSessionOptions().numIOTiles;

    if (numIOTiles > 0) {

      if (numIOTiles < 32 || numIOTiles > 192 || (numIOTiles % 2 != 0)) {
        throw error(
            "{} is an invalid number of IO tiles. "
            "Number of IO tiles must be an even number in range [32, 192]",
            numIOTiles);
      }
      logging::devicex::info(
          "Reserving {} IO tiles for GCL collective operations on each IPU",
          numIOTiles);

      if (numIOTiles >= tilesPerIPU) {
        throw error("Tiles per IPU {} should exceed number of IO tiles {}.",
                    tilesPerIPU,
                    numIOTiles);
      }

      const auto ioTiles =
          gcl::perIPUTiles(graph().getPoplarGraph(), 0, numIOTiles, true, true);
      const auto computeTiles = gcl::perIPUTiles(graph().getPoplarGraph(),
                                                 numIOTiles,
                                                 tilesPerIPU - numIOTiles,
                                                 true,
                                                 true);

      for (VGraphId ipu = 0; ipu < numIPUs; ++ipu) {
        unsigned startTile = static_cast<unsigned>(ipu) * tilesPerIPU;
        unsigned endTile   = (static_cast<unsigned>(ipu) + 1) * tilesPerIPU;
        auto ipuGraph      = graph().createVirtualGraph(startTile, endTile);

        virtualGraphs.emplace_back(ipuGraph.createVirtualGraph(computeTiles),
                                   ipuGraph.createVirtualGraph(ioTiles));
        logging::devicex::info("Created virtual graph {} with {} tiles",
                               ipu,
                               tilesPerIPU - numIOTiles);
      }
    } else {
      for (VGraphId ipu = 0; ipu < numIPUs; ++ipu) {
        unsigned startTile = static_cast<unsigned>(ipu) * tilesPerIPU;
        unsigned endTile   = (static_cast<unsigned>(ipu) + 1) * tilesPerIPU;
        virtualGraphs.emplace_back(
            graph().createVirtualGraph(startTile, endTile));
        logging::devicex::info(
            "Created virtual graph {} from {} to {}", ipu, startTile, endTile);
      }
    }

    // Make sure that the virtual graph information is valid
    const auto schedule = ir().getOpSchedule({}, RequireOptimalSchedule::Yes);
    logging::devicex::debug(
        "Asserting that the virtual graph information is valid");
    for (Op *op : schedule) {
      if (op->hasVirtualGraphId()) {
        VGraphId index = op->getVirtualGraphId();
        if (index < 0 || index >= numIPUs) {
          throw error("{} has been assigned to an invalid virtual graph {}. "
                      "numIPUs = {}.",
                      op->debugName(),
                      index,
                      numIPUs);
        }
      }
    }
  } else {
    auto numIPUs = graph().getPoplarGraph().getTarget().getNumIPUs();
    if (numIPUs > 1 &&
        numIPUs != ir().getSessionOptions().replicatedGraphCount) {
      throw error("If virtual graphs are disabled, the replicated graph count "
                  "({}) needs to be equal to the number of IPUs ({})",
                  ir().getSessionOptions().replicatedGraphCount,
                  numIPUs);
    }
  }

  // Create a constant tensor which will be used if opxTrace is enabled
  if (opxTrace) {
    opxTraceTensor = getConst(graph(), poplar::HALF, {1}, 0, "traceTensor");
  }

  // create an Opx for every Op
  const auto schedule = ir().getOpSchedule({}, RequireOptimalSchedule::Yes);

  logging::devicex::info("Turning Ops into Opxes");
  for (Op *op : schedule) {
    logging::devicex::trace("Creating OPX for {}", op->debugName());
    opxs[op->id] = createOpx(op);
  }
  progressLogger.preplanningStart();
  // If the model contains convolutions or matmuls, generate plans for all of
  // them at the same time in advance of growing the ops. It saves time.
  if (dv_p->prePlanConvolutions) {
    const auto preplanTimer =
        ir().timePartitionLogger().scopedStopwatch("Convolution preplanning");
    prePlanConvolutions();
  }
  progressLogger.preplanningEnd();
  if (dv_p->prePlanMatMuls) {
    const auto preplanTimer =
        ir().timePartitionLogger().scopedStopwatch("Matmul preplanning");
    prePlanMatMuls();
  }

  PriTasks tasks;

  // weights and accl tensors (i.e. variables):
  // 1) make tensor,
  // THEN
  // 2) make stream from host,
  // 3) create write prog,
  // 4) make stream to host,
  // 5) create read prog.
  // OR
  // 2) set initial value (if using synthetic data).
  for (auto id : ir().getTensorIds(TensorType::Variable)) {
    Tensor *tensor = ir().getTensor(id);
    if (tensor->tensorLocationInfo.isRemote() || tensor->hasProducer()) {
      continue;
    }

    // 1
    tasks.add(initTensorTask(getInitTensorCreators(tensor)));

    if (!ir().streamingIsDisabledForTensor(id)) {
      // 2
      tasks.add(streamFromHostTask(tensor->id, {tensor}));
      // 3
      tasks.add(fromHostTask(tensor, progs.streamWeightsFromHostFragment()));
      // 4
      tasks.add(streamToHostTask(tensor->id, {tensor}, false));
      // 5
      tasks.add(toHostTask(
          tensor, progs.weightsToHostFragment(), ToHostStreamType::NonAnchor));
    } else {
      // 2
      tasks.add(setInitTensorValTask(tensor));
    }
  }

  // constants:
  // 1) make tensor,
  // 2) set initial value.
  for (auto id : ir().getTensorIds(TensorType::Const)) {
    logging::devicex::debug("Adding initTensorTask for Const {}", id);
    Tensor *tensor = ir().getTensor(id);
    // 1
    tasks.add(initTensorTask(getInitTensorCreators(tensor)));
    // 2
    tasks.add(setInitTensorValTask(tensor));
  }

  // Externally created outputs:
  // 1) ActGrad tensors
  for (Op *op : ir().getAllOps()) {
    if (dynamic_cast<SubgraphOp *>(op)) {
      // Separate procedure for subgraph output tensor initTensorTasks
      continue;
    }
    PopOpx *opx = getOpx(op->id);
    for (auto t_inds : op->output->indicesMap()) {
      if (opx->outputCreatedExternally(t_inds.second.front())) {
        logging::devicex::trace("Adding {} output initTensorTask for {}",
                                op->debugName(),
                                t_inds.first->id);
        tasks.add(initTensorTask(getInitTensorCreators(t_inds.first)));
      }
    }
  }

  // Host load tensor stream tasks.
  if (ir().getSessionOptions().useHostCopyOps) {
    for (auto idAndTensors : ir().getHostLoadTensors()) {
      logging::devicex::debug(
          "Adding streamFromHostTask for host load stream tensor id {}",
          idAndTensors.first);
      tasks.add(streamFromHostTask(idAndTensors.first, idAndTensors.second));
    }
  }

  // stream-to-device tensors :
  // 1) make tensor
  // THEN
  // 2) make stream
  // OR
  // 2) set initial value (if using synthetic data).
  for (auto id : ir().getTensorIds(TensorType::Stream)) {
    if (!tasks.contains(streamFromHostTaskId(id))) {
      Tensor *tensor = ir().getTensor(id);

      // 1
      tasks.add(initTensorTask(getInitTensorCreators(tensor)));
      logging::devicex::debug("Adding initTensorTask for Stream {}", id);

      // 2
      if (ir().useSyntheticData()) {
        tasks.add(setInitTensorValTask(tensor));
      } else {
        tasks.add(streamFromHostTask(tensor->id, {tensor}));
      }
    }
  }

  // Init the random seed
  if (RandomSetup::hasRandomSeed(ir()) and !ir().useSyntheticData()) {
    auto seedTen = ir().getTensor(RandomSetup::getStreamedSeedTensorId());
    tasks.add(fromHostTask(seedTen, progs.setRandomSeedFromHostFragment()));
    tasks.add(initRandomSeed());
  }

  if (ir().getSessionOptions().enableLoadAndOffloadRNGState) {
    tasks.add(rngStateLowering->initRngStateTensor());
    tasks.add(rngStateLowering->rngStateFromHost());
    tasks.add(rngStateLowering->rngStateToHost());
  }

  // Depending on anchor return types specified by the user, some
  // tensors may need to be added to the graph to keep track of
  // batch count.
  if (ir().getDataFlow().isBatchCountingRequired()) {
    tasks.add(initBatchCounterTensorsTask(progs.initFragment()));
    tasks.add(updateBatchCountTask(progs.preForwardFragment()));
  }

  // stream-to-host tensors : 1) make streams 2) make copy programs
  // note that the order in which tasks are added does not matter,
  // they will be topologically sorted before running
  if (!ir().useSyntheticData() && !ir().getSessionOptions().useHostCopyOps) {
    for (auto anchorId : ir().getRootAnchors()) {
      Tensor *tensor = ir().getTensor(anchorId);

      tasks.add(streamToHostTask(tensor->id, {tensor}, true));

      // 2
      switch (ir().getDataFlow().art(anchorId).id()) {
      // Copy program runs after every batch
      case (AnchorReturnTypeId::All): {
        tasks.add(toHostTask(tensor,
                             getAnchorReturnFragment(tensor),
                             ToHostStreamType::NonSumAnchor));
        break;
      }
      // Copy program runs at the end of every N batches
      case (AnchorReturnTypeId::EveryN): {
        if (ir().getSessionOptions().enablePipelining) {
          throw error(
              "AnchorReturnType::EVERYN is not valid for pipelined models");
        } else {
          tasks.add(toHostEveryNBatchesTask(
              tensor,
              ir().getDataFlow().art(anchorId).rp(),
              tensor->tensorType() == TensorType::Variable
                  ? progs.backwardFragment()
                  : progs.forwardOrBackwardFragment(tensor->scheduledPreLoss)));
        }
        break;
      }
      // Copy program runs at the end of the step
      case (AnchorReturnTypeId::Final): {
        tasks.add(toHostTask(tensor,
                             progs.toHostFinalCopyFragment(),
                             ToHostStreamType::NonSumAnchor));
        break;
      }
      case (AnchorReturnTypeId::Sum): {
        tasks.add(
            anchorReturnTypeSumTask(tensor, getAnchorReturnFragment(tensor)));
        tasks.add(toHostTask(tensor,
                             progs.toHostFinalCopyFragment(),
                             ToHostStreamType::SumAnchor));
        break;
      }
      }
    }

    for (Tensor *tensor : ir().dataStreamTensors()) {
      if (ir().getSessionOptions().implicitPipeliningEnabled()) {
        PipelineStage ps = *tensor->consumers.findLowestPipelineStage();
        auto &sq = progs.pipelineToDeviceStreamFragment(ps, tensor->str());
        tasks.add(fromHostTask(tensor, sq));
      } else {
        auto &sq = progs.forwardOrBackwardFragment(tensor->scheduledPreLoss);
        tasks.add(fromHostTask(tensor, sq));
      }
    }
  } else if (ir().getSessionOptions().useHostCopyOps) {
    for (auto idAndTensors : ir().getHostStoreTensors()) {
      logging::devicex::debug(
          "Adding streamToHostTask for host load stream tensor id {}",
          idAndTensors.first);
      tasks.add(
          streamToHostTask(idAndTensors.first, idAndTensors.second, true));
    }
  }

  if (!ir().useSyntheticData()) {
    // create Program to write optimizer tensors to device
    for (auto tensor : ir().optimizerTensors()) {
      tasks.add(fromHostTask(tensor, progs.streamOptimizerFromHostFragment()));
    }
  }

  addOpTasks(tasks);

  if (ir().getSessionOptions().implicitPipeliningEnabled()) {
    addPipelinedCopyTasks(tasks);
  }

  // Two-step task linearisation:
  //
  // Purpose: Avoids circular dependencies when growing ops and initialising
  //          tensors with their tile mapping (layout) by deferring the
  //          growOpx call on Ops that call subgraphs as long as possible.
  //
  // Example:
  //
  // Graph A (nested subgraph):
  //        Data path x:    Data path y:
  //              InitOp    InitOp
  //                |         |
  //         weight_init    bias_init
  //                |         |
  //           RemoteLoad    RemoteLoad
  //                |         |
  //       weight_loaded    bias_loaded
  //      (graph output)    (graph output)
  //
  // Graph B (subgraph):
  //
  //         data    Call(A)
  //          |      |     |
  //          |   weight  bias
  //          |      |     |
  //         Convolution   |
  //                 |     |
  //            conv_out   |
  //                 |     |
  //                 AddBias
  //                    |
  //                   out
  //
  // Graph C (main graph):
  //              in0       in1
  //               |         |
  //             Call(B)     |
  //               |      Call(B)
  //               |         |
  //              out0      out1
  //
  // With one-step task linearisation, a circular dependency would arise
  // because the tile mapping of "bias" depends on "conv_out" existing, since
  // AddBias will clone the tile mapping of "conv_out" to "bias".
  // To create "conv_out", however, "weight" needs to have a tile mapping,
  // created by "Convolution". But "weight" is a copy of the "weight_loaded"
  // output of subgraph "A". That means "weight_init" is the tensor to which
  // the tensor layout of "Convolution" will be unwound to.
  // However, Call(A) is the producer of both "weight" and "bias", and would
  // therefore have to be grown before "Convolution", causing a circular
  // dependency:
  // Call(A) > Convolution > AddBias
  // (due to Opx output directed acyclic graph)
  // Convolution > AddBias > Call(A)
  // (due to AddBias creating the tensor mapping for "bias" based on "conv_out")
  //
  // Two-step linearisation separates growing Opx and creating Poplar
  // tensors, from adding the Poplar sequences, resulting from growing Opx,
  // together to form a linear order, in which the Opx are executed.
  //
  // With two-step task linearisation, there are multiple types of dependencies:
  // Call(A) depends on all Op tasks within graph A (SUBGRAPH dependency)
  // Call(B) depends on all Op tasks within graph B (SUBGRAPH dependency)
  // Convolution depends on "data" and "weight" (OUTPUT or TENSOR dependency,
  //       depending on the creator and populator of those tensors)
  // Convolution depends on Call(A) (SCHEDULE dependency)
  //
  // Step 1:
  // For growing Opx, we can ignore SCHEDULE dependencies, because data paths
  // flowing through subgraphs (from A to B in the example above):
  //
  // InitOp->weight_init->RemoteLoad->weight_loaded->weight->Convolution->...
  //
  // can be grown independently from growing Call(A), thereby removing the
  // circular dependency.
  //
  // The linearized order in which Opx are grown for the example above becomes:
  // InitOp(x) > RemoteLoad(x) > Convolution > InitOp(y) > RemoteLoad(y) >
  // AddBias > Call(A) > Call(B) > Call(B)
  //
  // Step 2:
  // As soon as all data paths in a given graph (here A, B or C) are grown,
  // so the SUBGRAPH dependencies for CallOpx are fulfilled,
  // and we have one or more Poplar sequences for every opTask in the graph.
  // The Poplar sequences can be emplaced in graph functions (for subgraphs)
  // or the main sequence (main graph) according to SCHEDULER dependencies
  // (following the IR scheduler order).
  // All TENSOR dependencies can be ignored for emplacing sequences, since they
  // are only necessary to create the weight tensors (InitOp outputs) in step 1.
  //
  // The linearized order in which sequences are emplaced for the example above
  // becomes:
  // InitOp(x) > RemoteLoad(x) > InitOp(y) > RemoteLoad(y) > Call(A)
  // > Convolution > AddBias > Call(B) > Call(B)

  // Mappings for each task from final sequence to intermediate sequence
  std::map<TaskId, SequenceMap> seqs;
  std::vector<TaskId> taskOrder;

  logging::devicex::debug("Creating linear task schedule with OUTPUT, "
                          "SUBGRAPH and TENSOR dependencies.");
  auto createSchedule = tasks.getLinearised({DependencyType::Output,
                                             DependencyType::SubGraph,
                                             DependencyType::Tensor},
                                            *this,
                                            true);

  logging::devicex::debug("Creating linear task schedule with OUTPUT, "
                          "SUBGRAPH and SCHEDULER dependencies.");
  auto emplaceSchedule = tasks.getLinearised({DependencyType::Output,
                                              DependencyType::SubGraph,
                                              DependencyType::Scheduler},
                                             *this,
                                             false);

  auto emplaceTaskSeqs = [&](std::set<TaskId> filter) {
    // 2.) Add intermediate sequences in final sequence
    // Linearised, ignoring TENSOR creation dependencies (weight init deps)
    // because all tensors exist at this point

    for (auto &emplaceTask : emplaceSchedule) {
      if ((filter.size() == 0 ||
           filter.find(emplaceTask.name) != filter.end()) &&
          seqs.find(emplaceTask.name) != seqs.end()) {
        logging::devicex::trace("Adding sequences for task {}",
                                emplaceTask.name);
        auto &sequenceMap = seqs.at(emplaceTask.name);
        for (auto seq : sequenceMap.getFullSequenceMap()) {
          // Emplace intermediate sequence in final sequence
          seq.first->add(*seq.second);
        }
        // Erase sequences for task, so that each tasks's sequences
        // are only added once.
        seqs.erase(emplaceTask.name);
        taskOrder.push_back(emplaceTask.name);
      }
    }
  };

  // 1.) Create sequences and tensors
  // Linearised, ignoring SCHEDULER order dependencies, so that any circular
  // dependencies are avoided, when the scheduler order disagrees with the
  // tensor creation order

  int currentTask = 0;
  for (auto &createTask : createSchedule) {
    progressLogger.creatingSequence(currentTask++, createSchedule.size());

    logging::devicex::debug("Creating sequence for task {}", createTask.name);
    std::set<TaskId> subgraphTaskNames =
        createTask.getDependenciesOfTypes({DependencyType::SubGraph});
    if (subgraphTaskNames.size() > 0) {
      // Make sure the subgraph sequences are emplaced before the scopeFragments
      // of the called graphs are constructed.
      logging::devicex::trace("  Task depends on {} subgraph tasks",
                              subgraphTaskNames.size());
      emplaceTaskSeqs(subgraphTaskNames);
    }
    seqs.insert({createTask.name, createTask.f()});
  }
  // Emplace any main graph task sequences
  emplaceTaskSeqs({});

  verifyTaskOrder(taskOrder);

  // Log the order of tasks and associated ops for each execution context
  logging::devicex::debug(
      getContextOpString(ExecutionContext::WeightsFromHostFragment, taskOrder));
  logging::devicex::debug(
      getContextOpString(ExecutionContext::Normal, taskOrder));
  logging::devicex::debug(
      getContextOpString(ExecutionContext::AccumulateOuterFragment, taskOrder));
  logging::devicex::debug(
      getContextOpString(ExecutionContext::WeightsToHostFragment, taskOrder));
  logging::devicex::debug(getContextOpString(
      ExecutionContext::OptimizerFromHostFragment, taskOrder));

  if (ir().getSessionOptions().exportPoplarVertexGraph) {
    std::ofstream strm;
    strm.open("poplar_vertex_graph.dot", std::ios::out);
    graph().getPoplarGraph().outputVertexGraph(strm,
                                               toPoplarProgs(progs.progs()));
  }

  if (ir().getSessionOptions().exportPoplarComputationGraph) {
    std::ofstream strm;
    strm.open("poplar_compute_graph.dot", std::ios::out);
    graph().getPoplarGraph().outputComputeGraph(strm,
                                                toPoplarProgs(progs.progs()));
  }

  prepareGraphHasBeenCalled_ = true;
}

snap::program::Sequence &IrLowering::getAnchorReturnFragment(Tensor *tensor) {
  if (ir().getSessionOptions().implicitPipeliningEnabled()) {
    auto isOptimizerTensorCopy = [&](Op *x) {
      return x->isConvertibleTo<IpuCopyOp>() &&
             dynamic_cast<IpuCopyOp *>(x)->copiesOptimizerTensors();
    };

    if (tensor->hasProducer() &&
        tensor->getProducer()->settings.executionContext ==
            ExecutionContext::AccumulateOuterFragment) {
      return progs.accumulateOuterFragment();
    } else {
      PipelineStage ps;
      // Copies of optimizer tensors do not have a pipeline stage.
      if (tensor->hasProducer() &&
          !isOptimizerTensorCopy(tensor->getProducer())) {
        ps = tensor->getProducer()->getPipelineStage();
      } else if (tensor->tensorType() == TensorType::Stream) {
        ps = *tensor->consumers.findLowestPipelineStage();
      } else if (tensor->tensorType() == TensorType::Variable) {
        ps = *tensor->consumers.findHighestPipelineStage();
      }
      // Edge cases where we have a const or optimizer tensor.
      else {
        ps = *tensor->consumers.findHighestPipelineStage();
      }
      return progs.pipelineToHostStreamFragment(ps, tensor->str());
    }
  } else {
    return (tensor->tensorType() == TensorType::Variable ||
            !tensor->hasProducer())
               ? progs.backwardFragment()
               : progs.forwardOrBackwardFragment(tensor->scheduledPreLoss);
  }
}

std::string IrLowering::getPoplarGraphDebugName() {
  // Will return the user provided session name.
  return ir().getSessionName();
}

poplar::Executable IrLowering::getExecutable() {
  if (!prepareGraphHasBeenCalled_) {
    throw internal_error("IrLowering::prepareGraph() must be called before"
                         " IrLowering::getExecutable() is called.");
  }

  if (cachedExecutable) {
    // return the executable in cachedExecutable while ensuring
    // cachedExecutable is set to nonstd::nullopt
    nonstd::optional<poplar::Executable> result = nonstd::nullopt;
    boost::swap(cachedExecutable, result);
    logging::devicex::info("Returning CachedExecutable");

    progressLogger.complete();
    return std::move(result.value());
  } else {
    try {
      logging::devicex::info("Starting compilation");

      auto executable = poplar::compileGraph(graph().getPoplarGraph(),
                                             toPoplarProgs(progs.progs()),
                                             engineOptions,
                                             std::ref(progressLogger),
                                             getPoplarGraphDebugName());

      logging::devicex::info("Graph compiled");
      return executable;
    } catch (const poplar::graph_memory_allocation_error &e) {
      // If the compilations throws an exception due to memory
      // allocation i.e. the program does not fit show graph profile and
      // re-throw the exception In certain cases poplar will throw the error
      // without a graph profile. The following engine option needs to be set to
      // enable the graph profile in this case "debug.allowOutOfMemory":"true"
      progressLogger.complete();
      logging::devicex::err("Memory allocation error : {}", e.what());
      throw devicex_memory_allocation_err(e, reportOptions);
    }
  }
}

void IrLowering::loadPoplarExecutable(std::istream &in) {
  POPART_TRACEPOINT();

  cachedExecutable.emplace(
      popx::serialization::deserializePoplarExecutable(in));
  usingCachedExecutable_ = true;
}

TaskId IrLowering::streamFromHostTaskId(TensorId id) {
  return TaskId(TaskId::Type::StreamFromHostTask, id);
}

TaskId IrLowering::setInitTensorValTaskId(TensorId id) {
  return TaskId(TaskId::Type::SetInitTensorValTask, id);
}

TaskId IrLowering::streamToHostTaskId(TensorId id, bool isAnchorStream) {
  if (isAnchorStream) {
    return TaskId(TaskId::Type::AnchorStreamToHostTask, id);
  } else {
    return TaskId(TaskId::Type::WeightStreamToHostTask, id);
  }
}

TaskId IrLowering::fromHostTaskId(TensorId id) {
  return TaskId(TaskId::Type::FromHostTask, id);
}

TaskId IrLowering::toHostTaskId(TensorId id, bool isAnchorStream) {
  if (isAnchorStream) {
    return TaskId(TaskId::Type::AnchorToHostTask, id);
  } else {
    return TaskId(TaskId::Type::WeightToHostTask, id);
  }
}

TaskId IrLowering::anchorSumTaskId(const TensorId &id) {
  return TaskId(TaskId::Type::AnchorSumTask, id);
}

TaskId IrLowering::initBatchCounterTensorsTaskId() {
  return TaskId(TaskId::Type::InitBatchCounterTensorsTask);
}

TaskId IrLowering::updateBatchCountTaskId() {
  return TaskId(TaskId::Type::UpdateBatchCountTask);
}

TaskId IrLowering::initRandomSeedTaskId() {
  return TaskId(TaskId::Type::InitRandomSeedTask);
}

TaskId IrLowering::initTensorTaskId(TensorId id) {
  return TaskId(TaskId::Type::InitTensorTask, id);
}

TaskId IrLowering::opTaskId(Op *op) const {
  return TaskId(TaskId::Type::FromOpTask, op->id, op->opid);
}

TaskId IrLowering::opTensorTaskId(Op *op, Tensor *tensor) const {
  auto opx        = getOpx(op->id);
  auto growPartId = opx->getOutGrowPartId(tensor);
  if (growPartId != unusedGrowPartId) {
    return TaskId(TaskId::Type::FromOpTask, op->id, op->opid, growPartId);
  } else {
    return opTaskId(op);
  }
}

TaskId IrLowering::opPartTaskId(Op *op, OpxGrowPartId growPartId) const {
  if (growPartId != unusedGrowPartId) {
    return TaskId(TaskId::Type::FromOpTask, op->id, op->opid, growPartId);
  } else {
    return opTaskId(op);
  }
}

TaskId IrLowering::pipelinedCopyTaskId(Op *op) {
  return TaskId(TaskId::Type::PipelinedCopyTask, op->id, op->opid);
}

PriTask IrLowering::fromHostTask(Tensor *tensor, snap::program::Sequence &sq) {
  double priority;
  if (ir().getSessionOptions().groupHostSync) {
    priority = std::numeric_limits<double>::max();
  } else {
    priority = -1e6; // writes to device: always as late as possible (default)
  }
  auto f = [&sq, tensor, this]() {
    SequenceMap seqs(graph());
    logging::devicex::debug("Adding poplar::program::Copy from host " +
                            tensor->id);

    if (tensors_.hasViewChangers(tensor->id)) {
      if (tensors_.get(tensor->id).numElements() >
          tensors_.getView(tensor->id).numElements()) {
        // The view is not covering the whole tensor, therefore it is necessary
        // to zero-init it
        popops::zero(graph().getPoplarGraph(),
                     tensors_.get(tensor->id).getPoplarTensor(),
                     seqs.getSequence(&sq).getPoplarSequence(),
                     {"copyFromHost"});
      }
    }

    seqs.getSequence(&sq).getPoplarSequence().add(
        // Tensors with views: Use the view instead, so that e.g.
        // replicated tensor sharding padding is ignored
        poplar::program::Copy(fromHostStreams.at(tensor->id),
                              tensors_.getView(tensor->id).getPoplarTensor(),
                              doRearrangeOnHost(tensor),
                              {"copyFromHost"}));
    return seqs;
  };
  return {priority,
          fromHostTaskId(tensor->id),
          {
              {streamFromHostTaskId(tensor->id),
               DependencyType::Tensor}, // poplar::Stream created
              {initTensorTaskId(tensor->id),
               DependencyType::Tensor} // snap::Tensor created
          },
          f};
}

PriTask IrLowering::toHostTask(Tensor *tensor,
                               snap::program::Sequence &sq,
                               ToHostStreamType stype) const {

  auto f = [&sq, tensor, this, stype]() {
    // Have to use pGraph instead of IrLowering::graph() because we need a
    // non-const snap::Graph.
    if (pGraph == nullptr) {
      throw error("snap::Graph is null");
    }
    SequenceMap seqs(*pGraph);
    logging::devicex::debug("Adding poplar::program::Copy to host "
                            "(Type: {}) {}",
                            static_cast<int>(stype),
                            tensor->id);

    auto pToHostStreams = &toHostAnchorStreams;
    if (stype == ToHostStreamType::NonAnchor) {
      pToHostStreams = &toHostWeightStreams;
    }
    const auto &poplarStream = pToHostStreams->at(tensor->id);
    // Tensors with views: Use the view instead, so that e.g.
    // replicated tensor sharding padding is ignored
    const auto &anchorTensor =
        stype == ToHostStreamType::SumAnchor
            ? tensors_.getView(anchorSumPrefix() + tensor->id)
            : tensors_.getView(tensor->id);
    // verify that number of elements of poplar Tensor and poplar Stream are the
    // same
    auto nElmsStream = poplarStream.numElements();
    auto nElmsTensor = anchorTensor.numElements();
    if (nElmsStream != nElmsTensor) {
      throw internal_error("[Devicex::toHostTask] "
                           "The snap::Tensor {} has {}, whereas the "
                           "poplar::Stream has {}. These should be the same.",
                           tensor->id,
                           nElmsTensor,
                           nElmsStream);
    }

    seqs.getSequence(&sq).getPoplarSequence().add(
        poplar::program::Copy(anchorTensor.getPoplarTensor(),
                              poplarStream,
                              doRearrangeOnHost(tensor),
                              {"copyToHost"}));
    return seqs;
  };

  auto finalPopulator = taskWhichPopulates(tensor->id);
  if (stype != ToHostStreamType::NonAnchor &&
      tensor->tensorType() == TensorType::Variable) {
    for (auto op : tensor->consumers.getOps()) {
      if (dynamic_cast<VarUpdateOp *>(op)) {
        finalPopulator = opTaskId(op);
      }
    }
  }

  auto taskId = toHostTaskId(tensor->id, stype != ToHostStreamType::NonAnchor);

  logging::devicex::debug(
      "Final populator for {} is {} ", taskId, finalPopulator);

  std::vector<PriTaskDependency> deps = {
      // the dependencies:
      // snap::Tensor exists
      taskWhichCreates(tensor->id),
      // poplar::Stream creation task,
      {streamToHostTaskId(tensor->id, stype != ToHostStreamType::NonAnchor),
       DependencyType::Output},
      // snap::Tensor has its final values
      {finalPopulator, DependencyType::Scheduler}};

  if (stype == ToHostStreamType::SumAnchor) {
    deps.push_back({anchorSumTaskId(tensor->id), DependencyType::Tensor});
  }

  deps.push_back(taskWhichCreates(tensor->id));

  double priority;
  if (ir().getSessionOptions().groupHostSync) {
    priority = -std::numeric_limits<double>::max();
  } else {
    priority = +1e6; // writes to host: always as soon as possible (default)
  }
  return {priority, taskId, deps, f};
}

PriTask IrLowering::anchorReturnTypeSumTask(Tensor *tensor,
                                            snap::program::Sequence &sq) {
  auto f = [&sq, tensor, this]() {
    SequenceMap seqs(graph());

    const auto &poplarTensor     = tensors_.get(tensor->id);
    const TensorId accumulatorId = anchorSumPrefix() + tensor->id;
    auto accumulatorTensor =
        snap::Tensor{graph().getPoplarGraph().clone(
                         poplarTensor.getPoplarTensor(), accumulatorId),
                     graph()};
    tensors_.insertUnsafe(accumulatorId, accumulatorTensor);

    logging::devicex::debug("Adding AnchorSum operations to {}", tensor->id);
    popops::addInPlace(graph().getPoplarGraph(),
                       accumulatorTensor.getPoplarTensor(),
                       poplarTensor.getPoplarTensor(),
                       seqs.getSequence(&sq).getPoplarSequence(),
                       "AnchorSum_" + tensor->id);
    // Zero the accumulator
    popops::zero(graph().getPoplarGraph(),
                 accumulatorTensor.getPoplarTensor(),
                 seqs.getSequence(&progs.initFragment()).getPoplarSequence(),
                 "AnchorSumZero_" + tensor->id);

    return seqs;
  };

  auto finalPopulator = taskWhichPopulates(tensor->id);
  if (tensor->tensorType() == TensorType::Variable) {
    for (auto op : tensor->consumers.getOps()) {
      if (dynamic_cast<VarUpdateOp *>(op)) {
        finalPopulator = opTaskId(op);
      }
    }
  }
  auto taskId = anchorSumTaskId(tensor->id);
  return {+1e7, // Increments before any other host streams
          taskId,
          {// the dependencies:
           // snap::Tensor exists
           taskWhichCreates(tensor->id),
           // poplar::Stream creation task,
           {streamToHostTaskId(tensor->id, true), DependencyType::Output},
           // snap::Tensor has its final values
           {finalPopulator, DependencyType::Scheduler}},
          f};
}

PriTask IrLowering::initBatchCounterTensorsTask(snap::program::Sequence &sq) {

  auto f = [&sq, this]() {
    logging::devicex::debug("Adding batch counter tensors");

    snap::Tensor falseConst = getConst(graph(), poplar::BOOL, {}, 0, "false");

    // Add scalar tensors outside of the ir to track the batch
    // Id and decide when to execute the copy to the host
    for (ReturnPeriod N : ir().getDataFlow().rps()) {
      // Add to map so copy task can access
      batchCountingTensors[N] = getScalarVariable(graph(), poplar::INT, "");
      batchCountCheckingTensors[N] =
          getScalarVariable(graph(), poplar::BOOL, "");

      getConst(graph(), poplar::INT, {}, N, "batchCounter");

      poputil::mapTensorLinearly(graph().getPoplarGraph(),
                                 batchCountingTensors[N].getPoplarTensor());
      poputil::mapTensorLinearly(
          graph().getPoplarGraph(),
          batchCountCheckingTensors[N].getPoplarTensor());

      // Set the initial values of the tensors_.
      popops::zero(graph().getPoplarGraph(),
                   batchCountingTensors[N].getPoplarTensor(),
                   sq.getPoplarSequence(),
                   logging::format("initBatchCountTensors[{}]", N));
      sq.add(snap::program::Copy(
          falseConst, batchCountCheckingTensors[N], false, {"copyFalse"}));
    }

    // Make sure const 1 tensor exists
    getConst(graph(), poplar::INT, {}, 1, "one");
    return SequenceMap(graph());
  };

  return {+1e6, // followed by writes to host: always as early as possible
          initBatchCounterTensorsTaskId(),
          {},
          f};
}

PriTask IrLowering::updateBatchCountTask(snap::program::Sequence &sq) {

  auto f = [&sq, this]() {
    SequenceMap seqs(graph());
    logging::devicex::debug("Adding batch count checker program");

    // Placeholder 'do nothing' branch if not running assign program
    snap::program::Sequence emptyseq(poplar::DebugContext{"empty"}, graph());

    // Increment the batch count at the at the earliest point
    // the anchor tensor is required, and check if it is a
    // copy batch
    for (ReturnPeriod N : ir().getDataFlow().rps()) {
      popops::addInPlace(graph().getPoplarGraph(),
                         batchCountingTensors[N].getPoplarTensor(),
                         getConst(graph(), poplar::INT, {}, 1, "batchCount/one")
                             .getPoplarTensor(),
                         seqs.getSequence(&sq).getPoplarSequence());

      batchCountCheckingTensors[N] = snap::Tensor{
          popops::eq(graph().getPoplarGraph(),
                     batchCountingTensors[N].getPoplarTensor(),
                     getConst(graph(), poplar::INT, {}, N, "batchCount/n")
                         .getPoplarTensor(),
                     seqs.getSequence(&sq).getPoplarSequence()),
          graph()};

      // Reset batch count once it has reached N
      auto zero = getConst(graph(), poplar::INT, {}, 0, "batchCount/zero");
      snap::program::Copy trueBody(
          zero, batchCountingTensors[N], false, {"copyZero"});
      seqs.getSequence(&sq).getPoplarSequence().add(
          poplar::program::If(batchCountCheckingTensors[N].getPoplarTensor(),
                              trueBody.getPoplarProgram(),
                              emptyseq.getPoplarSequence(),
                              {"batchCountResetCheck"}));
    }
    return seqs;
  };

  return {+1e6, // followed by writes to host: always as early as possible
          updateBatchCountTaskId(),
          {
              {initBatchCounterTensorsTaskId(),
               DependencyType::Tensor} // snap::Tensor creation task
          },
          f};
}

std::map<PipelineStage, VGraphId> IrLowering::getPipelineToVGraphIdMap() const {
  // Create a map of pipeline stage to virtual graph ids
  std::map<PipelineStage, VGraphId> pipeline_vgraph_map;
  for (auto &id_op : ir().getMainGraph().getOps()) {
    auto op = id_op.second.get();

    // Not sure why, but in test
    // 'pipeline_test.py::test_acts_match_restored_acts', an IpuCopy did not
    // have a virtual graph id set
    if (!op->isConvertibleTo<IpuCopyOp>()) {
      auto ps       = op->getPipelineStage();
      auto vgraphid = op->getVirtualGraphId();

      pipeline_vgraph_map[ps] = vgraphid;
    }
  }

  std::stringstream ss;
  ss << "Pipeline stages running on virtual graphs:";
  for (auto &ps_vgraph : pipeline_vgraph_map) {
    auto ps       = ps_vgraph.first;
    auto vgraphid = ps_vgraph.second;
    ss << logging::format("\n  ps {} on virtual graph {}", ps, vgraphid);
  }
  logging::devicex::debug(ss.str());

  return pipeline_vgraph_map;
}

PriTask IrLowering::toHostEveryNBatchesTask(Tensor *tensor,
                                            int N,
                                            snap::program::Sequence &sq) {

  auto f = [&sq, tensor, N, this]() {
    SequenceMap seqs(graph());
    logging::devicex::debug(
        "Adding conditional poplar::program::Copy to host " + tensor->id);

    snap::Tensor isNthBatch = batchCountCheckingTensors.at(N);

    snap::program::Sequence copyseq(poplar::DebugContext{"copy"}, graph());
    copyseq.getPoplarSequence().add(
        poplar::program::Copy(tensors_.get(tensor->id).getPoplarTensor(),
                              toHostAnchorStreams.at(tensor->id),
                              doRearrangeOnHost(tensor),
                              {"copyToHostEveryNBatches"}));

    // Placeholder 'do nothing' branch if not running copy program
    snap::program::Sequence emptyseq(poplar::DebugContext{"empty"}, graph());

    seqs.getSequence(&sq).getPoplarSequence().add(
        poplar::program::If(isNthBatch.getPoplarTensor(),
                            copyseq.getPoplarSequence(),
                            emptyseq.getPoplarSequence(),
                            {"nthBatchCheck"}));
    return seqs;
  };

  bool isAnchorStream = true;
  return {
      +1e6, // writes to host: always as early as possible
      toHostTaskId(tensor->id, isAnchorStream),
      {// the dependencies:
       // snap::Tensor needs to exist
       taskWhichCreates(tensor->id),
       // updating snap::Tensor task,
       {updateBatchCountTaskId(), DependencyType::Output},
       // poplar::Stream creation task,
       {streamToHostTaskId(tensor->id, isAnchorStream), DependencyType::Output},
       // snap::Tensor value setting task
       {taskWhichPopulates(tensor->id), DependencyType::Scheduler}},
      f};
}

bool IrLowering::doRearrangeOnHost(Tensor *tensor) const {
  if (tensor->tensorType() == TensorType::Variable) {
    return true;
  } else if (tensor->tensorType() == TensorType::Stream) {
    return ir().getSessionOptions().rearrangeStreamsOnHost;
  } else if (ir().isAnchored(tensor->id)) {
    auto art = ir().getDataFlow().getAnchorReturnTypeMap().at(
        ir().getAnchorRemap().getRight(tensor->id));
    return ir().getSessionOptions().rearrangeAnchorsOnHost;
  } else if (ir().isRootAnchor(tensor->id)) {
    auto art = ir().getDataFlow().getAnchorReturnTypeMap().at(tensor->id);
    return ir().getSessionOptions().rearrangeAnchorsOnHost;
  }
  return true;
}

poplar::ReplicatedStreamMode
IrLowering::getReplicatedStreamMode(Tensor *tensor) const {
  poplar::ReplicatedStreamMode mode = poplar::ReplicatedStreamMode::BROADCAST;

  if (tensor->tensorType() == TensorType::Variable) {
    // If returned != 1 then streaming will have to handle different
    // replicas more dynamically, and broadcast will not work for the
    // necessary transfers.
    auto replicas   = ir().getSessionOptions().replicatedGraphCount;
    auto groupCount = tensor->getVariableSettings().groupCount(replicas);

    mode = groupCount != 1 ? poplar::ReplicatedStreamMode::REPLICATE
                           : poplar::ReplicatedStreamMode::BROADCAST;
  } else if (tensor->tensorType() == TensorType::Variable) {
    // If it is a variable we 'broadcast' the same tensor
    // to all replicants
    mode = poplar::ReplicatedStreamMode::BROADCAST;

  } else {
    if (ir().getSessionOptions().useHostCopyOps) {
      switch (tensor->getReplicatedStreamMode()) {
      case ReplicatedStreamMode::Broadcast:
        mode = poplar::ReplicatedStreamMode::BROADCAST;
        break;
      case ReplicatedStreamMode::Replicate:
        mode = poplar::ReplicatedStreamMode::REPLICATE;
        break;
      }
      return mode;
    }

    if (tensor->tensorType() == TensorType::Stream) {

      switch (tensor->getReplicatedStreamMode()) {
      case ReplicatedStreamMode::Broadcast:
        mode = poplar::ReplicatedStreamMode::BROADCAST;
        break;
      case ReplicatedStreamMode::Replicate:
        mode = poplar::ReplicatedStreamMode::REPLICATE;
        break;
      }

    } else {
      throw error("Tensor {} of type {} are not stream to device",
                  tensor->id,
                  tensor->tensorType());
    }
  }

  return mode;
}

unsigned IrLowering::getBufferingDepth(Tensor *tensor) const {

  // We should default to 1 when re-arranging on host because not doing this
  // could result in problems. We still let the user override this value
  // but they will get an error if it is >1.
  auto &sessionOpts   = ir().getSessionOptions();
  auto bufferingDepth = sessionOpts.defaultPrefetchBufferingDepth;

  // Default to buffering depth 1 if rearranging on host.
  if (doRearrangeOnHost(tensor)) {
    bufferingDepth = 1;
  }

  // Get bufferingDepth from SessionOptions.
  bufferingDepth =
      sessionOpts.getPrefetchBufferingDepth(tensor->id, bufferingDepth);

  if (bufferingDepth > 1) {
    if (doRearrangeOnHost(tensor)) {
      // There is a problem. This tensor is set to re-arrange on the
      // host but we've configured the engine option
      // "exchange.streamBufferOverlap" to "hostRearrangeOnly", meaning
      // that Poplar could overlap the memory of streams that are
      // rearranged on the host. This makes it incompatible with
      // bufferingDepths >1.
      throw error("Unable to support a buffering depth >1 for tensor {} "
                  "because the stream is set to rearrange on the host (and "
                  "PopART allows streams that are rearranged on the host to "
                  "overlap in memory, making this unsafe)",
                  tensor->id);
    }
  }

  return bufferingDepth;
}

void IrLowering::initPoplarGraph() {
  POPART_TRACEPOINT();

  const auto initPoplarGraphTimer =
      ir().timePartitionLogger().scopedStopwatch("Initializing poplar Graph");

  poplar::Target popTarget;
  unsigned replicationFactor = 0;

  if (ir().getSessionOptions().enableDistributedReplicatedGraphs) {
    auto globalReplicationFactor = getGlobalReplicationFactor();
    auto localReplicationFactor  = getReplicationFactor();
    auto numInstances = globalReplicationFactor / localReplicationFactor;

    auto globalNumIpus  = deviceInfo->getNumIpus() * numInstances;
    auto archString     = deviceInfo->getTarget().getTargetArchString();
    const auto &options = deviceInfo->getTarget().getTargetOptions();

    replicationFactor = globalReplicationFactor;

    logging::devicex::debug("Creating distributed replicated graph with global "
                            "replication factor {}",
                            replicationFactor);
    switch (deviceInfo->getType()) {
    case DeviceType::Ipu:
    case DeviceType::OfflineIpu: {
      popTarget = poplar::Target::createIPUTarget(
          static_cast<unsigned>(globalNumIpus), archString, options);
      break;
    }
    case DeviceType::Cpu:
    case DeviceType::IpuModel:
    case DeviceType::Sim:
    default:
      throw error(
          "Only IPU Hardware devices supported with distributed replicated "
          "graphs. Unsupported device type {}",
          deviceInfo->toString());
    }
  } else {
    popTarget         = deviceInfo->getTarget();
    replicationFactor = getReplicationFactor();

    logging::devicex::debug("Creating graph with replication factor {}",
                            replicationFactor);
  }

  pGraph.reset(new snap::Graph(popTarget,
                               poplar::replication_factor(replicationFactor)));
}

poplar::Type popType(const TensorInfo &info) {
  switch (info.dataType()) {
  case DataType::FLOAT: {
    return poplar::FLOAT;
  }
  case DataType::INT32: {
    return poplar::INT;
  }
  case DataType::FLOAT16: {
    return poplar::HALF;
  }
  case DataType::BOOL: {
    return poplar::BOOL;
  }
  case DataType::UINT32: {
    return poplar::UNSIGNED_INT;
  }
  case DataType::INT8: {
    return poplar::SIGNED_CHAR;
  }
  case DataType::UINT8: {
    return poplar::UNSIGNED_CHAR;
  }

  case DataType::UNDEFINED:
  case DataType::UINT16:
  case DataType::INT16:
  case DataType::INT64:
  case DataType::STRING:
  case DataType::BFLOAT16:
  case DataType::DOUBLE:
  case DataType::UINT64:
  case DataType::COMPLEX64:
  case DataType::COMPLEX128:
  default:
    throw error("The data type " + info.data_type() +
                " is not supported in Poplar");
  }
}

// using TensorInfo's data_type() function to get a string of the DataType
poplar::Type popType(DataType type) { return popType(TensorInfo(type, {1})); }

std::set<TensorId> IrLowering::getLinearlyCreatedInputTensors() const {
  return linearlyCreatedInputTensors;
}
std::set<TensorId> IrLowering::getEfficientlyCreatedInputTensors() const {
  return efficientlyCreatedInputTensors;
}

PopStreamId IrLowering::h2dId(TensorId id) { return "h2d_" + id; }

PopStreamId IrLowering::d2hId(TensorId id, bool isAnchorStream) {

  std::string anchorPrefix = isAnchorStream ? "anchor" : "weight";

  return anchorPrefix + "_d2h_" + id;
}

PopStreamId IrLowering::gradientStoreStreamId(TensorId id) {
  return gradientStoreStreamPrefix + id;
}

PopStreamId IrLowering::gradientLoadStreamId(TensorId id) {
  return gradientLoadStreamPrefix + id;
}

PopStreamId IrLowering::weightLoadStreamId(TensorId id) {
  return weightLoadStreamPrefix + id;
}

std::string IrLowering::cycleCountStreamId(std::string id) {
  return "d2h_" + std::string(cycleCountPrefix()) + "_" + id;
}

poplar::DataStream &IrLowering::insertGradientStoreStream(TensorId tensorId,
                                                          TensorInfo tensorInfo,
                                                          snap::Graph &graph) {
  auto streamMapEntry = toHostGradientStreams.find(tensorId);

  if (streamMapEntry == toHostGradientStreams.end()) {
    toHostGradientStreams.emplace(
        tensorId,
        poplar::DataStream(graph.getPoplarGraph().addDeviceToHostFIFO(
            gradientStoreStreamId(tensorId),
            popType(tensorInfo),
            tensorInfo.nelms())));
    streamMapEntry = toHostGradientStreams.find(tensorId);
  } else {
    throw error("Tensor Id " + tensorId +
                " already exists in toHostGradientStreams");
  }

  return streamMapEntry->second;
}

poplar::DataStream &IrLowering::insertGradientLoadStream(TensorId tensorId,
                                                         TensorInfo tensorInfo,
                                                         snap::Graph &graph) {
  auto streamMapEntry = fromHostGradientStreams.find(tensorId);

  if (streamMapEntry == fromHostGradientStreams.end()) {
    fromHostGradientStreams.emplace(
        tensorId,
        poplar::DataStream(graph.getPoplarGraph().addHostToDeviceFIFO(
            gradientLoadStreamId(tensorId),
            popType(tensorInfo),
            tensorInfo.nelms(),
            poplar::ReplicatedStreamMode::BROADCAST)));
    streamMapEntry = fromHostGradientStreams.find(tensorId);
  } else {
    throw error("Tensor Id " + tensorId +
                " already exists in fromHostGradientStreams");
  }

  return streamMapEntry->second;
}

poplar::DataStream &IrLowering::insertWeightLoadStream(TensorId tensorId,
                                                       TensorInfo tensorInfo,
                                                       snap::Graph &graph) {
  auto streamMapEntry = fromHostWeightLoadStreams.find(tensorId);

  if (streamMapEntry == fromHostWeightLoadStreams.end()) {
    fromHostWeightLoadStreams.emplace(
        tensorId,
        poplar::DataStream(graph.getPoplarGraph().addHostToDeviceFIFO(
            weightLoadStreamId(tensorId),
            popType(tensorInfo),
            tensorInfo.nelms(),
            poplar::ReplicatedStreamMode::BROADCAST)));
    streamMapEntry = fromHostWeightLoadStreams.find(tensorId);
  } else {
    throw error("Tensor Id " + tensorId +
                " already exists in weightStoreStreams");
  }

  return streamMapEntry->second;
}

} // namespace popx
} // namespace popart
