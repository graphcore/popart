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

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/range/algorithm/find.hpp>
#include <boost/range/algorithm_ext.hpp>

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
#include <poputil/exceptions.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>
#include <popart/logging.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/op/if.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/remote.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/subgraph.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/popx/exporter.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/callx.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/poplaroptionsx.hpp>
#include <popart/popx/pritask.hpp>
#include <popart/recompute.hpp>
#include <popart/stepio.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/tojson.hpp>
#include <popart/topocons.hpp>

#include <popart/op/hostreducevarupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/popx/op/ipucopyx.hpp>
#include <popart/tensornames.hpp>

#include <popart/poparttracepoint.hpp>

#include <stepiosplitter.hpp>

namespace popart {
namespace popx {

namespace {

void progressLogger(int progress, int total) {
  if (total != 0) {
    float percentage = std::floor(100.0f * static_cast<float>(progress) /
                                  static_cast<float>(total));
    logging::devicex::info("Engine compilation {}% complete", percentage);
  }
}

class SavedInfo {
public:
  SavedInfo(const IrLowering *lowering)
      : irHash(std::hash<Ir>{}(lowering->ir())) {}

  void serialize(std::ostream &os) { os << irHash; }

  static SavedInfo deserialize(std::istream &is) {
    SavedInfo result;
    is >> result.irHash;
    return result;
  }

  bool operator==(const SavedInfo &rhs) const { return irHash == rhs.irHash; }

  std::string toString() const {
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(2 * sizeof(std::size_t))
       << irHash;
    return ss.str();
  }

  std::size_t irHash;

private:
  SavedInfo() : irHash(0) {}
};

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
          x->getPipelineStage() != op->getPipelineStage()) {
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

void gclEnvironmentDeprecationWarning(const std::string &envVarName,
                                      const std::string &optName) {
  logging::warn("You are using a deprecated environment variable \"{}\". This "
                "will be removed in an upcoming release. Please use the "
                "session option 'SessionOptions::{}' instead",
                envVarName,
                optName);
}

} // namespace

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

  if (exception.graphProfile.type() == poplar::ProfileValue::Type::MAP &&
      exception.graphProfile.size() != 0) {

    std::stringstream ss;
    poplar::printProfileSummary(ss, exception.graphProfile, {}, reportOptions);
    return ss.str();
  } else {
    throw error("Need to set the 'debug.allowOutOfMemory' engine option to "
                "true to get the graph report");
  }
}

std::string devicex_memory_allocation_err::getGraphReport(bool useCbor) const {

  if (exception.graphProfile.type() == poplar::ProfileValue::Type::MAP &&
      exception.graphProfile.size() != 0) {

    std::stringstream ss;
    if (useCbor) {
      serializeToCBOR(ss, exception.graphProfile);
    } else {
      serializeToJSON(ss, exception.graphProfile);
    }
    return ss.str();
  } else {
    throw error("Need to set the 'debug.allowOutOfMemory' engine option to "
                "true to get the graph report");
  }
}

IrLowering::IrLowering(const Ir &ir, std::shared_ptr<DeviceInfo> deviceInfo_)
    : _ir(ir), deviceInfo(deviceInfo_), prepareGraphHasBeenCalled_(false),
      tensors_(ir), progs(PopPrograms(this)) {
  POPART_TRACEPOINT();

  // Set the opxTrace flag based on the environment variable
  auto POPART_OPX_TRACE = getPopartEnvVar("OPX_TRACE");
  opxTrace = POPART_OPX_TRACE ? strncmp(POPART_OPX_TRACE, "1", 1) == 0 : false;

  if (ir.getExecutionMode() == Ir::ExecutionMode::Training) {
    lstmOptions.set("inferenceOnly", "false");
  } else {
    lstmOptions.set("inferenceOnly", "true");
  }

  if (ir.getSessionOptions().enablePipelining) {
    pInfo =
        PipelineInfo(static_cast<int64_t>(ir.getDataFlow().batchesPerStep()),
                     ir.getSessionOptions().accumulationFactor,
                     ir.getNumPipelineStages(),
                     ir.canTrain(),
                     ir.getSessionOptions().enableGradientAccumulation);
  }

  const auto &userGclOptions = ir.getSessionOptions().gclOptions;
  validateGclOptions(userGclOptions);

  // Prefer to use `userGclOptions' over environment variables.
  if (userGclOptions.find("useSynclessCollectives") != userGclOptions.end()) {
    gclOptions.set("useSynclessCollectives",
                   userGclOptions.at("useSynclessCollectives"));
  } else if (std::getenv("GCL_REAL_COLLECTIVES")) {
    gclEnvironmentDeprecationWarning("GCL_REAL_COLLECTIVES",
                                     "gclOptions[\"useSynclessCollectives\"]");
    gclOptions.set("useSynclessCollectives", "true");
  }

  // Prefer to use `maxBytesPerTile' over environment variables.
  if (userGclOptions.find("maxBytesPerTile") != userGclOptions.end()) {
    auto &val = userGclOptions.at("maxBytesPerTile");
    gclOptions.set("maxBytesPerTile", val);
  } else if (auto *val = std::getenv("GCL_MAX_BYTES_PER_TILE")) {
    gclEnvironmentDeprecationWarning("GCL_MAX_BYTES_PER_TILE",
                                     "gclOptions[\"maxBytesPerTile\"]");
    gclOptions.set("maxBytesPerTile", val);
  }
}

// Constructor when deserializing the lowered state
IrLowering::IrLowering(const Ir &ir)
    : _ir(ir), deviceInfo(nullptr), prepareGraphHasBeenCalled_(true),
      tensors_(ir), progs(PopPrograms(this)) {}

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

void IrLowering::verifyTaskOrder(const std::vector<TaskId> &taskOrder) const {
  logging::debug("Verifying task order");
  int errors = 0;
  std::set<Op *> seen;
  std::set<Op *> recomputeSeen;

  for (auto taskId : taskOrder) {
    auto id_ops = contextOpRegistry.find({ExecutionContext::Normal, taskId});
    if (id_ops == contextOpRegistry.end()) {
      continue;
    }
    auto &taskOps = id_ops->second;

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
      // This is a recompute op.
      else {
        for (auto before : op->getGraph().topoCons->getBefores(op)) {
          if (recomputeSeen.find(before) == recomputeSeen.end()) {
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
    logging::devicex::warn("Encountered {} errors when verifying task order",
                           errors);
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

void IrLowering::trySaveTensorTileMap() const {
  auto popartTensorTileMap = getPopartEnvVar("TENSOR_TILE_MAP");
  if (popartTensorTileMap && strcmp(popartTensorTileMap, "") != 0) {
    saveTensorTileMap(popartTensorTileMap);
  }
}

void IrLowering::saveTensorTileMap(const std::string &mapFileName) const {
  auto tt = getTensorTileMap();

  std::string finalPath =
      io::appendDirFn(ir().getSessionOptions().logDir, mapFileName);

  std::ofstream ofs(finalPath, std::ofstream::out);
  if (!ofs.is_open()) {
    throw error("Unable to open file '{}'", finalPath);
  }

  writeJSON(tt, ofs);
}

TensorTileMap IrLowering::getTensorTileMap() const {
  if (!tensorTileMap.empty()) {
    return tensorTileMap;
  }

  TensorTileMap map;
  for (const auto &t : tensors_.getTensors()) {
    std::vector<TensorIntervalList> mapping;
    for (auto tile : graph().getTileMapping(*t.second.get())) {
      TensorIntervalList intervalList;
      std::transform(tile.begin(),
                     tile.end(),
                     std::back_inserter(intervalList),
                     [](poplar::Interval i) {
                       return std::pair<size_t, size_t>(i.begin(), i.end());
                     });
      mapping.emplace_back(intervalList);
    }
    map.insert(std::make_pair(t.first, mapping));
  }
  return map;
}

void IrLowering::setTensorTileMap(const TensorTileMap &m) { tensorTileMap = m; }

void IrLowering::instrumentWithHardwareCycleCounter(
    poplar::program::Sequence &sq,
    int64_t tileId,
    std::string id) {
  poplar::Tensor cycleCountTensor = poplar::cycleCount(
      graph(), sq, static_cast<unsigned int>(tileId), cycleCountPrefix());

  // Create stream
  auto st = graph().addDeviceToHostFIFO(cycleCountStreamId(id),
                                        cycleCountTensor.elementType(),
                                        cycleCountTensor.numElements());

  cycleCountIds.push_back(id);

  // Add program fragment to copy to host stream
  auto cyclesToHostStream = poplar::program::Copy(cycleCountTensor, st, true);
  progs.cycleCountTensorToHostFragment().add(cyclesToHostStream);
}

poplar::Tensor IrLowering::getConst(poplar::Graph &graph,
                                    const poplar::Type &type,
                                    const std::vector<size_t> &shape,
                                    double val,
                                    const std::string &name) {
  static unsigned tileCounter = 0;

  auto tensor     = graph.addConstant(type, shape, val, name);
  auto tilesTotal = graph.getTarget().getTilesPerIPU();
  auto tile       = tileCounter % tilesTotal;
  tileCounter++;

  graph.setTileMapping(tensor, tile);
  return tensor;
}

poplar::Tensor IrLowering::getScalarVariable(poplar::Graph &graph,
                                             const poplar::Type &type,
                                             const std::string &name) {
  static int tileCounter = -1;

  auto tensor     = graph.addVariable(type, {}, name);
  auto tilesTotal = graph.getTarget().getTilesPerIPU();
  auto tile       = (tilesTotal + (tileCounter % tilesTotal)) % tilesTotal;
  tileCounter--;

  graph.setTileMapping(tensor, tile);
  return tensor;
}

PipelineInfo::PipelineInfo(int64_t _batchesPerStep,
                           int64_t _gradAcclFactor,
                           int64_t _numPipelineStages,
                           bool _doTraining,
                           bool _doGradAccl)
    : doTraining(_doTraining), doGradAccl(_doGradAccl) {

  auto fillFlushPhaseCycles = _numPipelineStages - 1;
  fillPhase.start           = 0;
  fillPhase.end             = fillFlushPhaseCycles - 1;

  int64_t mainCycles;
  if (doGradAccl) {
    mainCycles = _gradAcclFactor - fillFlushPhaseCycles;
  } else {
    mainCycles = _batchesPerStep - fillFlushPhaseCycles;
  }
  if (mainCycles < 1) {
    throw internal_error(
        "Pipeline mainCycles should not be less than 1. Current value is {}.",
        mainCycles);
  }

  mainPhase.start = fillPhase.end + 1;
  mainPhase.end   = mainPhase.start + mainCycles - 1;

  flushPhase.start = mainPhase.end + 1;
  flushPhase.end   = flushPhase.start + fillFlushPhaseCycles - 1;
}

bool PipelineInfo::doStage(PipelineCycle pCycle, PipelineStage pStage) const {
  bool doStageLower = (pCycle >= pStage);
  bool doStageUpper = (pCycle < pStage + flushPhase.start);

  return (doStageLower && doStageUpper);
}

poplar::Graph &IrLowering::getVirtualGraph(VGraphId virtualGraphIndex,
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

std::unique_ptr<Opx> IrLowering::createOpx(Op *op) {
  if (dv_p == nullptr) {
    throw error("IrLowering::setDevice has not been called.");
  }

  auto opx = OpxManager::createOpx(op, dv_p);

  if (!opx) {
    if (op->opid == Onnx::Operators::Constant_1 ||
        op->opid == Onnx::Operators::Constant_9) {
      throw internal_error("No Opx for {}", op->opid);
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

// The Id of the task which adds a Tensor to a poplar::Graph
std::pair<TaskId, DependencyType> IrLowering::taskWhichCreates(TensorId id) {
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
      return {opTaskId(tensor->getProducer()), DependencyType::Tensor};
    }
  }
}

TaskId IrLowering::taskWhichPopulates(TensorId id) const {
  Tensor *tensor = ir().getTensor(id);

  // OpTasks both initialize a Tensor, and generate the code to set its value
  if (tensor->hasProducer()) {
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
      auto conOpId   = op->id;
      const Opx *opx = getOpx(conOpId);

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

          const SubgraphOpx *callopx = dynamic_cast<const SubgraphOpx *>(opx);

          // Get delegated endpoints
          SubgraphOp *subgraphOp = &callopx->getOp<SubgraphOp>();
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

ICreatorCandidatePtr IrLowering::getTensorCreator(Tensor *tensor) const {
  // Search of the graph to get the candidate Opxs that
  // know how to create this tensor.
  // The pathFromInput argument is an empty vector, as
  // we are starting the search from the root (input)

  logging::devicex::trace("Get tensor creator for {}, {} elements",
                          tensor->id,
                          tensor->info.nelms());

  std::vector<ICreatorCandidatePtr> candidates = getCreatorEndpoints(tensor);

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
      return candidates.front();
    } else {
      logging::devicex::trace("Multiple candidates needed.");
      // Multiple creators need to be concatenated to form the full tensor.
      std::shared_ptr<InputMultiCreatorCandidate> multiCandidate =
          std::make_shared<InputMultiCreatorCandidate>();
      for (auto candidate : candidates) {
        // Important to add candidates sorted by priorty.
        // Highest first - ICreatorCandidate::greaterThan.
        multiCandidate->addCreatorCandidate(candidate);
      }
      logging::devicex::trace("Using multi-candidate {}.",
                              multiCandidate->str());
      return multiCandidate;
    }
  } else {
    logging::devicex::trace("No suitable candidate.");
    return nullptr;
  }
}

// Design decision : leave the option for a Tensor to be
// created based on complex global criteria open.
PriTask IrLowering::initTensorTask(Tensor *tensor) {
  auto candidate = getTensorCreator(tensor);

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
    // assumes an op has max 1000 inputs
    priMod =
        static_cast<double>(priMod0) + static_cast<double>(priMod1) / 1000.0;
  }

  // 1. A unique candidate creator will create the tensor
  // 2. The tensor will be unwound (have its layout modified)
  //    by view-changing opxs on the path from the input to
  //    the candidate candidate
  if (candidate) {

    auto f = [this, candidate, tensor]() {
      logging::devicex::debug(
          "Creating poplar::Tensor {}, with layout allocated by {}",
          tensor->id,
          candidate->str());

      auto inputAndView = candidate->createInput(tensor->str() + "_tmp");

      // Try if an existing Poplar tensor can be reused
      if (tryInitTensorByPostIRAliasing(tensor->id, inputAndView.second)) {
        return SequenceMap();
      }

      if (!inputAndView.second.empty()) {
        // Underlying poplar::Tensor does not match IR expectations, supply
        // view-changing transformation
        tensors_.setViewChangers(tensor->id, inputAndView.second);
      }

      // The clone makes sure to only keep the necessary parts of the unwound
      // tensor alive, and contiguate it,
      // reducing IPU memory liveness and fragmentation (see T18661)
      poplar::Tensor input = graph().clone(inputAndView.first);

      tensors_.insert(tensor->id, input);
      efficientlyCreatedInputTensors.insert(tensor->id);
      return SequenceMap();
    };
    // the inputs of creator which must have poplar::Tensors
    // before creator creates input tensor at index inIndex.
    std::vector<std::pair<TaskId, DependencyType>> deps;
    for (TensorId tenId : candidate->mustExistBeforeCreate()) {
      auto dep = taskWhichCreates(tenId);
      deps.push_back(dep);
    }

    // Discussion with David Norman suggests creating tensors as
    // late as possible gives better IPU memory use, so
    // giving this low priority.
    return {-1e6 + priMod,
            initTensorTaskId(tensor->id), // the task name
            deps,
            f};
  } else {

    auto f = [this, tensor]() {
      // Try if an existing Poplar tensor can be reused
      if (tryInitTensorByPostIRAliasing(tensor->id, ViewChangers())) {
        return SequenceMap();
      }

      logging::devicex::debug("Creating poplar::Tensor '{}' linearly. No "
                              "operator specific allocator found",
                              tensor->id);

      // Find the ipu the op that consumes with tensor is on and create the
      // tensor on that graph
      auto consumerOps = tensor->consumers.getOps();

      std::vector<std::pair<Op *, bool>> relatedOps;
      relatedOps.reserve(consumerOps.size() + 1);

      for (auto *op : consumerOps) {
        relatedOps.emplace_back(op, true);
      }

      if (tensor->hasProducer()) {
        relatedOps.emplace_back(tensor->getProducer(), false);
      }

      if (logging::shouldLog(logging::Module::devicex, logging::Level::Trace)) {
        std::stringstream ss;
        for (auto &op : relatedOps) {
          if (op.second) {
            auto index = op.first->input->indicesMap().at(tensor)[0];
            ss << std::endl
               << "    {" << op.first->debugName() << ", VGID: "
               << (op.first->hasVirtualGraphId()
                       ? op.first->getIntrospectionInVirtualGraphId(index).first
                       : -1)
               << "}";
          }
        }
        logging::devicex::trace(
            "[initTensorTask] Tensor: {}, type: {}, consumed by ops: [{}]",
            tensor->id,
            tensor->getTensorTypeInfo()->type_s(),
            ss.str());
      }

      std::vector<VGraphId> ipus;
      for (auto &op : relatedOps) {
        VGraphIdAndTileSet vgid{-1, TileSet::Compute};
        if (op.first->hasVirtualGraphId()) {
          if (op.second) {
            // Consumer OP
            // VirtualGraphId with subgraph call introspection
            // for the current tensor
            auto index = op.first->input->indicesMap().at(tensor)[0];
            vgid       = op.first->getIntrospectionInVirtualGraphId(index);
          } else {
            // Producer OP
            vgid = {op.first->getVirtualGraphId(), op.first->settings.tileSet};
          }
        }

        // The copyToIpu op assume that the tensor will already
        // have been copied to the ipu from another op
        if (op.first->opid != Onnx::CustomOperators::IpuCopy) {

          if (ipus.end() == std::find(ipus.begin(), ipus.end(), vgid.first)) {

            auto &graph = vgid.first > -1
                              ? getVirtualGraph(vgid.first, vgid.second)
                              : getOpx(op.first->id)->graph();

            auto newTensor = graph.addVariable(
                popType(tensor->info), tensor->info.shape_szt(), tensor->str());
            linearMapper.mapTensor(graph, newTensor);

            tensors_.insert(tensor->id, newTensor);
            linearlyCreatedInputTensors.insert(tensor->id);
            ipus.push_back(vgid.first);
          }
        }
      }
      return SequenceMap();
    };

    // Discussion with David Norman suggests creating tensors as
    // late as possible gives better IPU memory use, so
    // giving this low priority.
    return {-1e6 + priMod, initTensorTaskId(tensor->id), {}, f};
  }
}

PriTask IrLowering::initRandomSeed() {
  auto streamedSeedId = GetRandomSeedOp::getStreamedSeedTensorId();
  auto updatedSeedId  = GetRandomSeedOp::getUpdatedSeedTensorId();

  auto initRandomSeedTask = [this, updatedSeedId]() {
    logging::devicex::debug("Initializing random seed.");
    SequenceMap seqs;
    poprand::setSeed(graph(),
                     tensors_.get(updatedSeedId),
                     0,
                     seqs[&progs.setRandomSeedFromHostFragment()],
                     logging::format("{}/set", updatedSeedId));
    return seqs;
  };

  std::vector<std::pair<TaskId, DependencyType>> deps;
  deps.push_back(taskWhichCreates(updatedSeedId));
  // Stream the seed tensor to device before using to set PRNGs
  deps.push_back({fromHostTaskId(streamedSeedId), DependencyType::Scheduler});

  return {
      +1e6,                   // high priority
      initRandomSeedTaskId(), // name of this task
      deps,                   // depends on
      initRandomSeedTask      // what to run when the task is executed
  };
}

PriTask IrLowering::rngStateFromHost() {
  auto rngStateFromHostTask = [this]() {
    int rngSize = (graph().getTarget().getNumTiles()) *
                  graph().getTarget().getNumWorkerContexts() * 4;
    auto streamRngFromHost =
        graph().addHostToDeviceFIFO("h2d_rngStateTensor",
                                    poplar::UNSIGNED_INT,
                                    rngSize,
                                    poplar::ReplicatedStreamMode::REPLICATE);

    logging::devicex::debug("Initializing RNG h2d.");

    SequenceMap seqs;
    seqs[&progs.rngStateFromHostFragment()].add(
        poplar::program::Copy(streamRngFromHost, rngStateTensor));
    poplar::setHwSeeds(graph(),
                       rngStateTensor,
                       seqs[&progs.rngStateFromHostFragment()],
                       "RNG set");
    logging::devicex::debug("RNG size {}", rngSize);
    return seqs;
  };
  return {0,
          rngStateFromHostTaskId(),
          {{initRngStateTensorTaskId(), DependencyType::Tensor}},
          rngStateFromHostTask};
}

PriTask IrLowering::initRngStateTensor() {
  // Add a new tensor to the graph to store the Hw seed
  auto initRngStateTensorTask = [this]() {
    SequenceMap seqs;
    auto workersPerIPU = graph().getTarget().getNumWorkerContexts();
    auto numTiles      = graph().getTarget().getNumTiles();
    rngStateTensor     = graph().addVariable(
        poplar::UNSIGNED_INT, {numTiles, workersPerIPU, 4}, "rngStateTensor");
    linearMapper.mapTensor(graph(), rngStateTensor);
    return SequenceMap();
  };
  return {+1e6, initRngStateTensorTaskId(), {}, initRngStateTensorTask};
}

PriTask IrLowering::rngStateToHost() {
  auto rngStateToHostTask = [this]() {
    int rngSize = graph().getTarget().getNumTiles() *
                  graph().getTarget().getNumWorkerContexts() * 4;
    auto streamRngToHost = graph().addDeviceToHostFIFO(
        "d2h_rngStateTensor", poplar::UNSIGNED_INT, rngSize);

    logging::devicex::debug("Initializing RNG d2h.");
    logging::devicex::debug("RNG size {}", rngSize);
    SequenceMap seqs;
    rngStateTensor = poplar::getHwSeeds(
        graph(), seqs[&progs.rngStateToHostFragment()], "RNG get");
    seqs[&progs.rngStateToHostFragment()].add(
        poplar::program::Copy(rngStateTensor, streamRngToHost));
    return seqs;
  };

  return {0,
          rngStateToHostTaskId(),
          {{initRngStateTensorTaskId(), DependencyType::Tensor}},
          rngStateToHostTask};
}

template <typename T> void IrLowering::setInitVal(Tensor *tensor) {

  graph().setInitialValue<T>(
      tensors_.get(tensor->id),
      poplar::ArrayRef<T>(static_cast<const T *>(tensor->tensorData()->data()),
                          tensor->info.nelms()));
}

// Using specialised poplar function for setting init val for FLOAT16
void IrLowering::setInitValHalf(Tensor *tensor) {

  graph().setInitialValueHalf(
      tensors_.get(tensor->id),
      poplar::ArrayRef<uint16_t>(
          static_cast<const uint16_t *>(tensor->tensorData()->data()),
          tensor->info.nelms()));
}

PriTask IrLowering::setInitTensorValTask(Tensor *tensor) {
  // See T6254. Currently we just use setInitialValue for all constant tensors
  auto f = [this, tensor]() {
    logging::devicex::debug("Setting initial value for tensor {}",
                            tensor->str());

    // see T5925 for making a more compact way of matching
    // types than using this switch statement
    switch (tensor->info.dataType()) {
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

    case DataType::UNDEFINED:
    case DataType::UINT8:
    case DataType::INT8:
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
    return SequenceMap();
  };

  return {// priority unimportant
          0,
          // name of this task
          setInitTensorValTaskId(tensor->id),
          // poplar::Tensor must exist. Other that this, this task can be
          // performed any time
          {{initTensorTaskId(tensor->id), DependencyType::Tensor}},
          f};
}

PriTask IrLowering::streamFromHostTask(Tensor *tensor) {
  auto f = [this, tensor]() {
    std::vector<VGraphId> ipus;

    auto consumerOps = tensor->consumers.getOps();

    for (auto *op : consumerOps) {
      // Assume another op will copy the tensor for an ipucopy
      if (op->opid != Onnx::CustomOperators::IpuCopy) {
        auto &graph = getOpx(op->id)->graph();

        VGraphId vgid = -1;
        if (op->hasVirtualGraphId()) {
          // VirtualGraphId with subgraph call introspection
          // for the current tensor
          auto index = op->input->indicesMap().at(tensor)[0];
          vgid       = op->getIntrospectionInVirtualGraphId(index).first;
        }

        // Only stream the tensor once for all op's that consume it on an ipu
        if (std::find(ipus.begin(), ipus.end(), vgid) == ipus.end()) {

          poplar::OptionFlags options{};

          // Get bufferingDepth from SessionOptions.
          auto depth =
              ir().getSessionOptions().getPrefetchBufferingDepth(tensor->id);

          if (depth > 1) {
            if (doRearrangeOnHost(tensor)) {
              // There is a problem. This tensor is set to re-arrange on the
              // host but we've configured the engine option
              // "exchange.streamBufferOverlap" to "hostRearrangeOnly", meaning
              // that Poplar could overlap the memory of streams that are
              // rearranged on the host. This makes it incompatible with
              // bufferingDepths >1.
              throw error(
                  "Unable to support a buffering depth >1 for tensor {} "
                  "because the stream is set to rearrange on the host (and "
                  "PopART allows streams that are rearranged on the host to "
                  "overlap in memory, making this unsafe)",
                  tensor->id);
            }

            // Configure the buffering depth of the stream.
            options.set("bufferingDepth", std::to_string(depth));
          }

          logging::devicex::debug("Creating host-to-device FIFO {} copied to "
                                  "ipu:{} (with buffering depth {})",
                                  tensor->id,
                                  vgid,
                                  depth);

          poplar::ReplicatedStreamMode mode =
              poplar::ReplicatedStreamMode::BROADCAST;

          if (tensor->tensorType() == TensorType::Variable) {
            // If it is a variable we 'broadcast' the same tensor
            // to all replicants
            mode = poplar::ReplicatedStreamMode::BROADCAST;

          } else if (tensor->tensorType() == TensorType::Stream) {

            switch (tensor->getReplicatedStreamMode()) {
            case Tensor::ReplicatedStreamMode::Broadcast:
              mode = poplar::ReplicatedStreamMode::BROADCAST;
              break;
            case Tensor::ReplicatedStreamMode::Replicate:
              mode = poplar::ReplicatedStreamMode::REPLICATE;
              break;
            }

          } else {
            throw error("Tensor {} of type {} are not stream to device",
                        tensor->id,
                        tensor->tensorType());
          }

          fromHostStreams.emplace(
              tensor->id,
              graph.addHostToDeviceFIFO(h2dId(tensor->id),
                                        popType(tensor->info),
                                        tensor->info.nelms(),
                                        mode,
                                        options));

          ipus.push_back(vgid);
        }
      }
    }
    return SequenceMap();
  };

  return {
      0,                                // priority unimportant
      streamFromHostTaskId(tensor->id), // name of this task
      // poplar::Tensor must exist
      {{initTensorTaskId(tensor->id), DependencyType::Tensor}},
      f // what to run when the task is executed
  };
}

PriTask IrLowering::streamToHostTask(Tensor *tensor, bool isAnchorStream) {
  auto f = [this, tensor, isAnchorStream]() {
    logging::devicex::debug("Creating device-to-host FIFO for poplar::Tensor "
                            "{} (isAnchorStream = {}) with {} elements",
                            tensor->id,
                            isAnchorStream,
                            tensor->info.nelms());

    auto pToHostStreams = &toHostAnchorStreams;
    if (!isAnchorStream) {
      pToHostStreams = &toHostWeightStreams;
    }

    pToHostStreams->emplace(
        tensor->id,
        graph().addDeviceToHostFIFO(d2hId(tensor->id, isAnchorStream),
                                    popType(tensor->info),
                                    tensor->info.nelms()));
    return SequenceMap();
  };

  return {
      0,                                              // priority unimportant
      streamToHostTaskId(tensor->id, isAnchorStream), // name of this task
      {taskWhichCreates(tensor->id)}, // poplar::Tensor must exist
      f                               // what to run when the task is executed,
  };
}

bool IrLowering::hasRemoteBuffer(RemoteBufferId id) const {
  return remoteBuffers.find(id) != remoteBuffers.end();
}

const std::string IrLowering::getRemoteBufferName(RemoteBufferId id) {
  return "RB_" + std::to_string(id);
}

const std::pair<poplar::RemoteBuffer, nonstd::optional<poplar::Tensor>> &
IrLowering::getRemoteBuffer(RemoteBufferId id) const {
  return remoteBuffers.at(id);
}

void IrLowering::createRemoteBuffer(RemoteBufferId id, poplar::Tensor tensor) {
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

  remoteBuffers.insert(
      {id,
       {graph().addRemoteBuffer(name, type, size, repeats, true),
        nonstd::optional<poplar::Tensor>(tensor)}});
}

std::shared_ptr<CollectiveBalancedReorder>
IrLowering::getCollectiveBalancedReorder(TensorId tensor_id) {
  return collectiveReorders[tensor_id];
}

const CollectiveBalancedHostRearrangement &
IrLowering::getCollectiveBalancedHostRearrangement(
    const TensorId &tensor_id) const {
  return collectiveReorders.at(tensor_id)->getHostRearrangement();
}

void IrLowering::setCollectiveBalancedReorder(
    TensorId tensor_id,
    std::shared_ptr<CollectiveBalancedReorder> cbr) {
  collectiveReorders[tensor_id] = cbr;
}

bool IrLowering::containsFragment(const Graph &graph) const {
  return progs.containsFragment(graph);
}

void IrLowering::createFragment(const Graph &graph) {
  return progs.createFragment(graph);
}

poplar::Function &IrLowering::getFragmentFunction(const Graph &called_graph) {
  logging::devicex::trace("[getFragmentFunction] Getting function for graph {}",
                          called_graph.id.str());
  return progs.getFragmentFunction(called_graph, graph());
}

void IrLowering::addPipelinedCopyTasks(PriTasks &tasks) {
  auto schedule =
      ir().getMainGraph().getOpSchedule({}, RequireOptimalSchedule::Yes);
  std::string prevTaskId = "";

  for (auto iter = schedule.rbegin(); iter != schedule.rend(); iter++) {
    auto &op = *iter;
    if (op->isConvertibleTo<IpuCopyOp>() && !op->copiesOptimizerTensors()) {
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
    SequenceMap seqs;
    logging::debug("Adding pipelined copies for op {}", copyOp->debugName());
    auto &prog = progs.pipelineIpuCopyFragment(
        logging::format("{}, {}, PipelineStage({})",
                        copyOp->debugName(),
                        copyOp->getFromToStr(),
                        copyOp->getPipelineStage()));
    copyOpx->growPipelined(seqs[&prog]);
    return seqs;
  };

  std::vector<std::pair<TaskId, DependencyType>> deps;
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

  // Ensure there is a program fragment for every Ir Graph
  logging::devicex::trace("[addOpTasks] Graphs: {}",
                          ir().getGraphSchedule().size());
  for (auto graph : ir().getGraphSchedule()) {
    if (!containsFragment(*graph)) {
      createFragment(*graph);
    }
  }

  auto mainGraphSchedule =
      ir().getMainGraph().getOpSchedule({}, RequireOptimalSchedule::Yes);

  // repeating logic in Ir::getOpSchedule (can be simplified there?)
  std::vector<Op *> allOps;
  std::set<const Graph *> addedGraphs;
  std::function<void(const Graph *)> addGraph;
  addGraph = [&allOps, &addedGraphs, &addGraph](const Graph *graph) {
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

  for (auto op : mainGraphSchedule) {
    for (auto calledGraph : op->getCalledGraphs()) {
      addGraph(calledGraph);
    }
    allOps.push_back(op);
  }

  double priority     = 0.0;
  TaskId prevOpTaskId = "";

  std::set<std::pair<Op *, ExecutionPhase>> seenRecomputeOps;
  FindRequiredRecomputes recomputeFinder(allOps);
  // Iterate through Ops according to the Ir's schedule
  for (auto op : allOps) {
    for (auto graph : op->getCalledGraphs()) {
      auto opInputs = op->getInputsForGraph(*graph);
      for (int i = 0; i < opInputs.size(); i++) {
        auto graphInput = graph->getInputId(i);
        if (!tasks.contains(initTensorTaskId(graphInput))) {
          tasks.add(initTensorByCloningTask(op, opInputs.at(i), graphInput));
        }
      }

      auto opOutputs = getOpx(op->id)->getOutputsToPrepare();
      for (int i = 0; i < opOutputs.size(); i++) {
        auto opOutput = opOutputs[i];
        if (!tasks.contains(initTensorTaskId(std::get<1>(opOutput)))) {
          if (std::get<0>(opOutput).empty()) {
            // No tensor to clone or alias from
            tasks.add(initTensorTask(ir().getTensor(std::get<1>(opOutput))));
          } else {
            // Tensor can be cloned or aliased
            if (std::get<2>(opOutput)) {
              tasks.add(initTensorByAliasingTask(
                  op, std::get<0>(opOutput), std::get<1>(opOutput)));
            } else {
              tasks.add(initTensorByCloningTask(
                  op, std::get<0>(opOutput), std::get<1>(opOutput)));
            }
          }
        }
      }
    }

    auto rerunSchedule = recomputeFinder.getRequiredRecomputeOps(op);
    if (!rerunSchedule.empty()) {
      requiredRecomputes[opTaskId(op)] = rerunSchedule;
    }

    auto task = opTask(op, priority, prevOpTaskId);

    tasks.add(task);
    prevOpTaskId = task.name;
    priority -= 1.;
  }
}

bool IrLowering::tryInitTensorByPostIRAliasing(
    TensorId dstId,
    const ViewChangers &viewChangers) {
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
        logging::devicex::debug("Creating poplar::Tensor '{}' "
                                "by aliasing from poplar::Tensor '{}'",
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

PriTask
IrLowering::initTensorByCloningTask(Op *op, TensorId srcId, TensorId dstId) {
  Opx *opx = getOpx(op->id);

  auto f = [srcId, dstId, opx, this]() {
    // Try if an existing Poplar tensor can be reused
    if (tryInitTensorByPostIRAliasing(dstId,
                                      tensors_.hasViewChangers(srcId)
                                          ? tensors_.getViewChangers(srcId)
                                          : ViewChangers())) {
      return SequenceMap();
    }

    logging::debug("Cloning tensor {} to {}", srcId, dstId);
    auto src = opx->get(srcId);
    auto dst = opx->graph().clone(src, dstId);

    if (tensors_.hasViewChangers(srcId)) {
      tensors_.setViewChangers(dstId, tensors_.getViewChangers(srcId));
    }
    tensors_.insert(dstId, dst);
    return SequenceMap();
  };

  std::vector<std::pair<TaskId, DependencyType>> deps;
  auto creatorTask = taskWhichCreates(srcId);
  deps.push_back(creatorTask);

  return {-1e6, initTensorTaskId(dstId), deps, f};
}

PriTask
IrLowering::initTensorByAliasingTask(Op *op, TensorId srcId, TensorId dstId) {
  auto f = [srcId, dstId, this]() {
    logging::debug("Aliasing tensor {} to {}", srcId, dstId);
    tensors_.insertAliased(dstId, srcId);
    return SequenceMap();
  };

  std::vector<std::pair<TaskId, DependencyType>> deps;
  auto creatorTask = taskWhichCreates(srcId);
  deps.push_back(creatorTask);

  return {-1e6, initTensorTaskId(dstId), deps, f};
}

PriTask IrLowering::opTask(Op *op, double priority, TaskId prevOpTaskId) {
  // although priority should guarantee that this
  // task is only run after inputs are all created,
  // we add a dependency to the input tensors, just
  // in case someone plays with the priorities.
  // Moreover, we must state the copy-from-host deps
  std::vector<std::pair<TaskId, DependencyType>> deps;
  for (auto t_inds : op->input->indicesMap()) {
    Tensor *tensor = t_inds.first;

    std::pair<TaskId, DependencyType> creatorTask =
        taskWhichCreates(tensor->id);

    std::pair<TaskId, DependencyType> populatorTask = {
        taskWhichPopulates(tensor->id), DependencyType::Scheduler};

    // Make sure we only add the creatorTask once in the dependency list
    if (std::find(deps.begin(), deps.end(), creatorTask) == deps.end()) {
      deps.push_back(creatorTask);
    }
    if (std::find(deps.begin(), deps.end(), populatorTask) == deps.end()) {
      deps.push_back(populatorTask);
    }
  }

  // Add initTensorTask dependencies for externally created output tensors
  Opx *opx = getOpx(op->id);
  for (auto t_inds : op->output->indicesMap()) {
    if (opx->outputCreatedExternally(t_inds.second.front())) {
      Tensor *tensor = t_inds.first;

      logging::devicex::trace("Operation {} depends on it's output tensor {} "
                              "being externally created.",
                              op->debugName(),
                              tensor->id);

      std::pair<TaskId, DependencyType> creatorTask = {
          initTensorTaskId(tensor->id), DependencyType::Tensor};

      // Make sure we only add the creatorTask once in the dependency list
      if (std::find(deps.begin(), deps.end(), creatorTask) == deps.end()) {
        deps.push_back(creatorTask);
      }
    }
  }

  auto addGraphOpsToDeps = [&](const Graph *graph) {
    for (auto graphOp : graph->getOpSchedule({}, RequireOptimalSchedule::Yes)) {
      std::pair<TaskId, DependencyType> taskId = {opTaskId(graphOp),
                                                  DependencyType::SubGraph};
      if (std::find(deps.begin(), deps.end(), taskId) == deps.end()) {
        deps.push_back(taskId);
      }
    }
  };

  for (auto &graph : op->getCalledGraphs()) {
    addGraphOpsToDeps(graph);
  }

  // Depends on previous op task. This preserves op ordering from ir.
  // Note: the first opTask has no previous opTask
  if (prevOpTaskId != "") {
    std::pair<TaskId, DependencyType> prevTask = {prevOpTaskId,
                                                  DependencyType::Scheduler};
    // Add dependency only if not already added
    if (std::find(deps.begin(), deps.end(), prevTask) == deps.end()) {
      deps.push_back(prevTask);
    }
  }

  auto taskId = opTaskId(op);
  if (requiredRecomputes.find(taskId) != requiredRecomputes.end()) {
    for (const auto &recompOp : requiredRecomputes[taskId]) {
      std::pair<TaskId, DependencyType> recompTask = {opTaskId(recompOp),
                                                      DependencyType::Output};
      if (std::find(deps.begin(), deps.end(), recompTask) == deps.end()) {
        deps.push_back(recompTask);
      }
    }
  }

  auto f = [op, this]() {
    SequenceMap seqs;
    const auto &containingGraph = op->getGraph();
    // if this Op is not in the main scope
    if (!containingGraph.id.str().empty()) {
      Opx *opx = getOpx(op->id);
      logging::devicex::debug("Creating output tensors for non-main {} in {}",
                              opx->op_p->debugName(),
                              containingGraph.id.str());
      // Record each scope task fragment separately first.
      growOpx(opx, seqs[&progs.scopeFragment(containingGraph)]);
    } else if (ir().getSessionOptions().enablePipelining) {
      pipelinedOpTaskFunc(opTaskId(op), op, seqs);
    } else {
      opTaskFunc(opTaskId(op), op, seqs);
    }
    return seqs;
  };

  return {priority, opTaskId(op), deps, f};
}

void IrLowering::growOpx(Opx *opx, poplar::program::Sequence &seq) {
  logging::devicex::trace("Calling growOpx for Op {} with debugName {}",
                          opx->op_p->str(),
                          opx->op_p->debugName());

  if (opxTrace) {
    seq.add(poplar::program::PrintTensor(opx->op_p->str() + "/enter",
                                         opxTraceTensor));
  }

  // Build a map of all tensors that are inputs and not
  // supposed to be modified by this operation
  std::map<InIndex, std::pair<poplar::Tensor, poplar::Tensor>>
      nonModifiedTensors;
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
        poplar::Tensor inTensor = opx->get(inputMap.second->id);
        // Check that this isn't a phony tensor
        if (inTensor.numElements() > 0) {
          // Clone the input tensor with its current values
          // to check if the original input tensor has been modified
          // during opx->grow(seq)
          auto inTensorClone = graph().clone(inTensor);
          seq.add(poplar::program::Copy(inTensor, inTensorClone));
          nonModifiedTensors[inputMap.first] =
              std::make_pair(inTensor, inTensorClone);
        }
      }
    }
  }

  // Grow code for the Op
  opx->grow(seq);

  if (ir().getSessionOptions().opxModifyChecking) {
    for (auto &nonModified : nonModifiedTensors) {
      // Compare input tensor with the input tensor clone before executing the
      // Op, skip checking non-finite numbers
      // (which always evaluate to non-equal)
      auto check = popops::map(
          opx->graph(),
          popops::expr::NotEqual(
              popops::expr::Select(popops::expr::_1,
                                   popops::expr::Const(0),
                                   popops::expr::IsFinite(popops::expr::_1)),
              popops::expr::Select(popops::expr::_2,
                                   popops::expr::Const(0),
                                   popops::expr::IsFinite(popops::expr::_2))),
          {nonModified.second.first, nonModified.second.second},
          seq,
          opx->debugPrefix("opxModifyChecking"),
          {});
      auto checkReduced = check.flatten();
      // Convert result to boolean scalar
      if (check.numElements() > 1) {
        checkReduced = popops::reduce(opx->graph(),
                                      checkReduced,
                                      {0},
                                      {popops::Operation::LOGICAL_OR},
                                      seq,
                                      opx->debugPrefix("opxModifyChecking"));
      } else {
        checkReduced = checkReduced.squeeze({0});
      }
      auto ifProg = poplar::program::ErrorProgram(
          logging::format("Op {} claims input {} is not modified, "
                          "but the Poplar tensors disagree.",
                          opx->op_p->debugName(),
                          nonModified.first),
          check);
      auto elseProg = poplar::program::Sequence();
      seq.add(poplar::program::If(checkReduced, ifProg, elseProg));
    }
  }

  if (opxTrace) {
    seq.add(poplar::program::PrintTensor(opx->op_p->str() + "/exit",
                                         opxTraceTensor));
  }
}

void IrLowering::opTaskFunc(TaskId taskId, Op *op, SequenceMap &seqs) {
  Opx *opx                 = getOpx(op->id);
  ExecutionContext context = op->settings.executionContext;

  if (op->copiesOptimizerTensors()) {
    growOpx(opx, seqs[&progs.streamOptimizerFromHostFragment()]);
  }

  // Special case for running operators before the main loop.
  else if (context == ExecutionContext::WeightsFromHostFragment) {
    growOpx(opx, seqs[&progs.streamWeightsFromHostFragment()]);
    contextOpRegistry[{context, taskId}].push_back(op);
  }

  // Special case for running operators after the main loop.
  else if (context == ExecutionContext::WeightsToHostFragment) {
    growOpx(opx, seqs[&progs.weightsToHostFragment()]);
    contextOpRegistry[{context, taskId}].push_back(op);
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
      seqs[&progs.forwardFragment()].add(progs.recomputeFragment(op->id));
    }

    // Pre-loss, not recompute or checkpoint
    else {
      throw internal_error("Unrecognised recompute type");
    }
    contextOpRegistry[{context, taskId}].push_back(op);
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

    // 2 special case Ops when their is a gradient accumulator / velocity.
    // If we are doing gradient accumulation, we need to ensure the reset
    // and var update aren't run every time. Instead, these fragments sit
    // outside the "main" loop of the fowards and backwards passes.
    // special case Op 1:
    if (ir().getSessionOptions().enableGradientAccumulation &&
        context == ExecutionContext::AccumulateOuterFragment) {
      outerLoopFragEmpty = false;
      growOpx(opx, seqs[&progs.accumulateOuterFragment()]);
      contextOpRegistry[{context, taskId}].push_back(op);
    } else {
      auto found = requiredRecomputes.find(taskId);
      if (found != requiredRecomputes.end()) {
        auto &rerunSchedule = found->second;
        for (auto opToRerun : rerunSchedule) {
          logging::devicex::debug("Adding (second) recompute Op {}",
                                  opToRerun->debugName());

          seqs[&progs.backwardFragment()].add(
              progs.recomputeFragment(opToRerun->id));
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
      contextOpRegistry[{context, taskId}].push_back(op);
    }
  }

  else {
    throw internal_error("Unknown SchedulePreLoss in prepare, should "
                         "updateVertices have been called recently?");
  }
}

void IrLowering::pipelinedOpTaskFunc(TaskId taskId, Op *op, SequenceMap &seqs) {
  Opx *opx                 = getOpx(op->id);
  ExecutionContext context = op->settings.executionContext;

  if (op->copiesOptimizerTensors()) {
    growOpx(opx, seqs[&progs.streamOptimizerFromHostFragment()]);
  }

  else if (ir().getSessionOptions().enableGradientAccumulation &&
           context == ExecutionContext::AccumulateOuterFragment) {
    outerLoopFragEmpty = false;
    growOpx(opx, seqs[&progs.accumulateOuterFragment()]);
    contextOpRegistry[{context, taskId}].push_back(op);
  }

  // Special case for running operators before the main loop.
  else if (context == ExecutionContext::WeightsFromHostFragment) {
    growOpx(opx, seqs[&progs.streamWeightsFromHostFragment()]);
    contextOpRegistry[{context, taskId}].push_back(op);
  }

  // Special case for running operators before the main loop.
  else if (context == ExecutionContext::WeightsToHostFragment) {
    // Special case for running operators before the main loop.
    growOpx(opx, seqs[&progs.weightsToHostFragment()]);
    contextOpRegistry[{context, taskId}].push_back(op);
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
        seqs[&progs.pipelineForwardFragment(op->getPipelineStage(),
                                            "recompute of " + opToRerun->str())]
            .add(progs.recomputeFragment(opToRerun->id));

        contextOpRegistry[{context, taskId}].push_back(opToRerun);
      }
    }

    if (op->isConvertibleTo<IpuCopyOp>()) {
      // IpuCopyOps are handled as a special case in pipelining. Here,
      // the destination tensor is created using the
      // `createPipelinedOutput` method. Later, for each pipeline cycle
      // the copy appears in, a new copy program is added to the cycles
      // sequence using `IpuCopyOpx::growPipelined`.
      dynamic_cast<IpuCopyOpx *>(opx)->createPipelinedOutput();
    } else if (op->isConvertibleTo<RestoreOp>()) {
      // Restore Operations are required to run at the start of a pipelineStage
      growOpx(opx,
              progs.pipelineRestoreFragment(op->getPipelineStage(), op->str()));
    } else if (op->settings.recomputeType == RecomputeType::Checkpoint ||
               op->settings.recomputeType == RecomputeType::Undefined) {
      logging::devicex::debug(
          "Adding post-turning check-point Op {} {} in pipelinedOpTaskFunc",
          op->str(),
          op->debugName());
      auto seqsKey =
          &progs.pipelineForwardFragment(op->getPipelineStage(), op->str());
      logging::devicex::debug("Obtained pipeline forward frag for ",
                              op->debugName());
      auto found_ = seqs.find(seqsKey);
      if (found_ == seqs.end()) {
        seqs[seqsKey] = poplar::program::Sequence{};
        found_        = seqs.find(seqsKey);
      }
      logging::devicex::debug(
          "Growing {} {} in pipelinedOpTaskFunc", op->str(), op->debugName());

      growOpx(opx, found_->second);
    } else if (op->settings.recomputeType == RecomputeType::Recompute) {
      logging::devicex::debug("Adding (first) recompute Op {}",
                              op->debugName());

      growOpx(opx, progs.createRecomputeFragment(op->id));

      seqs[&progs.pipelineForwardFragment(op->getPipelineStage(), op->str())]
          .add(progs.recomputeFragment(op->id));
    }
    contextOpRegistry[{context, taskId}].push_back(op);
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

  unsigned globalReplicationFactor = 1;
  if (ir().getSessionOptions().enableDistributedReplicatedGraphs) {
    globalReplicationFactor =
        static_cast<unsigned>(ir().getSessionOptions().globalReplicationFactor);
  }

  else {
    // A check on user input consistency
    if (static_cast<unsigned>(
            ir().getSessionOptions().globalReplicationFactor) > 1) {
      throw error("enableDistributedReplicatedGraphs is false, but "
                  "globalReplicationFactor > 1. "
                  "Either enable global replicated graphs, or set the "
                  "globalReplicationFactor "
                  "to 1");
    }
  }

  return globalReplicationFactor;
}

bool IrLowering::isReplicatedGraph() const {
  const bool isLocallyReplicated  = getReplicationFactor() > 1;
  const bool isGloballyReplicated = getGlobalReplicationFactor() > 1;
  return isLocallyReplicated || isGloballyReplicated;
}

unsigned IrLowering::getAccumulationFactor() const {

  unsigned accumulationFactor = 1;
  if (ir().getSessionOptions().enableGradientAccumulation) {
    accumulationFactor =
        static_cast<unsigned>(ir().getSessionOptions().accumulationFactor);
  }

  else {
    // A check on user input consistency
    if (static_cast<unsigned>(ir().getSessionOptions().accumulationFactor) >
        1) {
      throw error(
          "enableGradientAccumulation is false, but accumulationFactor > 1. "
          "Either enable gradient accumulation, or set the accumulation factor "
          "to 1");
    }
  }

  return accumulationFactor;
}

PipelineInfo IrLowering::pipelineInfo() const { return pInfo; }

// Floating point settings are not suported on CPU
void IrLowering::setFloatingPointBehaviour(poplar::Graph &graph) {

  if (ir().getSessionOptions().enableFloatingPointChecks) {
    if (deviceInfo->getType() == DeviceType::Ipu) {
      logging::devicex::info("Enabling all floating point checks");
      // Not enabling stochasitc rounding, that is done in a seperate call
      poplar::FloatingPointBehaviour behaviour(true, true, true, false, true);
      poplar::setFloatingPointBehaviour(
          graph, progs.initFragment(), behaviour, "/init");
    } else {
      logging::devicex::warn(
          "Floating point checks cannot be enabled for non IPU devices");
    }
  }
}

// Stocastic rounding is only supported on the IPU
void IrLowering::setStochasticRoundingBehaviour(poplar::Graph &graph) {

  if (ir().getSessionOptions().enableStochasticRounding) {
    if (deviceInfo->getType() == DeviceType::Ipu) {
      logging::devicex::info("Enabling stochastic rounding");
      bool behaviour = true;
      poplar::setStochasticRounding(
          graph, progs.initFragment(), behaviour, "/init");
    } else {
      logging::devicex::warn(
          "Stochastic rounding cannot be enabled for non IPU devices");
    }
  }
}

void IrLowering::prepareGraph() {
  POPART_TRACEPOINT();
  if (prepareGraphHasBeenCalled_) {
    logging::devicex::info("Poplar graph has already been prepared");
    return;
  }

  logging::devicex::info("Poplar version: {}", poplar::versionString());
  logging::devicex::info("Poplar release githash: {}", poplar::packageHash());

  tryLoadExecutable();
  logging::devicex::info("Loaded executable");

  initPoplarGraph();

  logging::devicex::info("Poplar graph initialised");

  popops::addCodelets(graph());
  poplin::addCodelets(graph());
  popnn::addCodelets(graph());
  poprand::addCodelets(graph());

  // Add custom codelets as per the user provided list of paths. Allow poplar to
  // infer the file type from the extension. Also feed through the compile
  // flags.
  for (auto codelet : ir().getSessionOptions().customCodelets) {
    logging::devicex::info("Adding codelet: {}", codelet);
    graph().addCodelets(codelet,
                        poplar::CodeletFileType::Auto,
                        ir().getSessionOptions().customCodeletCompileFlags);
  }

  setFloatingPointBehaviour(graph());
  setStochasticRoundingBehaviour(graph());

  // Initialize the liveness analyzer
  livenessAnalyzer.reset(new liveness::LivenessAnalyzer(&ir()));
  aliasZeroCopy.reset(
      new liveness::AliasZeroCopy(&ir(), livenessAnalyzer.get()));

  livenessAnalyzer->apply();

  if (ir().getSessionOptions().aliasZeroCopy) {
    aliasZeroCopy->apply();
  }

  if (ir().virtualGraphsEnabled()) {
    auto numIPUs     = graph().getTarget().getNumIPUs();
    auto tilesPerIPU = graph().getTarget().getTilesPerIPU();

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

      if (numIOTiles > tilesPerIPU) {
        throw error("Number of IO tiles {} exceeds tiles per IPU {}. ",
                    numIOTiles,
                    tilesPerIPU);
      }

      const auto computeTiles =
          gcl::perIPUTiles(graph(), numIOTiles, tilesPerIPU - numIOTiles, true);

      std::vector<unsigned> ioTiles;
      ioTiles.reserve(numIOTiles);
      unsigned j = 0;
      for (unsigned i = 0; i < tilesPerIPU; ++i) {
        if (j < computeTiles.size() && computeTiles[j] == i) {
          ++j;
        } else {
          ioTiles.push_back(i);
        }
      }

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
    for (Op *op : ir().getOpSchedule({}, RequireOptimalSchedule::Yes)) {
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
    auto numIPUs = graph().getTarget().getNumIPUs();
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

  logging::devicex::info("Turning Ops into Opxes");

  // create an Opx for every Op
  for (Op *op : ir().getOpSchedule({}, RequireOptimalSchedule::Yes)) {
    logging::devicex::trace("Creating OPX for {}", op->debugName());
    opxs[op->id] = createOpx(op);
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
    if (tensor->tensorLocationInfo.isRemote())
      continue;

    // 1
    tasks.add(initTensorTask(tensor));

    if (!ir().streamingIsDisabledForTensor(id)) {
      // 2
      tasks.add(streamFromHostTask(tensor));
      // 3
      tasks.add(fromHostTask(tensor, progs.streamWeightsFromHostFragment()));
      // 4
      tasks.add(streamToHostTask(tensor, false));
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
    tasks.add(initTensorTask(tensor));
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
    Opx *opx = getOpx(op->id);
    for (auto t_inds : op->output->indicesMap()) {
      if (opx->outputCreatedExternally(t_inds.second.front())) {
        logging::devicex::trace("Adding {} output initTensorTask for {}",
                                op->debugName(),
                                t_inds.first->id);
        tasks.add(initTensorTask(t_inds.first));
      }
    }
  }

  // stream-to-device tensors :
  // 1) make tensor
  // THEN
  // 2) make stream
  // OR
  // 2) set initial value (if using synthetic data).
  for (auto id : ir().getTensorIds(TensorType::Stream)) {
    Tensor *tensor = ir().getTensor(id);

    // 1
    tasks.add(initTensorTask(tensor));
    logging::devicex::debug("Adding initTensorTask for Stream {}", id);

    // 2
    if (ir().useSyntheticData()) {
      tasks.add(setInitTensorValTask(tensor));
    } else {
      tasks.add(streamFromHostTask(tensor));
    }
  }

  // Init the random seed
  if (ir().getRequiresRandomSeed() and !ir().useSyntheticData()) {
    auto seedTen = ir().getTensor(GetRandomSeedOp::getStreamedSeedTensorId());
    tasks.add(fromHostTask(seedTen, progs.setRandomSeedFromHostFragment()));
    tasks.add(initRandomSeed());
  }

  if (ir().getSessionOptions().enableLoadAndOffloadRNGState) {
    tasks.add(initRngStateTensor());
    tasks.add(rngStateFromHost());
    tasks.add(rngStateToHost());
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
  if (ir().useSyntheticData() == false) {
    for (auto anchorId : ir().getDataFlow().anchors()) {
      Tensor *tensor = ir().getTensor(anchorId);

      tasks.add(streamToHostTask(tensor, true));

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

    // create Program to write optimizer tensors to device
    for (auto tensor : ir().optimizerTensors()) {
      tasks.add(fromHostTask(tensor, progs.streamOptimizerFromHostFragment()));
    }

    for (Tensor *tensor : ir().dataStreamTensors()) {
      if (ir().getSessionOptions().enablePipelining) {
        PipelineStage ps = *tensor->consumers.findLowestPipelineStage();
        auto &sq = progs.pipelineToDeviceStreamFragment(ps, tensor->str());
        tasks.add(fromHostTask(tensor, sq));
      } else {
        auto &sq = progs.forwardOrBackwardFragment(tensor->scheduledPreLoss);
        tasks.add(fromHostTask(tensor, sq));
      }
    }
  }

  addOpTasks(tasks);

  if (ir().getSessionOptions().enablePipelining) {
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
                                             DependencyType::Tensor});

  logging::devicex::debug("Creating linear task schedule with OUTPUT, "
                          "SUBGRAPH and SCHEDULER dependencies.");
  auto emplaceSchedule = tasks.getLinearised({DependencyType::Output,
                                              DependencyType::SubGraph,
                                              DependencyType::Scheduler});

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
        for (auto seq : seqs[emplaceTask.name]) {
          // Emplace intermediate sequence in final sequence
          seq.first->add(seq.second);
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

  for (auto &createTask : createSchedule) {
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
    seqs[createTask.name] = createTask.f();
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

  if (ir().getSessionOptions().exportPoplarVertexGraph) {
    std::ofstream strm;
    strm.open("poplar_vertex_graph.dot", std::ios::out);
    graph().outputVertexGraph(strm, progs.progs());
  }

  if (ir().getSessionOptions().exportPoplarComputationGraph) {
    std::ofstream strm;
    strm.open("poplar_compute_graph.dot", std::ios::out);
    graph().outputComputeGraph(strm, progs.progs());
  }

  prepareGraphHasBeenCalled_ = true;
}

void IrLowering::compileAndExport(const std::string &executablePath,
                                  const std::string &weightsPath) {
  POPART_TRACEPOINT();
  if (!getDeviceInfo()->canCompileOffline()) {
    std::ostringstream oss;
    oss << getDeviceInfo()->getType();

    throw error("Offline compilation is not supported for device type {}",
                oss.str());
  }
  if (weightsPath.empty() && executablePath.empty()) {
    throw error(
        "At least one of weightsPath or executablePath must be provided");
  }
  if (!exporterIsAvailable()) {
    throw error("Exporter not available");
  }

  prepareGraph();

  if (!ir().getSessionOptions().compileEngine) {
    throw error(
        "The engine must be compiled before the executable can be exported");
  }
  if (!weightsPath.empty()) {
    if (dv_p == nullptr) {
      throw error("IrLowering::setDevice has not been called.");
    }
    exportWeights(*dv_p, weightsPath);
  }
  if (executablePath.empty()) {
    return;
  }
  try {
    // Regroup programs in 3 programs: HOST_TO_DEVICE / MAIN_SEQUENCE /
    // DEVICE_TO_HOST
    std::vector<poplar::program::Program> fusedProgs;
    const auto programs = progs.progs();
    poplar::program::Sequence fusedH2d, fusedMain, fusedD2h;

    static_assert(PopPrograms::ProgramIndex::N == 8,
                  "Make sure all the programs are added to one of the 3 "
                  "sequences below.");

    fusedH2d.add(programs[PopPrograms::ProgramIndex::WeightsFromHost]);
    fusedH2d.add(programs[PopPrograms::ProgramIndex::OptimizerFromHost]);
    fusedH2d.add(programs[PopPrograms::ProgramIndex::SetRandomSeedFromHost]);

    fusedMain.add(programs[PopPrograms::ProgramIndex::Program]);

    fusedD2h.add(programs[PopPrograms::ProgramIndex::WeightstoHost]);
    fusedD2h.add(programs[PopPrograms::ProgramIndex::CycleCountTensortoHost]);

    fusedProgs.push_back(fusedH2d);
    fusedProgs.push_back(fusedMain);
    fusedProgs.push_back(fusedD2h);

    auto executable = poplar::compileGraph(
        graph(), fusedProgs, engineOptions, progressLogger);
    auto numIPUs = graph().getTarget().getNumIPUs();
    logging::devicex::info("Exporting compiled executable");
    if (dv_p == nullptr) {
      throw error("IrLowering::setDevice has not been called.");
    }
    exportExecutable(executable,
                     *dv_p,
                     engineOptions,
                     deviceInfo->getOptionFlags(),
                     SavedInfo(this).toString(),
                     numIPUs,
                     executablePath);
  } catch (const poplar::graph_memory_allocation_error &e) {
    // If the creation of the engine throw an exception due to memory
    // allocation i.e. the program does not fit show graph profile and
    // re-throw the exception In certain cases poplar will throw the error
    // without a graph profile. The following engine option needs to be set to
    // enable the graph profile in this case "debug.allowOutOfMemory":"true"

    trySaveTensorTileMap();

    logging::devicex::err("Memory allocation error : {}", e.what());
    throw devicex_memory_allocation_err(e, reportOptions);
  }
}

poplar::program::Sequence &IrLowering::getAnchorReturnFragment(Tensor *tensor) {
  if (ir().getSessionOptions().enablePipelining) {
    auto isOptimizerTensorCopy = [&](Op *x) {
      return x->isConvertibleTo<IpuCopyOp>() &&
             dynamic_cast<IpuCopyOp *>(x)->copiesOptimizerTensors();
    };

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
  } else {
    return tensor->tensorType() == TensorType::Variable
               ? progs.backwardFragment()
               : progs.forwardOrBackwardFragment(tensor->scheduledPreLoss);
  }
}

poplar::Executable IrLowering::getExecutable() {
  if (!prepareGraphHasBeenCalled_) {
    throw internal_error("Devicex::prepareGraph() must be called before"
                         " Devicex::getExecutable() is called.");
  }

  if (cachedExecutable) {
    // return the executable in cachedExecutable while ensuring
    // cachedExecutable is set to nonstd::nullopt
    nonstd::optional<poplar::Executable> result = nonstd::nullopt;
    boost::swap(cachedExecutable, result);
    logging::devicex::info("Returing CachedExecutable");

    return std::move(result.value());
  } else {
    try {
      logging::devicex::warn("Starting Engine compilation");

      auto executable = poplar::compileGraph(
          graph(), progs.progs(), engineOptions, progressLogger);

      logging::devicex::warn("Graph compiled");

      trySaveExecutable(executable);
      return executable;
    } catch (const poplar::graph_memory_allocation_error &e) {
      // If the compilations throws an exception due to memory
      // allocation i.e. the program does not fit show graph profile and
      // re-throw the exception In certain cases poplar will throw the error
      // without a graph profile. The following engine option needs to be set to
      // enable the graph profile in this case "debug.allowOutOfMemory":"true"

      trySaveTensorTileMap();

      logging::devicex::err("Memory allocation error : {}", e.what());
      throw devicex_memory_allocation_err(e, reportOptions);
    }
  }
}

std::string IrLowering::getPoplarCachePath() {
  return ir().getSessionOptions().cachePath + ".poplar";
}

std::string IrLowering::getPopartCachePath() {
  return ir().getSessionOptions().cachePath + ".popart";
}

void IrLowering::trySaveExecutable(poplar::Executable &executable) {
  POPART_TRACEPOINT();
  auto cachePath    = ir().getSessionOptions().cachePath;
  auto cacheEnabled = ir().getSessionOptions().enableEngineCaching;

  if (cacheEnabled && !cachePath.empty() &&
      deviceInfo->getType() == DeviceType::Ipu) {

    // If target directory does not exist, create it
    auto cachePathObj = boost::filesystem::path(cachePath);
    if (cachePathObj.has_parent_path()) {
      auto cacheDir = cachePathObj.parent_path();
      if (!boost::filesystem::exists(cacheDir)) {
        logging::devicex::warn("Specified cache directory not found. "
                               "Creating {} directory ",
                               cacheDir);
        if (!boost::filesystem::create_directory(cacheDir))
          throw error("Cannot create cache directory. Aborting.");
      }
    }
    // save the poplar executable
    auto poplarCachePath = getPoplarCachePath();
    std::ofstream poplarFs(poplarCachePath, std::ofstream::binary);
    logging::devicex::warn("Saving poplar Executable to '{}'", poplarCachePath);
    executable.serialize(poplarFs);

    // save the popart ir hash
    auto popartCachePath = getPopartCachePath();
    std::ofstream popartFs(popartCachePath, std::ofstream::binary);
    logging::devicex::warn("Saving popart ir hash to '{}'", popartCachePath);
    SavedInfo savedInfo(this);
    savedInfo.serialize(popartFs);
  }
}

void IrLowering::tryLoadExecutable() {
  POPART_TRACEPOINT();
  auto warn = [&](const std::string &msg) {
    logging::devicex::warn("Unable to load cached poplar::Executable, {}", msg);
  };

  auto cachePath    = ir().getSessionOptions().cachePath;
  auto cacheEnabled = ir().getSessionOptions().enableEngineCaching;

  if (cacheEnabled && !cachePath.empty() &&
      deviceInfo->getType() == DeviceType::Ipu) {
    // load the popart ir hash
    auto popartCachePath = getPopartCachePath();
    std::ifstream popartFs(popartCachePath, std::ifstream::binary);
    if (popartFs.is_open()) {
      if (SavedInfo(this) == SavedInfo::deserialize(popartFs)) {
        auto poplarCachePath = getPoplarCachePath();
        std::ifstream poplarFs(poplarCachePath, std::ifstream::binary);
        if (poplarFs.is_open()) {
          logging::devicex::warn("Loading poplar Executable from '{}'",
                                 cachePath);
          cachedExecutable.emplace(poplar::Executable::deserialize(poplarFs));
          usingCachedExecutable_ = true;
        } else {
          warn(logging::format("could not open file `{}'", poplarCachePath));
        }
      } else {
        warn("ir hashes differ");
      }
    } else {
      warn(logging::format("could not open file `{}'", popartCachePath));
    }
  }
}

TaskId IrLowering::streamFromHostTaskId(TensorId id) {
  return "streamFromHostTask_" + id;
}

TaskId IrLowering::setInitTensorValTaskId(TensorId id) {
  return "setInitTensorValTask_" + id;
}

TaskId IrLowering::streamToHostTaskId(TensorId id, bool isAnchorStream) {
  std::string anchorPrefix = isAnchorStream ? "anchor" : "weight";
  return anchorPrefix + "StreamToHostTask_" + id;
}

TaskId IrLowering::fromHostTaskId(TensorId id) { return "fromHostTask_" + id; }

TaskId IrLowering::toHostTaskId(TensorId id, bool isAnchorStream) {
  if (isAnchorStream) {
    return "anchorToHostTask_" + id;
  }
  return "weightToHostTask_" + id;
}

TaskId IrLowering::anchorSumTaskId(const TensorId &id) {
  return "anchorSumTask_" + id;
}

TaskId IrLowering::initBatchCounterTensorsTaskId() {
  return "initBatchCounterTensorsTask";
}

TaskId IrLowering::updateBatchCountTaskId() { return "updateBatchCountTask"; }

TaskId IrLowering::initRandomSeedTaskId() { return "initRandomSeedTask"; }

TaskId IrLowering::rngStateFromHostTaskId() { return "rngStateFromHostTask"; }

TaskId IrLowering::rngStateToHostTaskId() { return "rngStateToHostTask"; }

TaskId IrLowering::initRngStateTensorTaskId() {
  return "initRngStateTensorTask";
}

TaskId IrLowering::initTensorTaskId(TensorId id) {
  return "initTensorTaskId_" + id;
}

TaskId IrLowering::opTaskId(Op *op) {

  std::stringstream ss;
  ss << "fromOpTask_" << op->id << '_' << op->opid;
  return ss.str();
}

TaskId IrLowering::pipelinedCopyTaskId(Op *op) {

  std::stringstream ss;
  ss << "pipelinedCopyTask_" << op->id << "_" << op->opid;
  return ss.str();
}

PriTask IrLowering::fromHostTask(Tensor *tensor,
                                 poplar::program::Sequence &sq) const {
  double priority;
  if (ir().getSessionOptions().groupHostSync) {
    priority = std::numeric_limits<double>::max();
  } else {
    priority = -1e6; // writes to device: always as late as possible (default)
  }
  auto f = [&sq, tensor, this]() {
    SequenceMap seqs;
    logging::devicex::debug("Adding poplar::program::Copy from host " +
                            tensor->id);

    seqs[&sq].add(poplar::program::Copy(fromHostStreams.at(tensor->id),
                                        tensors_.get(tensor->id),
                                        doRearrangeOnHost(tensor)));
    return seqs;
  };
  return {priority,
          fromHostTaskId(tensor->id),
          {
              {streamFromHostTaskId(tensor->id),
               DependencyType::Tensor}, // poplar::Stream created
              {initTensorTaskId(tensor->id),
               DependencyType::Tensor} // poplar::Tensor created
          },
          f};
}

PriTask IrLowering::toHostTask(Tensor *tensor,
                               poplar::program::Sequence &sq,
                               ToHostStreamType stype) const {

  auto f = [&sq, tensor, this, stype]() {
    SequenceMap seqs;
    logging::devicex::debug("Adding poplar::program::Copy to host "
                            "(Type: {}) {}",
                            static_cast<int>(stype),
                            tensor->id);

    auto pToHostStreams = &toHostAnchorStreams;
    if (stype == ToHostStreamType::NonAnchor) {
      pToHostStreams = &toHostWeightStreams;
    }
    const auto &poplarStream = pToHostStreams->at(tensor->id);
    const auto &anchorTensor =
        stype == ToHostStreamType::SumAnchor
            ? tensors_.get(anchorSumPrefix() + tensor->id)
            : tensors_.get(tensor->id);
    // verify that number of elements of poplar Tensor and poplar Stream are the
    // same
    auto nElmsStream = poplarStream.numElements();
    auto nElmsTensor = anchorTensor.numElements();
    if (nElmsStream != nElmsTensor) {
      throw internal_error("[Devicex::toHostTask] "
                           "The poplar::Tensor {} has {}, whereas the "
                           "poplar::Stream has {}. These should be the same.",
                           tensor->id,
                           nElmsTensor,
                           nElmsStream);
    }

    seqs[&sq].add(poplar::program::Copy(
        anchorTensor, poplarStream, doRearrangeOnHost(tensor)));
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

  std::vector<std::pair<TaskId, DependencyType>> deps = {
      // the dependencies:
      // poplar::Stream creation task,
      {streamToHostTaskId(tensor->id, stype != ToHostStreamType::NonAnchor),
       DependencyType::Output},
      // poplar::Tensor has its final values
      {finalPopulator, DependencyType::Scheduler}};

  if (stype == ToHostStreamType::SumAnchor) {
    deps.push_back({anchorSumTaskId(tensor->id), DependencyType::Tensor});
  }
  double priority;
  if (ir().getSessionOptions().groupHostSync) {
    priority = -std::numeric_limits<double>::max();
  } else {
    priority = +1e6; // writes to host: always as soon as possible (default)
  }
  return {priority, taskId, deps, f};
}

PriTask IrLowering::anchorReturnTypeSumTask(Tensor *tensor,
                                            poplar::program::Sequence &sq) {
  auto f = [&sq, tensor, this]() {
    SequenceMap seqs;

    const auto &poplarTensor     = tensors_.get(tensor->id);
    const TensorId accumulatorId = anchorSumPrefix() + tensor->id;
    auto accumulatorTensor       = graph().clone(poplarTensor, accumulatorId);
    tensors_.insertUnsafe(accumulatorId, accumulatorTensor);

    logging::devicex::debug("Adding AnchorSum operations to {}", tensor->id);
    popops::addInPlace(graph(),
                       accumulatorTensor,
                       poplarTensor,
                       seqs[&sq],
                       "AnchorSum_" + tensor->id);
    // Zero the accumulator
    popops::zero(graph(),
                 accumulatorTensor,
                 seqs[&progs.initFragment()],
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
           // poplar::Stream creation task,
           {streamToHostTaskId(tensor->id, true), DependencyType::Output},
           // poplar::Tensor has its final values
           {finalPopulator, DependencyType::Scheduler}},
          f};
}

PriTask IrLowering::initBatchCounterTensorsTask(poplar::program::Sequence &sq) {

  auto f = [&sq, this]() {
    logging::devicex::debug("Adding batch counter tensors");

    poplar::Tensor falseConst = getConst(graph(), poplar::BOOL, {}, 0, "false");

    // Add scalar tensors outside of the ir to track the batch
    // Id and decide when to execute the copy to the host
    for (ReturnPeriod N : ir().getDataFlow().rps()) {
      // Add to map so copy task can access
      batchCountingTensors[N] = getScalarVariable(graph(), poplar::INT, "");
      batchCountCheckingTensors[N] =
          getScalarVariable(graph(), poplar::BOOL, "");

      getConst(graph(), poplar::INT, {}, N, "batchCounter");

      poputil::mapTensorLinearly(graph(), batchCountingTensors[N]);
      poputil::mapTensorLinearly(graph(), batchCountCheckingTensors[N]);

      // Set the initial values of the tensors_.
      popops::zero(graph(),
                   batchCountingTensors[N],
                   sq,
                   logging::format("initBatchCountTensors[{}]", N));
      sq.add(poplar::program::Copy(falseConst, batchCountCheckingTensors[N]));
    }

    // Make sure const 1 tensor exists
    getConst(graph(), poplar::INT, {}, 1, "one");
    return SequenceMap();
  };

  return {+1e6, // followed by writes to host: always as early as possible
          initBatchCounterTensorsTaskId(),
          {},
          f};
}

PriTask IrLowering::updateBatchCountTask(poplar::program::Sequence &sq) {

  auto f = [&sq, this]() {
    SequenceMap seqs;
    logging::devicex::debug("Adding batch count checker program");

    // Placeholder 'do nothing' branch if not running assign program
    poplar::program::Sequence emptyseq;

    // Increment the batch count at the at the earliest point
    // the anchor tensor is required, and check if it is a
    // copy batch
    for (ReturnPeriod N : ir().getDataFlow().rps()) {
      popops::addInPlace(
          graph(),
          batchCountingTensors[N],
          getConst(graph(), poplar::INT, {}, 1, "batchCount/one"),
          seqs[&sq]);

      batchCountCheckingTensors[N] =
          popops::eq(graph(),
                     batchCountingTensors[N],
                     getConst(graph(), poplar::INT, {}, N, "batchCount/n"),
                     seqs[&sq]);

      // Reset batch count once it has reached N
      auto zero = getConst(graph(), poplar::INT, {}, 0, "batchCount/zero");
      seqs[&sq].add(poplar::program::If(
          batchCountCheckingTensors[N],
          poplar::program::Copy(zero, batchCountingTensors[N]),
          emptyseq));
    }
    return seqs;
  };

  return {+1e6, // followed by writes to host: always as early as possible
          updateBatchCountTaskId(),
          {
              {initBatchCounterTensorsTaskId(),
               DependencyType::Tensor} // poplar::Tensor creation task
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
                                            poplar::program::Sequence &sq) {

  auto f = [&sq, tensor, N, this]() {
    SequenceMap seqs;
    logging::devicex::debug(
        "Adding conditional poplar::program::Copy to host " + tensor->id);

    poplar::Tensor isNthBatch = batchCountCheckingTensors.at(N);

    poplar::program::Sequence copyseq;
    copyseq.add(poplar::program::Copy(tensors_.get(tensor->id),
                                      toHostAnchorStreams.at(tensor->id),
                                      doRearrangeOnHost(tensor)));

    // Placeholder 'do nothing' branch if not running copy program
    poplar::program::Sequence emptyseq;

    seqs[&sq].add(poplar::program::If(isNthBatch, copyseq, emptyseq));
    return seqs;
  };

  bool isAnchorStream = true;
  return {
      +1e6, // writes to host: always as early as possible
      toHostTaskId(tensor->id, isAnchorStream),
      {// the dependencies:
       // updating poplar::Tensor task,
       {updateBatchCountTaskId(), DependencyType::Output},
       // poplar::Stream creation task,
       {streamToHostTaskId(tensor->id, isAnchorStream), DependencyType::Output},
       // poplar::Tensor value setting task
       {taskWhichPopulates(tensor->id), DependencyType::Scheduler}},
      f};
}

bool IrLowering::doRearrangeOnHost(Tensor *tensor) const {
  if (tensor->tensorType() == TensorType::Variable) {
    return true;
  } else if (tensor->tensorType() == TensorType::Stream) {
    return false;
  } else if (ir().isAnchored(tensor->id)) {
    return ir().getSessionOptions().rearrangeAnchorsOnHost;
  }
  return true;
}

void IrLowering::initPoplarGraph() {
  POPART_TRACEPOINT();
  poplar::Target popTarget;
  unsigned replicationFactor = 0;

  if (ir().getSessionOptions().enableDistributedReplicatedGraphs) {
    auto globalReplicationFactor = getGlobalReplicationFactor();
    auto localReplicationFactor = ir().getSessionOptions().replicatedGraphCount;
    auto numInstances = globalReplicationFactor / localReplicationFactor;

    auto globalNumIpus  = deviceInfo->getNumIpus() * numInstances;
    auto &ipuSystemType = ir().getSessionOptions().ipuSystemType;

    replicationFactor = globalReplicationFactor;

    logging::devicex::debug("Creating distributed replicated graph with global "
                            "replication factor {}",
                            replicationFactor);
    switch (deviceInfo->getType()) {
    case DeviceType::Ipu: {
      popTarget = poplar::Target::createIPUTarget(
          static_cast<unsigned>(globalNumIpus), ipuSystemType);
      break;
    }
    case DeviceType::Cpu:
    case DeviceType::IpuModel:
    case DeviceType::OfflineIpu:
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

  pGraph.reset(new poplar::Graph(
      popTarget, poplar::replication_factor(replicationFactor)));
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

  case DataType::UNDEFINED:
  case DataType::UINT8:
  case DataType::INT8:
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
    throw error("Is there a poplar type for " + info.data_type() + "?");
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

poplar::RemoteBuffer &
IrLowering::getOrCreateHostReduceRemoteBuffer(TensorId tensorId,
                                              TensorInfo tensorInfo,
                                              poplar::Graph &graph) {
  auto entry = hostReduceRemoteBuffers.find(tensorId);

  if (entry == hostReduceRemoteBuffers.end()) {
    auto remoteBuffer = graph.addRemoteBuffer(
        tensorId, popType(tensorInfo), tensorInfo.nelms(), 1, true);

    hostReduceRemoteBuffers.emplace(tensorId, remoteBuffer);
    entry = hostReduceRemoteBuffers.find(tensorId);
  }

  return entry->second;
}

poplar::DataStream &
IrLowering::insertGradientStoreStream(TensorId tensorId,
                                      TensorInfo tensorInfo,
                                      poplar::Graph &graph) {
  auto streamMapEntry = toHostGradientStreams.find(tensorId);

  if (streamMapEntry == toHostGradientStreams.end()) {
    if (ir().getSessionOptions().hostAllReduceRemoteBuffer) {
      toHostGradientStreams.emplace(
          tensorId,
          poplar::DataStream(graph.addDeviceToHostFIFO(
              gradientStoreStreamId(tensorId), poplar::CHAR, 1)));
    } else {
      toHostGradientStreams.emplace(
          tensorId,
          poplar::DataStream(
              graph.addDeviceToHostFIFO(gradientStoreStreamId(tensorId),
                                        popType(tensorInfo),
                                        tensorInfo.nelms())));
    }
    streamMapEntry = toHostGradientStreams.find(tensorId);
  } else {
    throw error("Tensor Id " + tensorId +
                " already exists in toHostGradientStreams");
  }

  return streamMapEntry->second;
}

poplar::DataStream &IrLowering::insertGradientLoadStream(TensorId tensorId,
                                                         TensorInfo tensorInfo,
                                                         poplar::Graph &graph) {
  auto streamMapEntry = fromHostGradientStreams.find(tensorId);

  if (streamMapEntry == fromHostGradientStreams.end()) {
    if (ir().getSessionOptions().hostAllReduceRemoteBuffer) {
      fromHostGradientStreams.emplace(
          tensorId,
          poplar::DataStream(graph.addHostToDeviceFIFO(
              gradientLoadStreamId(tensorId), poplar::CHAR, 1)));
    } else {
      fromHostGradientStreams.emplace(
          tensorId,
          poplar::DataStream(graph.addHostToDeviceFIFO(
              gradientLoadStreamId(tensorId),
              popType(tensorInfo),
              tensorInfo.nelms(),
              poplar::ReplicatedStreamMode::BROADCAST)));
    }
    streamMapEntry = fromHostGradientStreams.find(tensorId);
  } else {
    throw error("Tensor Id " + tensorId +
                " already exists in fromHostGradientStreams");
  }

  return streamMapEntry->second;
}

poplar::DataStream &IrLowering::insertWeightLoadStream(TensorId tensorId,
                                                       TensorInfo tensorInfo,
                                                       poplar::Graph &graph) {
  auto streamMapEntry = fromHostWeightLoadStreams.find(tensorId);

  if (streamMapEntry == fromHostWeightLoadStreams.end()) {
    fromHostWeightLoadStreams.emplace(
        tensorId,
        poplar::DataStream(graph.addHostToDeviceFIFO(
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

const std::vector<TensorId> &IrLowering::getHostReduceStreamIds() const {
  return hostReduceStreamIds;
}

std::vector<TensorId> &IrLowering::getHostReduceStreamIds() {
  return hostReduceStreamIds;
}

const std::map<TensorId, poplar::RemoteBuffer> &
IrLowering::getHostReduceRemoteBuffers() const {
  return hostReduceRemoteBuffers;
}

} // namespace popx
} // namespace popart
