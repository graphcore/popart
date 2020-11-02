// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/range/algorithm/find.hpp>
#include <cmath>
#include <memory>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/boundary.hpp>
#include <popart/op/call.hpp>
#include <popart/op/dropout.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/remote.hpp>
#include <popart/subgraph/iosubgraphcostmodel.hpp>
#include <popart/subgraph/outliner.hpp>
#include <popart/subgraph/prunematches.hpp>
#include <popart/subgraph/subgraphutil.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/subgraphoutline.hpp>
#include <popart/vendored/optional.hpp>

using boost::find;
using boost::algorithm::any_of;

namespace popart {
namespace {

using OpMap = std::unordered_map<OpId, Op *>;

// Create a map of OpIds to Op*.
// This map includes ops in all graphs in the Ir.
OpMap createOpMap(const Ir &ir) {
  OpMap opMap;
  for (auto &graphId_graphPtr : ir.getGraphs()) {
    auto graphPtr = graphId_graphPtr.second.get();
    for (auto &opId_opPtr : graphPtr->getOps()) {
      auto op = opId_opPtr.second.get();
      opMap.insert({op->id, op});
    }
  }
  return opMap;
}

template <typename T> void sortMatches(std::vector<T> &matches) {
  std::sort(matches.begin(), matches.end(), [=](T &p1, T &p2) {
    return std::tuple<int, std::vector<int>::size_type, int>(
               p2.length, p1.starts.size(), p2.starts.front()) <
           std::tuple<int, std::vector<int>::size_type, int>(
               p1.length, p2.starts.size(), p1.starts.front());
  });
}

// TODO T8888: templatize every function in this namespace so that can be used
// outside of popart, then put it in the popart/subgraph directory. Then add
// tests with the Blip class (as other basic outlining functions are).
namespace localoutline {
namespace {
using namespace fwtools::subgraph;

std::vector<int64_t> getBoundariesCrossed(const SessionOptions &opts,
                                          int64_t start,
                                          int64_t end,
                                          const std::vector<Op *> &schedule) {
  std::vector<int64_t> crossing;

  OptionalVGraphId vgid;
  OptionalVGraphId last_vgid;

  OptionalExecutionPhase phase;
  OptionalExecutionPhase last_phase;

  ExecutionContext exec_cont;
  ExecutionContext last_exec_cont;

  OptionalBatchSerializedPhase batchserial;
  OptionalBatchSerializedPhase last_batchserial;

  RecomputeType recompute      = RecomputeType::Undefined;
  RecomputeType last_recompute = RecomputeType::Undefined;

  auto aof_schedule = opts.accumulateOuterFragmentSettings.schedule;

  bool check_vgid_in_aof =
      (aof_schedule !=
       AccumulateOuterFragmentSchedule::OverlapCycleOptimized) &&
      (aof_schedule != AccumulateOuterFragmentSchedule::OverlapMemoryOptimized);

  bool overlap_phase = opts.executionPhaseSettings.phases > 1 &&
                       opts.executionPhaseSettings.schedule ==
                           ExecutionPhaseSchedule::BatchClusteredIO;

  bool overlap_batch_serial =
      opts.batchSerializationSettings.factor > 1 &&
      (opts.batchSerializationSettings.batchSchedule ==
           BatchSerializationBatchSchedule::OverlapOnCompute ||
       opts.batchSerializationSettings.batchSchedule ==
           BatchSerializationBatchSchedule::OverlapOnIo);

  for (int64_t i = start; i < end; ++i) {
    Op *op           = schedule[i];
    last_vgid        = vgid;
    last_phase       = phase;
    last_exec_cont   = exec_cont;
    last_batchserial = batchserial;
    last_recompute   = recompute;

    // Enable barriers between different sections of the schedules:
    // - Improves subgraph structures by dividing the schedule into
    //   logical units
    // - Speeds up the outlining match algorithm
    bool check_vgid        = true;
    bool check_phase       = true;
    bool check_batchserial = true;

    if (overlap_phase) {
      // Disable vgid and phase barriers when using the BatchClusteredIO
      // schedule, so that subgraphs can span multiple phases and virtual graphs
      check_vgid  = false;
      check_phase = false;
    }

    if (overlap_batch_serial) {
      // Don't insert barriers for outlining if a BSP overlap schedule is used,
      // since operations in different BSPs will be mixed in the schedule
      check_batchserial = false;
    }

    exec_cont   = op->settings.executionContext;
    batchserial = op->getOptionalBatchSerializedPhase();
    recompute   = op->settings.recomputeType == RecomputeType::Recompute
                    ? RecomputeType::Recompute
                    : RecomputeType::Checkpoint;

    check_vgid &= check_vgid_in_aof ||
                  exec_cont != ExecutionContext::AccumulateOuterFragment;

    if (i > start &&
        ((exec_cont != last_exec_cont) || (recompute != last_recompute) ||
         (check_vgid && vgid != last_vgid) ||
         (check_phase && phase != last_phase) ||
         (check_batchserial && batchserial != last_batchserial))) {
      crossing.push_back(i - start);
    }
  }
  return crossing;
}

// Cost model for computing runs of parallelizable sequences
class SoftParallelismModel {
public:
  SoftParallelismModel(std::vector<Op *> schedule_) : schedule(schedule_) {
    parallelSchedule.resize(schedule.size());
    for (size_t i = 0; i < schedule.size(); ++i) {
      // Op at position i can overlap with itself: [i, i + 1)
      parallelSchedule[i].first  = i;
      parallelSchedule[i].second = i + 1;
      for (size_t j = i; j >= 1; --j) {
        if (canSoftParallelize(i, j - 1)) {
          // Extend parallelizable sequence to [j - 1, i + 1)
          parallelSchedule[i].first = j - 1;
        } else {
          break;
        }
      }
      for (size_t j = i + 1; j < schedule.size(); ++j) {
        if (canSoftParallelize(i, j)) {
          // Extend parallelizable sequence to [i, j + 1)
          parallelSchedule[i].second = j + 1;
        } else {
          break;
        }
      }
    }
  }

  void log() {
    logging::transform::trace("[SubgraphOutline] Soft parallelism:");
    for (size_t i = 0; i < parallelSchedule.size(); ++i) {
      std::string a(parallelSchedule.at(i).first, ' ');
      std::string b(
          parallelSchedule.at(i).second - parallelSchedule.at(i).first, 'x');
      std::string c(schedule.size() - parallelSchedule.at(i).second, ' ');
      logging::transform::trace(
          "{}{}{} ({}: {})", a, b, c, i, schedule.at(i)->debugName());
    }
  }

  std::set<VGraphId> getVirtualGraphIds(Op *op) {
    std::set<VGraphId> vgids;
    for (auto &in : op->input->tensorMap()) {
      vgids.insert(op->getIntrospectionInVirtualGraphId(in.first).first);
    }
    for (auto &out : op->output->tensorMap()) {
      vgids.insert(op->getIntrospectionOutVirtualGraphId(out.first).first);
    }
    return vgids;
  }

  std::set<TileSet> getTileSets(Op *op) {
    std::set<TileSet> tileSets;
    for (auto &in : op->input->tensorMap()) {
      tileSets.insert(op->getIntrospectionInVirtualGraphId(in.first).second);
    }
    for (auto &out : op->output->tensorMap()) {
      tileSets.insert(op->getIntrospectionOutVirtualGraphId(out.first).second);
    }
    return tileSets;
  }

  bool virtualGraphOverlap(Op *op0, Op *op1) {
    auto set0 = getVirtualGraphIds(op0);
    auto set1 = getVirtualGraphIds(op1);

    std::set<VGraphId> intersect;
    std::set_intersection(set0.begin(),
                          set0.end(),
                          set1.begin(),
                          set1.end(),
                          std::inserter(intersect, intersect.begin()));
    return intersect.size();
  }

  bool tileSetOverlap(Op *op0, Op *op1) {
    auto set0 = getTileSets(op0);
    auto set1 = getTileSets(op1);

    std::set<TileSet> intersect;
    std::set_intersection(set0.begin(),
                          set0.end(),
                          set1.begin(),
                          set1.end(),
                          std::inserter(intersect, intersect.begin()));
    return intersect.size();
  }

  bool isRemoteOp(Op *op) {
    return op->isConvertibleTo<RemoteLoadOp>() ||
           op->isConvertibleTo<RemoteStoreOp>() ||
           op->isConvertibleTo<RemoteExchangeOp>();
  }

  bool canSoftParallelize(size_t p0, size_t p1) {
    Op *op0 = schedule.at(p0);
    Op *op1 = schedule.at(p1);

    if (op0->id == op1->id) {
      // Op can overlap with itself
      return true;
    }

    if (op0->settings.executionContext != op1->settings.executionContext) {
      // Can't overlap if in different execution contexts
      return false;
    }

    if (virtualGraphOverlap(op0, op1) && tileSetOverlap(op0, op1)) {
      // Can't overlap in software if the tile sets on the same IPU overlap
      return false;
    } else if (op0->isConvertibleTo<BoundaryOp>() ||
               op1->isConvertibleTo<BoundaryOp>() || !isRemoteOp(op0) ||
               (isRemoteOp(op0) && p1 < p0)) {
      return false;
    } else {
      // Can only overlap if op0 is a Remote Op and the other ops follows after
      return true;
    }
  }

  const std::vector<std::pair<size_t, size_t>> &getParallelSchedule() const {
    return parallelSchedule;
  }

private:
  // Op schedule to operate on
  std::vector<Op *> schedule;

  // For Op at schedule position i, the start and end positions encasing i
  // that benefit from being outlined together with schedule.at(i)
  std::vector<std::pair<size_t, size_t>> parallelSchedule;
};

} // namespace

namespace {
//  Outlining matches are not supposed to cross certain boundaries:
// a.) Across recompute/non-recompute operators
// b.) Across execution phases
// c.) Across graphs
void insertBoundariesOps(const SessionOptions &opts,
                         const std::vector<Op *> &schedule) {
  if (!schedule.empty()) {
    for (int64_t i = 0; i < schedule.size() - 1; ++i) {
      auto crossed = getBoundariesCrossed(opts, i, i + 2, schedule);
      // The graph comparison has been omitted from `getBoundariesCrossed` as it
      // is a special case that needs handling when adding the boundary op.
      if (crossed.size() > 0 ||
          schedule[i]->getGraph().id != schedule[i + 1]->getGraph().id) {
        auto &graph     = schedule[i]->getGraph();
        auto boundaryOp = std::make_unique<BoundaryOp>(Op::Settings(graph, ""));
        auto boundary   = boundaryOp.get();
        auto phase      = schedule[i]->getOptionalExecutionPhase();
        boundary->setExecutionPhase(phase);
        boundary->settings.executionContext =
            schedule[i]->settings.executionContext;
        VGraphId vgid = 0;
        boundary->setVirtualGraphId(vgid);
        graph.moveIntoGraph(std::move(boundaryOp));
        // Insert topo cons to pin boundary between ops
        graph.topoCons.get()->insert(schedule[i], boundary);
        if (schedule[i]->getGraph().id == schedule[i + 1]->getGraph().id) {
          graph.topoCons.get()->insert(boundary, schedule[i + 1]);
        }
        // Ensures inserting boundaries does not mess with priorities
        boundary->settings.schedulePriority =
            schedule[i]->settings.schedulePriority;
      }
    }
  }
}
} // namespace

} // namespace localoutline

class Match {
public:
  class Instance {
  public:
    Instance(const std::vector<OpId> &, Graph &, const OpMap &);

    std::vector<OpId> ops;

    std::vector<Tensor *> external_inputs;
    std::vector<Tensor *> external_outputs;
    std::set<Tensor *> all_outputs;

    // bool contains(const Op *) const;
    int getIndex(const Op *) const;

  private:
    void addExternalOutput(Tensor *);
    void addExternalInput(Tensor *);
  };

  Match(const fwtools::subgraph::Match &,
        const std::vector<Op *> &,
        const OpMap &);

  std::vector<Instance> instances;
  int length;
};

class Replacement {
public:
  Replacement(const std::vector<OpId> &ops_, Op *replacement_op_)
      : ops(ops_), replacement_op(replacement_op_) {}

  std::vector<OpId> ops;
  Op *replacement_op;
};

Match::Instance::Instance(const std::vector<OpId> &ops_,
                          Graph &graph,
                          const OpMap &opMap)
    : ops(ops_) {
  std::set<Op *> op_set;
  for (auto opid : ops) {
    auto op = opMap.at(opid);
    op_set.insert(op);
  }

  auto &ir = graph.getIr();

  for (auto opid : ops) {
    auto op = opMap.at(opid);

    for (auto &index_tensor : op->input->tensorMap()) {
      auto input = index_tensor.second;

      if (!input->hasProducer() ||
          op_set.find(input->getProducer()) == op_set.end()) {
        addExternalInput(input);
      }
    }

    auto hasExternalConsumer = [&](Tensor *tensor) {
      auto consumers = tensor->consumers.getOps();
      return any_of(consumers, [&](Op *consumer) {
        return op_set.find(consumer) == op_set.end();
      });
    };

    for (auto &index_tensor : op->output->tensorMap()) {
      auto output = index_tensor.second;

      if (hasExternalConsumer(output) || ir.isAnchored(output->id)) {
        addExternalOutput(output);
      }

      all_outputs.insert(output);
    }
  }
}

/*
bool Match::Instance::contains(const Op *op) const {
  for (auto opid : ops) {
    if (op->id == opid) {
      return true;
    }
  }
  return false;
}
*/

int Match::Instance::getIndex(const Op *op) const {
  for (int i = 0; i < ops.size(); i++) {
    if (op->id == ops[i]) {
      return i;
    }
  }
  return -1;
}

void Match::Instance::addExternalInput(Tensor *tensor) {
  // dont add inputs more than once
  if (find(external_inputs, tensor) == external_inputs.end()) {
    external_inputs.push_back(tensor);
  }
}

void Match::Instance::addExternalOutput(Tensor *tensor) {
  // dont add outputs more than once
  if (find(external_outputs, tensor) == external_outputs.end()) {
    external_outputs.push_back(tensor);
  }
}

Match::Match(const fwtools::subgraph::Match &match,
             const std::vector<Op *> &ops,
             const OpMap &opMap)
    : length(match.length) {
  for (auto &start : match.starts) {
    std::vector<OpId> m;

    for (int i = 0; i < match.length; i++) {
      auto idx = start + i;
      m.push_back(ops[idx]->id);
    }

    instances.emplace_back(m, ops[0]->getGraph(), opMap);
  }
}

void updateTopoCons(const std::vector<OpId> &ops,
                    const OpId &replacement_op,
                    Graph &graph) {

  if (logging::shouldLog(logging::Module::none, logging::Level::Trace)) {
    std::vector<std::string> graph_ops;
    for (OpId opid : ops) {
      graph_ops.push_back(graph.getOp(opid)->debugName());
    }

    logging::transform::trace(
        "[SubgraphOutline] Updating TopoCons for {} ops {}",
        graph.getOp(replacement_op)->debugName(),
        graph_ops);
  }

  // dont include any of the ops being replaced
  auto include_op = [&](OpId opid) { return find(ops, opid) == ops.end(); };

  auto OpCompare = [](const std::pair<Op *, bool> &a,
                      const std::pair<Op *, bool> &b) {
    return std::pair<OpId, bool>(a.first->id, a.second) <
           std::pair<OpId, bool>(b.first->id, b.second);
  };
  std::set<std::pair<Op *, bool>, decltype(OpCompare)> befores(OpCompare);
  std::set<std::pair<Op *, bool>, decltype(OpCompare)> afters(OpCompare);

  // Get all befores and afters that are not in ops
  for (auto &opid : ops) {
    for (auto before : graph.topoCons->getBefores(graph.getOp(opid))) {
      if (include_op(before->id)) {
        befores.insert({before, false});
      }
    }
    for (auto after : graph.topoCons->getAfters(graph.getOp(opid))) {
      if (include_op(after->id)) {
        afters.insert({after, false});
      }
    }
    for (auto before : graph.topoCons->getTiedBefores(graph.getOp(opid))) {
      if (include_op(before->id)) {
        befores.insert({before, true});
      }
    }
    for (auto after : graph.topoCons->getTiedAfters(graph.getOp(opid))) {
      if (include_op(after->id)) {
        afters.insert({after, true});
      }
    }
  }

  // Remove the existing topocons
  for (auto &opid : ops) {
    graph.topoCons->remove(graph.getOp(opid));
  }

  // Add the topoCons for the replacement Op
  for (auto before : befores) {
    graph.topoCons->insert(
        before.first, graph.getOp(replacement_op), before.second);
  }
  for (auto after : afters) {
    graph.topoCons->insert(
        graph.getOp(replacement_op), after.first, after.second);
  }
}

static Op *replaceWithCallOp(const Match::Instance &instance,
                             Graph &graph,
                             Graph &subgraph) {

  // Copy some attributes with heuristics from the instance ops
  nonstd::optional<Scope> scope;
  OptionalVGraphId ipu_copy_vgid;
  OptionalVGraphId vgid;
  OptionalExecutionPhase phase;
  OptionalBatchSerializedPhase batchserial;
  bool conflicting_batchserial = false;
  OptionalPipelineStage pipeline_stage;
  nonstd::optional<RecomputeType> recompute;
  nonstd::optional<ExecutionContext> execution_context;

  for (const OpId &opid : instance.ops) {
    Op *op = graph.getOp(opid);
    if (!scope) {
      scope = op->getScope();
    }
    IpuCopyOp *copy = dynamic_cast<IpuCopyOp *>(op);
    if (copy && !ipu_copy_vgid) {
      ipu_copy_vgid = copy->getMinSourceIpu();
    }
    if (!vgid) {
      vgid = op->getOptionalVGraphId();
    }
    if (!phase) {
      phase = op->getOptionalExecutionPhase();
    }
    if (!pipeline_stage) {
      pipeline_stage = op->getOptionalPipelineStage();
    }
    if (!recompute) {
      recompute = op->settings.recomputeType;
    }
    if (batchserial && op->hasBatchSerializedPhase() &&
        op->getBatchSerializedPhase() != batchserial) {
      conflicting_batchserial = true;
    }
    if (!execution_context) {
      execution_context = op->settings.executionContext;
    }
  }

  // Fallback to IPU copy VGID
  if (!vgid) {
    vgid = ipu_copy_vgid;
  }

  if (conflicting_batchserial) {
    batchserial.reset();
  }

  // Create the call op. Note that toLoss and fromLoss are set in the
  // constructor
  auto up_call_op =
      std::make_unique<CallOp>(Onnx::CustomOperators::Call_1, graph, subgraph);
  auto call_op_id = graph.moveIntoGraph(std::move(up_call_op));
  CallOp *call_op = dynamic_cast<CallOp *>(graph.getOp(call_op_id));
  if (scope) {
    call_op->settings.scope = scope.value();
  }
  if (recompute) {
    call_op->settings.recomputeType = recompute.value();
  }
  call_op->setVirtualGraphId(vgid);
  call_op->setExecutionPhase(phase);
  call_op->setPipelineStage(pipeline_stage);
  call_op->setBatchSerializedPhase(batchserial);
  if (execution_context) {
    call_op->settings.executionContext = execution_context.value();
  }

  // Set the position w.r.t loss, if possible. If any of the internal ops
  // is connected to the final loss, then so is this CallOp. Note that we use
  // the Ops in the instance of this Match, and not the canonical subgraph.
  for (auto opid : instance.ops) {
    auto instanceOp = graph.getOp(opid);
    if (instanceOp->toLoss == PathToLoss::Yes) {
      call_op->toLoss = PathToLoss::Yes;
    }
    if (instanceOp->fromLoss == PathFromLoss::Yes) {
      call_op->fromLoss = PathFromLoss::Yes;
    }
  }

  // Check aliasing and modifying before disconnecting the old ops
  for (int i = 0; i < instance.external_inputs.size(); i++) {
    Tensor *inTensor = instance.external_inputs[i];

    std::map<Op *, int64_t> beforeOps;
    for (Op *consumer : inTensor->consumers.getOps()) {
      if (std::find(instance.ops.begin(), instance.ops.end(), consumer->id) !=
          instance.ops.end()) {
        beforeOps.insert({consumer, 0});
      }
    }
    for (Op *consumer : inTensor->consumers.getOps()) {
      for (Op *after : graph.topoCons->getAfters(consumer)) {
        if (beforeOps.find(after) != beforeOps.end()) {
          beforeOps[after]++;
        }
      }
    }
    std::vector<std::pair<Op *, int64_t>> consumersOrdered;
    for (auto it = beforeOps.begin(); it != beforeOps.end(); ++it)
      consumersOrdered.push_back(*it);

    std::sort(
        consumersOrdered.begin(),
        consumersOrdered.end(),
        [](const std::pair<Op *, int64_t> &a,
           const std::pair<Op *, int64_t> &b) { return a.second < b.second; });

    view::Regions modifiedRegions;
    view::AccessType accessType = view::AccessType::None;

    // As soon as a consumer modified the whole input, we can stop
    for (auto &consumerOrdered : consumersOrdered) {
      Op *c        = consumerOrdered.first;
      auto indices = c->input->indices(inTensor);
      for (InIndex index : indices) {
        auto regions = consumerOrdered.first->modifies(index);

        // If an op consumes the tensor without specifying modifies we assume
        // (conservatively) full read access to the tensor
        if (regions.empty() ||
            std::any_of(regions.begin(),
                        regions.end(),
                        [](const view::Region &r) { return r.isEmpty(); })) {
          accessType = view::combine({accessType, view::AccessType::Read});
        }

        modifiedRegions.insert(
            modifiedRegions.end(), regions.begin(), regions.end());
      }
      modifiedRegions = view::mergeRegions(modifiedRegions);
      for (auto &r : modifiedRegions) {
        view::AccessType regionAccessType = r.getAccessType();
        if (!r.isEmpty() && (regionAccessType == view::AccessType::None ||
                             regionAccessType == view::AccessType::Read)) {
          throw error("Unexpected modified region access type None or Read");
        }
        accessType = view::combine({accessType, regionAccessType});
      }
      if (modifiedRegions.size() > 0 &&
          modifiedRegions.front() ==
              view::Region::getFull(inTensor->info.shape()) &&
          accessType == view::AccessType::Write) {
        // The whole input tensor has been touched, conclude
        //  If the whole tensor has been write-accessed first, we say that
        //  the CallOp consumes the tensor write-only.
        //  If any read access to the tensor happens before the write-only
        //  access, the modified tensor is read-write.
        //  Read-only does not make sense, since we ask about modified regions.
        // Examples:
        //  1.) VarUpdate will cause read-write access to modified input.
        //  2.) RemoteLoad will cause write-only access to modified input.
        break;
      }
    }
    // Update access type
    for (auto &r : modifiedRegions) {
      r.setAccessType(accessType);
    }
    call_op->addModified(i, modifiedRegions);

    for (int j = 0; j < instance.external_outputs.size(); j++) {
      Tensor *outTensor = instance.external_outputs[j];

      if (inTensor->id == outTensor->id) {
        throw internal_error(
            "[SubgraphOutline] {} is both subgraph input and output.",
            inTensor);
      }

      // alias Regions in input Tensor:
      auto fwdAliasRegions =
          graph.getTensors().getChainsFromTo(inTensor, outTensor);
      auto bwdAliasRegions =
          graph.getTensors().getChainsFromTo(outTensor, inTensor);

      call_op->addAlias(i, j, fwdAliasRegions, bwdAliasRegions);
      if (logging::shouldLog(logging::Module::transform,
                             logging::Level::Trace)) {
        logging::transform::trace("[SubgraphOutline] Alias {} ({}) -> {} ({})",
                                  i,
                                  inTensor->id,
                                  j,
                                  outTensor->id);
      }
    }
  }

  double priority = -std::numeric_limits<double>::infinity();

  // Disconnect the old ops
  for (auto opid : instance.ops) {
    auto old_op = graph.getOp(opid);
    old_op->disconnectAllInputs();
    old_op->disconnectAllOutputs();
    priority = std::max(priority, old_op->settings.schedulePriority);
  }

  // CallOp's priority should be the max priority of the Op's that it replaces
  call_op->settings.schedulePriority = priority;

  // Connect inputs
  for (int i = 0; i < instance.external_inputs.size(); i++) {
    auto input = instance.external_inputs.at(i);
    call_op->connectInTensor(i, input->id);
  }

  // Connect outputs
  for (int i = 0; i < instance.external_outputs.size(); i++) {
    auto output = instance.external_outputs.at(i);
    call_op->connectOutTensor(i, output->id);
  }

  updateTopoCons(instance.ops, call_op_id, graph);

  // Erase the old ops
  for (auto opid : instance.ops) {
    graph.eraseOp(opid);
  }

  call_op->setup();

  return call_op;
}

class InstanceConstraints {
public:
  /*
  InstanceConstraints(const Match::Instance &instance, Graph &graph) {
    for (auto opid : instance.ops) {
      auto op = graph.getOp(opid);

      for (auto &before : graph.topoCons->getBefores(op)) {
        if (instance.contains(before)) {
          insertInternal(instance.getIndex(before), instance.getIndex(op));
        }
      }

      for (auto &after : graph.topoCons->getAfters(op)) {
        if (instance.contains(after)) {
          insertInternal(instance.getIndex(op), instance.getIndex(after));
        }
      }
    }
  }

  void insertInternal(int before, int after) {
    auto foundBefore = internalBefores.find(after);
    if (foundBefore == internalBefores.end()) {
      internalBefores.insert({after, {before}});
    } else {
      foundBefore->second.insert(before);
    }

    auto foundAfter = internalAfters.find(before);
    if (foundAfter == internalAfters.end()) {
      internalAfters.insert({before, {after}});
    } else {
      foundAfter->second.insert(after);
    }
  }


  bool operator!=(const InstanceConstraints &rhs) { return !(*this == rhs); }


  bool operator==(const InstanceConstraints &rhs) {
    return (internalBefores == rhs.internalBefores) &&
           (internalAfters == rhs.internalAfters);
  }

  */

  std::map<int, std::set<int>> internalBefores;
  std::map<int, std::set<int>> internalAfters;
};

/*
std::ostream &operator<<(std::ostream &os, const InstanceConstraints &ic) {
  os << "InstanceConstraints:";

  os << "\n  internalBefores:";
  for (auto &i_befores : ic.internalBefores) {
    auto i        = i_befores.first;
    auto &befores = i_befores.second;
    os << logging::format("\n    {}:", i);
    for (auto &before : befores) {
      os << logging::format("\n      {}", before);
    }
  }

  os << "\n  internalAfters:";
  for (auto &i_afters : ic.internalAfters) {
    auto i       = i_afters.first;
    auto &afters = i_afters.second;
    os << logging::format("\n    {}:", i);
    for (auto &after : afters) {
      os << logging::format("\n      {}", after);
    }
  }

  return os;
};

void verifyTopologicalConstraints(const Match &match, Graph &graph) {
  logging::debug("Checking topological constraints");
  InstanceConstraints c0(match.instances.at(0), graph);
  for (int i = 1; i < match.instances.size(); i++) {
    InstanceConstraints c(match.instances.at(i), graph);

    if (c0 != c) {

      std::vector<std::string> c0_ops;
      std::vector<std::string> c_ops;

      for (auto opid : match.instances.at(0).ops) {
        c0_ops.push_back(graph.getOp(opid)->debugName());
      }

      for (auto opid : match.instances.at(i).ops) {
        c_ops.push_back(graph.getOp(opid)->debugName());
      }

      throw internal_error("Internal constraints between match "
                  "instance \n{} \nand \n{} \n do not match "
                  "(Ops: {} {}, {} {}).",
                  c0,
                  c,
                  match.instances.at(0).ops,
                  c0_ops,
                  match.instances.at(i).ops,
                  c_ops);
    }
  }
}
*/

void verifyMatchInstances(const Match &match) {
  logging::debug("Checking match instances for inconsistencies");
  auto &external_inputs = match.instances[0].external_inputs;
  for (auto &instance : match.instances) {
    if (instance.external_inputs.size() != external_inputs.size()) {
      throw error("Instances of match have different external input sizes "
                  "({} vs. {}).",
                  external_inputs.size(),
                  instance.external_inputs.size());
    }
  }
}

Graph &createSubgraph(const Match &match, Ir &ir, const OpMap &opMap) {

  auto subgraph_id    = ir.createUniqueSubgraphId({""});
  auto &subgraph      = ir.createGraph(subgraph_id);
  auto subgraph_scope = subgraph.getScope();
  auto &instance      = match.instances[0];

  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {
    std::stringstream ss;
    for (int i = 0; i < instance.ops.size(); i++) {
      auto opid = instance.ops.at(i);
      auto op   = opMap.at(opid);
      ss << std::endl
         << "    " << op->debugName() << ", "
         << "VGID: " << (op->hasVirtualGraphId() ? op->getVirtualGraphId() : -1)
         << ", "
         << "ExecutionPhase: "
         << (op->getOptionalExecutionPhase() ? *op->getOptionalExecutionPhase()
                                             : -1);
    }
    logging::transform::trace("[SubgraphOutline] Creating subgraph: {}, "
                              "replacing {} instances, "
                              "with ops: [{}]",
                              subgraph_id,
                              match.instances.size(),
                              ss.str());
  }

  // clone all the ops and move into subgraph
  std::map<Op *, Op *> clone_map;
  std::vector<Op *> clones;
  for (int i = 0; i < instance.ops.size(); i++) {
    auto opid                     = instance.ops.at(i);
    auto op                       = opMap.at(opid);
    auto clone                    = op->clone();
    clone->settings.graph         = subgraph;
    clone->settings.scope         = subgraph_scope;
    clone->settings.recomputeType = RecomputeType::Checkpoint;
    // There are some attributes that have little meaning in subgraphs. Unset
    // them here.
    clone->settings.executionPhase   = ExecutionPhase();
    clone->settings.pipelineStage    = PipelineStage();
    clone->settings.executionContext = ExecutionContext::Subgraph;
    auto cloneid                     = subgraph.moveIntoGraph(std::move(clone));
    Op *clone_op                     = subgraph.getOp(cloneid);
    clone_map.insert({op, clone_op});
    clones.push_back(clone_op);
  }

  // Map out constraints by schedule match positions for all instances.
  // If different constraints per instance exist, they either clash or can
  // coexist. We assume instance.ops preserves schedule order.
  std::set<std::tuple<int, int, bool>> constraints;
  for (auto &instanceForConstraints : match.instances) {
    for (int i = 0; i < instanceForConstraints.ops.size(); i++) {
      auto opid   = instanceForConstraints.ops.at(i);
      auto op     = opMap.at(opid);
      auto &graph = op->getGraph();
      {
        auto afters = graph.topoCons->getAfters(op);
        for (Op *after_op : afters) {
          auto j = instanceForConstraints.getIndex(after_op);
          if (j > 0) {
            // i before j
            constraints.insert({i, j, false});
          }
        }
      }
      {
        auto afters = graph.topoCons->getTiedAfters(op);
        for (Op *after_op : afters) {
          auto j = instanceForConstraints.getIndex(after_op);
          if (j > 0) {
            // i before j
            constraints.insert({i, j, true});
          }
        }
      }
    }
  }

  // Preserve topological constraints between ops being added to the subgraph
  for (auto &constraint : constraints) {
    subgraph.topoCons->insert(clones[std::get<0>(constraint)],
                              clones[std::get<1>(constraint)],
                              std::get<2>(constraint));
  }

  // duplicate all the output tensors
  std::map<Tensor *, Tensor *> tensor_map;
  for (auto output : instance.all_outputs) {
    auto new_id = (subgraph_scope / output->id).str();

    auto clone = output->clone(subgraph);
    clone->id  = new_id;
    subgraph.getTensors().moveIntoTensors(std::move(clone));
    auto clone_ptr = subgraph.getTensors().get(new_id);
    tensor_map.insert({output, clone_ptr});
  }

  // create graph inputs
  for (auto tensor : instance.external_inputs) {
    auto input_id = subgraph.addScope(tensor->id);
    subgraph.addInput(input_id, tensor->info);
    auto t = subgraph.getTensors().get(input_id);
    if (tensor_map.find(tensor) != tensor_map.end()) {
      throw error(
          "tensor {} is already in tensor map, cannot rebind to {} -> {}",
          tensor->id,
          tensor->id,
          input_id);
    }
    tensor_map.insert({tensor, t});
  }

  // create graph outputs
  for (auto tensor : instance.external_outputs) {
    auto out_id = tensor_map.at(tensor)->id;
    subgraph.markAsOutput(out_id);
  }

  // hook up graph inputs and outputs
  for (auto opid : instance.ops) {
    auto op    = opMap.at(opid);
    auto clone = clone_map.at(op);

    // connect inputs
    for (auto &idx_tensor : op->input->tensorMap()) {
      auto idx             = idx_tensor.first;
      auto tensor          = idx_tensor.second;
      auto clone_tensor_id = tensor_map.at(tensor)->id;
      auto *copyOp         = dynamic_cast<IpuCopyOp *>(op);
      auto *cloneCopyOp    = dynamic_cast<IpuCopyOp *>(clone);
      if (copyOp && cloneCopyOp) {
        auto sourceIpu = copyOp->getSourceIpu(tensor->id);
        cloneCopyOp->connectInTensor(idx, clone_tensor_id, sourceIpu);
      } else {
        clone->connectInTensor(idx, clone_tensor_id);
      }
    }

    // connect outputs
    for (auto &idx_tensor : op->output->tensorMap()) {
      auto idx             = idx_tensor.first;
      auto tensor          = idx_tensor.second;
      auto clone_tensor_id = tensor_map.at(tensor)->id;
      clone->connectOutTensor(idx, clone_tensor_id);
    }
  }

  return subgraph;
}

static void
handleRandomReferences(const Match &match, Ir &ir, const OpMap &opMap) {
  std::map<std::string, std::set<RandomReferenceId>> idsToMerge;
  for (auto &instance : match.instances) {
    for (auto opid : instance.ops) {
      auto old_op = dynamic_cast<DropoutOp *>(opMap.at(opid));
      if (old_op) {
        idsToMerge[old_op->getSubgraphEquivId()].insert(
            old_op->getReferenceId());
      }
    }
  }

  for (auto &equivSet : idsToMerge) {
    ir.mergeRandomReferenceIds(equivSet.second);
  }
}

// Create a subgraph for the match and
// replace instances of the match with a CallOp
static std::vector<Replacement>
applyMatch(const Match &match, Ir &ir, const OpMap &opMap) {
  verifyMatchInstances(match);

  // TODO: Verify. This is possibly too strict. Can probably be dropped.
  // verifyTopologicalConstraints(match, graph);

  auto &subgraph = createSubgraph(match, ir, opMap);

  handleRandomReferences(match, ir, opMap);

  std::vector<Replacement> replacements;

  // Replace the matches with call ops
  for (auto &instance : match.instances) {
    Graph &instanceGraph = opMap.at(instance.ops.at(0))->getGraph();
    auto call_op         = replaceWithCallOp(instance, instanceGraph, subgraph);
    replacements.push_back({instance.ops, call_op});
  }

  return replacements;
}

// Returns a vector of Match instance
// sorted so the smallest matches are at the back
std::vector<Match>
getRinseMatches(const std::vector<Op *> &ops,
                const std::vector<std::pair<size_t, size_t>> &sequences,
                float threshold,
                float sequenceBreakCost,
                bool copyCostPruning,
                const OpMap &opMap) {

  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {
    std::vector<int> intSchedule = fwtools::subgraph::getIntSchedule(ops);
    for (size_t i = 0; i < ops.size(); ++i) {
      Op *op = ops[i];
      logging::transform::trace(
          "[SubgraphOutline] "
          "Index: {}, ID: {}, Op: {}, "
          "VGID: {}, execution phase: {}",
          i,
          intSchedule[i],
          op->debugName(),
          op->hasVirtualGraphId() ? op->getVirtualGraphId() : -1,
          op->getOptionalExecutionPhase() ? *op->getOptionalExecutionPhase()
                                          : -1);
    }
    logging::transform::trace("[SubgraphOutline] Int schedule: {}",
                              intSchedule);
  }

  auto fw_matches = fwtools::subgraph::getRinseMatches(
      ops,
      sequences,
      threshold,
      sequenceBreakCost,
      fwtools::subgraph::getDefaultOutlinerAlgorithm());
  int64_t num_matches_0 = fw_matches.size();

  // TODO: T Copy cost pruning can cause crossing matches,
  // and is therefore buggy/broken.
  if (copyCostPruning) {
    fw_matches = fwtools::subgraph::prune::
        pruneMatches<Op, popart::outline::IoSubgraphCostModel>(
            fw_matches, ops, threshold);
  }
  int64_t num_matches_1 = fw_matches.size();

  logging::transform::trace("[SubgraphOutline] Matches before pruning: {}, "
                            "matches after IOSize: {} ",
                            num_matches_0,
                            num_matches_1);

  // Remove matches only wrapping another CallOp
  std::vector<fwtools::subgraph::Match> filtered_fw_matches;
  std::copy_if(fw_matches.begin(),
               fw_matches.end(),
               std::back_inserter(filtered_fw_matches),
               [&ops](const fwtools::subgraph::Match &match) {
                 return !(
                     match.length == 1 &&
                     ops.at(match.starts.front())->isConvertibleTo<CallOp>());
               });
  fw_matches = filtered_fw_matches;

  // Sort the matches so the smallest subgraphs are at the back.
  // `matches' is treated like a stack, so this will ensure the smallest
  // subgraphs are processed first `matches' cannot be std::stack as it needs
  // to be iterated over
  sortMatches<fwtools::subgraph::Match>(fw_matches);

  std::vector<Match> matches;

  for (auto &match : fw_matches) {
    logging::transform::trace("[SubgraphOutline] Match length: {}, starts: {}",
                              match.length,
                              match.starts);
    matches.emplace_back(match, ops, opMap);
  }

  return matches;
}

// Replace the op ids that have been removed from the graph with callop opid
void applyReplacement(Match::Instance &instance, Replacement &replacement) {
  auto start = std::search(instance.ops.begin(),
                           instance.ops.end(),
                           replacement.ops.begin(),
                           replacement.ops.end());
  if (start != instance.ops.end()) {
    instance.ops.erase(start, start + replacement.ops.size());
    instance.ops.insert(start, replacement.replacement_op->id);
  } else {
    for (OpId id : replacement.ops) {
      if (std::find(instance.ops.begin(), instance.ops.end(), id) !=
          instance.ops.end()) {
        throw error("Instance {} crossing replacement {}",
                    instance.ops,
                    replacement.ops);
      }
    }
  }
}

// In each match, replace the op ids that have been removed from the graph
// with the replacement callops opid
void applyReplacements(std::vector<Match> &matches,
                       std::vector<Replacement> &replacements) {
  for (auto &match : matches) {
    for (auto &instance : match.instances) {
      for (auto &replacement : replacements) {
        applyReplacement(instance, replacement);
      }
    }
  }
}

std::vector<Op *> getFullSchedule(const Ir &ir) {
  std::vector<Op *> schedule;
  for (auto &graphId_graphPtr : ir.getGraphs()) {
    Graph *graphPtr = graphId_graphPtr.second.get();
    auto sched      = graphPtr->getOpSchedule({}, RequireOptimalSchedule::Yes);
    schedule.insert(schedule.end(), sched.begin(), sched.end());
  }
  return schedule;
}

void removeBoundaryOps(const Ir &ir) {
  for (auto &graphId_graphPtr : ir.getGraphs()) {
    auto graph    = graphId_graphPtr.second.get();
    auto schedule = graph->getOpSchedule({}, RequireOptimalSchedule::No);
    for (Op *op : schedule) {
      if (dynamic_cast<BoundaryOp *>(op)) {
        graph->topoCons->remove(graph->getOp(op->id));
        graph->eraseOp(op->id);
      }
    }
  }
}

void updateOpMap(OpMap &opMap,
                 const Match &match,
                 const std::vector<Replacement> &replacements) {
  // Remove all the ops in the match instances from opMap as they should
  // have been removed.
  for (auto &instance : match.instances) {
    for (const OpId &opid : instance.ops) {
      opMap.erase(opid);
    }
  }

  // Add the replacement ops to opMap.
  for (auto &replacement : replacements) {
    auto x = replacement.replacement_op;
    opMap.insert({x->id, x});
  }
}

} // namespace

bool SubgraphOutline::apply(Graph &graph) const {

  auto &ir = graph.getIr();

  OpMap opMap = createOpMap(ir);

  auto schedule = getFullSchedule(ir);

  // Change schedule to include boundaries that can't be outlined
  localoutline::insertBoundariesOps(ir.getSessionOptions(), schedule);

  // Get updated schedule with boundaries
  schedule = getFullSchedule(ir);

  // Get the software parallel schedule to generate sequences that should
  // be outlined as a whole
  localoutline::SoftParallelismModel softParallelismModel(schedule);
  // softParallelismModel.log();

  auto matches =
      getRinseMatches(schedule,
                      softParallelismModel.getParallelSchedule(),
                      ir.getSessionOptions().outlineThreshold,
                      ir.getSessionOptions().outlineSequenceBreakCost,
                      ir.getSessionOptions().enableOutliningCopyCostPruning,
                      opMap);

  if (logging::shouldLog(logging::Module::none, logging::Level::Trace)) {
    unsigned i = 0;
    for (auto &match : matches) {
      std::stringstream ss;
      for (auto &instance : match.instances) {
        ss << "["
           << logging::join(instance.ops.begin(), instance.ops.end(), ", ")
           << "]";
      }
      logging::transform::trace("[SubgraphOutline] Match {}: {}", i, ss.str());
      ++i;
    }
  }

  // matches needs to be treated like a stack
  while (!matches.empty()) {
    auto match = matches.back();
    matches.pop_back();

    auto replacements = applyMatch(match, ir, opMap);
    applyReplacements(matches, replacements);

    // If there are more matches, we need to update opMap with the new call ops.
    if (!matches.empty()) {
      updateOpMap(opMap, match, replacements);
    }
  }

  removeBoundaryOps(ir);

  graph.getTensors().removeIsolated(true);

  return true;
}

std::size_t SubgraphOutline::id() {
  return typeid(SubgraphOutline).hash_code();
}

namespace {
bool init = Transform::registerTransform(new SubgraphOutline);
}

} // namespace popart
