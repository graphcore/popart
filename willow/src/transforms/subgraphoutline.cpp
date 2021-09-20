// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/range/algorithm/find.hpp>
#include <cmath>
#include <memory>

#include <popart/aliasesmap.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/boundary.hpp>
#include <popart/op/call.hpp>
#include <popart/op/dropout.hpp>
#include <popart/op/exchange/exchange.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/subgraph/iosubgraphcostmodel.hpp>
#include <popart/subgraph/match.hpp>
#include <popart/subgraph/outliner.hpp>
#include <popart/subgraph/prunematches.hpp>
#include <popart/subgraph/subgraphutil.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/subgraphoutline.hpp>
#include <popart/util.hpp>
#include <popart/vendored/optional.hpp>

using boost::find;
using boost::algorithm::any_of;

namespace popart {
namespace {

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
      logging::transform::trace("{}{}{} ({}: {}, VGID: {}, TileSet: {})",
                                a,
                                b,
                                c,
                                i,
                                schedule.at(i)->debugName(),
                                schedule.at(i)->hasVirtualGraphId()
                                    ? schedule.at(i)->getVirtualGraphId()
                                    : unusedVGraphId,
                                schedule.at(i)->settings.tileSet);
    }
  }

  std::set<VGraphId> getVirtualGraphIds(Op *op) {
    std::set<VGraphId> vgids;
    for (auto &in : op->input->tensorMap()) {
      std::set<OpId> visited;
      vgids.insert(
          op->getIntrospectionInVirtualGraphId(in.first, visited).first);
    }
    for (auto &out : op->output->tensorMap()) {
      std::set<OpId> visited;
      vgids.insert(
          op->getIntrospectionOutVirtualGraphId(out.first, visited).first);
    }
    return vgids;
  }

  std::set<TileSet> getTileSets(Op *op) {
    std::set<TileSet> tileSets;
    for (auto &in : op->input->tensorMap()) {
      std::set<OpId> visited;
      tileSets.insert(
          op->getIntrospectionInVirtualGraphId(in.first, visited).second);
    }
    for (auto &out : op->output->tensorMap()) {
      std::set<OpId> visited;
      tileSets.insert(
          op->getIntrospectionOutVirtualGraphId(out.first, visited).second);
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

  bool isExchangeOp(Op *op) { return op->isConvertibleTo<ExchangeBaseOp>(); }

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
               op1->isConvertibleTo<BoundaryOp>() || !isExchangeOp(op0) ||
               (isExchangeOp(op0) && p1 < p0)) {
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
        if (schedule[i]->getGraph().id == schedule[i + 1]->getGraph().id) {
          // Insert topo cons to pin boundary between ops
          graph.topoCons.get()->insert(schedule[i], boundary);
          graph.topoCons.get()->insert(boundary, schedule[i + 1]);
        } else {
          // Insert topo cons to pin boundary at the end of the graph
          int64_t j = i;
          while (j >= 0 &&
                 schedule[j]->getGraph().id == schedule[i]->getGraph().id) {
            graph.topoCons.get()->insert(schedule[j], boundary);
            --j;
          }
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
  Match(const fwtools::subgraph::Match &, const std::vector<Op *> &);

  // If not all instances of a match export the same outputs to the parent
  // graph, adjust all instances to export the superset of all required
  // outputs to the parent graph
  void equalizeInstanceOutputs();

  std::vector<SubgraphableOpCluster> instances;
  int length;
};

class Replacement {
public:
  Replacement(const std::vector<OpId> &ops_, Op *replacement_op_)
      : ops(ops_), replacement_op(replacement_op_) {}

  std::vector<OpId> ops;
  Op *replacement_op;
};

Match::Match(const fwtools::subgraph::Match &match,
             const std::vector<Op *> &ops)
    : length(match.length) {
  for (auto &start : match.starts) {
    std::vector<OpId> m;

    for (int i = 0; i < match.length; i++) {
      auto idx = start + i;
      m.push_back(ops[idx]->id);
    }

    instances.emplace_back(m, &(ops[start]->getGraph()));
  }
}

void Match::equalizeInstanceOutputs() {
  std::set<std::pair<int64_t, OutIndex>> outOpAndIndex;

  for (auto &instance : instances) {
    for (Tensor *out : instance.external_outputs) {
      auto it = std::find(
          instance.ops.begin(), instance.ops.end(), out->getProducer()->id);
      auto opIndex      = std::distance(instance.ops.begin(), it);
      Op *op            = instance.graph->getOp(instance.ops.at(opIndex));
      OutIndex outIndex = op->output->indices(out).front();
      outOpAndIndex.insert({opIndex, outIndex});
    }
  }
  for (auto &instance : instances) {
    int64_t outputs_before = instance.external_outputs.size();
    instance.external_outputs.clear();
    for (auto &outIdx : outOpAndIndex) {
      auto opIndex      = outIdx.first;
      OutIndex outIndex = outIdx.second;
      Op *op            = instance.graph->getOp(instance.ops.at(opIndex));
      Tensor *out       = op->output->tensor(outIndex);
      instance.external_outputs.push_back(out);
    }
    int64_t outputs_after = instance.external_outputs.size();
    logging::trace("[SubgraphOutline] Instance outputs {} -> {}{}",
                   outputs_before,
                   outputs_after,
                   outputs_before != outputs_after ? " changed" : "");
  }
}

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

static void handleRandomReferences(const Match &match, Ir &ir) {
  std::map<std::string, std::set<RandomReferenceId>> idsToMerge;
  for (auto &instance : match.instances) {
    for (auto opid : instance.ops) {
      auto old_op = dynamic_cast<DropoutOp *>(instance.graph->getOp(opid));
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
static std::vector<Replacement> applyMatch(const Match &match, Ir &ir) {
  verifyMatchInstances(match);

  // TODO: Verify. This is possibly too strict. Can probably be dropped.
  // verifyTopologicalConstraints(match, graph);

  std::map<Op *, int> index_map;

  auto &subgraph =
      SubgraphOutline::createSubgraph(match.instances, ir, index_map);

  handleRandomReferences(match, ir);

  std::vector<Replacement> replacements;
  AliasesMap aliasesMap{&ir};

  // Replace the matches with call ops
  for (auto &instance : match.instances) {
    auto call_op = SubgraphOutline::replaceWithCallOp(
        instance, subgraph, index_map, aliasesMap);
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
                bool copyCostPruning) {

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
    matches.emplace_back(match, ops);
  }

  return matches;
}

// Replace the op ids that have been removed from the graph with callop opid
void applyReplacement(SubgraphableOpCluster &instance,
                      Replacement &replacement) {
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
  for (Op *op : ir.getAllOps()) {
    if (op->isConvertibleTo<BoundaryOp>()) {
      auto &graph = op->getGraph();
      graph.topoCons->remove(op);
      graph.eraseOp(op->id);
    }
  }
}

} // namespace

SubgraphableOpCluster::SubgraphableOpCluster(const std::vector<OpId> &ops_,
                                             Graph *graph_)
    : ops(ops_), graph(graph_) {
  std::set<Op *> op_set;
  for (auto opid : ops) {
    auto op = graph->getOp(opid);
    op_set.insert(op);
  }

  auto &ir = graph->getIr();

  for (auto opid : ops) {
    auto op = graph->getOp(opid);

    for (auto &index_tensor : op->input->tensorMap()) {
      auto input = index_tensor.second;

      if (!input->hasProducer() ||
          op_set.find(input->getProducer()) == op_set.end() ||
          input->isGraphInput()) {
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

      if (hasExternalConsumer(output) || ir.isAnchored(output->id) ||
          output->isGraphOutput()) {
        addExternalOutput(output);
      }

      all_outputs.insert(output);
    }
  }
}

void SubgraphableOpCluster::addExternalInput(Tensor *tensor) {
  // dont add inputs more than once
  if (find(external_inputs, tensor) == external_inputs.end()) {
    external_inputs.push_back(tensor);
  }
}

void SubgraphableOpCluster::addExternalOutput(Tensor *tensor) {
  // dont add outputs more than once
  if (find(external_outputs, tensor) == external_outputs.end()) {
    external_outputs.push_back(tensor);
  }
}

Graph &SubgraphOutline::createSubgraph(
    const std::vector<SubgraphableOpCluster> instances,
    Ir &ir,
    std::map<Op *, int> &index_map,
    std::string subgraphId) {
  auto subgraph_id    = ir.createUniqueSubgraphId({subgraphId});
  auto &subgraph      = ir.createGraph(subgraph_id);
  auto subgraph_scope = subgraph.getScope();
  auto &instance      = instances[0];

  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {
    std::stringstream ss;
    for (int i = 0; i < instance.ops.size(); i++) {
      Graph &graph = instance.getGraph();
      auto opid    = instance.ops.at(i);
      auto op      = graph.getOp(opid);
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
                              instances.size(),
                              ss.str());
  }

  // clone all the ops and move into subgraph
  std::map<Op *, Op *> clone_map;
  std::vector<Op *> clones;
  for (int i = 0; i < instance.ops.size(); i++) {
    auto opid                     = instance.ops.at(i);
    auto op                       = instance.getGraph().getOp(opid);
    auto clone                    = op->clone();
    clone->settings.graph         = subgraph;
    clone->settings.scope         = subgraph_scope;
    clone->settings.recomputeType = RecomputeType::Checkpoint;
    // There are some attributes that have little meaning in subgraphs. Unset
    // them here.
    clone->settings.executionPhase   = OptionalExecutionPhase();
    clone->settings.pipelineStage    = OptionalPipelineStage();
    clone->settings.executionContext = ExecutionContext::Subgraph;
    auto cloneid                     = subgraph.moveIntoGraph(std::move(clone));
    Op *clone_op                     = subgraph.getOp(cloneid);
    clone_map.insert({op, clone_op});
    index_map.insert({clone_op, i});
    clones.push_back(clone_op);
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
    auto input_id = addScope(subgraph, tensor->id);
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
    auto op    = instance.getGraph().getOp(opid);
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

Op *SubgraphOutline::replaceWithCallOp(const SubgraphableOpCluster &instance,
                                       Graph &subgraph,
                                       const std::map<Op *, int> &index_map,
                                       AliasesMap &aliasesMap) {

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
    Op *op = instance.getGraph().getOp(opid);
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
  auto up_call_op = std::make_unique<CallOp>(
      Onnx::CustomOperators::Call_1,
      subgraph,
      Op::Settings{instance.getGraph(), "", instance.getGraph().getScope()});
  auto call_op_id = instance.getGraph().moveIntoGraph(std::move(up_call_op));
  CallOp *callOp =
      dynamic_cast<CallOp *>(instance.getGraph().getOp(call_op_id));
  if (scope) {
    callOp->settings.scope = scope.value();
  }
  if (recompute) {
    callOp->settings.recomputeType = recompute.value();
  }
  callOp->setVirtualGraphId(vgid);
  callOp->setExecutionPhase(phase);
  callOp->setPipelineStage(pipeline_stage);
  callOp->setBatchSerializedPhase(batchserial);
  if (execution_context) {
    callOp->settings.executionContext = execution_context.value();
  }

  // Set the position w.r.t loss, if possible. If any of the internal ops
  // is connected to the final loss, then so is this CallOp. Note that we use
  // the Ops in the instance of this Match, and not the canonical subgraph.
  for (auto opid : instance.ops) {
    auto instanceOp = instance.graph->getOp(opid);
    if (instanceOp->toLoss == PathToLoss::Yes) {
      callOp->toLoss = PathToLoss::Yes;
    }
    if (instanceOp->fromLoss == PathFromLoss::Yes) {
      callOp->fromLoss = PathFromLoss::Yes;
    }
  }

  auto &aliases = aliasesMap.getAliases(instance.getGraph());

  // Check aliasing and modifying before disconnecting the old ops
  for (int i = 0; i < instance.external_inputs.size(); i++) {
    Tensor *inTensor = instance.external_inputs[i];

    auto modifiedRegions =
        inTensor->modifiedRegionsByOps(instance.ops, aliases);
    callOp->addModified(i, modifiedRegions);

    for (int j = 0; j < instance.external_outputs.size(); j++) {
      Tensor *outTensor = instance.external_outputs[j];

      if (inTensor->id == outTensor->id) {
        throw internal_error(
            "[SubgraphOutline] {} is both subgraph input and output.",
            inTensor);
      }

      // alias Regions in input Tensor:
      auto fwdAliasRegions = aliases.getChainsFromTo(inTensor, outTensor);
      auto bwdAliasRegions = aliases.getChainsFromTo(outTensor, inTensor);

      callOp->addAlias(i, j, fwdAliasRegions, bwdAliasRegions);
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
    auto oldOp = instance.getGraph().getOp(opid);
    oldOp->disconnectAllInputs();
    oldOp->disconnectAllOutputs();
    priority = std::max(priority, oldOp->settings.schedulePriority);
  }

  // CallOp's priority should be the max priority of the Op's that it replaces
  callOp->settings.schedulePriority = priority;

  // Connect inputs
  for (int i = 0; i < instance.external_inputs.size(); i++) {
    auto input = instance.external_inputs.at(i);
    callOp->connectInTensor(i, input->id);
  }

  // Connect outputs
  for (int i = 0; i < instance.external_outputs.size(); i++) {
    auto output = instance.external_outputs.at(i);
    callOp->connectOutTensor(i, output->id);
  }

  std::map<Op *, std::vector<Op *>> opRemaps;

  // Remap between instance ops and subgraph ops (used to transfer topocons)
  for (auto &opAndIndex : index_map) {
    opRemaps.insert({instance.graph->getOp(instance.ops.at(opAndIndex.second)),
                     {opAndIndex.first}});
  }

  TopoCons::transferToSubgraph(callOp, opRemaps);

  // Erase the old ops
  for (auto opid : instance.ops) {
    instance.getGraph().eraseOp(opid);
  }

  callOp->setup();

  return callOp;
}

bool SubgraphOutline::apply(Graph &graph) const {

  auto &ir = graph.getIr();

  auto schedule = getFullSchedule(ir);

  // Change schedule to include boundaries that can't be outlined
  localoutline::insertBoundariesOps(ir.getSessionOptions(), schedule);

  // Get updated schedule with boundaries
  schedule = getFullSchedule(ir);

  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op = schedule.at(i);
    logging::transform::trace("[SubgraphOutline] {} - graph {}, opid {}, op {}",
                              i,
                              op->getGraph().id,
                              op->id,
                              op->debugName());
  }

  // Get the software parallel schedule to generate sequences that should
  // be outlined as a whole
  localoutline::SoftParallelismModel softParallelismModel(schedule);
  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {
    softParallelismModel.log();
  }

  auto matches =
      getRinseMatches(schedule,
                      softParallelismModel.getParallelSchedule(),
                      ir.getSessionOptions().outlineThreshold,
                      ir.getSessionOptions().outlineSequenceBreakCost,
                      ir.getSessionOptions().enableOutliningCopyCostPruning);

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

    // Make sure all instances have the same outputs
    match.equalizeInstanceOutputs();

    auto replacements = applyMatch(match, ir);
    applyReplacements(matches, replacements);
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
