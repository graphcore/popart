// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/range/algorithm/find.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <subgraph/wrappedop.hpp>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>
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
#include <popart/op/expand.hpp>
#include <popart/op/if.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/nop.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/subgraph/iosubgraphcostmodel.hpp>
#include <popart/subgraph/match.hpp>
#include <popart/subgraph/outliner.hpp>
#include <popart/subgraph/prunematches.hpp>
#include <popart/subgraph/subgraphutil.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/subgraphoutline.hpp>
#include <popart/util.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/aliases.hpp"
#include "popart/analysis/replicaequal/replicaequalanalysis.hpp"
#include "popart/basicoptionals.hpp"
#include "popart/chains.hpp"
#include "popart/datatype.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/opdebuginfo.hpp"
#include "popart/operators.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/scope.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/subgraph/subgraphnames.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"
#include "popart/tensors.hpp"
#include "popart/transforms/transform.hpp"
#include "popart/vendored/any.hpp" // IWYU pragma: keep
#include "popart/vertex.hpp"

namespace popart {
class AliasModel;
} // namespace popart

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

  nonstd::optional<ExecutionContext> exec_cont;
  nonstd::optional<ExecutionContext> last_exec_cont;

  OptionalBatchSerializedPhase batchserial;
  OptionalBatchSerializedPhase last_batchserial;

  RecomputeType recompute      = RecomputeType::Undefined;
  RecomputeType last_recompute = RecomputeType::Undefined;

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
    bool check_batchserial = true;

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

    if (i > start &&
        ((exec_cont != last_exec_cont) || (recompute != last_recompute) ||
         (check_batchserial && batchserial != last_batchserial))) {
      crossing.push_back(i - start);
    }
  }
  return crossing;
}

// Cost model for computing runs of parallelizable sequences
class SoftParallelismModel {
public:
  SoftParallelismModel(const std::vector<Op *> &schedule_)
      : schedule(schedule_) {
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
static std::vector<Replacement>
applyMatch(const Match &match, Ir &ir, AliasesMap &aliasesMap) {
  verifyMatchInstances(match);

  // TODO: Verify. This is possibly too strict. Can probably be dropped.
  // verifyTopologicalConstraints(match, graph);

  std::map<Op *, int> index_map;

  auto &subgraph =
      SubgraphOutline::createSubgraph(match.instances, ir, index_map);

  handleRandomReferences(match, ir);

  std::vector<Replacement> replacements;

  // Replace the matches with call ops
  for (auto &instance : match.instances) {
    auto call_op = SubgraphOutline::replaceWithCallOp(
        instance, subgraph, index_map, aliasesMap);
    replacements.push_back({instance.ops, call_op});
  }

  return replacements;
}

// Function to filter out Matches which, for some instance, have an alias
// between an input and an output that do not agree on Shape. This is because
// currently CallOp aliases are grown into Poprithms' alias models as CrossLinks
// which assume that any aliases have matching shapes, so violating this
// assumption would lead to an exception.
bool matchIsAliasCompatible(const Match &match, AliasesMap &aliasesMap) {

  bool aliasCompatible = true;
  for (const auto &instance : match.instances) {
    // Get the aliases object for the right graph.
    auto &aliases = aliasesMap.getAliases(instance.getGraph());

    for (int i = 0; i < instance.external_inputs.size(); i++) {
      Tensor *inTensor = instance.external_inputs[i];
      for (int j = 0; j < instance.external_outputs.size(); j++) {
        Tensor *outTensor = instance.external_outputs[j];
        // If shapes match then there isn't a problem.
        if (aliasCompatible &&
            inTensor->info.shape() != outTensor->info.shape()) {
          // If there is a non-empty alias, we must filter this match out.
          auto regions = aliases.getChainsFromTo(inTensor, outTensor);
          if (!regions.isEmpty()) {
            aliasCompatible = false;
          }
        }
      }
    }
  }

  return aliasCompatible;
}

// Returns a vector of Match instance
// sorted so the smallest matches are at the back
std::vector<Match>
getRinseMatches(const std::vector<popart::Op *> &ops,
                const std::vector<fwtools::subgraph::WrappedOp *> &wrappedOps,
                const std::vector<std::pair<size_t, size_t>> &sequences,
                float threshold,
                float sequenceBreakCost,
                bool copyCostPruning,
                AliasesMap &aliasesMap) {

  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {
    std::vector<int> intSchedule =
        fwtools::subgraph::getIntSchedule(wrappedOps);
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
      wrappedOps,
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
               [&wrappedOps](const fwtools::subgraph::Match &match) {
                 return !(match.length == 1 &&
                          wrappedOps.at(match.starts.front())
                              ->unwrap()
                              ->isConvertibleTo<CallOp>());
               });
  fw_matches = filtered_fw_matches;

  // Sort the matches so the smallest subgraphs are at the back.
  // `matches' is treated like a stack, so this will ensure the smallest
  // subgraphs are processed first `matches' cannot be std::stack as it needs
  // to be iterated over
  sortMatches<fwtools::subgraph::Match>(fw_matches);

  std::vector<Match> matches;
  for (auto &match : fw_matches) {
    // Filter out those cases where there are aliases between inputs and outputs
    // but the shapes of those inputs and outputs are not identical. This is
    // because, when we grow the Poprithms' AliasModel for the CallOp's that
    // call the subgraph that replaces the Op clusters, Poprithms assumes that
    // these shapes match.
    Match wrappedMatch{match, ops};
    if (matchIsAliasCompatible(wrappedMatch, aliasesMap)) {
      logging::transform::trace(
          "[SubgraphOutline] Match length: {}, starts: {}",
          match.length,
          match.starts);
      matches.push_back(wrappedMatch);
    }
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
    auto firstItAfterLastErase =
        instance.ops.erase(start, start + replacement.ops.size());
    instance.ops.insert(firstItAfterLastErase, replacement.replacement_op->id);
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
    auto new_id =
        addScope(subgraph, removeScope(output->getGraph(), output->id));

    auto clone = output->clone(subgraph);
    clone->id  = new_id;
    subgraph.getTensors().moveIntoTensors(std::move(clone));
    auto clone_ptr = subgraph.getTensors().get(new_id);
    tensor_map.insert({output, clone_ptr});
  }

  // create graph inputs
  for (auto tensor : instance.external_inputs) {
    auto input_id =
        addScope(subgraph, removeScope(tensor->getGraph(), tensor->id));
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

static void addCallOutlineDebugInfo(Op *call,
                                    const SubgraphableOpCluster &instance,
                                    const std::map<Op *, int> &index_map) {
  // Debug context IDs of PopArt ops that were replaced by a Call...
  {
    std::ostringstream oss("[", std::ios::ate);
    for (auto const opId : instance.ops) {
      oss << instance.graph->getOp(opId)->debugInfo.getId() << ",";
    }
    if (oss.tellp() > 1) {
      oss.seekp(-1, oss.cur); // Replace last comma
    }
    oss << "]";
    call->debugInfo.setValue("replacedDebugContextIds", oss.str());
  }
  // ...and corresponding debug context IDs of PopArt ops in the Call function
  {
    std::ostringstream oss("[", std::ios::ate);
    for (auto const &opAndIndex : index_map) {
      oss << opAndIndex.first->debugInfo.getId() << ",";
    }
    if (oss.tellp() > 1) {
      oss.seekp(-1, oss.cur); // Replace last comma
    }
    oss << "]";
    call->debugInfo.setValue("outlinedDebugContextIds", oss.str());
  }
}

Op *SubgraphOutline::replaceWithCallOp(const SubgraphableOpCluster &instance,
                                       Graph &subgraph,
                                       const std::map<Op *, int> &index_map,
                                       AliasesMap &aliasesMap) {

  // Create the call op. Note that toLoss and fromLoss are set in the
  // constructor
  auto up_call_op = std::make_unique<CallOp>(
      Onnx::CustomOperators::Call_1,
      subgraph,
      Op::Settings{instance.getGraph(), "", instance.getGraph().getScope()});
  auto call_op_id = instance.getGraph().moveIntoGraph(std::move(up_call_op));
  CallOp *callOp =
      dynamic_cast<CallOp *>(instance.getGraph().getOp(call_op_id));

  setSubgraphOpSettingsFromClusterInstance(callOp, instance);
  addCallOutlineDebugInfo(callOp, instance, index_map);

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

  // Disconnect the old ops
  for (auto opid : instance.ops) {
    auto oldOp = instance.getGraph().getOp(opid);
    oldOp->disconnectAllInputs();
    oldOp->disconnectAllOutputs();
  }

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

  std::map<Op *, std::vector<Op *>, POpCmp> opRemaps;

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

Graph &SubgraphOutline::createEmptySubgraph(
    const SubgraphableOpCluster &instance,
    Ir &ir,
    std::string subgraphId,
    const std::map<InIndex, OutIndex> &identityInputToOutputIndiciesMapping,
    const std::map<OutIndex, float> &outputIndiciesAndValues,
    AliasModel &aliasModel) {
  auto &subgraph      = ir.createGraph(GraphId(subgraphId));
  auto subgraph_scope = subgraph.getScope();
  Op::Settings subgraphSettings(subgraph, subgraphId, subgraph.getScope());

  // duplicate all the output tensors
  std::map<Tensor *, Tensor *> tensor_map;
  for (auto output : instance.all_outputs) {
    auto new_id =
        addScope(subgraph, removeScope(output->getGraph(), output->id));

    auto clone = output->clone(subgraph);
    clone->id  = new_id;
    subgraph.getTensors().moveIntoTensors(std::move(clone));
    auto clone_ptr = subgraph.getTensors().get(new_id);
    tensor_map.insert({output, clone_ptr});
  }

  // create graph inputs
  for (auto tensor : instance.external_inputs) {
    auto input_id =
        addScope(subgraph, removeScope(tensor->getGraph(), tensor->id));
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

  // Create nopOps and hook up them to graph inputs and outputs
  // based on user specifications.
  for (auto opid : instance.ops) {
    auto op = instance.getGraph().getOp(opid);
    for (const auto &inOutNop : identityInputToOutputIndiciesMapping) {
      auto nopOp     = subgraph.createOp<NopOp>(Onnx::CustomOperators::Nop_1,
                                            subgraphSettings);
      auto tensorIn  = op->input->tensorMap().at(inOutNop.first);
      auto tensorOut = op->output->tensorMap().at(inOutNop.second);
      nopOp->connectInTensor(NopOp::getInIndex(), tensor_map.at(tensorIn)->id);
      nopOp->connectOutTensor(NopOp::getOutIndex(),
                              tensor_map.at(tensorOut)->id);
      nopOp->inheritPlacementAttributes(false, aliasModel);
      nopOp->setVirtualGraphId(op->getOptionalVGraphId());
      nopOp->setup();
    }
  }

  for (auto opid : instance.ops) {
    auto op = instance.getGraph().getOp(opid);
    for (const auto &outputIndexAndValue : outputIndiciesAndValues) {
      auto tensorOut = op->output->tensorMap().at(outputIndexAndValue.first);
      Shape shapeOut = tensorOut->info.shape();
      DataType dataTypeOut = tensorOut->info.dataType();

      TensorId outputValueId =
          addScope(subgraph,
                   op->str() + "_outputIndex_" +
                       std::to_string(outputIndexAndValue.first));

      TensorInfo tensorInfoOut(dataTypeOut, Shape{});
      addConstInitFromFloat(outputIndexAndValue.second,
                            outputValueId,
                            tensorInfoOut,
                            subgraph.getTensors());

      TensorId shapeExpandTensorId = addScope(subgraph, "shapeExpandTensorId");
      std::vector<int64_t> shapeOut1D =
          shapeOut.empty()
              ? std::vector<int64_t>{1}
              : std::vector<int64_t>{static_cast<int64_t>(shapeOut.size())};
      TensorInfo shapeExpandInfo{DataType::INT64, shapeOut1D};
      std::vector<int64_t> shapeExpandData =
          shapeOut.empty() ? std::vector<int64_t>{0} : shapeOut;
      subgraph.getTensors().addConstInit(
          shapeExpandTensorId, shapeExpandInfo, shapeExpandData.data());

      auto expandOp = subgraph.createOp<ExpandOp>(Onnx::Operators::Expand_8,
                                                  subgraphSettings);
      expandOp->connectInTensor(ExpandOp::getInTensorIndex(), outputValueId);
      expandOp->connectInTensor(ExpandOp::getInShapeIndex(),
                                shapeExpandTensorId);
      expandOp->connectOutTensor(ExpandOp::getOutIndex(),
                                 tensor_map.at(tensorOut)->id);
      expandOp->inheritPlacementAttributes(false, aliasModel);
      expandOp->setVirtualGraphId(op->getOptionalVGraphId());
      expandOp->setup();
    }
  }

  return subgraph;
}

void SubgraphOutline::setSubgraphOpSettingsFromClusterInstance(
    Op *op,
    const SubgraphableOpCluster &instance) {

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

  if (scope) {
    op->settings.scope = scope.value();
  }
  if (recompute) {
    op->settings.recomputeType = recompute.value();
  }
  op->setVirtualGraphId(vgid);
  op->setExecutionPhase(phase);
  op->setPipelineStage(pipeline_stage);
  op->setBatchSerializedPhase(batchserial);
  if (execution_context) {
    op->settings.executionContext = execution_context.value();
  }

  // Set the position w.r.t loss, if possible. If any of the internal ops
  // is connected to the final loss, then so is this op. Note that we use
  // the Ops in the instance of this Match, and not the canonical subgraph.
  for (auto opid : instance.ops) {
    auto instanceOp = instance.graph->getOp(opid);
    if (instanceOp->toLoss == PathToLoss::Yes) {
      op->toLoss = PathToLoss::Yes;
    }
    if (instanceOp->fromLoss == PathFromLoss::Yes) {
      op->fromLoss = PathFromLoss::Yes;
    }
  }

  double priority = -std::numeric_limits<double>::infinity();
  for (auto opid : instance.ops) {
    auto oldOp = instance.getGraph().getOp(opid);
    priority   = std::max(priority, oldOp->settings.schedulePriority);
  }

  // Op's priority should be the max priority of the Op's that it replaces
  op->settings.schedulePriority = priority;
}

Op *SubgraphOutline::replaceWithEmptyElseBranchIfOp(
    const SubgraphableOpCluster &instance,
    Graph &subgraph,
    Graph &emptySubgraph,
    const std::map<Op *, int> &index_map,
    AliasesMap &aliasesMap,
    Tensor *flag) {

  Graph &graph = instance.getGraph();
  // Relation between IfOp inputs and its subgraphs inputs is
  // given by mappings.
  // IfOp has inputs:
  // 0 - condition flag
  // 1 - t0
  // ...
  // k+1 - tk
  // ThenBranch inputs:
  // 0 - t0
  // ...
  // k - tk
  // ElseBranch inputs:
  // none
  //
  // IfOp to ThenBranch inputs mapping:
  // {{1, 0}, {2, 1} ,{3, 2} ...}
  // This maps input 1 of the IfOp to input 0 of the ThenBranch and so on.
  // IfOp to ElseBranch inputs mapping:
  // {{}}
  std::map<int, int> subgraphInputIndices;
  for (int i = 0; i < instance.external_inputs.size(); i++) {
    subgraphInputIndices.insert(std::pair<int, int>(i + 1, i));
  }

  std::map<int, int> subgraphOutputIndices;
  for (int i = 0; i < instance.all_outputs.size(); i++) {
    subgraphOutputIndices.insert(std::pair<int, int>(i, i));
  }

  BranchInfo branchInfoSubgraph{
      subgraph.getGraphId(), subgraphInputIndices, subgraphOutputIndices};

  BranchInfo branchInfoEmptySubgraph{
      emptySubgraph.getGraphId(), subgraphInputIndices, subgraphOutputIndices};

  auto ifOp = graph.createOp<IfOp>(Onnx::Operators::If_1,
                                   branchInfoSubgraph,
                                   branchInfoEmptySubgraph,
                                   Op::Settings(graph, ""));

  setSubgraphOpSettingsFromClusterInstance(ifOp, instance);

  // ToDo T50509 can we add aliasing/inplace as in replaceWithCallOp case?

  // Disconnect the old op
  for (auto opid : instance.ops) {
    auto oldOp = instance.getGraph().getOp(opid);
    oldOp->disconnectAllInputs();
    oldOp->disconnectAllOutputs();
  }

  // Connect inputs
  ifOp->connectInTensor(IfOp::getConditionInIndex(), flag->id);
  for (int i = 0; i < instance.external_inputs.size(); i++) {
    auto input = instance.external_inputs.at(i);
    ifOp->connectInTensor(i + 1, input->id);
  }

  // Connect outputs
  for (int i = 0; i < instance.external_outputs.size(); i++) {
    auto output = instance.external_outputs.at(i);
    ifOp->connectOutTensor(i, output->id);
  }

  std::map<Op *, std::vector<Op *>, POpCmp> opRemaps;

  // Remap between instance ops and subgraph ops (used to transfer topocons)
  for (auto &opAndIndex : index_map) {
    opRemaps.insert({instance.graph->getOp(instance.ops.at(opAndIndex.second)),
                     {opAndIndex.first}});
  }

  TopoCons::transferToSubgraph(ifOp, opRemaps);

  // Erase the old ops
  for (auto opid : instance.ops) {
    instance.getGraph().eraseOp(opid);
  }

  ifOp->setup();

  return ifOp;
}

bool SubgraphOutline::apply(Graph &graph) const {

  auto &ir = graph.getIr();

  auto opSched = getFullSchedule(ir);

  // Change schedule to include boundaries that can't be outlined
  localoutline::insertBoundariesOps(ir.getSessionOptions(), opSched);

  // Get updated schedule with boundaries
  opSched = getFullSchedule(ir);

  ReplicaEqualAnalysis reAnalysis{graph.getIr()};
  reAnalysis.apply();

  auto wrappedOpSched =
      fwtools::subgraph::toWrappedOpSched(graph.getIr(), reAnalysis, opSched);

  for (size_t i = 0; i < opSched.size(); ++i) {
    Op *op = opSched.at(i);
    logging::transform::trace("[SubgraphOutline] {} - graph {}, opid {}, op {}",
                              i,
                              op->getGraph().id,
                              op->id,
                              op->debugName());
  }

  // Get the software parallel schedule to generate sequences that should
  // be outlined as a whole
  localoutline::SoftParallelismModel softParallelismModel(opSched);
  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {
    softParallelismModel.log();
  }

  AliasesMap aliasesMap{&ir};

  auto matches =
      getRinseMatches(opSched,
                      wrappedOpSched.rawPtrs,
                      softParallelismModel.getParallelSchedule(),
                      ir.getSessionOptions().outlineThreshold,
                      ir.getSessionOptions().outlineSequenceBreakCost,
                      ir.getSessionOptions().enableOutliningCopyCostPruning,
                      aliasesMap);

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

    auto replacements = applyMatch(match, ir, aliasesMap);
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
