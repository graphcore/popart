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
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/subgraph/iosubgraphcostmodel.hpp>
#include <popart/subgraph/outliner.hpp>
#include <popart/subgraph/prunematches.hpp>
#include <popart/subgraph/subgraphutil.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/subgraphoutline.hpp>

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

std::vector<int64_t> getBoundariesCrossed(int64_t start,
                                          int64_t end,
                                          const std::vector<Op *> &schedule) {
  std::vector<int64_t> crossing;
  PingPongPhase phase{-1LL};
  PingPongPhase last_phase{-1LL};
  RecomputeType recompute      = RecomputeType::UNDEFINED;
  RecomputeType last_recompute = RecomputeType::UNDEFINED;

  for (int64_t i = start; i < end; ++i) {
    Op *op         = schedule[i];
    last_phase     = phase;
    last_recompute = recompute;
    phase          = op->hasPingPongPhase() ? op->getPingPongPhase() : -1;
    recompute      = op->settings.recomputeType == RecomputeType::RECOMPUTE
                    ? RecomputeType::RECOMPUTE
                    : RecomputeType::CHECKPOINT;
    if (i > start && (phase != last_phase || recompute != last_recompute)) {
      crossing.push_back(i - start);
    }
  }
  return crossing;
}

} // namespace

namespace {
//  Outlining matches are not supposed to cross certain boundaries:
// a.) Across recompute/non-recompute operators
// b.) Across PingPong phases
void insertBoundariesOps(const std::vector<Op *> &schedule) {
  if (!schedule.empty()) {
    for (int64_t i = 0; i < schedule.size() - 1; ++i) {
      auto crossed = getBoundariesCrossed(i, i + 2, schedule);
      if (crossed.size() > 0) {
        auto &graph     = schedule[i]->getGraph();
        auto boundaryOp = std::make_unique<BoundaryOp>(Op::Settings(graph, ""));
        auto boundary   = boundaryOp.get();
        auto phase      = schedule[i]->getOptionalPingPongPhase();
        boundary->setPingPongPhase(phase);
        VGraphId vgid = 0;
        boundary->setVirtualGraphId(vgid);
        graph.moveIntoGraph(std::move(boundaryOp));
        graph.topoCons.get()->insert(schedule[i], boundary);
        graph.topoCons.get()->insert(boundary, schedule[i + 1]);
      }
    }
  }
}

std::vector<Match> separateTopLevelMatches(const std::vector<Match> &inMatches,
                                           size_t scheduleSize) {
  logging::trace("[SubgraphOutline] Separate top level matches start.");
  std::vector<Match> filtered;

  std::vector<int64_t> covered(scheduleSize, 0);

  std::map<int, std::vector<Match>> matchesByLength;
  for (auto &match : inMatches) {
    matchesByLength[match.length].push_back(match);
  }

  for (auto iter = matchesByLength.rbegin(); iter != matchesByLength.rend();
       ++iter) {
    for (auto &match : iter->second) {
      std::vector<Start> topLevelStarts;
      std::vector<Start> coveredStarts;
      for (Start start : match.starts) {
        if (covered[start] > 0) {
          coveredStarts.push_back(start);
        } else {
          topLevelStarts.push_back(start);
        }
        for (Start i = start; i < start + match.length; ++i) {
          covered[i] += 1;
        }
      }
      if (topLevelStarts.size() > 0) {
        if (coveredStarts.size() > 0) {
          // Mix of top level and covered starts; repeat top level matches
          filtered.push_back(Match(topLevelStarts, match.length));
        }
      }
      filtered.push_back(match);
    }
  }
  logging::trace("[SubgraphOutline] Separate top level matches end.");
  return filtered;
}
} // namespace

} // namespace localoutline

class Match {
public:
  class Instance {
  public:
    Instance(const std::vector<OpId> &, Graph &);

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

  Match(const fwtools::subgraph::Match &, const std::vector<Op *> &);

  std::vector<Instance> instances;
  int length;
};

class Replacement {
public:
  Replacement(const std::vector<OpId> &ops_, const OpId &replacement_op_)
      : ops(ops_), replacement_op(replacement_op_) {}

  std::vector<OpId> ops;
  OpId replacement_op;
};

Match::Instance::Instance(const std::vector<OpId> &ops_, Graph &graph)
    : ops(ops_) {
  std::set<Op *> op_set;
  for (auto opid : ops) {
    auto op = graph.getOp(opid);
    op_set.insert(op);
  }

  auto &ir = graph.getIr();

  for (auto opid : ops) {
    auto op = graph.getOp(opid);

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
             const std::vector<Op *> &ops)
    : length(match.length) {
  for (auto &start : match.starts) {
    std::vector<OpId> m;

    for (int i = 0; i < match.length; i++) {
      auto idx = start + i;
      m.push_back(ops[idx]->id);
    }

    instances.emplace_back(m, ops[0]->getGraph());
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

    logging::trace("[SubgraphOutline] Updating TopoCons for {} ops {}",
                   graph.getOp(replacement_op)->debugName(),
                   graph_ops);
  }

  // dont include any of the ops being replaced
  auto include_op = [&](OpId opid) { return find(ops, opid) == ops.end(); };

  std::set<Op *> befores;
  std::set<Op *> afters;

  // Get all befores and afters that are not in ops
  for (auto &opid : ops) {
    for (auto before : graph.topoCons->getBefores(graph.getOp(opid))) {
      if (include_op(before->id)) {
        befores.insert(before);
      }
    }
    for (auto after : graph.topoCons->getAfters(graph.getOp(opid))) {
      if (include_op(after->id)) {
        afters.insert(after);
      }
    }
  }

  // Remove the existing topocons
  for (auto &opid : ops) {
    graph.topoCons->remove(graph.getOp(opid));
  }

  // Add the topoCons for the replacement Op
  for (auto before : befores) {
    graph.topoCons->insert(before, graph.getOp(replacement_op));
  }
  for (auto after : afters) {
    graph.topoCons->insert(graph.getOp(replacement_op), after);
  }
}

static OpId replaceWithCallOp(const Match::Instance &instance,
                              Graph &graph,
                              Graph &subgraph) {

  // Copy some attributes with heuristics from the instance ops
  boost::optional<Scope> scope;
  boost::optional<VGraphId> ipu_copy_vgid;
  boost::optional<VGraphId> vgid;
  boost::optional<PingPongPhase> phase;
  boost::optional<BatchSerializedPhase> batchserial;
  bool conflicting_batchserial = false;
  boost::optional<PipelineStage> pipeline_stage;
  boost::optional<RecomputeType> recompute;

  for (const OpId &opid : instance.ops) {
    Op *op = graph.getOp(opid);
    if (!scope.is_initialized()) {
      scope = op->getScope();
    }
    IpuCopyOp *copy = dynamic_cast<IpuCopyOp *>(op);
    if (copy && !ipu_copy_vgid.is_initialized()) {
      ipu_copy_vgid = copy->getSourceIpu();
    }
    if (!vgid.is_initialized()) {
      vgid = op->getOptionalVirtualGraphId();
    }
    if (!phase.is_initialized()) {
      phase = op->getOptionalPingPongPhase();
    }
    if (!pipeline_stage.is_initialized()) {
      pipeline_stage = op->getOptionalPipelineStage();
    }
    if (!recompute.is_initialized()) {
      recompute = op->settings.recomputeType;
    }
    if (batchserial.is_initialized() && op->hasBatchSerializedPhase() &&
        op->getBatchSerializedPhase() != batchserial) {
      conflicting_batchserial = true;
    }
  }

  // Fallback to IPU copy VGID
  if (!vgid.is_initialized()) {
    vgid = ipu_copy_vgid;
  }

  if (conflicting_batchserial) {
    batchserial.reset();
  }

  // Create the call op. Note that toLoss and fromLoss are set in the
  // constructor
  auto up_call_op =
      std::make_unique<CallOp>(Onnx::CustomOperators::Call_1, graph, subgraph);
  auto call_op_id         = graph.moveIntoGraph(std::move(up_call_op));
  CallOp *call_op         = dynamic_cast<CallOp *>(graph.getOp(call_op_id));
  call_op->settings.scope = scope.get();
  call_op->settings.recomputeType = recompute.get();
  call_op->setVirtualGraphId(vgid);
  call_op->setPingPongPhase(phase);
  call_op->setPipelineStage(pipeline_stage);
  call_op->setBatchSerializedPhase(batchserial);

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

  // Check aliasing before disconnecting the old ops
  for (int i = 0; i < instance.external_inputs.size(); i++) {
    Tensor *inTensor = instance.external_inputs[i];
    for (int j = 0; j < instance.external_outputs.size(); j++) {
      Tensor *outTensor = instance.external_outputs[j];

      if (inTensor->id == outTensor->id) {
        throw internal_error(
            "[SubgraphOutline] {} is both subgraph input and output.",
            inTensor);
      }

      // alias Regions in input Tensor:
      auto fwdAliasRegions =
          graph.getTensors().getAliasRegions(inTensor, outTensor);
      auto bwdAliasRegions =
          graph.getTensors().getAliasRegions(outTensor, inTensor);

      for (const auto &r : fwdAliasRegions) {
        if (r.rank() != outTensor->info.rank()) {
          throw error(
              "Invalid Region of rank {} in updateTopoCons at InIndex {} "
              "where the input Tensor is of rank {}. The Input Tensor is {}, "
              "and it is entering CallOp {}. The Input Tensor has TensorInfo "
              "{}.",
              r.rank(),
              i,
              outTensor->info.rank(),
              outTensor->str(),
              call_op->str(),
              outTensor->info);
        }
      }
      for (const auto &r : bwdAliasRegions) {
        if (r.rank() != inTensor->info.rank()) {
          throw error(
              "Invalid Region of rank {} in updateTopoCons at InIndex {} "
              "where the input Tensor is of rank {}. The Input Tensor is {}, "
              "and it is entering CallOp {}. The Input Tensor has TensorInfo "
              "{}.",
              r.rank(),
              i,
              inTensor->info.rank(),
              inTensor->str(),
              call_op->str(),
              inTensor->info);
        }
      }

      call_op->addAlias(i, j, fwdAliasRegions, bwdAliasRegions);
      if (logging::shouldLog(logging::Module::transform,
                             logging::Level::Trace)) {
        logging::trace("[SubgraphOutline] Alias {} ({}) -> {} ({})",
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

  return call_op->id;
}

static int subgraph_uid = 0;

void reset_subgraph_id() { subgraph_uid = 0; }

int generate_subgraph_unique_id() { return subgraph_uid++; }

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

Graph &createSubgraph(const Match &match, Graph &graph) {

  auto &ir         = graph.getIr();
  auto subgraph_id = logging::format(
      "{}_subgraph({})", graph.id, generate_subgraph_unique_id());
  auto &subgraph      = ir.createGraph(subgraph_id);
  auto subgraph_scope = subgraph.getScope();
  auto &instance      = match.instances[0];

  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {
    std::stringstream ss;
    for (int i = 0; i < instance.ops.size(); i++) {
      auto opid = instance.ops.at(i);
      auto op   = graph.getOp(opid);
      ss << std::endl
         << "    " << op->debugName() << ", "
         << "VGID: " << (op->hasVirtualGraphId() ? op->getVirtualGraphId() : -1)
         << ", "
         << "PingPong phase: "
         << (op->getOptionalPingPongPhase()
                 ? op->getOptionalPingPongPhase().get()
                 : -1);
    }
    logging::trace("[SubgraphOutline] Creating subgraph: {}, "
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
    auto op                       = graph.getOp(opid);
    auto clone                    = op->clone();
    clone->settings.graph         = subgraph;
    clone->settings.scope         = subgraph_scope;
    clone->settings.recomputeType = RecomputeType::CHECKPOINT;
    auto cloneid                  = subgraph.moveIntoGraph(std::move(clone));
    Op *clone_op                  = subgraph.getOp(cloneid);
    clone_map.insert({op, clone_op});
    clones.push_back(clone_op);
  }

  // Map out constraints by schedule match positions for all instances.
  // If different constraints per instance exist, they either clash or can
  // coexist. We assume instance.ops preserves schedule order.
  std::set<std::pair<int, int>> constraints;
  for (auto &instanceForConstraints : match.instances) {
    for (int i = 0; i < instanceForConstraints.ops.size(); i++) {
      auto opid   = instanceForConstraints.ops.at(i);
      auto op     = graph.getOp(opid);
      auto afters = graph.topoCons->getAfters(op);
      for (Op *after_op : afters) {
        auto j = instanceForConstraints.getIndex(after_op);
        if (j > 0) {
          // i before j
          constraints.insert({i, j});
        }
      }
    }
  }

  // Preserve topological constraints between ops being added to the subgraph
  for (auto &constraint : constraints) {
    subgraph.topoCons->insert(clones[constraint.first],
                              clones[constraint.second]);
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
    auto input_id = subgraph.addInput(tensor->info);
    auto t        = subgraph.getTensors().get(input_id);
    if (tensor_map.find(tensor) != tensor_map.end()) {
      throw error(
          "tensor {} is already in tensor map, cannot rebind to {} -> {}",
          tensor->id,
          tensor->id,
          t->id);
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
    auto op    = graph.getOp(opid);
    auto clone = clone_map.at(op);

    // connect inputs
    for (auto &idx_tensor : op->input->tensorMap()) {
      auto idx             = idx_tensor.first;
      auto tensor          = idx_tensor.second;
      auto clone_tensor_id = tensor_map.at(tensor)->id;
      auto *copyOp         = dynamic_cast<IpuCopyOp *>(op);
      auto *cloneCopyOp    = dynamic_cast<IpuCopyOp *>(clone);
      if (copyOp && cloneCopyOp) {
        auto sourceIpu = copyOp->getSourceIpus().at(tensor->id);
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

// Create a subgraph for the match and
// replace instances of the match with a CallOp
static std::vector<Replacement> applyMatch(const Match &match, Graph &graph) {
  verifyMatchInstances(match);

  // TODO: Verify. This is possibly too strict. Can probably be dropped.
  // verifyTopologicalConstraints(match, graph);

  auto &subgraph = createSubgraph(match, graph);

  std::vector<Replacement> replacements;

  // Replace the matches with call ops
  for (auto &instance : match.instances) {
    auto call_op_id = replaceWithCallOp(instance, graph, subgraph);
    replacements.push_back({instance.ops, call_op_id});
  }

  return replacements;
}

// Returns a vector of Match instance
// sorted so the smallest matches are at the back
std::vector<Match> getRinseMatches(const std::vector<Op *> &ops,
                                   float threshold,
                                   bool copyCostPruning,
                                   bool topLevelSeparation) {

  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {
    std::vector<int> intSchedule = fwtools::subgraph::getIntSchedule(ops);
    for (size_t i = 0; i < ops.size(); ++i) {
      Op *op = ops[i];
      logging::trace("[SubgraphOutline] "
                     "Index: {}, ID: {}, Op: {}, "
                     "VGID: {}, PingPong phase: {}",
                     i,
                     intSchedule[i],
                     op->debugName(),
                     op->hasVirtualGraphId() ? op->getVirtualGraphId() : -1,
                     op->getOptionalPingPongPhase()
                         ? op->getOptionalPingPongPhase().get()
                         : -1);
    }
    logging::trace("[SubgraphOutline] Int schedule: {}", intSchedule);
  }

  auto fw_matches = fwtools::subgraph::getRinseMatches(
      ops, threshold, fwtools::subgraph::getDefaultOutlinerAlgorithm());
  int64_t num_matches_0 = fw_matches.size();

  // TODO: T Copy cost pruning can cause crossing matches,
  // and is therefore buggy/broken.
  if (copyCostPruning) {
    fw_matches = fwtools::subgraph::prune::
        pruneMatches<Op, popart::outline::IoSubgraphCostModel>(
            fw_matches, ops, threshold);
  }
  int64_t num_matches_1 = fw_matches.size();

  // TODO: Enable this only when aliasZeroCopy is enabled, which requires
  // separation of top-level and non-top-level matches currently.
  if (topLevelSeparation) {
    fw_matches = localoutline::separateTopLevelMatches(fw_matches, ops.size());
  }
  int64_t num_matches_2 = fw_matches.size();

  // Remove the offsets caused by the boundary OPs from the matches
  // outline::removeBoundariesOps(fw_matches, opsWithBoundaries);

  logging::trace("[SubgraphOutline] Matches before pruning: {}, "
                 "matches after IOSize: {}, "
                 "matches after TopLevel: {}",
                 num_matches_0,
                 num_matches_1,
                 num_matches_2);

  // Sort the matches so the smallest subgraphs are at the back.
  // `matches' is treated like a stack, so this will ensure the smallest
  // subgraphs are processed first `matches' cannot be std::stack as it needs
  // to be iterated over
  sortMatches<fwtools::subgraph::Match>(fw_matches);

  std::vector<Match> matches;

  for (auto &match : fw_matches) {
    logging::trace("[SubgraphOutline] Match length: {}, starts: {}",
                   match.length,
                   match.starts);
    matches.emplace_back(match, ops);
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
    instance.ops.insert(start, replacement.replacement_op);
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

} // namespace

bool SubgraphOutline::apply(Graph &graph) const {

  // Make sure we start with a 0 subgraph id
  reset_subgraph_id();

  auto &ir = graph.getIr();

  std::vector<Op *> schedule = graph.getOpSchedule({});

  // Change schedule to include boundaries that can't be outlined
  localoutline::insertBoundariesOps(schedule);

  // Get updated schedule with boundaries
  schedule = graph.getOpSchedule({});

  auto matches =
      getRinseMatches(schedule,
                      ir.getSessionOptions().outlineThreshold,
                      ir.getSessionOptions().enableOutliningCopyCostPruning,
                      ir.getSessionOptions().pingPongPhases > 1);

  if (logging::shouldLog(logging::Module::none, logging::Level::Trace)) {
    unsigned i = 0;
    for (auto &match : matches) {
      std::stringstream ss;
      for (auto &instance : match.instances) {
        ss << "["
           << logging::join(instance.ops.begin(), instance.ops.end(), ", ")
           << "]";
      }
      logging::trace("[SubgraphOutline] Match {}: {}", i, ss.str());
      ++i;
    }
  }

  // matches needs to be treated like a stack
  while (!matches.empty()) {
    auto match = matches.back();
    matches.pop_back();

    auto replacements = applyMatch(match, graph);
    applyReplacements(matches, replacements);
  }

  // Remove all boundaries
  schedule = graph.getOpSchedule({});
  for (Op *op : schedule) {
    if (dynamic_cast<BoundaryOp *>(op)) {
      graph.topoCons->remove(graph.getOp(op->id));
      graph.eraseOp(op->id);
    }
  }

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
