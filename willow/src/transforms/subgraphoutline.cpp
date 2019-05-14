#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/range/algorithm/find.hpp>

#include <poponnx/error.hpp>
#include <poponnx/graph.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/op/call.hpp>
#include <poponnx/topocons.hpp>

#include <poponnx/subgraph/outliner.hpp>
#include <poponnx/transforms/subgraphoutline.hpp>

using boost::find;
using boost::algorithm::any_of;

namespace poponnx {

namespace {

class Match {
public:
  class Instance {
  public:
    Instance(const std::vector<OpId> &, Graph &);

    std::vector<OpId> ops;

    std::vector<Tensor *> external_inputs;
    std::vector<Tensor *> external_outputs;
    std::set<Tensor *> all_outputs;

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

    // can not use op->input->tensors() here.
    // order is very important and op->input->tensors() doesn't gaurantee
    // returning tensors in index order
    for (int i = 0; i < op->input->n(); i++) {
      auto input = op->inTensor(i);

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

    // can not use op->output->tensors() here.
    // order is very important and op->output->tensors() doesn't gaurantee
    // returning tensors in index order
    for (int i = 0; i < op->output->n(); i++) {
      auto output = op->outTensor(i);

      if (hasExternalConsumer(output) || ir.isAnchored(output->id)) {
        addExternalOutput(output);
      }

      all_outputs.insert(output);
    }
  }
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

  // Add the topocons for the replacement op
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

  // Vote for new Phase amongst Ops (unanimous agreement required)
  Phase phase = graph.getOp(instance.ops[0])->getPhase();
  for (OpId id : instance.ops) {
    if (phase != graph.getOp(id)->getPhase()) {
      phase = Phase::UNDEFINED;
      break;
    }
  }

  // Copy some attributes from the first op in the instance
  auto scope    = graph.getOp(instance.ops.at(0))->getScope();
  auto vgraphid = graph.getOp(instance.ops.at(0))->getVirtualGraphId();

  // Disconnect the old ops
  for (auto opid : instance.ops) {
    auto op = graph.getOp(opid);
    op->disconnectAllInputs();
    op->disconnectAllOutputs();
  }

  // Create the call op
  auto up_call_op = make_unique<CallOp>(graph, subgraph);
  auto call_op_id = graph.moveIntoGraph(std::move(up_call_op));
  auto call_op    = graph.getOp(call_op_id);
  call_op->setPhase(phase);
  call_op->settings.scope = scope;
  call_op->setVirtualGraphId(vgraphid);

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

int generate_subgraph_unique_id() {
  static int uid = 0;
  return uid++;
}

Graph &createSubgraph(const Match &match, Graph &graph) {
  auto &ir = graph.getIr();
  auto subgraph_id =
      fmt::format("{}_subgraph({})", graph.id, generate_subgraph_unique_id());
  auto &subgraph      = ir.createGraph(subgraph_id);
  auto subgraph_scope = Scope() / graph.id.str() / subgraph_id;
  auto &instance      = match.instances[0];

  // clone all the ops and move into subgraph
  // many ops don't have a clone method implemented yet
  // so if there is an clone_error, erase the graph and return none
  std::map<Op *, Op *> clone_map;
  for (auto opid : instance.ops) {
    auto op    = graph.getOp(opid);
    auto clone = op->clone();
    clone->setPhase(op->getPhase());
    clone->settings.graph = subgraph;
    clone->settings.scope = subgraph_scope;
    auto cloneid          = subgraph.moveIntoGraph(std::move(clone));
    clone_map.insert({op, subgraph.getOp(cloneid)});
  }

  // duplicate all the output tensors
  std::map<Tensor *, Tensor *> tensor_map;
  for (auto output : instance.all_outputs) {
    auto new_id = (subgraph_scope / output->id).str();

    auto clone = output->clone();
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
          "tensor {} is already in tensor map, can not rebind to {} -> {}",
          tensor->id,
          tensor->id,
          t->id);
    }
    tensor_map.insert({tensor, t});
  }

  // create graph outputs
  for (auto tensor : instance.external_outputs) {
    auto out_id = tensor_map.at(tensor)->id;
    subgraph.addOutput(out_id);
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
      clone->connectInTensor(idx, clone_tensor_id);
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
                                   float threshold) {
  auto fw_matches = fwtools::subgraph::getRinseMatches(
      ops, threshold, fwtools::subgraph::getDefaultOutlinerAlgorithm());

  // Sort the matches so the smallest subgraphs are at the back.
  // `matches' is treated like a stack, so this will ensure the smallest
  // subgraphs are processed first `matches' can not be std::stack as it needs
  // to be iterated over
  std::sort(fw_matches.begin(),
            fw_matches.end(),
            [=](fwtools::subgraph::Match &p1, fwtools::subgraph::Match &p2) {
              return p1.length > p2.length;
            });

  std::vector<Match> matches;

  for (auto &match : fw_matches) {
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
  auto &ir = graph.getIr();

  std::vector<Op *> outlinedOps = graph.getOpSchedule({});

  auto matches =
      getRinseMatches(outlinedOps, ir.getSessionOptions().outlineThreshold);

  // matches needs to be treated like a stack
  while (!matches.empty()) {
    auto match = matches.back();
    matches.pop_back();

    auto replacements = applyMatch(match, graph);
    applyReplacements(matches, replacements);
  }

  graph.getTensors().removeIsolated();

  return true;
}

std::size_t SubgraphOutline::id() {
  return typeid(SubgraphOutline).hash_code();
}

namespace {
bool init = Transform::registerTransform(new SubgraphOutline);
}

} // namespace poponnx
