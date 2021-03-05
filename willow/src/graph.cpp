// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>

#include <boost/algorithm/string.hpp>
#include <boost/range/algorithm.hpp>
#include <onnx/onnx_pb.h>

#include <popart/ces/constexpr.hpp>
#include <popart/ces/onnxconstexpr.hpp>
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/opmanager.hpp>
#include <popart/pbwrap.hpp>
#include <popart/poparttracepoint.hpp>
#include <popart/scheduler.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

// Ops required for Graph::getCalledOps
#include <popart/op/call.hpp>
#include <popart/op/if.hpp>

// The layers required to construct the backwards pass
#include <popart/op/accumulate.hpp>
#include <popart/op/accumulatorupdate.hpp>
#include <popart/op/conv.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/remote.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/sgd1acclupdate.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/varupdate.hpp>

#include <onnxpasses/onnxtoonnx.hpp>

// Prototypes
namespace {
using namespace popart;

// For some Op A with OpId a: (a, A) -> (a, [consumer OpIds of a]).
std::pair<OpId, std::unordered_set<OpId>> getConsumerOpIdsInGraph(
    const Graph *graph,
    const std::pair<const OpId, std::unique_ptr<Op>> &opid_op);

} // namespace

namespace popart {

Graph::~Graph() = default;

// map of grad Tensor to the list of Tensors that
// must be summed to create the grad Tensor
using GradTensorsPartsMap = std::map<Tensor *, std::vector<Tensor *>>;

class TensorGradMapRegister {
public:
  void insert(Tensor *nonGrad, Tensor *grad);
  GradTensorsPartsMap popComplete();

  GradTensorsPartsMap partial;
  GradTensorsPartsMap complete;
};

class BackwardPassCreator {
public:
  BackwardPassCreator(Graph &fwdGraph_, Graph &bwdGraph_);

private:
  void growGradGraph();
  std::vector<Op *> growGradOps(Op *nonGradOp);
  bool opIsReadyToCreateGradients(Op *);
  void registerBwdOp(Op *fwdOp, Op *bwdOp);
  Op *growGradSumOp(Tensor *target, const std::vector<Tensor *> &partials);
  TensorId getGradId(const TensorId &);
  void populateGradInInfo(
      const std::map<TensorId, TensorId> &bwdInputIdToFwdTensorId);

  static void cloneGraph(const Graph &from, Graph &to);
  static void doPrune(Graph &);

  Graph &fwdGraph;
  Graph &bwdGraph;

  // A map of fwd tensors to their corresponding gradient tensors
  std::map<TensorId, TensorId> gradTensorMap;
  TensorGradMapRegister gradRegister;
};

Graph::Graph(Ir &ir_, const GraphId &id_)
    : id(id_), onnxToOnnx(std::make_unique<onnxpasses::Canonnxalizer>()),
      ir(ir_) {
  up_tensors.reset(new Tensors(*this));
  topoCons.reset(new TopoCons());
  scheduler.reset(new Scheduler());
}

void Graph::setOnnxToOnnx(
    std::unique_ptr<onnxpasses::IOnnxToOnnx> onnxToOnnx_) {
  onnxToOnnx = std::move(onnxToOnnx_);
}

const std::map<OpId, std::unique_ptr<Op>> &Graph::getOps() const { return ops; }
std::map<OpId, std::unique_ptr<Op>> &Graph::getOps() { return ops; }

const int64_t Graph::NoVGraph = -1;

const std::set<int64_t> Graph::getAllVirtualGraphIds() const {

  std::set<int64_t> vGraphIds;

  // Get the vgraph ids from all non-IpuCopyOps
  for (auto &id_op : getOps()) {
    auto op = id_op.second.get();
    if (!op->isConvertibleTo<IpuCopyOp>()) {
      vGraphIds.insert(getVirtualGraphId(*id_op.second));
    }
  }
  return vGraphIds;
}

const std::map<int64_t, int> Graph::getVirtualGraphCounts() const {
  std::map<int64_t, int> vGraphCounts;

  for (auto &idOp : getOps()) {
    int64_t vGraphId = getVirtualGraphId(*idOp.second);

    if (vGraphCounts.count(vGraphId) == 0) {
      vGraphCounts[vGraphId] = 0;
    }

    vGraphCounts[vGraphId]++;
  }

  return vGraphCounts;
}

Op *Graph::getOp(OpId opId) const {
  auto found = ops.find(opId);
  if (found == ops.end()) {
    throw error("No Op `" + std::to_string(opId) + "'");
  }
  return found->second.get();
}

const Tensors &Graph::getTensors() const { return *(up_tensors.get()); }
Tensors &Graph::getTensors() { return *(up_tensors.get()); }

void Graph::addInput(const InIndex &index,
                     const TensorId &tensorId,
                     const TensorInfo &tensorInfo,
                     bool overwrite) {
  getTensors().addActGrad(tensorId);
  auto tensor  = getTensors().get(tensorId);
  tensor->info = tensorInfo;
  if (overwrite) {
    if (graph_inputs.size() < index + 1) {
      graph_inputs.resize(index + 1);
    }
    graph_inputs.at(index) = tensorId;
  } else {
    graph_inputs.insert(graph_inputs.begin() + index, tensorId);
  }
}

void Graph::addInput(const TensorId &tensorId, const TensorInfo &tensorInfo) {
  getTensors().addActGrad(tensorId);
  auto tensor  = getTensors().get(tensorId);
  tensor->info = tensorInfo;
  graph_inputs.push_back(tensorId);
}

TensorId Graph::addInput(const TensorInfo &tinfo) {
  auto tensorId = logging::format("input_{}", graph_inputs.size());
  auto scopedId = addScope(tensorId);
  addInput(scopedId, tinfo);
  return scopedId;
}

bool Graph::hasInputId(const TensorId &id) const {
  return std::find(graph_inputs.begin(), graph_inputs.end(), id) !=
         graph_inputs.end();
}

void Graph::markAsInput(const TensorId &tensorId) {
  if (!getTensors().contains(tensorId)) {
    throw error("Could not find tensor '{}' to mark as input", tensorId);
  }
  graph_inputs.push_back(tensorId);
}

void Graph::removeInput(const TensorId &tensorId) {
  auto found = boost::range::find(graph_inputs, tensorId);
  if (found == graph_inputs.end()) {
    throw error("Could not find tensor '{}' in graph {} outputs", tensorId, id);
  }
  graph_inputs.erase(found);
}

void Graph::removeInput(const InIndex &index) {
  graph_inputs.erase(graph_inputs.begin() + index);
}

OutIndex Graph::getOutputIndex(TensorId tensorId) const {
  auto it = std::find(graph_outputs.begin(), graph_outputs.end(), tensorId);
  if (it == graph_outputs.end()) {
    throw error("Could not find output tensor '{}'", tensorId);
  }
  return std::distance(graph_outputs.begin(), it);
}

bool Graph::hasOutputId(const TensorId &id) const {
  return std::find(graph_outputs.begin(), graph_outputs.end(), id) !=
         graph_outputs.end();
}

void Graph::markAsOutput(const OutIndex &index,
                         const TensorId &tensorId,
                         bool overwrite) {
  if (!getTensors().contains(tensorId)) {
    throw error("Could not find tensor '{}' to mark as output", tensorId);
  }
  if (overwrite) {
    if (graph_outputs.size() < index + 1) {
      graph_outputs.resize(index + 1);
    }
    graph_outputs.at(index) = tensorId;
  } else {
    graph_outputs.insert(graph_outputs.begin() + index, tensorId);
  }
}

void Graph::markAsOutput(const TensorId &tensorId) {
  if (!getTensors().contains(tensorId)) {
    throw error("Could not find tensor '{}' to mark as output", tensorId);
  }
  graph_outputs.push_back(tensorId);
}

void Graph::removeOutput(const TensorId &tensorId) {
  auto found = boost::range::find(graph_outputs, tensorId);
  if (found == graph_outputs.end()) {
    throw error("Could not find tensor '{}' in graph {} outputs", tensorId, id);
  }
  graph_outputs.erase(found);
}

void Graph::removeOutput(const OutIndex &index) {
  graph_outputs.erase(graph_outputs.begin() + index);
}

std::vector<const Graph *> Graph::getCalledGraphs() const {
  std::vector<const Graph *> called;

  for (auto &id_op : getOps()) {
    auto op = id_op.second.get();
    for (auto graph : op->getCalledGraphs()) {
      called.push_back(graph);
    }
  }

  return called;
}

void Graph::constructFromOnnxGraph(
    const ONNX_NAMESPACE::GraphProto &onnx_graph) {

  auto g0 = onnxToOnnx->getCanonnxalized(onnx_graph);

  for (const auto &node : g0.node()) {
    if (OnnxConstExprUtil::isConst(node)) {
      OnnxConstExprUtil::processNode(node, this);
    } else {
      Op *op = growFromNode(node);
      // process ops as they are created
      // Reshape requires a const input tensor at creation time
      // if const folding is left till after the ir is completly constructed
      // then Reshape may not get a const input tensor at creation time
      if (ConstExprUtil::isComputable(op, *this)) {
        ConstExprUtil::processOp(op, *this);
      }
    }
  }
}

Op *Graph::growFromNode(const Node &node) {
  Op *op = OpManager::createOpInGraph(node, *this);
  op->setup();
  return op;
}

Scope Graph::getScope() const { return Scope() / id.str(); }

TensorId Graph::addScope(const TensorId &tensorId) const {
  return (getScope() / tensorId).str();
}

TensorId Graph::removeScope(const TensorId &scopedId) const {
  using boost::algorithm::starts_with;

  auto scopeStr = getScope().str();
  if (!starts_with(scopedId, scopeStr)) {
    throw error(
        "Cannot remove scope from {} as it does not start with scope {}",
        scopedId,
        scopeStr);
  }
  return scopedId.substr(scopeStr.size() + 1);
}

Graph &Graph::getBackwardsGraph(const GraphId &bwdId) {
  if (ir.hasGraph(bwdId)) {
    return ir.getGraph(bwdId);
  } else {
    auto &bwdGraph = ir.createGraph(bwdId);
    BackwardPassCreator(*this, bwdGraph);
    return bwdGraph;
  }
}

void TensorGradMapRegister::insert(Tensor *nonGrad, Tensor *grad) {
  auto found = partial.find(nonGrad);
  if (found == partial.end()) {
    partial.insert({nonGrad, {grad}});
  } else {
    found->second.push_back(grad);
  }

  if (partial.at(nonGrad).size() == nonGrad->consumers.getTotal()) {
    complete.insert({nonGrad, partial.at(nonGrad)});
    partial.erase(nonGrad);
  }
}

GradTensorsPartsMap TensorGradMapRegister::popComplete() {
  auto toRet = complete;
  complete   = {};
  return toRet;
}

OpId Graph::moveIntoGraph(std::unique_ptr<Op> op) {
  // Op may be moved in from a different graph
  op->settings.graph = *this;

  OpId opid = op->id;
  ops[opid] = std::move(op);
  return opid;
}

void Graph::connectInputsFromInputMapWrapper(const InputMapWrapper &in,
                                             OpId opid) {
  connectInputs(in, opid);
}

void Graph::connectOutputsFromOutputMapWrapper(const OutputMapWrapper &out,
                                               OpId opid) {
  connectOutputs(out, opid);
}

void Graph::eraseOp(OpId opid) {
  auto found = ops.find(opid);
  if (found == ops.end()) {
    throw internal_error("no op {} to erase", std::to_string(opid));
  }
  // Clean up topo cons for removed op, because the caller can't be trusted
  // to clean this up properly, resulting in horrible accidents.
  topoCons->remove(found->second.get());
  ops.erase(opid);
}

// T12001
// Remove AddInplace, VarUpdate should be only modifier
void Graph::setVarUpdateConstraints() {
  // For every Op, for every input, is it input modified?
  for (const auto &id_up : getOps()) {
    auto proposalOp = id_up.second.get();
    for (auto inIndex_tensor : proposalOp->input->tensorMap()) {
      auto proposalIndex  = inIndex_tensor.first;
      auto proposalTensor = inIndex_tensor.second;

      auto regions = proposalOp->modifies(proposalIndex);
      if (std::any_of(regions.begin(),
                      regions.end(),
                      [](const view::Region &r) { return !r.isEmpty(); })) {

        // The input is modified.
        auto modifiedTensor = proposalTensor;
        auto modifier       = proposalOp;

        // Collect all tensors aliased to modifiedTensor, but not downstream of
        // the modifier. The consumers of these aliasing Ops will need
        // topological constraints.

        std::set<TensorId> excludes;
        // Visit any tensor downstream of the modifier
        graphutils::traverse(
            modifier->output->tensors(),
            [&excludes](Tensor *t) {
              excludes.insert(t->id);
              return true;
            },
            [](Op *, Tensor *, Tensor *) { return true; },
            graphutils::TraversalType::BreadthFirst,
            graphutils::VisitType::Pre,
            graphutils::TraversalDirection::Forward);

        auto OpCompare = [](Op *a, Op *b) { return a->id < b->id; };
        std::set<Op *, decltype(OpCompare)> befores(OpCompare);

        auto applyTopoCons = [&excludes, &befores, &modifier](Tensor *t) {
          if (excludes.find(t->id) != excludes.end()) {
            return false;
          }

          for (Op *consumer : t->consumers.getOps()) {

            // Accl Updater doesn't come before anything
            if (consumer->isConvertibleTo<SGD1AcclUpdateOp>()) {
              continue;
            }
            // Don't have consumer -> modifier if consumer is VarUpdater (we
            // need better aliasing and modifying analysis here to disable this,
            // because of the TightVarMerge)
            if (!modifier->isConvertibleTo<SGD1AcclUpdateOp>() &&
                consumer->isConvertibleTo<VarUpdateOp>()) {
              continue;
            }

            // Modifiers that don't force all consumers to occur before
            if (modifier->isConvertibleTo<RemoteLoadOp>() ||
                modifier->isConvertibleTo<RemoteExchangeOp>() ||
                modifier->isConvertibleTo<AccumulateOp>() ||
                modifier->isConvertibleTo<AccumulatorUpdateOp>()) {
              continue;
            }

            // Consumers that don't need to run before modifiers
            if (consumer->isConvertibleTo<RemoteLoadOp>() ||
                consumer->isConvertibleTo<RemoteExchangeOp>() ||
                consumer->isConvertibleTo<RemoteStoreOp>() ||
                consumer->isConvertibleTo<SliceInplaceOp>()) {
              continue;
            }

            if (consumer == modifier) {
              continue;
            }

            if (consumer->getGraph().id != modifier->getGraph().id) {
              continue;
            }

            befores.emplace(consumer);
          }
          return true;
        };

        // For all consumers of aliasing modifiedTensor tensors, add the
        // topological constraint
        modifiedTensor->anyAlias(applyTopoCons);

        for (auto before : befores) {
          topoCons->insert(before, modifier);
        }
      }
    }
  }
}

// T12001 don't use topoCons
void Graph::setConvFlipWeightConstraints() {
  // The ConvFlipWeights op is used exclusively in the backwards pass as an
  // input to the bwd conv or multiconv op. Since it acts only on an input
  // to the graph, it has no dependencies. Constrain it to schedule after all
  // other ops producing tensors consumed by the bwd conv.
  for (auto &id_op : getOps()) {
    auto op = id_op.second.get();
    if (op->isConvertibleTo<ConvFlipWeightsOp>()) {
      for (Tensor *wT : op->output->tensors()) {
        if (wT->consumers.getTotal() == 1) {
          Op *bwConv = wT->consumers.getOps().at(0);
          for (Tensor *consumedByBwdConvT : bwConv->input->tensors()) {
            if (consumedByBwdConvT->id == wT->id) {
              continue;
            } else {
              // Apply constraint: All other ops producing tensors
              // consumed by the bwd conv must happen before the
              // flipweights
              // Note: don't insert dependecies on other ConvFlipWeights ops
              // that produce inputs to the MultiConvOp, so as not to create
              // a cycle in the graph.
              Op *producerToBwdConvOp = consumedByBwdConvT->getProducer();
              if (!producerToBwdConvOp->isConvertibleTo<ConvFlipWeightsOp>()) {
                topoCons->insert(producerToBwdConvOp, op);
              }
            }
          }
        } else {
          // Multiple (i.e. unexpected number of) consumers of flipweights
          // op. Do not apply constraints, so might schedule of these ops
          // might not be optimized for liveness
          logging::ir::warn(
              "ConvFlipWeightsOp, {}, has an unexpected number of consumers. "
              "Not constraining its schedule. This may result in a schedule "
              "not optimized for minimum max-liveness.",
              op->str());
        }
      }
    }
  }
}

std::vector<Op *> Graph::getOpSchedule(
    const OpsBeforeKey &gCons,
    const RequireOptimalSchedule requireOptimalSchedule) const {
  POPART_TRACEPOINT();
  const auto respectExecutionPhases = ir.getExecutionPhasesReady();
  const auto swapLimit   = getIr().getSessionOptions().swapLimitScheduler;
  const std::string &ktb = getIr().getSessionOptions().kahnTieBreaker;
  const auto timeLimit   = getIr().getSessionOptions().timeLimitScheduler;

  return scheduler->getSchedule(gCons,
                                *this,
                                requireOptimalSchedule,
                                respectExecutionPhases,
                                timeLimit,
                                swapLimit,
                                ktb);
}

void Graph::freezeSchedule(const OpsBeforeKey &gCons) {
  auto schedule = getOpSchedule(gCons, RequireOptimalSchedule::Yes);
  for (size_t i = 1; i < schedule.size(); ++i) {
    topoCons->insert(schedule.at(i - 1), schedule.at(i), false);
  }
}

// Are the Ops with all the dependencies a DAG?
bool Graph::isSchedulable(const OpsBeforeKey &gCons,
                          bool respectExecutionPhases) const {
  return scheduler->isSchedulable(gCons, *this, respectExecutionPhases);
}

bool Graph::hasUserRecomputeOps() const {
  for (auto &id_op : getOps()) {
    if (id_op.second.get()->settings.recomputeType ==
        RecomputeType::Recompute) {
      return true;
    }
  }
  return false;
}

std::vector<std::set<Op *>>
Graph::getLiveSets(const std::vector<Op *> &topoOps) const {

  // the key op waits for the ops in val
  // so the key op is later in the sort.
  std::map<Op *, std::vector<Op *>, POpCmp> waiting;

  // the number of ops that are waiting for key
  // this is NOT the size of the values of is_waiting_for
  std::map<Op *, int, POpCmp> nWaiting;

  for (Op *op : topoOps) {
    nWaiting[op] = 0;
    waiting[op]  = {};
  }
  for (Op *op : topoOps) {
    for (auto t_inds : op->input->indicesMap()) {
      Tensor *tensor = t_inds.first;
      if (tensor->hasProducer()) {
        Op *prod = tensor->getProducer();
        // have we noted that op is waiting for prod yet? if not,
        if (std::find(waiting[op].begin(), waiting[op].end(), prod) ==
            waiting[op].end()) {
          // make note
          waiting[op].push_back(prod);
          // increase the number of ops waiting for prod
          ++nWaiting[prod];
        }
      }
    }
  }

  std::set<Op *> live = {};
  std::vector<std::set<Op *>> liveSets;
  for (Op *newOp : topoOps) {
    for (Op *isEarlier : waiting[newOp]) {
      if (live.count(isEarlier) == 0) {
        throw internal_error(
            "Op {} should still be live (newOp waits for its output)",
            isEarlier->str());
      }
      --nWaiting[isEarlier];
      if (nWaiting[isEarlier] == 0) {
        live.erase(isEarlier);
      }
    }
    live.insert(newOp);
    liveSets.push_back(live);
  }
  return liveSets;
}

int64_t Graph::getVirtualGraphId(const Op &op) {
  if (op.hasVirtualGraphId()) {
    return op.getVirtualGraphId();
  } else {
    return NoVGraph;
  }
}

BackwardPassCreator::BackwardPassCreator(Graph &fwdGraph_, Graph &bwdGraph_)
    : fwdGraph(fwdGraph_), bwdGraph(bwdGraph_) {
  growGradGraph();

  // create map of bwdGraph input TensorIds to fwdGraph input/gradient TensorIds
  // must be done before pruning
  std::map<TensorId, TensorId> bwdInputIdToFwdTensorId;
  for (int i = 0; i < fwdGraph.getInputIds().size(); i++) {
    auto fwdIn = fwdGraph.getInputId(i);
    auto bwdIn = bwdGraph.getInputId(i);
    bwdInputIdToFwdTensorId.insert({bwdIn, fwdIn});
  }
  for (auto &fwdOut : fwdGraph.getOutputIds()) {
    auto gradId = getGradId(fwdOut);
    bwdInputIdToFwdTensorId.insert({gradId, fwdOut});
  }

  doPrune(bwdGraph);

  populateGradInInfo(bwdInputIdToFwdTensorId);
}

void BackwardPassCreator::populateGradInInfo(
    const std::map<TensorId, TensorId> &bwdInputIdToFwdTensorId) {
  // Populate bwdGraph.gradInInfo
  using boost::range::find;
  std::map<TensorId, GradInOutMapper> partialGradInfo;
  for (int i = 0; i < fwdGraph.getInputIds().size(); i++) {
    auto id = fwdGraph.getInputId(i);
    partialGradInfo.insert({id, {-1, i, GradOpInType::In}});
  }
  for (int i = 0; i < fwdGraph.getOutputIds().size(); i++) {
    auto id = fwdGraph.getOutputId(i);
    partialGradInfo.insert({id, {-1, i, GradOpInType::GradOut}});
  }

  auto bwdInputIds = bwdGraph.getInputIds();
  for (int bIdx = 0; bIdx < bwdInputIds.size(); bIdx++) {
    auto bwdId       = bwdInputIds.at(bIdx);
    auto fwdTensorId = bwdInputIdToFwdTensorId.at(bwdId);
    auto found       = partialGradInfo.find(fwdTensorId);
    if (found != partialGradInfo.end()) {
      auto gradInOutMapper  = found->second;
      gradInOutMapper.iGrad = bIdx;
      bwdGraph.gradInInfo.push_back(gradInOutMapper);
    } else {
      throw error(
          "Could not find corresponding input tensor for graph input {}",
          bwdId);
    }
  }
}

void BackwardPassCreator::growGradGraph() {
  // clone ops from the fwdGraph into the bwdGraph
  cloneGraph(fwdGraph, bwdGraph);
  // cloned outputs are not required
  for (auto &id : bwdGraph.getOutputIds()) {
    bwdGraph.removeOutput(id);
  }

  // Create an input tensor for each output tensor of fwdGraph
  for (auto &scopedId : fwdGraph.getOutputIds()) {
    auto gradId   = getGradId(scopedId);
    auto gradInfo = fwdGraph.getTensors().get(scopedId)->info;
    bwdGraph.addInput(gradId, gradInfo);
    gradTensorMap.insert({scopedId, gradId});
  }

  // Add all ops in the fwdGraph to pending ops
  std::set<Op *> pendingOps;
  for (auto &id_op : fwdGraph.getOps()) {
    auto op = id_op.second.get();
    pendingOps.insert(op);
  }

  while (!pendingOps.empty()) {
    // get all the ops that are ready to grow grad ops
    std::vector<Op *> readyOps;
    for (auto op : pendingOps) {
      if (opIsReadyToCreateGradients(op)) {
        readyOps.push_back(op);
      }
    }
    // remove ready ops from pending
    for (auto op : readyOps) {
      pendingOps.erase(pendingOps.find(op));
    }
    // grow grad ops for op
    for (auto fwdOp : readyOps) {
      auto bwdOps = growGradOps(fwdOp);

      for (auto bwdOp : bwdOps) {
        registerBwdOp(fwdOp, bwdOp);
      }
    }
  }

  // connect up outputs
  for (auto &scopedId : fwdGraph.getInputIds()) {
    if (gradTensorMap.find(scopedId) == gradTensorMap.end()) {
      throw error("Could not find tensor {} in gradTensorMap", scopedId);
    }
    auto gradId = getGradId(scopedId);
    bwdGraph.markAsOutput(gradId);
  }
}

void BackwardPassCreator::cloneGraph(const Graph &from, Graph &to) {
  // clone all the ops
  std::map<Op *, Op *> cloneMap;
  for (auto &id_op : from.getOps()) {
    auto op                 = id_op.second.get();
    auto clone              = op->clone();
    clone->toLoss           = op->toLoss;
    clone->fromLoss         = op->fromLoss;
    clone->scheduledPreLoss = op->scheduledPreLoss;
    clone->settings.graph   = to;
    clone->settings.scope   = to.getScope();
    auto cloneId            = to.moveIntoGraph(std::move(clone));
    cloneMap.insert({op, to.getOp(cloneId)});
  }

  // clone all the tensors
  std::map<Tensor *, Tensor *> tensorMap;
  for (auto &id : from.getTensors().getAllTensorIds()) {
    auto tensor = from.getTensors().get(id);

    auto newId = to.addScope(from.removeScope(id));

    auto tensorClone = tensor->clone(to);
    tensorClone->id  = newId;
    to.getTensors().moveIntoTensors(std::move(tensorClone));
    auto tensorClonePtr = to.getTensors().get(newId);
    tensorMap.insert({tensor, tensorClonePtr});
  }

  // hook up op inputs and outputs
  for (auto &id_op : from.getOps()) {
    auto op    = id_op.second.get();
    auto clone = cloneMap.at(op);

    // connect inputs
    for (auto &idx_tensor : op->input->tensorMap()) {
      auto idx             = idx_tensor.first;
      auto tensor          = idx_tensor.second;
      auto clone_tensor_id = tensorMap.at(tensor)->id;
      clone->connectInTensor(idx, clone_tensor_id);
    }

    // connect outputs
    for (auto &idx_tensor : op->output->tensorMap()) {
      auto idx             = idx_tensor.first;
      auto tensor          = idx_tensor.second;
      auto clone_tensor_id = tensorMap.at(tensor)->id;
      clone->connectOutTensor(idx, clone_tensor_id);
    }
  }

  // add graph inputs and outputs
  for (auto &id : from.getInputIds()) {
    auto unscopedId = from.removeScope(id);
    auto newId      = to.addScope(unscopedId);
    to.markAsInput(newId);
  }

  for (auto &id : from.getOutputIds()) {
    auto unscopedId = from.removeScope(id);
    auto newId      = to.addScope(unscopedId);
    to.markAsOutput(newId);
  }
}

void BackwardPassCreator::registerBwdOp(Op *fwdOp, Op *bwdOp) {
  for (auto &idx_tensor : bwdOp->output->tensorMap()) {
    auto bwdOutIndex = idx_tensor.first;
    auto bwdTensor   = idx_tensor.second;
    auto fwdInIndex  = bwdOp->getNonGradInIndex(bwdOutIndex);
    auto fwdTensor   = fwdOp->inTensor(fwdInIndex);
    gradRegister.insert(fwdTensor, bwdTensor);
  }

  for (auto &fwdTensor_partials : gradRegister.popComplete()) {
    auto fwdTensor = fwdTensor_partials.first;
    auto &partials = fwdTensor_partials.second;
    auto sumOp     = growGradSumOp(fwdTensor, partials);
    gradTensorMap.insert({fwdTensor->id, sumOp->outId(0)});
  }
}

std::vector<OpId> Graph::getOpIds() const {
  const auto &opMap = getOps();
  std::vector<OpId> opIds;
  opIds.reserve(opMap.size());
  for (const auto &x : opMap) {
    opIds.push_back(x.first);
  }
  return opIds;
}

Op *BackwardPassCreator::growGradSumOp(Tensor *nonGradTensor,
                                       const std::vector<Tensor *> &partials) {
  std::unique_ptr<popart::Op> gradSum = OpManager::createOp(
      Domain::ai_onnx,
      "Sum",
      bwdGraph.getIr().getOpSetVersionFromModel(Domain::ai_onnx),
      bwdGraph,
      "GradSum");

  OpId opId = bwdGraph.moveIntoGraph(std::move(gradSum));

  std::vector<TensorId> inputs;
  inputs.reserve(partials.size());
  for (auto &tensor : partials) {
    inputs.push_back(tensor->id);
  }

  auto gradId = getGradId(nonGradTensor->id);

  std::vector<TensorId> outputs{gradId};

  bwdGraph.connectInputs(InputVecWrapper(inputs), opId);
  bwdGraph.connectOutputs(OutputVecWrapper(outputs), opId);
  Op *op = bwdGraph.getOps()[opId].get();
  op->setup();
  return op;
}

TensorId BackwardPassCreator::getGradId(const TensorId &id) {
  auto x = fwdGraph.removeScope(id);
  x      = popart::getGradId(x);
  return bwdGraph.addScope(x);
}

bool BackwardPassCreator::opIsReadyToCreateGradients(Op *op) {
  for (auto output : op->output->tensors()) {
    if (gradTensorMap.find(output->id) == gradTensorMap.end()) {
      return false;
    }
  }

  return true;
}

std::vector<Op *> BackwardPassCreator::growGradOps(Op *nonGradOp) {
  auto nonGradOpId = nonGradOp->id;
  auto bwdOps      = nonGradOp->getGradOps();
  if (bwdOps.empty()) {
    throw error("Cannot get gradients for {}", nonGradOp->debugName());
  }

  std::vector<Op *> result;

  for (auto &uPtrOp : bwdOps) {
    Op *gradOp    = uPtrOp.get();
    OpId gradOpId = bwdGraph.moveIntoGraph(std::move(uPtrOp));

    // Reset priority, since fwd priority should not influence bwd priority
    //
    // TODO: Uncomment this. This prevented explicit priorities on certain
    // gradient ops being set which was necessary as a short term fix for
    // sharded training regressions seen in T17036. This could be replaced
    // once explicit priorities are no longer needed for this purpose. T17311
    // should fix this.
    //
    // gradOp->settings.schedulePriority = 0.0;

    gradOp->setScope(bwdGraph.getScope());

    if (nonGradOp->settings.recomputeType == RecomputeType::Recompute &&
        bwdGraph.getIr().autoRecomputationEnabled() &&
        bwdGraph.getIr().getSessionOptions().executionPhaseSettings.phases <
            2) {
      throw error("Grad Ops should be grown before recompute annotation");
    }

    // connect inputs of gradOp
    {
      // inputs to gradOp (to populate in this scope):
      std::map<int, std::string> m_inputs;
      for (auto &inOutMapper : gradOp->gradInputInfo()) {

        int indexGrad     = inOutMapper.iGrad;
        int indexFwd      = inOutMapper.iNonGrad;
        GradOpInType type = inOutMapper.type;

        // the input at index 'indexGrad' to gradOp is
        switch (type) {
        //  (1) the INPUT at index 'indexFwd' of nonGradOp
        //  This will be a tensor internal to fwdGraph
        case GradOpInType::In: {
          auto fwdId          = nonGradOp->inId(indexFwd);
          auto bwdId          = bwdGraph.addScope(fwdGraph.removeScope(fwdId));
          m_inputs[indexGrad] = bwdId;
          break;
        }

        //  (2) the OUTPUT at index 'indexFwd' of nonGradOp
        //  This will be a tensor internal to fwdGraph
        case GradOpInType::Out: {
          auto fwdId          = nonGradOp->outId(indexFwd);
          auto bwdId          = bwdGraph.addScope(fwdGraph.removeScope(fwdId));
          m_inputs[indexGrad] = bwdId;
          break;
        }

        //  (3) the GRADIENT of the OUTPUT
        //      at index 'indexFwd' of nonGradOp.
        case GradOpInType::GradOut: {
          auto fwdId = nonGradOp->outId(indexFwd);
          auto found = gradTensorMap.find(fwdId);
          if (found == gradTensorMap.end()) {
            throw error("Could not find TensorId '{}' in gradTensorMap", fwdId);
          }
          m_inputs[indexGrad] = found->second;
          break;
        }
        }
      }

      bwdGraph.connectInputs(InputMapWrapper(m_inputs), gradOpId);
    }

    // connect outputs of gradOp
    {
      std::vector<TensorId> v_outputs;
      for (auto out_in : gradOp->gradOutToNonGradIn()) {
        int gradOut   = out_in.first;
        int nonGradIn = out_in.second;

        if (!nonGradOp->input->tensor(nonGradIn)) {
          throw error("Invalid configuration of gradOp {}. nonGradOp ({}) "
                      "OUTPUT {} is not defined ",
                      gradOp->debugName(),
                      nonGradOp->debugName(),
                      nonGradIn);
        }

        TensorId inId  = nonGradOp->inId(nonGradIn);
        TensorId outId = getEdgeGradId(inId, nonGradOpId, nonGradIn);
        if (v_outputs.size() < gradOut + 1) {
          v_outputs.resize(gradOut + 1, "");
        }
        v_outputs[gradOut] = outId;
      }
      bwdGraph.connectOutputs(OutputVecWrapper(v_outputs), gradOpId);
    }
    gradOp->setup();

    result.push_back(gradOp);
  }

  return result;
}

void BackwardPassCreator::doPrune(Graph &graph) {
  using boost::range::find;

  const auto outputIds = graph.getOutputIds();
  auto isNotOutput     = [&outputIds](const TensorId &tId) {
    return find(outputIds, tId) == outputIds.end();
  };

  while (true) {
    // set to true if a tensor or op is removed
    bool continueLoop = false;

    auto &tensors = graph.getTensors();

    // Remove tensors that are not inputs or outputs and have no consumers
    const auto tensorIds = tensors.getAllTensorIds();
    for (auto id : tensorIds) {
      auto tensor = graph.getTensors().get(id);
      if (tensor->consumers.getTotal() == 0 && isNotOutput(id)) {
        if (tensor->hasProducer()) {
          auto producer = tensor->getProducer();
          producer->disconnectOutTensor(tensor);
        }
        tensors.remove(id);
        continueLoop = true;
      }
    }

    // Remove ops with no outputs
    const auto opIds = graph.getOpIds();
    for (const auto id : opIds) {
      const auto op = graph.getOp(id);
      if (op->output->n() == 0) {
        op->disconnectAllInputs();
        graph.eraseOp(id);
        continueLoop = true;
      }
    }

    if (!continueLoop) {
      break;
    }
  }

  // Remove inputs ids that have been pruned
  auto inputIds = graph.getInputIds();
  for (auto &id : inputIds) {
    if (!graph.getTensors().contains(id)) {
      auto inputIter = find(graph.getInputIds(), id);
      graph.graph_inputs.erase(inputIter);
    }
  }
}

void Graph::replaceTensor(const TensorId &oldId, const TensorId &newId) {
  Tensor *oldt = getTensors().get(oldId);
  Tensor *newt = getTensors().get(newId);

  for (Op *c : oldt->consumers.getOps()) {
    auto indices = c->input->indices(oldt);
    c->disconnectInTensor(oldt);
    for (auto index : indices) {
      c->connectInTensor(index, newt->id);
    }
    c->setup();
  }

  for (size_t i = 0; i < graph_outputs.size(); ++i) {
    if (graph_outputs.at(i) == oldId) {
      graph_outputs.at(i) = newId;
    }
  }
}

std::vector<Op *> Graph::getCallSiteOps() const {
  auto ops = ir.getAllOps();
  std::vector<Op *> callSites;
  for (Op *op : ops) {
    for (const Graph *calledGraph : op->getCalledGraphs()) {
      if (calledGraph->id.str() == id.str()) {
        callSites.push_back(op);
      }
    }
  }
  return callSites;
}

std::vector<Op *> Graph::getCallSiteOps(size_t num) const {
  std::vector<Op *> ops_;

  std::set<const Graph *> visited;

  // Depth first search for call sites
  std::vector<Op *> opStack;

  // Start at first op of main graph
  auto schedule =
      ir.getMainGraph().getOpSchedule({}, RequireOptimalSchedule::Yes);
  opStack.insert(opStack.end(), schedule.rbegin(), schedule.rend());

  while (!opStack.empty()) {
    Op *op = opStack.back();
    opStack.pop_back();
    for (const Graph *calledGraph : op->getCalledGraphs()) {
      if (calledGraph->id.str() == id.str()) {
        ops_.push_back(op);
        if (num > 0 && ops_.size() == num) {
          return ops_;
        }
      } else if (visited.find(calledGraph) == visited.end()) {
        schedule = calledGraph->getOpSchedule({}, RequireOptimalSchedule::Yes);
        opStack.insert(opStack.end(), schedule.rbegin(), schedule.rend());
        visited.insert(calledGraph);
      }
    }
  }

  return ops_;
}

std::map<OpId, std::unordered_set<OpId>> Graph::getEdgeMap() const {
  const auto &opid_op_map = this->getOps();

  const auto getConsumerOpIds = [this](const auto &opid_op) {
    return getConsumerOpIdsInGraph(this, opid_op);
  };

  std::map<OpId, std::unordered_set<OpId>> edges;

  std::transform(opid_op_map.cbegin(),
                 opid_op_map.cend(),
                 std::inserter(edges, edges.end()),
                 getConsumerOpIds);

  return edges;
}

std::string Graph::getGraphString() const {
  std::string graphStr = id.str() == ""
                             ? "the main graph"
                             : std::string("subgraph '") + id.str() + "'";
  return graphStr;
}

} // namespace popart

namespace {
using namespace popart;

std::pair<OpId, std::unordered_set<OpId>> getConsumerOpIdsInGraph(
    const Graph *graph,
    const std::pair<const OpId, std::unique_ptr<Op>> &opid_op) {
  const auto opid = opid_op.first;
  auto *op        = opid_op.second.get();

  std::unordered_set<OpId> consumers;

  const auto addAllIdsToConsumers = [&consumers](const auto &ops) -> void {
    std::transform(ops.cbegin(),
                   ops.cend(),
                   std::inserter(consumers, consumers.end()),
                   [](Op *const &op) { return op->id; });
  };

  // 1. Add all topoCons consumers.

  const auto topoConsConsumerOps = graph->topoCons->getAfters(op);

  addAllIdsToConsumers(topoConsConsumerOps);

  // 2. Add all graph consumers.

  for (const auto *t_out : op->output->tensors()) {
    const auto graphConsumerOps = t_out->consumers.getOps();

    addAllIdsToConsumers(graphConsumerOps);
  }

  return {opid, consumers};
}

} // namespace
