#include <boost/algorithm/string.hpp>
#include <boost/range/algorithm.hpp>
#include <onnx/onnx_pb.h>

#include <popart/ces/constexpr.hpp>
#include <popart/ces/onnxconstexpr.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/opmanager.hpp>
#include <popart/pbwrap.hpp>
#include <popart/scheduler.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

// Ops required for Graph::getCalledOps
#include <popart/op/call.hpp>
#include <popart/op/if.hpp>

// The layers required to construct the backwards pass
#include <popart/op/conv.hpp>
#include <popart/op/flatten.hpp>
#include <popart/op/varupdate.hpp>

namespace popart {

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

Graph::Graph(Ir &ir_, const GraphId &id_) : id(id_), ir(ir_) {
  up_tensors.reset(new Tensors(*this));
  topoCons.reset(new TopoCons());
  scheduler.reset(new Scheduler());
}

const std::map<OpId, std::unique_ptr<Op>> &Graph::getOps() const { return ops; }
std::map<OpId, std::unique_ptr<Op>> &Graph::getOps() { return ops; }

Op *Graph::getOp(OpId opId) {
  auto found = ops.find(opId);
  if (found == ops.end()) {
    throw error("No Op `" + std::to_string(opId) + "'");
  }
  return found->second.get();
}

const Tensors &Graph::getTensors() const { return *(up_tensors.get()); }
Tensors &Graph::getTensors() { return *(up_tensors.get()); }

void Graph::addInput(const TensorId &tensorId, const TensorInfo &tensorInfo) {
  getTensors().addActGrad(tensorId);
  auto tensor  = getTensors().get(tensorId);
  tensor->info = tensorInfo;
  graph_inputs.push_back(tensorId);
}

TensorId Graph::addInput(const TensorInfo &tinfo) {
  auto tensorId = fmt::format("input_{}", graph_inputs.size());
  auto scopedId = addScope(tensorId);
  addInput(scopedId, tinfo);
  return scopedId;
}

void Graph::markAsInput(const TensorId &tensorId) {
  if (!getTensors().contains(tensorId)) {
    throw error("Could not find tensor '{}' to mark as input");
  }
  graph_inputs.push_back(tensorId);
}

void Graph::markAsOutput(const TensorId &tensorId) {
  if (!getTensors().contains(tensorId)) {
    throw error("Could not find tensor '{}' to mark as output");
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

void Graph::constructFromOnnxGraph(const onnx::GraphProto &onnx_graph) {
  for (const auto &node : onnx_graph.node()) {
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

  OpId opId = moveIntoGraph(addOp(node));

  connectInputs(node, opId);
  connectOutputs(node, opId);

  // finally, set the output tensor info for the output
  // tensors, and any other Op specific class variables
  Op *fromNodeOp = ops.at(opId).get();
  fromNodeOp->setup();
  return fromNodeOp;
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

std::unique_ptr<Op> Graph::addOp(const Node &node) {

  int version = ir.getOpSetVersionFromModel(node.domain());

  std::unique_ptr<Op> p = OpManager::createOp(node.domain(),
                                              node.op_type(),
                                              version,
                                              *this,
                                              node.name(),
                                              getScope(),
                                              node.attribute());
  if (p != nullptr)
    return p;
  else {
    if (node.op_type() == Onnx::AiOnnx::OpSet9::Constant.type) {
      throw error("ILE. Constant Ops are not to be added");
    } else {
      throw error("No class for {}.{}:{}",
                  (node.domain() == "" ? Domain::ai_onnx : node.domain()),
                  node.op_type(),
                  version);
    }
  }
}

void Graph::eraseOp(OpId opid) {
  auto found = ops.find(opid);
  if (found == ops.end()) {
    throw error("ILE: no op " + std::to_string(opid) + " to erase");
  }

  ops.erase(opid);
}

void Graph::setVarUpdateConstraints() {
  // impose the constraint that inplace consumers
  // are the last consumers of the vars

  for (auto var : getTensors().getOfType(TensorType::Variable)) {

    // First, determine which consumer is the updater,
    // or a flatten-inplace if merged var updating.
    std::vector<Op *> varInplaceConsumers;
    for (Op *consumer : var->consumers.getOps()) {
      if (dynamic_cast<VarUpdateOp *>(consumer) ||
          dynamic_cast<FlattenInplaceOp *>(consumer)) {
        varInplaceConsumers.push_back(consumer);
        break;
      }
    }
    if (varInplaceConsumers.size() == 0) {
      logging::debug("Failed to find any updaters of {}", var->str());
    } else if (varInplaceConsumers.size() > 1) {
      throw error("Found more than 1 potential updater of {}, bailing",
                  var->str());
    } else {
      // Good, there is a unique consumer which is inplace
      Op *varInplaceConsumer = varInplaceConsumers.back();

      // Set the constraints
      for (Op *consumer : var->consumers.getOps()) {
        if (consumer != varInplaceConsumer) {
          topoCons->insert(consumer, varInplaceConsumer);
        }
      }
    }
  }
}

void Graph::setConvFlipWeightConstraints() {
  // The ConvFlipWeights op is used exclusively in the backwards pass as an
  // input to the bwd conv op. Since it acts only on an input to the graph,
  // it has no dependencies. Constrain it to schedule after all other ops
  // producing tensors consumed by the bwd conv.
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
              Op *producerToBwdConvOp = consumedByBwdConvT->getProducer();
              topoCons->insert(producerToBwdConvOp, op);
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

std::vector<Op *> Graph::getOpSchedule(const OpsBeforeKey &gCons) const {
  auto sorted = scheduler->getPartialOpSchedule(gCons, *this);
  if (sorted.size() != getOps().size()) {
    throw error("failure to sort topologically in getOpSchedule ({} != {})",
                sorted.size(),
                getOps().size());
  }
  return sorted;
}

// Are the Ops with all the dependencies a DAG?
bool Graph::isSchedulable(const OpsBeforeKey &gCons) const {
  auto sorted = scheduler->getPartialOpSchedule(gCons, *this);
  return sorted.size() == getOps().size();
}

bool Graph::hasUserRecomputeOps() const {
  for (auto &id_op : getOps()) {
    if (id_op.second.get()->settings.recomputeType ==
        RecomputeType::RECOMPUTE) {
      return true;
    }
  }
  return false;
}

std::vector<std::set<Op *>>
Graph::getLiveSets(const std::vector<Op *> &topoOps) const {

  // the key op waits for the ops in val
  // so the key op is later in the sort.
  std::map<Op *, std::vector<Op *>> waiting;

  // the number of ops that are waiting for key
  // this is NOT the size of the values of is_waiting_for
  std::map<Op *, int> nWaiting;

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
        throw error(
            "ILE: Op {} should still be live (newOp waits for its output)",
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
    partialGradInfo.insert({id, {-1, i, GradOpInType::IN}});
  }
  for (int i = 0; i < fwdGraph.getOutputIds().size(); i++) {
    auto id = fwdGraph.getOutputId(i);
    partialGradInfo.insert({id, {-1, i, GradOpInType::GRADOUT}});
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

    auto tensorClone = tensor->clone();
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

    gradOp->setScope(bwdGraph.getScope());

    if (nonGradOp->settings.recomputeType == RecomputeType::RECOMPUTE &&
        bwdGraph.getIr().autoRecomputationEnabled()) {
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
        case GradOpInType::IN: {
          auto fwdId          = nonGradOp->inId(indexFwd);
          auto bwdId          = bwdGraph.addScope(fwdGraph.removeScope(fwdId));
          m_inputs[indexGrad] = bwdId;
          break;
        }

        //  (2) the OUTPUT at index 'indexFwd' of nonGradOp
        //  This will be a tensor internal to fwdGraph
        case GradOpInType::OUT: {
          auto fwdId          = nonGradOp->outId(indexFwd);
          auto bwdId          = bwdGraph.addScope(fwdGraph.removeScope(fwdId));
          m_inputs[indexGrad] = bwdId;
          break;
        }

        //  (3) the GRADIENT of the OUTPUT
        //      at index 'indexFwd' of nonGradOp.
        case GradOpInType::GRADOUT: {
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

  auto outputIds   = graph.getOutputIds();
  auto isNotOutput = [&outputIds](const TensorId &tId) {
    return find(outputIds, tId) == outputIds.end();
  };

  auto removeTensor = [&graph](Tensor *tensor) {
    if (tensor->hasProducer()) {
      auto producer = tensor->getProducer();
      producer->disconnectOutTensor(tensor);
    }
    graph.getTensors().remove(tensor->id);
  };

  while (true) {
    // set to true if a tensor or op is removed
    bool continueLoop = false;

    // Remove tensors that are not inputs or outputs and have no consumers
    for (auto &id : graph.getTensors().getAllTensorIds()) {
      auto tensor = graph.getTensors().get(id);
      if (tensor->consumers.getTotal() == 0 && isNotOutput(id)) {
        removeTensor(tensor);
        continueLoop = true;
      }
    }

    // Remove ops with no outputs
    for (auto &id_op : graph.getOps()) {
      auto id = id_op.first;
      auto op = id_op.second.get();

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

} // namespace popart
