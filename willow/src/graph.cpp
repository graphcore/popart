#include <onnx/onnx_pb.h>

#include <poponnx/ces/constexpr.hpp>
#include <poponnx/ces/onnxconstexpr.hpp>
#include <poponnx/graph.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/pbwrap.hpp>
#include <poponnx/scheduler.hpp>
#include <poponnx/tensors.hpp>
#include <poponnx/topocons.hpp>

// Ops required for Graph::getCalledOps
#include <poponnx/op/call.hpp>
#include <poponnx/op/if.hpp>

// The layers required to construct the backwards pass
#include <poponnx/op/conv.hpp>
#include <poponnx/op/flatten.hpp>
#include <poponnx/op/varupdate.hpp>

namespace poponnx {

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
  auto tenid    = fmt::format("input_{}", graph_inputs.size());
  auto scopedid = (Scope() / id.str() / tenid).str();
  addInput(scopedid, tinfo);
  return scopedid;
}

void Graph::addOutput(const TensorId &tensorId) {
  graph_outputs.push_back(tensorId);
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

void Graph::constructFromOnnxGraph(const onnx::GraphProto &onnx_graph,
                                   const Scope &scope) {
  for (const auto &node : onnx_graph.node()) {
    if (OnnxConstExprUtil::isConst(node)) {
      OnnxConstExprUtil::processNode(node, this);
    } else {
      Op *op = growFromNode(node, scope);

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

Op *Graph::growFromNode(const Node &node, const Scope &scope) {

  OpId opId = moveIntoGraph(addOp(node, scope));

  connectInputs(node, opId);
  connectOutputs(node, opId);

  // finally, set the output tensor info for the output
  // tensors, and any other Op specific class variables
  Op *fromNodeOp = ops.at(opId).get();
  fromNodeOp->setup();
  return fromNodeOp;
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

std::unique_ptr<Op> Graph::addOp(const Node &node, const Scope &scope) {

  int version = ir.getOpSetVersionFromModel(node.domain());

  std::unique_ptr<Op> p = OpManager::createOp(node.domain(),
                                              node.op_type(),
                                              version,
                                              *this,
                                              node.name(),
                                              scope,
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
  for (auto &varId : getTensors().getIds(TensorType::Variable)) {

    Tensor *var = getTensors().get(varId);

    // First, determine which consumer is the updater,
    // or a flatten-inplace if merged var updating.
    std::vector<Op *> varInplaceConsumers;
    for (Op *consumer : var->consumers.getOps()) {
      if (dynamic_cast<VarUpdateOp *>(consumer) != nullptr ||
          dynamic_cast<FlattenInplaceOp *>(consumer) != nullptr) {
        varInplaceConsumers.push_back(consumer);
        break;
      }
    }
    if (varInplaceConsumers.size() == 0) {
      throw error("Failed to find any updaters of {}, bailing", var->str());
    } else if (varInplaceConsumers.size() > 1) {
      throw error("Found more than 1 potential updater of {}, bailing",
                  var->str());
    }
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
    Op *op = id_op.second.get();
    if (op->getRecomputeOutput()) {
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
            "ILE: op should still be live (newOp waits for its output)");
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

} // namespace poponnx
