#include <onnx/onnx_pb.h>
#include <poponnx/attributes.hpp>
#include <poponnx/ces/addce.hpp>
#include <poponnx/ces/castce.hpp>
#include <poponnx/ces/constexpr.hpp>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

void ConstExprOp::addConstInitTensor(const TensorId &id,
                                     const TensorInfo &info,
                                     const void *data) const {
  ir->getTensors().addConstInit(id, info, data);
}

Tensor *ConstExprOp::atInIndex(InIndex index) const {
  if (index >= node.input_size()) {
    throw error("Index {} too large on ConstExprOp::atInIndex", index);
  }
  TensorId id = node.input(index);
  if (!ir->getTensors().contains(id)) {
    throw error("No Tensor {} from ConstExprOp::atInIndex", id);
  }
  return ir->getTensors().get(id);
}

TensorId ConstExprOp::atOutIndex0() const { return node.output(0); }

ConstExprOp::ConstExprOp(const onnx::NodeProto &n, Ir *i)
    : node(n), ir(i), nAtts(node.attribute()) {}

bool ConstExprClassifier::isConstExprTensor(TensorId id) const {
  auto found = M.find(id);
  if (found == M.end()) {
    throw error("ILE: No Tensor " + id + " in ConstExprClassifier::M");
  }
  return found->second;
}

void ConstExprUtil::processNode(const onnx::NodeProto &node, Ir *ir) {

  // For now we need to look at the domain and set the version. We really should
  // get the version from the model
  unsigned version = 9;
  if (node.domain() == Domain::ai_graphcore) {
    version = 1;
  }

  OperatorIdentifier opid(node.domain(), node.op_type(), version);

  // TODO: Consider moving this into the op and register a constexprutil
  // function. See T5993

  if (opid == Onnx::Operators::Cast) {
    CastCe caster(node, ir);
    caster.insertOutput();
  } else if (opid == Onnx::Operators::Constant) {
    TensorId name = node.output(0);
    // We assume that a tensor coming from a Constant Node should
    // not have a gradient computed for it or be updated during training
    // We will implement a seperate tool to convert between
    // Constant Operator output and initializer in ONNX models (T6213)
    ir->getTensors().addConstInit(name, &node.attribute(0).t());

  } else if (opid == Onnx::Operators::Add) {
    ConstExprAdd adder(node, ir);
    adder.insertOutput();

  } else {
    throw error("No ConstExpr implementation of {}. "
                "Consider what OpType::ADD does (creates a Const Tensor) "
                "if you would like to implement a ConstExpr",
                opid);
  }
}

ConstExprClassifier
ConstExprUtil::getClassifier(const onnx::GraphProto &graph,
                             const std::vector<TensorId> &sourceTensors) {

  // build a rudimentary DAG from the onnxModel
  using NodeId = int;
  // use maps to connect Tensors <-> Nodes
  std::map<NodeId, std::vector<TensorId>> outputs;
  std::map<NodeId, std::vector<TensorId>> inputs;
  std::map<TensorId, std::set<NodeId>> consumers;
  std::map<TensorId, NodeId> producers;
  // populate the edge maps above
  for (NodeId nodeId = 0; nodeId < graph.node_size(); ++nodeId) {
    auto &node      = graph.node(nodeId);
    outputs[nodeId] = {};
    inputs[nodeId]  = {};
    for (auto o : node.output()) {
      outputs[nodeId].push_back(o);
      producers[o] = nodeId;
    }
    for (auto i : node.input()) {
      inputs[nodeId].push_back(i);
      if (consumers.find(i) == consumers.end()) {
        consumers[i] = {};
      }
      consumers[i].insert(nodeId);
    }
  }

  // we initialize all const-expr values to true, and then
  // forward traverse the graph from relevant inputs, setting
  // values to false as we discover they are not const-expr
  std::map<TensorId, bool> M;
  for (auto &tenId_nodeId : producers) {
    TensorId tenId = tenId_nodeId.first;
    M[tenId]       = true;
  }

  auto activeFront = sourceTensors;

  while (activeFront.size() > 0) {
    auto tenId = activeFront.back();
    activeFront.resize(activeFront.size() - 1);
    auto found = consumers.find(tenId);
    if (found != consumers.end()) {
      for (auto consumer : found->second) {
        for (auto out : outputs.at(consumer)) {
          if (M.at(out) == true) {
            M[out] = false;
            activeFront.push_back(out);
          }
        }
      }
    }
  }
  return ConstExprClassifier(std::move(M));
}

} // namespace poponnx
