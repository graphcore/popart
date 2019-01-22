#include <onnx/onnx_pb.h>
#include <poponnx/attributes.hpp>
#include <poponnx/ces/addce.hpp>
#include <poponnx/ces/castce.hpp>
#include <poponnx/ces/constexpr.hpp>
#include <poponnx/ces/shapece.hpp>
#include <poponnx/ces/transposece.hpp>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensors.hpp>

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

  // TODO: Consider moving this into the op and register a constexprutil
  // function. See T5993

  std::string nodeInfoStr = '`' + node.name() + "' (" + node.op_type() + ')';
  logging::ir::debug("Processing Node " + nodeInfoStr + " in ConstExprUtil");

  if (node.op_type() == "Constant") {
    TensorId name = node.output(0);
    // We assume that a tensor coming from a Constant Node should
    // not have a gradient computed for it or be updated during training
    // We will implement a seperate tool to convert between
    // Constant Operator output and initializer in ONNX models (T6213)
    ir->getTensors().addConstInit(name, &node.attribute(0).t());
  } else if (node.op_type() == "Cast") {
    CastCe caster(node, ir);
    caster.insertOutput();
  } else if (node.op_type() == "Add") {
    ConstExprAdd adder(node, ir);
    adder.insertOutput();
  } else if (node.op_type() == "Shape") {
    ConstExprShape expr(node, ir);
    expr.insertOutput();
  } else if (node.op_type() == "Transpose") {
    ConstExprTranspose transposer(node, ir);
    transposer.insertOutput();
  } else {
    throw error("No ConstExpr implementation of {}. "
                "Consider what OpType::ADD does (creates a Const Tensor) "
                "if you would like to implement a ConstExpr",
                node.op_type());
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
          auto node = graph.node(consumer);
          if (M.at(out) == true &&
              !isNodeOutputAlwaysConstExpr(node.op_type(),
                                           getOutIndex(node, out))) {
            M[out] = false;
            activeFront.push_back(out);
          }
        }
      }
    }
  }
  return ConstExprClassifier(std::move(M));
}

int ConstExprUtil::getOutIndex(const onnx::NodeProto &node,
                               const TensorId &tensor) {
  for (int i = 0; i < node.output_size(); i++) {
    if (tensor == node.output(i)) {
      return i;
    }
  }

  throw error("tensor {} is not an output of node {}", tensor, node.op_type());
}

bool ConstExprUtil::isNodeOutputAlwaysConstExpr(const OpType &op_type,
                                                OutIndex) {
  if (op_type == "Shape") {
    return true;
  }
  // here : any Operator whose output at OutIndex index is ALWAYS
  // computable at compile time should be added here
  return false;
}

} // namespace poponnx
