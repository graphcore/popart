#include <onnx/onnx_pb.h>
#include <poponnx/attributes.hpp>
#include <poponnx/ces/addce.hpp>
#include <poponnx/ces/castce.hpp>
#include <poponnx/ces/constexpr.hpp>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/optypes.hpp>
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
  OpType opType = getOpTypes().get(node.op_type(), node.domain());
  switch (opType) {

  case OpType::CAST: {
    CastCe caster(node, ir);
    caster.insertOutput();
    break;
  }

  case OpType::CONSTANT: {
    TensorId name = node.output(0);
    // We assume that a tensor coming from a Constant Node should
    // not have a gradient computed for it or be updated during training
    // We may need to change this assumption for some ONNX Model exporters
    ir->getTensors().insertConstId(name);
    ir->getTensors().addInit(name, &node.attribute(0).t());
    break;
  }

  case OpType::ADD: {
    ConstExprAdd adder(node, ir);
    adder.insertOutput();
    break;
  }

  case OpType::AVERAGEPOOL:
  case OpType::BATCHNORM:
  case OpType::CONV:
  case OpType::COS:
  case OpType::COSH:
  case OpType::DIV:
  case OpType::EXP:
  case OpType::GEMM:
  case OpType::IDENTITY:
  case OpType::NEGATE:
  case OpType::RECIPROCAL:
  case OpType::SQRT:
  case OpType::SQUARE:
  case OpType::SOFTMAX:
  case OpType::MAXPOOL:
  case OpType::MUL:
  case OpType::PAD:
  case OpType::REDUCESUM:
  case OpType::RELU:
  case OpType::RESHAPE:
  case OpType::SIGMOID:
  case OpType::SIN:
  case OpType::SUBTRACT:
  case OpType::SUBSAMPLE:
  case OpType::SUM:
  case OpType::SQUEEZE:
  case OpType::TAN:
  case OpType::TANH:
  case OpType::MATMUL:
  case OpType::TRANSPOSE:
    throw error("No ConstExpr implementation of " + node.op_type() + ". " +
                "Consider what OpType::ADD does (creates a Const Tensor) " +
                "if you would like to implement a ConstExpr");

  case OpType::ADDARG0GRAD:
  case OpType::ADDARG1GRAD:
  case OpType::ADDBIASBIASGRAD:
  case OpType::ADDBIASDATAGRAD:
  case OpType::COSGRAD:
  case OpType::DIVARG0GRAD:
  case OpType::DIVARG1GRAD:
  case OpType::EXPGRAD:
  case OpType::RESHAPEGRAD:
  case OpType::SQUEEZEGRAD:
  case OpType::REDUCESUMGRAD:
  case OpType::RELUGRAD:
  case OpType::AVERAGEPOOLGRAD:
  case OpType::CONVDATAGRAD:
  case OpType::CONVWEIGHTSGRAD:
  case OpType::NEGATEGRAD:
  case OpType::IDENTITYGRAD:
  case OpType::NLLGRAD:
  case OpType::L1GRAD:
  case OpType::MAXPOOLGRAD:
  case OpType::MULARG0GRAD:
  case OpType::MULARG1GRAD:
  case OpType::RECIPROCALGRAD:
  case OpType::SIGMOIDGRAD:
  case OpType::SINGRAD:
  case OpType::SCALE:
  case OpType::SCALEGRAD:
  case OpType::SOFTMAXGRAD:
  case OpType::SGDVARUPDATE:
  case OpType::SQRTGRAD:
  case OpType::CONSTSGDVARUPDATE:
  case OpType::SUBTRACTARG0GRAD:
  case OpType::SUBTRACTARG1GRAD:
  case OpType::TANHGRAD:
  case OpType::SUBSAMPLEGRAD:
  case OpType::TRANSPOSEGRAD:
  case OpType::MATMULLHSGRAD:
  case OpType::MATMULRHSGRAD:
  case OpType::BATCHNORMGRAD:
    throw error("No ConstExpr implementations for grad Ops");

  case OpType::NLL:
  case OpType::L1:
    throw error("No ConstExpr implementations for loss Ops");

  case OpType::ADDBIAS:
  case OpType::RELUINPLACE:
  case OpType::SOFTMAXGRADDIRECT:
    throw error("No ConstExpr implementations for non-ONNX Ops");

  default: { throw error("No ConstExpr for " + node.op_type()); }
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
