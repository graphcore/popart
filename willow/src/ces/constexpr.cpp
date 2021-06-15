// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <onnx/onnx_pb.h>
#include <onnxutil.hpp>
#include <popart/attributes.hpp>
#include <popart/ces/castce.hpp>
#include <popart/ces/concatce.hpp>
#include <popart/ces/constexpr.hpp>
#include <popart/ces/elementwisece.hpp>
#include <popart/ces/floorce.hpp>
#include <popart/ces/gatherce.hpp>
#include <popart/ces/identityce.hpp>
#include <popart/ces/reshapece.hpp>
#include <popart/ces/scalece.hpp>
#include <popart/ces/slicece.hpp>
#include <popart/ces/squeezece.hpp>
#include <popart/ces/transposece.hpp>
#include <popart/ces/unsqueezece.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>

namespace popart {

ConstExprOp::ConstExprOp(Op *op_) : op(op_) {}

Tensor *ConstExprOp::inTensor(InIndex index) const {
  return op->inTensor(index);
}

const TensorInfo &ConstExprOp::inInfo(InIndex index) const {
  return op->inInfo(index);
}

const Shape &ConstExprOp::inShape(InIndex index) const {
  return op->inShape(index);
}

const TensorInfo &ConstExprOp::outInfo0() const { return op->outInfo(0); }

void ConstExprUtil::processOp(Op *op, Graph &graph) {
  logging::ces::debug(
      "Processing Op `{}` ({}) in ConstExprUtil", op->id, op->opid.type);
  auto constOp = ConstExprOpManager::createConstExprOp(op);

  auto data = constOp->compute();
  makeTensorConstInit(op->outTensor(0)->id, data.data(), graph);
  op->disconnectAllInputs();

  if (op->input->n() > 0 || op->output->n() > 0) {
    throw error("processed op still has connected tensors");
  }

  graph.eraseOp(op->id);
  op = nullptr;
}

const Op *ConstExprOp::getBaseOp() const { return this->op; }

void ConstExprUtil::foldConstants(Graph &graph) {
  // get ops that may be computable
  std::unordered_set<Op *> computable_ops;
  for (auto &id_op : graph.getOps()) {
    auto &op = id_op.second;
    if (isComputable(op.get(), graph)) {
      computable_ops.insert(op.get());
    }
  }

  // try to fold ops, and where successful,
  // add consumers of ops output to `computable_ops`
  while (!computable_ops.empty()) {
    auto op = *computable_ops.begin();
    computable_ops.erase(op);

    // get the id here as if processOp is successful,
    // the tensor will be replaced
    auto out_id = op->outTensor(0)->id;
    processOp(op, graph);
    auto out_tensor = graph.getTensors().get(out_id);
    for (auto consumer : out_tensor->consumers.getOps()) {
      if (isComputable(consumer, graph)) {
        computable_ops.insert(consumer);
      }
    }
  }
}

void ConstExprUtil::makeTensorConstInit(const TensorId name,
                                        const void *data,
                                        Graph &graph) {
  // disconnect producer
  auto current_tensor = graph.getTensors().get(name);
  auto producer       = current_tensor->getProducer();
  producer->disconnectOutTensor(current_tensor);

  graph.getTensors().makeConstInit(name, data);
}

bool ConstExprUtil::isComputable(Op *op, Graph &graph) {
  if (!ConstExprOpManager::hasConstExprOp(op)) {
    logging::ces::trace("No ConstExpr implementation of {}, returning from "
                        "constant folding early.",
                        op->opid.type);
    return false;
  }

  // An op is computable as a const expression if all the inputs are Const
  // tensors, and none of the outputs are anchors. This would also be true for
  // Variable tensors during inference, unless the user calls resetHostWeights.
  // Because of this, am choosing to ignore case of Variable tensors during
  // inference.
  auto inputs  = op->input->tensors();
  auto outputs = op->output->tensors();

  // Op is not computable if any of the outputs are anchors
  if (std::any_of(outputs.begin(), outputs.end(), [&graph](Tensor *t) {
        return graph.getIr().isAnchored(t->id);
      })) {
    return false;
  }

  return std::all_of(inputs.begin(), inputs.end(), [](Tensor *t) {
    return t->tensorType() == TensorType::Const;
  });
}

ConstExprOpManager::ConstExprOpManager() { registerConstOps(); }

ConstExprOpManager &ConstExprOpManager::getInstance() {
  static ConstExprOpManager instance;
  return instance;
}

void ConstExprOpManager::registerConstExprOpImpl(const std::string &type,
                                                 ConstExprOpFactoryFunc func) {
  constExprOpMap.insert(std::make_pair(type, func));
}

void ConstExprOpManager::registerConstExprOp(const std::string &type,
                                             ConstExprOpFactoryFunc func) {
  getInstance().registerConstExprOpImpl(type, func);
}

template <typename T>
void ConstExprOpManager::registerConstOp(const std::string &type) {
  registerConstExprOpImpl(type, [](Op *op) -> std::unique_ptr<ConstExprOp> {
    return std::make_unique<T>(op);
  });
}

// TODO: T17818 Const ops expr should be able to be created from their inplace
// variants.
void ConstExprOpManager::registerConstOps() {
  registerConstOp<ConstExprAdd>("Add");
  registerConstOp<ConstExprMul>("Mul");
  registerConstOp<ConstExprSub>("Sub");
  registerConstOp<ConstExprFmod>("Fmod");
  registerConstOp<ConstExprDiv>("Div");
  registerConstOp<ConstExprCast>("Cast");
  registerConstOp<ConstExprScale>("Scale");
  registerConstOp<ConstExprSlice>("Slice");
  registerConstOp<ConstExprSlice>("SliceInplace");
  registerConstOp<ConstExprTranspose>("Transpose");
  registerConstOp<ConstExprConcat>("Concat");
  registerConstOp<ConstExprConcat>("ConcatInplace");
  registerConstOp<ConstExprUnsqueeze>("Unsqueeze");
  registerConstOp<ConstExprSqueeze>("Squeeze");
  registerConstOp<ConstExprIdentity>("Identity");
  registerConstOp<ConstExprIdentity>("Detach");
  registerConstOp<ConstExprReshape>("Reshape");
  registerConstOp<ConstExprReshape>("ReshapeInplace");
  registerConstOp<ConstExprGather>("Gather");
  registerConstOp<ConstExprIdentity>("Flatten");
  registerConstOp<ConstExprIdentity>("FlattenInplace");
  registerConstOp<ConstExprFloor>("Floor");
}

std::unique_ptr<ConstExprOp> ConstExprOpManager::createConstExprOp(Op *op) {

  auto &self = getInstance();
  auto it2   = self.constExprOpMap.find(op->opid.type);
  if (it2 != self.constExprOpMap.end()) {
    return it2->second(op);
  } else {
    throw internal_error("No ConstExpr implementation of {}. ", op->opid.type);
  }
}

bool ConstExprOpManager::hasConstExprOp(Op *op) {

  auto &self = getInstance();
  auto found = self.constExprOpMap.find(op->opid.type);
  return found != self.constExprOpMap.end();
}

} // namespace popart
