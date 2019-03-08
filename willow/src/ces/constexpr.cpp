#include <onnx/onnx_pb.h>
#include <poponnx/attributes.hpp>
#include <poponnx/ces/addce.hpp>
#include <poponnx/ces/castce.hpp>
#include <poponnx/ces/concatce.hpp>
#include <poponnx/ces/constexpr.hpp>
#include <poponnx/ces/reshapece.hpp>
#include <poponnx/ces/scalece.hpp>
#include <poponnx/ces/slicece.hpp>
#include <poponnx/ces/transposece.hpp>
#include <poponnx/ces/unsqueezece.hpp>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensors.hpp>

namespace poponnx {

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

void ConstExprUtil::processOp(Op *op, Ir *ir) {
  logging::ces::debug(
      "Processing Op `{}` ({}) in ConstExprUtil", op->id, op->opid.type);
  auto constOp = ConstExprOpManager::createConstExprOp(op);

  auto data = constOp->compute();
  makeTensorConstInit(op->outTensor(0)->id, data.data(), ir);
  op->disconnectAllInputs();

  if (op->input->n() > 0 || op->output->n() > 0) {
    throw error("processed op still has connected tensors");
  }

  ir->eraseOp(op->id);
  op = nullptr;
}

void ConstExprUtil::foldConstants(Ir *ir) {
  // get ops that may be computable
  std::unordered_set<Op *> computable_ops;
  for (auto &id_op : ir->getOps()) {
    auto &op = id_op.second;
    if (isComputable(op.get(), ir)) {
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
    processOp(op, ir);
    auto out_tensor = ir->getTensors().get(out_id);
    for (auto consumer : out_tensor->consumers.getOps()) {
      if (isComputable(consumer, ir)) {
        computable_ops.insert(consumer);
      }
    }
  }
}

void ConstExprUtil::makeTensorConstInit(const TensorId name,
                                        const void *data,
                                        Ir *ir) {
  // disconnect producer
  auto current_tensor = ir->getTensors().get(name);
  auto producer       = current_tensor->getProducer();
  producer->disconnectOutTensor(current_tensor);

  ir->getTensors().makeConstInit(name, data);
}

bool ConstExprUtil::isComputable(Op *op, Ir *ir) {
  auto mode   = ir->getExecutionMode();
  auto inputs = op->input->tensors();

  if (mode == Ir::ExecutionMode::TRAINING) {
    return std::all_of(inputs.begin(), inputs.end(), [](Tensor *t) {
      return t->tensorType() == TensorType::Const;
    });
  } else {
    return std::all_of(inputs.begin(), inputs.end(), [](Tensor *t) {
      return t->tensorType() == TensorType::Const ||
             t->tensorType() == TensorType::Variable;
    });
  }
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
    return make_unique<T>(op);
  });
}

void ConstExprOpManager::registerConstOps() {
  registerConstOp<ConstExprAdd>("Add");
  registerConstOp<ConstExprCast>("Cast");
  registerConstOp<ConstExprScale>("Scale");
  registerConstOp<ConstExprSlice>("Slice");
  registerConstOp<ConstExprTranspose>("Transpose");
  registerConstOp<ConstExprConcat>("Concat");
  registerConstOp<ConstExprUnsqueeze>("Unsqueeze");
  registerConstOp<ConstExprReshape>("Reshape");
}

std::unique_ptr<ConstExprOp> ConstExprOpManager::createConstExprOp(Op *op) {

  auto &self = getInstance();
  auto it2   = self.constExprOpMap.find(op->opid.type);
  if (it2 != self.constExprOpMap.end()) {
    return it2->second(op);
  } else {
    throw error("No ConstExpr implementation of {}. "
                "Consider what OpType::ADD does (creates a Const Tensor) "
                "if you would like to implement a ConstExpr",
                op->opid.type);
  }
}

} // namespace poponnx
