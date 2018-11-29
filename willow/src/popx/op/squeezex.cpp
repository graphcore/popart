#include <poponnx/error.hpp>
#include <poponnx/op/squeeze.hpp>
#include <poponnx/popx/op/squeezex.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {
namespace popx {

void SqueezeOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cloneNcopy(prog, inId(0));
  outTensor      = outTensor.reshape(op_p->output.tensor(0)->info.shape_szt());
  insert(outId(0), outTensor);
}

SqueezeOpx::SqueezeOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::SQUEEZE) {
    throw error("cannot create SqueezeOpx from " + op->op_type());
  }
}

SqueezeOp *SqueezeOpx::getSqueezeOp() const {
  return dynamic_cast<SqueezeOp *>(op_p);
}

SqueezeGradOpx::SqueezeGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::SQUEEZEGRAD) {
    throw error("cannot create SqueezeGradOpx from " + op->op_type());
  }
}

void SqueezeGradOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cloneNcopy(prog, inId(0));
  outTensor =
      outTensor.reshape(getSqueezeGradOp()->output.tensor(0)->info.shape_szt());
  insert(outId(0), outTensor);
}

SqueezeGradOp *SqueezeGradOpx::getSqueezeGradOp() const {
  return dynamic_cast<SqueezeGradOp *>(op_p);
}

} // namespace popx
} // namespace poponnx
