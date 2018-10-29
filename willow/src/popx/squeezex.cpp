#include <willow/error.hpp>
#include <willow/popx/squeezex.hpp>
#include <willow/squeeze.hpp>
#include <willow/tensor.hpp>

namespace willow {
namespace popx {

void SqueezeOpx::grow() const {
  auto outTensor = cloneNcopy(inId(0));
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

void SqueezeGradOpx::grow() const {
  auto outTensor = cloneNcopy(inId(0));
  outTensor.reshape(getSqueezeGradOp()->output.tensor(0)->info.shape_szt());
  insert(outId(0), outTensor);
}

SqueezeGradOp *SqueezeGradOpx::getSqueezeGradOp() const {
  return dynamic_cast<SqueezeGradOp *>(op_p);
}

} // namespace popx
} // namespace willow
