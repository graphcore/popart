#include <poponnx/error.hpp>
#include <poponnx/op/reshape.hpp>
#include <poponnx/popx/op/reshapex.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {
namespace popx {

// Test note : scale by 1.0001 in grad op makes the test fail. Good.
void ReshapeOpx::grow(poplar::program::Sequence &prog) const {
  // not in-place, so cloning input
  auto outTensor = cloneNcopy(prog, inId(ReshapeOp::getInIndex()));
  outTensor = outTensor.reshape(outInfo(ReshapeOp::getOutIndex()).shape_szt());
  insert(outId(ReshapeOp::getOutIndex()), outTensor);
}

ReshapeOpx::ReshapeOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::RESHAPE) {
    throw error("cannot create ReshapeOpx from " + op->op_type());
  }
}

ReshapeGradOpx::ReshapeGradOpx(Op *op, Devicex *devicex)
    : ReshapeOpx(op, devicex) {
  if (op->opType != OpType::RESHAPEGRAD) {
    throw error("cannot create ReshapeGradOpx from " + op->op_type());
  }
}

} // namespace popx
} // namespace poponnx
