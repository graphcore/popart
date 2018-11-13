#include <poponnx/error.hpp>
#include <poponnx/popx/sumx.hpp>
#include <poponnx/sum.hpp>

#include <popops/ElementWise.hpp>

namespace willow {
namespace popx {

SumOpx::SumOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::SUM) {
    throw error("cannot create SumOpx from " + op->op_type());
  }
}

SumOp *SumOpx::getSumOp() const { return dynamic_cast<SumOp *>(op_p); }

void SumOpx::grow() const {
  // if the total number of tensors is less than
  // "5", then perform a series of adds.
  if (getSumOp()->input.n() < 5) {
    poplar::Tensor sum = popops::map(graph(),
                                     popops::expr::BinaryOpType::ADD,
                                     get(inId(0)),
                                     get(inId(1)),
                                     step(),
                                     idStr());

    for (int i = 2; i < getSumOp()->input.n(); ++i) {
      popops::mapInPlace(graph(),
                         popops::expr::BinaryOpType::ADD,
                         sum,
                         get(inId(i)),
                         step(),
                         idStr());
    }
    insert(outId(0), sum);
  }

  else {
    throw error("Must implemented SumOpx::grow() for greater than 4 inputs");
  }
}

} // namespace popx
} // namespace willow
