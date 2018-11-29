#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/reciprocal.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/reciprocalx.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {
namespace popx {

ReciprocalOpx::ReciprocalOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (dynamic_cast<ReciprocalOp *>(op) == nullptr) {
    throw error("cannot create ReciprocalOpx from " + op->op_type());
  }
}

void ReciprocalOpx::grow(poplar::program::Sequence &prog) const {
  auto ones = dv_p->getConst(popType(op_p->input.tensor(0)->info), {1}, 1.0);

  insert(outId(0),
         popops::map(graph(),
                     popops::expr::BinaryOpType::DIVIDE,
                     ones,
                     get(inId(0)),
                     prog,
                     idStr()));
}

} // namespace popx
} // namespace poponnx
