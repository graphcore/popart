#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/cos.hpp>
#include <poponnx/popx/op/cosx.hpp>

namespace poponnx {
namespace popx {

CosOpx::CosOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (!op->isConvertibleTo<CosOp>()) {
    throw error("cannot create CosOpx from " + op->op_type());
  }
}

void CosOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(CosOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::UnaryOpType::COS,
                     get(inId(CosOp::getInIndex())),
                     prog,
                     idStr()));
}

} // namespace popx
} // namespace poponnx
