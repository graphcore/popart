#include <poponnx/error.hpp>
#include <poponnx/op/abs.hpp>
#include <poponnx/popx/op/absx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

namespace pe = popops::expr;

namespace poponnx {
namespace popx {

AbsOpx::AbsOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<AbsOp>(op, {Onnx::Operators::Abs_6});
}

void AbsOpx::grow(poplar::program::Sequence &prog) const {

  insert(outId(AbsOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::UnaryOpType::ABSOLUTE,
                     get(inId(AbsOp::getInIndex())),
                     prog,
                     idStr()));
}

namespace {
OpxCreator<AbsOpx> absOpxCreator(Onnx::Operators::Abs_6);
} // namespace

} // namespace popx
} // namespace poponnx
