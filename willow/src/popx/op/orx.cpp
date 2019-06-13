#include <poponnx/error.hpp>
#include <poponnx/op/or.hpp>
#include <poponnx/popx/devicex.hpp>

#include <poponnx/popx/op/orx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

OrOpx::OrOpx(Op *op, Devicex *devicex) : BinaryComparisonOpx(op, devicex) {
  verifyOp<OrOp>(op, {Onnx::Operators::Or_1, Onnx::Operators::Or_7});
}

void OrOpx::grow(poplar::program::Sequence &prog) const {

  insert(outId(OrOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::BinaryOpType::LOGICAL_OR,
                     get(inId(OrOp::getArg0InIndex())),
                     get(inId(OrOp::getArg1InIndex())),
                     prog,
                     idStr()));
}

namespace {

OpxCreator<OrOpx> greaterOpxCreator_7(Onnx::Operators::Or_1);
OpxCreator<OrOpx> greaterOpxCreator_9(Onnx::Operators::Or_7);

} // namespace

} // namespace popx
} // namespace poponnx
