#include <popops/Cast.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/less.hpp>
#include <poponnx/popx/devicex.hpp>

#include <poponnx/popx/op/lessx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

LessOpx::LessOpx(Op *op, Devicex *devicex) : BinaryComparisonOpx(op, devicex) {
  verifyOp<LessOp>(op, {Onnx::Operators::Less_7, Onnx::Operators::Less_9});
}

void LessOpx::grow(poplar::program::Sequence &prog) const {

  insert(outId(LessOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::BinaryOpType::LESS_THAN,
                     get(inId(LessOp::getArg0InIndex())),
                     get(inId(LessOp::getArg1InIndex())),
                     prog,
                     idStr()));
}

namespace {

OpxCreator<LessOpx> lessOpxCreator_7(Onnx::Operators::Less_7);
OpxCreator<LessOpx> lessOpxCreator_9(Onnx::Operators::Less_9);

} // namespace

} // namespace popx
} // namespace poponnx
