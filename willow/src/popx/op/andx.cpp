#include <poponnx/error.hpp>
#include <poponnx/op/and.hpp>
#include <poponnx/popx/devicex.hpp>

#include <poponnx/popx/op/andx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

AndOpx::AndOpx(Op *op, Devicex *devicex) : BinaryComparisonOpx(op, devicex) {
  verifyOp<AndOp>(op, {Onnx::Operators::And_1, Onnx::Operators::And_7});
}

void AndOpx::grow(poplar::program::Sequence &prog) const {

  insert(outId(AndOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::BinaryOpType::LOGICAL_AND,
                     get(inId(AndOp::getArg0InIndex())),
                     get(inId(AndOp::getArg1InIndex())),
                     prog,
                     idStr()));
}

namespace {

OpxCreator<AndOpx> greaterOpxCreator_7(Onnx::Operators::And_1);
OpxCreator<AndOpx> greaterOpxCreator_9(Onnx::Operators::And_7);

} // namespace

} // namespace popx
} // namespace poponnx
