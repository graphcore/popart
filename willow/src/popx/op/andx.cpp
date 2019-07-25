#include <popart/error.hpp>
#include <popart/op/and.hpp>
#include <popart/popx/devicex.hpp>

#include <popart/popx/op/andx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
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
} // namespace popart
