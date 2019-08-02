#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/square.hpp>
#include <popart/popx/op/squarex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

SquareOpx::SquareOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SquareOp>(op, Onnx::CustomOperators::Square);
}

void SquareOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0,
               popops::map(graph(),
                           popops::expr::UnaryOpType::SQUARE,
                           getInTensor(0),
                           prog,
                           debugPrefix()));
}

namespace {
OpxCreator<SquareOpx> squareOpxCreator(Onnx::CustomOperators::Square);
// OpxCreator<Opx> squareGradOpxCreator("SquareGrad", "SquareGradOp should be
// removed by pattern 'SqrtGradOp'");

} // namespace

} // namespace popx
} // namespace popart
