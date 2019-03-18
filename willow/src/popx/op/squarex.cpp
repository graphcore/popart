#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/square.hpp>
#include <poponnx/popx/op/squarex.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
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
                           idStr()));
}

namespace {
OpxCreator<SquareOpx> squareOpxCreator(Onnx::CustomOperators::Square);
// OpxCreator<Opx> squareGradOpxCreator("SquareGrad", "SquareGradOp should be
// removed by pattern 'SqrtGradOp'");

} // namespace

} // namespace popx
} // namespace poponnx
