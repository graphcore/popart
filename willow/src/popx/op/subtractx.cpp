#include <popart/error.hpp>
#include <popart/op/subtract.hpp>
#include <popart/popx/op/subtractx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

SubtractOpx::SubtractOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<SubtractOp>(op, {Onnx::Operators::Sub_6, Onnx::Operators::Sub_7});
}

void SubtractOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(SubtractOp::getOutIndex(),
               popops::map(graph(),
                           popops::expr::BinaryOpType::SUBTRACT,
                           getInTensor(SubtractOp::getArg0InIndex()),
                           getInTensor(SubtractOp::getArg1InIndex()),
                           prog,
                           idStr()));
}

SubtractArg0GradOpx::SubtractArg0GradOpx(Op *op, Devicex *devicex)
    : ReduceSumOpx(op, devicex) {
  verifyOp<SubtractArg0GradOp>(op, Onnx::GradOperators::SubArg0Grad);
}

namespace {
OpxCreator<SubtractOpx> subtractOpxCreator({Onnx::Operators::Sub_6,
                                            Onnx::Operators::Sub_7});
OpxCreator<SubtractArg0GradOpx>
    subtractArg0GradOpxCreator(Onnx::GradOperators::SubArg0Grad);
OpxCreator<Opx> subtractArg1GradOpxCreator(
    Onnx::GradOperators::SubArg1Grad,
    "SubtractArg1GradOpx should be removed by pattern 'SubtractArg1GradOp'");
} // namespace

} // namespace popx
} // namespace popart
