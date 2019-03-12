#include <poponnx/error.hpp>
#include <poponnx/op/div.hpp>
#include <poponnx/popx/op/divx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

DivOpx::DivOpx(Op *op, Devicex *devicex) : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<DivOp>(op, {Onnx::Operators::Div_6, Onnx::Operators::Div_7});
}

void DivOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(0),
         popops::map(graph(),
                     popops::expr::BinaryOpType::DIVIDE,
                     get(inId(DivOp::getArg0InIndex())),
                     get(inId(DivOp::getArg1InIndex())),
                     prog,
                     idStr()));
}

namespace {
OpxCreator<DivOpx> divOpxCreator({Onnx::Operators::Div_6,
                                  Onnx::Operators::Div_7});
OpxCreator<Opx> divArg0OpxCreator(
    Onnx::GradOperators::DivArg0Grad,
    "DivArg0Grad should be optimised out, \"DivArg0Grad\" pattern is required");
OpxCreator<Opx> divArg1OpxCreator(Onnx::GradOperators::DivArg1Grad,
                                  "DivArg1Grad should be optimised out, "
                                  "\"DivArg1GradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace poponnx
