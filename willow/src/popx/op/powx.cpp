#include <memory>
#include <poponnx/error.hpp>
#include <poponnx/op/pow.hpp>
#include <poponnx/popx/op/powx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

PowOpx::PowOpx(Op *op, Devicex *devicex) : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<PowOp>(op, {Onnx::Operators::Pow_1, Onnx::Operators::Pow_7});
}

void PowOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0,
               popops::map(graph(),
                           popops::expr::BinaryOpType::POWER,
                           getInTensor(PowOp::getArg0InIndex()),
                           getInTensor(PowOp::getArg1InIndex()),
                           prog,
                           idStr()));
}

namespace {
OpxCreator<PowOpx> powOpxCreator({Onnx::Operators::Pow_1,
                                  Onnx::Operators::Pow_7});
OpxCreator<Opx> powArg0OpxCreator(Onnx::GradOperators::PowArg0Grad,
                                  "PowArg0Grad should be optimised out, "
                                  "\"PowArg0GradOp\" pattern is required");
OpxCreator<Opx> powArg1OpxCreator(Onnx::GradOperators::PowArg1Grad,
                                  "PowArg1Grad should be optimised out, "
                                  "\"PowArg1GradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace poponnx
