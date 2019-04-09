#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/exp.hpp>
#include <poponnx/popx/op/expx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

ExpInplaceOpx::ExpInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, ExpComputex::get()) {
  verifyOp<ExpInplaceOp>(op, Onnx::CustomOperators::ExpInplace);
}

ExpOpx::ExpOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, ExpComputex::get()) {
  verifyOp<ExpOp>(op, Onnx::Operators::Exp_6);
}

poplar::Tensor ExpComputex::outplace(poplar::program::Sequence &p,
                                     poplar::Graph &g,
                                     const poplar::Tensor &t) const {

  return popops::map(g, popops::expr::UnaryOpType::EXPONENT, t, p, "");
}

void ExpComputex::inplace(poplar::program::Sequence &p,
                          poplar::Graph &g,
                          const poplar::Tensor &t) const {

  popops::mapInPlace(g, popops::expr::UnaryOpType::EXPONENT, t, p, "");
}

namespace {
OpxCreator<ExpOpx> expOpxCreator(Onnx::Operators::Exp_6);
OpxCreator<ExpInplaceOpx>
    expxInplaceOpxCreator(Onnx::CustomOperators::ExpInplace);
OpxCreator<Opx>
    expGradOpxCreator(Onnx::GradOperators::ExpGrad,
                      "ExpGradOp should be removed by pattern 'ExpGradOp'");
} // namespace

} // namespace popx
} // namespace poponnx
