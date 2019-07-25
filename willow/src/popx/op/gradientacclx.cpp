#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/gradientaccl.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/gradientacclx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

GradientAcclOpx::GradientAcclOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<GradientAcclOp>(op, {Onnx::CustomOperators::GradientAccumulation});
}

void GradientAcclOpx::grow(poplar::program::Sequence &prog) const {
  // Get the inputs
  auto accl = getInTensor(GradientAcclOp::getAcclInIndex());
  auto grad = getInTensor(GradientAcclOp::getGradInIndex());

  // Add the gradient to the accumulator inplace.
  popops::addInPlace(graph(), accl, grad, prog, debugPrefix("accl"));

  setOutTensor(0, getInTensor(GradientAcclOp::getAcclInIndex()));
}

ResetAcclOpx::ResetAcclOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ResetAcclOp>(op, {Onnx::CustomOperators::ResetAccumulation});
}

void ResetAcclOpx::grow(poplar::program::Sequence &prog) const {
  // Get the inputs

  // Reset the accumulator
  // T7787: This could be scaled instead to have a momentum optimizer

  popops::zero(graph(),
               getInTensor(ResetAcclOp::getAcclInIndex()),
               prog,
               debugPrefix("reset"));
  setOutTensor(0, getInTensor(ResetAcclOp::getAcclInIndex()));
}

namespace {
OpxCreator<GradientAcclOpx>
    GradientAcclOpxCreator({Onnx::CustomOperators::GradientAccumulation});

OpxCreator<ResetAcclOpx>
    ResetAcclOpxCreator({Onnx::CustomOperators::ResetAccumulation});
} // namespace

} // namespace popx
} // namespace popart
