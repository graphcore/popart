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

InputCreatorType GradientAcclOpx::getInputCreatorType(int index) const {
  // Can create the accumlation tensor
  if (index == GradientAcclOp::getAcclInIndex()) {
    return InputCreatorType::CANCREATE;
  } else {
    return Opx::getInputCreatorType(index);
  }
}

std::vector<TensorId> GradientAcclOpx::mustExistBeforeCreate(int index) const {
  // Need the gradient in tensor to clone the layout
  if (index == GradientAcclOp::getAcclInIndex()) {
    return {inId(GradientAcclOp::getGradInIndex())};
  } else {
    return Opx::mustExistBeforeCreate(index);
  }
}

bool GradientAcclOpx::createsEquiv(int ind0, const Opx *opx1, int ind1) const {
  if (opx1->op_p->opid != Onnx::CustomOperators::GradientAccumulation)
    return false;

  if (ind0 != ind1)
    return false;

  const GradientAcclOpx *rhs = dynamic_cast<const GradientAcclOpx *>(opx1);

  GradientAcclOp *op = dynamic_cast<GradientAcclOp *>(op_p);
  if (op == nullptr)
    return false;

  GradientAcclOp *rhsOp = dynamic_cast<GradientAcclOp *>(rhs->op_p);
  if (rhsOp == nullptr)
    return false;

  // Make sure that all the inputs/output match
  if (op->inInfo(GradientAcclOp::getAcclInIndex()) !=
          rhsOp->inInfo(GradientAcclOp::getAcclInIndex()) ||
      op->inInfo(GradientAcclOp::getGradInIndex()) !=
          rhsOp->inInfo(GradientAcclOp::getGradInIndex()) ||
      op->outInfo(GradientAcclOp::getAcclOutIndex()) !=
          rhsOp->outInfo(GradientAcclOp::getAcclOutIndex())) {
    return false;
  }

  return true;
}

poplar::Tensor GradientAcclOpx::createInput(int index,
                                            const std::string &name) const {
  if (index == GradientAcclOp::getAcclInIndex()) {
    poplar::Tensor var = getInTensor(GradientAcclOp::getGradInIndex());
    return graph().clone(var, name);
  } else {
    return Opx::createInput(index, name);
  }
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
