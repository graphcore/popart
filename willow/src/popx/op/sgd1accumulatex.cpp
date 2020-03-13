// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Zero.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/sgd1accumulate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/sgd1accumulatex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

SGD1AccumulateOpx::SGD1AccumulateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<SGD1AccumulateOp>(op, {Onnx::CustomOperators::SGD1Accumulate});
}

void SGD1AccumulateOpx::grow(poplar::program::Sequence &prog) const {

  auto sgd1AccumulateOp = getOp<SGD1AccumulateOp>();

  auto isConst = sgd1AccumulateOp.initDpsf1.isConst();

  auto accl = getInTensor(VarUpdateOp::getVarToUpdateInIndex());
  auto grad = getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());

  if (isConst) {
    auto val = sgd1AccumulateOp.initDpsf1.val();
    if (val == 0.0f) {
      throw internal_error(
          "dpsf1 of 0 is not allowed, should have been caught in "
          "the Ir, dpsf1 of 0 could be caused by dampening of 1, which "
          "means the gradient is multiplied by 0 (no learning)");
    }
    if (val - 1.0f == 0.0f) {
      // accl += grad
      popops::addInPlace(
          graph(), accl, grad, prog, debugPrefix("constAcclAdd1Ddpsf1"));
    } else {
      // accl += dpsf1 * grad
      popops::scaledAddTo(
          graph(), accl, grad, val, prog, debugPrefix("constScaledAddDpsf1"));
    }
  }

  else {
    auto dpsf = getInTensor(SGD1AccumulateOp::getDpsf1InIndex());
    popops::scaledAddTo(
        graph(), accl, grad, dpsf, prog, debugPrefix("nonConstSGD1AcclAddTo"));
  }

  // reference accl returned
  setOutTensor(VarUpdateOp::getUpdatedVarOutIndex(), accl);
}

poplar::Tensor SGD1AccumulateOpx::createInput(int inIndex,
                                              const std::string &name) const {

  if (inIndex != VarUpdateOp::getVarToUpdateInIndex()) {
    throw error(
        "SGD1AccumulateOpx::createInput, cannot create input at {}, it can "
        "only create the var to update input Tensor",
        inIndex);
  }
  return graph().clone(getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex()),
                       name);
}

InputCreatorType SGD1AccumulateOpx::getInputCreatorType(int inIndex) const {
  return inIndex == VarUpdateOp::getVarToUpdateInIndex()
             ? InputCreatorType::CANCREATE
             : Opx::getInputCreatorType(inIndex);
}

std::vector<TensorId>
SGD1AccumulateOpx::mustExistBeforeCreate(int index1) const {
  if (index1 != VarUpdateOp::getVarToUpdateInIndex()) {
    throw internal_error(
        "SGD1AccumulateOpx::mustExistBeforeCreate : Invalid index");
  }
  return {inId(VarUpdateWithUpdaterOp::getUpdaterInIndex())};
}

namespace {
OpxCreator<SGD1AccumulateOpx>
    SGD1AccumulateOpxCreator({Onnx::CustomOperators::SGD1Accumulate});
}

} // namespace popx
} // namespace popart
