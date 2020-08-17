// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/adamvarupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/adamvarupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

AdamVarUpdateOpx::AdamVarUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<AdamVarUpdateOp>(op, Onnx::CustomOperators::AdamVarUpdate);
}

void AdamVarUpdateOpx::grow(poplar::program::Sequence &prog) const {

  // see optimizer.hpp for the equations implemented here

  auto adamVarUpdateOp = getOp<AdamVarUpdateOp>();

  poplar::Tensor updater =
      getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());

  poplar::Tensor var = getInTensor(VarUpdateOp::getVarToUpdateInIndex());

  std::vector<poplar::Tensor> tensors;

  pe::Any lr(pe::Const(0.0f));

  if (adamVarUpdateOp.initLr.isConst()) {
    lr = pe::Const(adamVarUpdateOp.initLr.val());
  } else {
    tensors.push_back(getInTensor(AdamVarUpdateOp::getLrInIndex()));
    lr = pe::PlaceHolder(tensors.size());
  }

  // Lamb scaled learning rate: lr = lr * sqrt(r1)/sqrt(r2)
  if (hasInput(AdamVarUpdateOp::getLambR1SqInIndex()) &&
      hasInput(AdamVarUpdateOp::getLambR2SqInIndex())) {
    tensors.push_back(getInTensor(AdamVarUpdateOp::getLambR1SqInIndex()));
    auto r1sqindex = tensors.size();
    tensors.push_back(getInTensor(AdamVarUpdateOp::getLambR2SqInIndex()));
    auto r2sqindex = tensors.size();

    lr = pe::Mul(
        lr,
        pe::Select(
            pe::Const(1.0f),
            pe::Select(pe::Const(1.0f),
                       pe::Divide(pe::Sqrt(pe::PlaceHolder(r1sqindex)),
                                  pe::Sqrt(pe::PlaceHolder(r2sqindex))),
                       pe::Equal(pe::PlaceHolder(r2sqindex), pe::Const(0.0f))),
            pe::Equal(pe::PlaceHolder(r1sqindex), pe::Const(0.0f))));
  }

  if (tensors.size() == 0) {
    // Variable update: var -= lr * updater
    popops::scaledAddTo(
        graph(), var, updater, -adamVarUpdateOp.initLr.val(), prog);
  } else {
    // Calculate final non-const learning rate tensor from expression
    poplar::Tensor lrt = popops::map(graph(), pe::Neg(lr), tensors, prog);

    // Variable update: var -= lr * updater
    popops::scaledAddTo(graph(), var, updater, lrt, prog);
  }

  if (hasInViewChangers(AdamVarUpdateOp::getVarToUpdateInIndex())) {
    setOutViewChangers(
        AdamVarUpdateOp::getUpdatedVarOutIndex(),
        getInViewChangers(AdamVarUpdateOp::getVarToUpdateInIndex()));
  }
  // output is a reference to the updated input
  setOutTensor(AdamVarUpdateOp::getUpdatedVarOutIndex(),
               getInTensor(AdamVarUpdateOp::getVarToUpdateInIndex()));
}

namespace {
OpxCreator<AdamVarUpdateOpx>
    AdamVarUpdateOpxCreator(Onnx::CustomOperators::AdamVarUpdate);
}
} // namespace popx
} // namespace popart
