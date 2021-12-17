// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ScaledAdd.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/adamvarupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>
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

void AdamVarUpdateOpx::grow(snap::program::Sequence &prog) const {

  // see adam.hpp for the equations implemented here

  auto &adamVarUpdateOp = getOp<AdamVarUpdateOp>();

  poplar::Tensor updater =
      getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex())
          .getPoplarTensor();

  poplar::Tensor var =
      getInTensor(VarUpdateOp::getVarToUpdateInIndex()).getPoplarTensor();

  std::vector<poplar::Tensor> tensors;

  pe::Any lr(pe::Const(0.0f));
  pe::Any mwn(pe::Const(0.0f));

  if (adamVarUpdateOp.initLr.isConst()) {
    lr = pe::Const(adamVarUpdateOp.initLr.val());
  } else {
    tensors.push_back(
        getInTensor(AdamVarUpdateOp::getLrInIndex()).getPoplarTensor());
    lr = pe::PlaceHolder(tensors.size());
  }

  // Lamb scaled learning rate: lr = lr * sqrt(r1)/sqrt(r2)
  if (hasInput(AdamVarUpdateOp::getLambR1SqInIndex()) &&
      hasInput(AdamVarUpdateOp::getLambR2SqInIndex())) {
    tensors.push_back(
        getInTensor(AdamVarUpdateOp::getLambR1SqInIndex()).getPoplarTensor());
    auto r1sqindex = tensors.size();
    tensors.push_back(
        getInTensor(AdamVarUpdateOp::getLambR2SqInIndex()).getPoplarTensor());
    auto r2sqindex = tensors.size();

    if (adamVarUpdateOp.initMwn.isConst()) {
      mwn = pe::Const(adamVarUpdateOp.initMwn.val());
    } else {
      tensors.push_back(
          getInTensor(AdamVarUpdateOp::getMwnInIndex()).getPoplarTensor());
      mwn = pe::PlaceHolder(tensors.size());
    }

    // Following condition should always be satisfied as it is checked in
    // AdamDecompose::Apply
    if (!adamVarUpdateOp.initMwn.isConst() ||
        adamVarUpdateOp.initMwn.val() != 0) {

      // if (mwn == 0 || r1 == 0 || r2 == 0)
      //   ratio = 1
      // else
      //   ratio = min(mwn, r1) / r2
      pe::Any ratio =
          pe::Divide(pe::Min(pe::Sqrt(pe::PlaceHolder(r1sqindex)), mwn),
                     pe::Sqrt(pe::PlaceHolder(r2sqindex)));
      ratio =
          pe::Select(pe::Const(1.0f),
                     ratio,
                     pe::Equal(pe::PlaceHolder(r2sqindex), pe::Const(0.0f)));
      ratio =
          pe::Select(pe::Const(1.0f),
                     ratio,
                     pe::Equal(pe::PlaceHolder(r1sqindex), pe::Const(0.0f)));

      if (!adamVarUpdateOp.initMwn.isConst()) {
        ratio =
            pe::Select(pe::Const(1.0f), ratio, pe::Equal(mwn, pe::Const(0.0f)));
      }

      lr = pe::Mul(lr, ratio);
    } else {
      throw internal_error(
          "[AdamVarUpdatex] Constant zero max weight norm should not be "
          "lowered to AdamVarUpdatex with LambSquare inputs. This should have "
          "been prevented in AdamDecompose::Apply");
    }
  }

  if (tensors.size() == 0) {
    // Variable update: var -= lr * updater
    popops::scaledAddTo(graph().getPoplarGraph(),
                        var,
                        updater,
                        -adamVarUpdateOp.initLr.val(),
                        prog.getPoplarSequence(),
                        debugContext("varUpdate"));
  } else {
    // Calculate final non-const learning rate tensor from expression
    poplar::Tensor lrt = popops::map(graph().getPoplarGraph(),
                                     pe::Neg(lr),
                                     tensors,
                                     prog.getPoplarSequence(),
                                     debugContext("leaningRate"));

    // Variable update: var -= lr * updater
    popops::scaledAddTo(graph().getPoplarGraph(),
                        var,
                        updater,
                        lrt,
                        prog.getPoplarSequence(),
                        debugContext("varUpdate"));
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
