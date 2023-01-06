// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/ScaledAdd.hpp>
#include <popart/error.hpp>
#include <popart/op/adamvarupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/popx/op/adamvarupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/op/varupdatex.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace pe = popops::expr;

namespace popart {
class Op;

namespace popx {
class Devicex;

AdamVarUpdateOpx::AdamVarUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<AdamVarUpdateOp>(op, Onnx::CustomOperators::AdamVarUpdate);
}

void AdamVarUpdateOpx::grow(poplar::program::Sequence &prog) const {

  // see adam.hpp for the equations implemented here

  auto &adamVarUpdateOp = getOp<AdamVarUpdateOp>();

  poplar::Tensor updater =
      getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());

  poplar::Tensor var = getInTensor(VarUpdateOp::getVarToUpdateInIndex());

  std::vector<poplar::Tensor> tensors;

  pe::Any lr(pe::Const(0.0f));
  pe::Any mwn(pe::Const(0.0f));

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

    if (adamVarUpdateOp.initMwn.isConst()) {
      mwn = pe::Const(adamVarUpdateOp.initMwn.val());
    } else {
      tensors.push_back(getInTensor(AdamVarUpdateOp::getMwnInIndex()));
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
    popops::scaledAddTo(graph(),
                        var,
                        updater,
                        -adamVarUpdateOp.initLr.val(),
                        prog,
                        debugContext("varUpdate"));
  } else {
    // Calculate final non-const learning rate tensor from expression
    auto lrt = popops::map(
        graph(), pe::Neg(lr), tensors, prog, debugContext("leaningRate"));

    // Variable update: var -= lr * updater
    popops::scaledAddTo(
        graph(), var, updater, lrt, prog, debugContext("varUpdate"));
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
