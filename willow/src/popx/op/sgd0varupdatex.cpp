// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/ScaledAdd.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/popx/op/sgd0varupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/popx/op/varupdatex.hpp"

namespace pe = popops::expr;

namespace popart {
class Op;

namespace popx {
class Devicex;

SGD0VarUpdateOpx::SGD0VarUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<SGD0VarUpdateOp>(op, Onnx::CustomOperators::SGD0VarUpdate);
}

void SGD0VarUpdateOpx::grow(snap::program::Sequence &prog) const {

  // Weight update (matching pytorch implementation)
  //  w <- w * (1 - lr * wd) - (lr/ls) * weight_gradient
  //
  // lr = learning rate
  // ls = loss scaling
  // wd = weight decay
  //
  // This is expressed as
  //
  // w <- w * weightDecayScaleFactor - scaledLearningRate * weight_gradient
  //
  // The (1 - lr * wd) and (lr/ls) calculations are done in SGD::setTensorData

  const auto &vu_op = getOp<SGD0VarUpdateOp>();

  // (1) update weights with weight decay

  // non-const weight decay scale factor
  if (!vu_op.initWdsf0.isConst()) {

    snap::popops::mapInPlace(
        graph(),
        pe::Mul(pe::_1, pe::_2),
        {getInTensor(SGD0VarUpdateOp::getVarToUpdateInIndex()),
         getInTensor(SGD0VarUpdateOp::getWdsf0InIndex())},
        prog,
        debugContext("nonConstWeightDecay"));
  }

  // const weight decay scale factor
  else {
    float scaleFactor = vu_op.initWdsf0.val();
    if (scaleFactor != 1.0f) {
      snap::popops::mapInPlace(
          graph(),
          pe::Mul(pe::_1, pe::Const(scaleFactor)),
          {getInTensor(SGD0VarUpdateOp::getVarToUpdateInIndex())},
          prog,
          debugContext("constWeightDecay"));
    }
  }

  // (2) subtract scaled gradients
  poplar::Tensor weightDeltas =
      getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex())
          .getPoplarTensor();

  // non-const scaled learning rate case
  if (!vu_op.initSlr0.isConst()) {
    popops::scaledAddTo(
        graph().getPoplarGraph(),
        getInTensor(SGD0VarUpdateOp::getVarToUpdateInIndex())
            .getPoplarTensor(), // weights
        weightDeltas,           // weightDeltas
        popops::neg(
            graph().getPoplarGraph(),
            getInTensor(SGD0VarUpdateOp::getSlr0InIndex()).getPoplarTensor(),
            prog.getPoplarSequence(),
            debugContext("neg")),
        prog.getPoplarSequence(),
        debugContext("nonConstScaledSubtract"));
  }

  // const scaled learning rate case
  else {
    popops::scaledAddTo(
        graph().getPoplarGraph(),
        getInTensor(vu_op.getVarToUpdateInIndex()).getPoplarTensor(), // weights
        weightDeltas, // weightDeltas
        -vu_op.initSlr0.val(),
        prog.getPoplarSequence(),
        debugContext("scaledSubtract"));
  }

  if (hasInViewChangers(SGD0VarUpdateOp::getVarToUpdateInIndex())) {
    setOutViewChangers(
        SGD0VarUpdateOp::getUpdatedVarOutIndex(),
        getInViewChangers(SGD0VarUpdateOp::getVarToUpdateInIndex()));
  }
  // output is a reference to the updated input
  setOutTensor(SGD0VarUpdateOp::getUpdatedVarOutIndex(),
               getInTensor(SGD0VarUpdateOp::getVarToUpdateInIndex()));
}

namespace {
OpxCreator<SGD0VarUpdateOpx>
    sgd0VarUpdateOpxCreator(Onnx::CustomOperators::SGD0VarUpdate);
} // namespace

} // namespace popx
} // namespace popart
