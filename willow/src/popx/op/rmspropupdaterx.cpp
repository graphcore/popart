// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <vector>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/rmspropupdater.hpp>
#include <popart/popx/op/rmspropupdaterx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/popx/popopx.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace pe = popops::expr;

namespace popart {
class Op;

namespace popx {
class Devicex;

RMSPropUpdaterOpx::RMSPropUpdaterOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<RMSPropUpdaterOp>(op, Onnx::CustomOperators::RMSPropUpdater);
}

void RMSPropUpdaterOpx::grow(snap::program::Sequence &prog) const {

  // see adaptive.hpp for the equations implemented here

  auto &rmspropUpdaterOp = getOp<RMSPropUpdaterOp>();

  auto grad  = getInTensor(RMSPropUpdaterOp::getGradInIndex());
  auto accl1 = getInTensor(RMSPropUpdaterOp::getAccl1InIndex());

  std::vector<snap::Tensor> tensors = {grad, accl1};

  pe::Any rmsexpr(pe::Const(0.0f));
  if (hasInput(RMSPropUpdaterOp::getAccl2InIndex())) {
    // Centered version
    auto accl2 = getInTensor(RMSPropUpdaterOp::getAccl2InIndex());
    tensors.push_back(accl2);
    // Centered: Accl1 - Accl2^2
    rmsexpr =
        pe::Sub(pe::_2 - pe::Square(pe::Cast(pe::_3, accl1.elementType())));
  } else {
    // Non-centered: Accl1
    rmsexpr = pe::_2;
  }

  pe::Any epsexpr(pe::Const(0.0f));
  if (rmspropUpdaterOp.initEps.isConst()) {
    epsexpr = pe::Const(rmspropUpdaterOp.initEps.val());
  } else {
    tensors.push_back(getInTensor(RMSPropUpdaterOp::getEpsInIndex()));
    epsexpr = pe::PlaceHolder(tensors.size());
  }

  pe::Any denominatorexpr(pe::Const(0.0f));
  if (rmspropUpdaterOp.TFVariant) {
    // In TF variant of RMSProp, epsilon is added inside the square root.
    denominatorexpr = pe::Sqrt(pe::Add(rmsexpr, epsexpr));
  } else {
    denominatorexpr = pe::Add(pe::Sqrt(rmsexpr), epsexpr);
  }

  auto updater = snap::popops::map(
      graph(),
      pe::Cast(
          pe::Divide(pe::Cast(pe::_1, accl1.elementType()), denominatorexpr),
          grad.elementType()),
      tensors,
      prog,
      debugContext(""));

  if (hasInViewChangers(RMSPropUpdaterOp::getGradInIndex())) {
    setOutViewChangers(RMSPropUpdaterOp::getUpdaterOutIndex(),
                       getInViewChangers(RMSPropUpdaterOp::getGradInIndex()));
  }

  setOutTensor(RMSPropUpdaterOp::getUpdaterOutIndex(), updater);
}

namespace {
OpxCreator<RMSPropUpdaterOpx>
    RMSPropUpdaterOpxCreator(Onnx::CustomOperators::RMSPropUpdater);
}
} // namespace popx
} // namespace popart
