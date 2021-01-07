// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/rmspropupdater.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/rmspropupdaterx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

RMSPropUpdaterOpx::RMSPropUpdaterOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<RMSPropUpdaterOp>(op, Onnx::CustomOperators::RMSPropUpdater);
}

void RMSPropUpdaterOpx::grow(poplar::program::Sequence &prog) const {

  // see adaptive.hpp for the equations implemented here

  auto &rmspropUpdaterOp = getOp<RMSPropUpdaterOp>();

  poplar::Tensor grad  = getInTensor(RMSPropUpdaterOp::getGradInIndex());
  poplar::Tensor accl1 = getInTensor(RMSPropUpdaterOp::getAccl1InIndex());

  std::vector<poplar::Tensor> tensors = {grad, accl1};

  pe::Any rmsexpr(pe::Const(0.0f));
  if (hasInput(RMSPropUpdaterOp::getAccl2InIndex())) {
    // Centered version
    poplar::Tensor accl2 = getInTensor(RMSPropUpdaterOp::getAccl2InIndex());
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

  // grad / sqrt(Accl1 - Accl2^2 + eps)
  poplar::Tensor updater =
      popops::map(graph(),
                  pe::Cast(pe::Divide(pe::Cast(pe::_1, accl1.elementType()),
                                      pe::Add(pe::Sqrt(rmsexpr), epsexpr)),
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
