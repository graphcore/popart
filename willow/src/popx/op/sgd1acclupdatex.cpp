// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Zero.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/sgd1acclupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/sgd1acclupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

SGD1AcclUpdateOpx::SGD1AcclUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<SGD1AcclUpdateOp>(op, {Onnx::CustomOperators::SGD1AcclUpdate});
}

void SGD1AcclUpdateOpx::grow(poplar::program::Sequence &prog) const {

  //   See optimizer.hpp for derivation of the equations implemented here

  const auto &vu_op = getOp<SGD1AcclUpdateOp>();

  auto smm1Const  = vu_op.initSmm1.isConst();
  auto swdf1Const = vu_op.initSwd1.isConst();

  const auto &toUpdate = getInTensor(VarUpdateOp::getVarToUpdateInIndex());

  if (smm1Const) {
    auto smm1Val = vu_op.initSmm1.val();
    if (smm1Val == 0.0f) {
      popops::zero(graph().getPoplarGraph(),
                   toUpdate,
                   prog,
                   debugContext("resetZeroMm"));
    } else {
      // This may be a mix of half and float but the Mul can handle because
      // it is a "pe::Const"
      popops::mapInPlace(
          graph().getPoplarGraph(),
          pe::Mul(pe::_1, pe::Const(smm1Val)),
          {toUpdate},
          prog,
          debugContext("constMomentumScaling_" + std::to_string(smm1Val)));
    }
  } else {
    // In the case of SGD2, we may need to cast Smm
    // TODO: T40976 can we make it the right type (e.g. half) to begin with
    popops::mapInPlace(
        graph().getPoplarGraph(),
        pe::Mul(pe::_1, pe::Cast(pe::_2, toUpdate.elementType())),
        {toUpdate, getInTensor(SGD1AcclUpdateOp::getSmm1InIndex())},
        prog,
        debugContext("nonConstMomentumScaling"));
  }

  poplar::Tensor weights =
      getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());

  if (swdf1Const) {
    auto swd1Val = vu_op.initSwd1.val();
    if (swd1Val != 0.0f) {
      popops::scaledAddTo(
          graph().getPoplarGraph(),
          toUpdate,
          weights,
          swd1Val,
          prog,
          debugContext("constScaledAddSwd1_" + std::to_string(swd1Val)));
    }
  } else {
    popops::scaledAddTo(graph().getPoplarGraph(),
                        toUpdate,
                        weights,
                        getInTensor(SGD1AcclUpdateOp::getSwd1InIndex()),
                        prog,
                        debugContext("nonConstScaledAddSwd1"));
  }

  if (hasInViewChangers(VarUpdateOp::getVarToUpdateInIndex())) {
    setOutViewChangers(VarUpdateOp::getUpdatedVarOutIndex(),
                       getInViewChangers(VarUpdateOp::getVarToUpdateInIndex()));
  }
  // return a reference to the input
  setOutTensor(VarUpdateOp::getUpdatedVarOutIndex(), toUpdate);
}

namespace {
OpxCreator<SGD1AcclUpdateOpx>
    ResetAcclOpxCreator({Onnx::CustomOperators::SGD1AcclUpdate});
} // namespace

} // namespace popx
} // namespace popart
