// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/scaledvarupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/scaledvarupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

ScaledVarUpdateOpx::ScaledVarUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<ScaledVarUpdateOp>(op, Onnx::CustomOperators::ScaledVarUpdate);
}

poplar::Tensor ScaledVarUpdateOpx::getOrCreateLrTensor() const {
  auto adaptiveVarUpdateOp = getOp<ScaledVarUpdateOp>();

  if (hasInput(ScaledVarUpdateOp::getLrInIndex())) {
    return getInTensor(ScaledVarUpdateOp::getLrInIndex()).getPoplarTensor();
  } else if (hasInput(ScaledVarUpdateOp::getWdInIndex())) {
    poplar::Tensor wd =
        getInTensor(ScaledVarUpdateOp::getWdInIndex()).getPoplarTensor();
    poplar::Tensor lr =
        graph().getPoplarGraph().addConstant(wd.elementType(),
                                             wd.shape(),
                                             adaptiveVarUpdateOp.initLr.val(),
                                             debugContext("Lr"));
    graph().getPoplarGraph().setTileMapping(
        lr, graph().getPoplarGraph().getTileMapping(wd));
    return lr;
  } else {
    throw internal_error(
        "Unexpected condition in ScaledVarUpdateOpx::getOrCreateLrTensor");
  }
}

poplar::Tensor ScaledVarUpdateOpx::getOrCreateWdTensor() const {
  auto adaptiveVarUpdateOp = getOp<ScaledVarUpdateOp>();

  if (hasInput(ScaledVarUpdateOp::getWdInIndex())) {
    return getInTensor(ScaledVarUpdateOp::getWdInIndex()).getPoplarTensor();
  } else if (hasInput(ScaledVarUpdateOp::getLrInIndex())) {
    poplar::Tensor lr =
        getInTensor(ScaledVarUpdateOp::getLrInIndex()).getPoplarTensor();
    poplar::Tensor wd =
        graph().getPoplarGraph().addConstant(lr.elementType(),
                                             lr.shape(),
                                             adaptiveVarUpdateOp.initWd.val(),
                                             debugContext());
    graph().getPoplarGraph().setTileMapping(
        wd, graph().getPoplarGraph().getTileMapping(lr));
    return wd;
  } else {
    throw internal_error(
        "Unexpected condition in ScaledVarUpdateOpx::getOrCreateWdTensor");
  }
}

void ScaledVarUpdateOpx::grow(poplar::program::Sequence &prog) const {

  // see adaptive.hpp for the equations implemented here

  auto adaptiveVarUpdateOp = getOp<ScaledVarUpdateOp>();

  poplar::Tensor var =
      getInTensor(VarUpdateOp::getVarToUpdateInIndex()).getPoplarTensor();
  poplar::Tensor updater =
      getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex())
          .getPoplarTensor();

  if (adaptiveVarUpdateOp.lrInUpdater) {
    growWithLrInUpdater(prog, adaptiveVarUpdateOp, var, updater);
  } else {
    growWithLrAsInput(prog, adaptiveVarUpdateOp, var, updater);
  }

  if (hasInViewChangers(ScaledVarUpdateOp::getVarToUpdateInIndex())) {
    setOutViewChangers(
        ScaledVarUpdateOp::getUpdatedVarOutIndex(),
        getInViewChangers(ScaledVarUpdateOp::getVarToUpdateInIndex()));
  }
  // output is a reference to the updated input
  setOutTensor(ScaledVarUpdateOp::getUpdatedVarOutIndex(),
               getInTensor(ScaledVarUpdateOp::getVarToUpdateInIndex()));
}

void ScaledVarUpdateOpx::growWithLrAsInput(
    poplar::program::Sequence &prog,
    const ScaledVarUpdateOp &op,
    const poplar::Tensor &var,
    const poplar::Tensor &updater) const {
  if (op.initLr.isConst() && op.initWd.isConst()) {
    if (op.initWd.val() == 0.0f) {
      popops::scaledAddTo(graph().getPoplarGraph(),
                          var,
                          updater,
                          -op.initLr.val(),
                          prog,
                          debugContext("_c_0"));
    } else {
      popops::scaledAddTo(graph().getPoplarGraph(),
                          var,
                          1.0f - op.initLr.val() * op.initWd.val(),
                          updater,
                          -op.initLr.val(),
                          prog,
                          debugContext("_c_c"));
    }
  } else {
    poplar::Tensor lrt = getOrCreateLrTensor();

    // (-lr)                    (negate lr)
    lrt = popops::map(graph().getPoplarGraph(), pe::Neg(pe::_1), {lrt}, prog);

    if (op.initWd.isConst() && op.initWd.val() == 0.0f) {
      popops::scaledAddTo(graph().getPoplarGraph(),
                          var,
                          updater,
                          lrt,
                          prog,
                          debugContext("_t_0"));
    } else {
      poplar::Tensor wdt = getOrCreateWdTensor();

      // 1.0 + (-lr) * wd       (lr already negated)
      wdt = popops::map(graph().getPoplarGraph(),
                        pe::Add(pe::Const(1.0f), pe::Mul(pe::_1, pe::_2)),
                        {lrt, wdt},
                        prog);

      // var = (1.0 + (-lr) * wd) * var + (-lr) * updater
      popops::scaledAddTo(graph().getPoplarGraph(),
                          var,
                          wdt,
                          updater,
                          lrt,
                          prog,
                          debugContext("_t_t"));
    }
  }
}

void ScaledVarUpdateOpx::growWithLrInUpdater(
    poplar::program::Sequence &prog,
    const ScaledVarUpdateOp &op,
    const poplar::Tensor &var,
    const poplar::Tensor &updater) const {
  if (op.initWd.isConst()) {
    if (op.initWd.val() == 0.0f) {
      // var = var - updater
      popops::mapInPlace(graph().getPoplarGraph(),
                         pe::Sub(pe::_1, pe::_2),
                         {var, updater},
                         prog,
                         debugContext("__0"));
    } else {
      // var = var - (wd * var + updater)
      //     = var * (1.0 - wd) - updater
      popops::scaledAddTo(graph().getPoplarGraph(),
                          var,
                          1.0f - op.initWd.val(),
                          updater,
                          -1.0f,
                          prog,
                          debugContext("__c"));
    }
  } else {
    poplar::Tensor wdt = getOrCreateWdTensor();
    // var = var * (1.0 - wd) - updater
    wdt = popops::map(
        graph().getPoplarGraph(),
        pe::Sub(pe::Mul(pe::_1, pe::Sub(pe::Const(1.0f), pe::_3)), pe::_2),
        {var, updater, wdt},
        prog,
        debugContext("__t"));
  }
}

namespace {
OpxCreator<ScaledVarUpdateOpx>
    ScaledVarUpdateOpxCreator(Onnx::CustomOperators::ScaledVarUpdate);
}
} // namespace popx
} // namespace popart
