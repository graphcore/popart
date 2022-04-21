// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <snap/popops/ElementWise.hpp>
#include <popops/Cast.hpp>
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
class ScaledVarUpdateOp;
namespace popx {

ScaledVarUpdateOpx::ScaledVarUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<ScaledVarUpdateOp>(op, Onnx::CustomOperators::ScaledVarUpdate);
}

snap::Tensor ScaledVarUpdateOpx::getOrCreateLrTensor() const {
  auto adaptiveVarUpdateOp = getOp<ScaledVarUpdateOp>();

  if (hasInput(ScaledVarUpdateOp::getLrInIndex())) {
    return getInTensor(ScaledVarUpdateOp::getLrInIndex());
  } else if (hasInput(ScaledVarUpdateOp::getWdInIndex())) {
    snap::Tensor wd = getInTensor(ScaledVarUpdateOp::getWdInIndex());
    snap::Tensor lr = snap::Tensor{
        graph().getPoplarGraph().addConstant(wd.elementType(),
                                             wd.shape(),
                                             adaptiveVarUpdateOp.initLr.val(),
                                             debugContext("Lr")),
        graph()};
    graph().getPoplarGraph().setTileMapping(
        lr.getPoplarTensor(),
        graph().getPoplarGraph().getTileMapping(wd.getPoplarTensor()));
    return lr;
  } else {
    throw internal_error(
        "Unexpected condition in ScaledVarUpdateOpx::getOrCreateLrTensor");
  }
}

snap::Tensor ScaledVarUpdateOpx::getOrCreateWdTensor() const {
  auto adaptiveVarUpdateOp = getOp<ScaledVarUpdateOp>();

  if (hasInput(ScaledVarUpdateOp::getWdInIndex())) {
    return getInTensor(ScaledVarUpdateOp::getWdInIndex());
  } else if (hasInput(ScaledVarUpdateOp::getLrInIndex())) {
    snap::Tensor lr = getInTensor(ScaledVarUpdateOp::getLrInIndex());
    snap::Tensor wd = snap::Tensor{
        graph().getPoplarGraph().addConstant(lr.elementType(),
                                             lr.shape(),
                                             adaptiveVarUpdateOp.initWd.val(),
                                             debugContext()),
        graph()};
    graph().getPoplarGraph().setTileMapping(
        wd.getPoplarTensor(),
        graph().getPoplarGraph().getTileMapping(lr.getPoplarTensor()));
    return wd;
  } else {
    throw internal_error(
        "Unexpected condition in ScaledVarUpdateOpx::getOrCreateWdTensor");
  }
}

void ScaledVarUpdateOpx::grow(snap::program::Sequence &prog) const {

  // see adaptive.hpp for the equations implemented here

  auto adaptiveVarUpdateOp = getOp<ScaledVarUpdateOp>();

  snap::Tensor var = getInTensor(VarUpdateOp::getVarToUpdateInIndex());
  snap::Tensor updater =
      getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());

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

void ScaledVarUpdateOpx::growWithLrAsInput(snap::program::Sequence &prog,
                                           const ScaledVarUpdateOp &op,
                                           const snap::Tensor &var,
                                           const snap::Tensor &updater) const {
  auto updater_t = updater.getPoplarTensor();
  if (var.getPoplarTensor().elementType() != updater_t.elementType()) {
    updater_t = popops::cast(graph().getPoplarGraph(),
                             updater.getPoplarTensor(),
                             var.getPoplarTensor().elementType(),
                             prog.getPoplarSequence(),
                             debugContext("_castupdater"));
  }

  if (op.initLr.isConst() && op.initWd.isConst()) {
    if (op.initWd.val() == 0.0f) {
      popops::scaledAddTo(graph().getPoplarGraph(),
                          var.getPoplarTensor(),
                          updater_t,
                          -op.initLr.val(),
                          prog.getPoplarSequence(),
                          debugContext("_c_0"));
    } else {
      popops::scaledAddTo(graph().getPoplarGraph(),
                          var.getPoplarTensor(),
                          1.0f - op.initLr.val() * op.initWd.val(),
                          updater_t,
                          -op.initLr.val(),
                          prog.getPoplarSequence(),
                          debugContext("_c_c"));
    }
  } else {
    auto lrt = getOrCreateLrTensor();

    // (-lr)                    (negate lr)
    lrt = snap::popops::map(graph(), pe::Neg(pe::_1), {lrt}, prog);

    if (op.initWd.isConst() && op.initWd.val() == 0.0f) {
      popops::scaledAddTo(graph().getPoplarGraph(),
                          var.getPoplarTensor(),
                          updater_t,
                          lrt.getPoplarTensor(),
                          prog.getPoplarSequence(),
                          debugContext("_t_0"));
    } else {
      auto wdt = getOrCreateWdTensor();

      // 1.0 + (-lr) * wd       (lr already negated)
      wdt = snap::popops::map(graph(),
                              pe::Add(pe::Const(1.0f), pe::Mul(pe::_1, pe::_2)),
                              {lrt, wdt},
                              prog);

      // var = (1.0 + (-lr) * wd) * var + (-lr) * updater
      popops::scaledAddTo(graph().getPoplarGraph(),
                          var.getPoplarTensor(),
                          wdt.getPoplarTensor(),
                          updater_t,
                          lrt.getPoplarTensor(),
                          prog.getPoplarSequence(),
                          debugContext("_t_t"));
    }
  }
}

void ScaledVarUpdateOpx::growWithLrInUpdater(
    snap::program::Sequence &prog,
    const ScaledVarUpdateOp &op,
    const snap::Tensor &var,
    const snap::Tensor &updater) const {
  auto updater_t = updater;
  if (var.getPoplarTensor().elementType() != updater_t.elementType()) {
    updater_t = snap::Tensor{popops::cast(graph().getPoplarGraph(),
                                          updater.getPoplarTensor(),
                                          var.getPoplarTensor().elementType(),
                                          prog.getPoplarSequence(),
                                          debugContext("_castupdater")),
                             graph()};
  }
  if (op.initWd.isConst()) {
    if (op.initWd.val() == 0.0f) {
      // var = var - updater
      snap::popops::mapInPlace(graph(),
                               pe::Sub(pe::_1, pe::_2),
                               {var, updater_t},
                               prog,
                               debugContext("__0"));
    } else {
      // var = var - (wd * var + updater)
      //     = var * (1.0 - wd) - updater
      popops::scaledAddTo(graph().getPoplarGraph(),
                          var.getPoplarTensor(),
                          1.0f - op.initWd.val(),
                          updater_t.getPoplarTensor(),
                          -1.0f,
                          prog.getPoplarSequence(),
                          debugContext("__c"));
    }
  } else {
    auto wdt = getOrCreateWdTensor();
    // var = var * (1.0 - wd) - updater
    snap::popops::mapInPlace(
        graph(),
        pe::Sub(pe::Mul(pe::_1,
                        pe::Sub(pe::Const(1.0f),
                                pe::Cast(pe::_3, var.elementType()))),
                pe::_2),
        {var, updater_t, wdt},
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
