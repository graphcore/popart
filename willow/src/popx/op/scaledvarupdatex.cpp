// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/ScaledAdd.hpp>
#include <popart/error.hpp>
#include <popart/op/scaledvarupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/popx/op/scaledvarupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/optimizervalue.hpp"
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

ScaledVarUpdateOpx::ScaledVarUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<ScaledVarUpdateOp>(op, Onnx::CustomOperators::ScaledVarUpdate);
}

poplar::Tensor ScaledVarUpdateOpx::getOrCreateLrTensor() const {
  auto adaptiveVarUpdateOp = getOp<ScaledVarUpdateOp>();

  if (hasInput(ScaledVarUpdateOp::getLrInIndex())) {
    return getInTensor(ScaledVarUpdateOp::getLrInIndex());
  } else if (hasInput(ScaledVarUpdateOp::getWdInIndex())) {
    poplar::Tensor wd = getInTensor(ScaledVarUpdateOp::getWdInIndex());
    poplar::Tensor lr = graph().addConstant(
        wd.elementType(), wd.shape(), adaptiveVarUpdateOp.initLr.val());
    graph().setTileMapping(lr, graph().getTileMapping(wd));
    return lr;
  } else {
    throw internal_error(
        "Unexpected condition in ScaledVarUpdateOpx::getOrCreateLrTensor");
  }
}

poplar::Tensor ScaledVarUpdateOpx::getOrCreateWdTensor() const {
  auto adaptiveVarUpdateOp = getOp<ScaledVarUpdateOp>();

  if (hasInput(ScaledVarUpdateOp::getWdInIndex())) {
    return getInTensor(ScaledVarUpdateOp::getWdInIndex());
  } else if (hasInput(ScaledVarUpdateOp::getLrInIndex())) {
    poplar::Tensor lr = getInTensor(ScaledVarUpdateOp::getLrInIndex());
    poplar::Tensor wd = graph().addConstant(lr.elementType(),
                                            lr.shape(),
                                            adaptiveVarUpdateOp.initWd.val(),
                                            debugContext());
    graph().setTileMapping(wd, graph().getTileMapping(lr));
    return wd;
  } else {
    throw internal_error(
        "Unexpected condition in ScaledVarUpdateOpx::getOrCreateWdTensor");
  }
}

void ScaledVarUpdateOpx::grow(poplar::program::Sequence &prog) const {

  // see adaptive.hpp for the equations implemented here

  auto adaptiveVarUpdateOp = getOp<ScaledVarUpdateOp>();

  poplar::Tensor var = getInTensor(VarUpdateOp::getVarToUpdateInIndex());
  poplar::Tensor updater =
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

void ScaledVarUpdateOpx::growWithLrAsInput(
    poplar::program::Sequence &prog,
    const ScaledVarUpdateOp &op,
    const poplar::Tensor &var,
    const poplar::Tensor &updater) const {
  auto updater_t = updater;
  if (var.elementType() != updater_t.elementType()) {
    updater_t = popops::cast(graph(),
                             updater,
                             var.elementType(),
                             prog,
                             debugContext("_castupdater"));
  }

  if (op.initLr.isConst() && op.initWd.isConst()) {
    if (op.initWd.val() == 0.0f) {
      popops::scaledAddTo(graph(),
                          var,
                          updater_t,
                          -op.initLr.val(),
                          prog,
                          debugContext("_c_0"));
    } else {
      popops::scaledAddTo(graph(),
                          var,
                          1.0f - op.initLr.val() * op.initWd.val(),
                          updater_t,
                          -op.initLr.val(),
                          prog,
                          debugContext("_c_c"));
    }
  } else {
    auto lrt = getOrCreateLrTensor();

    // (-lr)                    (negate lr)
    lrt = popops::map(graph(), pe::Neg(pe::_1), {lrt}, prog);

    if (op.initWd.isConst() && op.initWd.val() == 0.0f) {
      popops::scaledAddTo(
          graph(), var, updater_t, lrt, prog, debugContext("_t_0"));
    } else {
      auto wdt = getOrCreateWdTensor();

      // 1.0 + (-lr) * wd       (lr already negated)
      wdt = popops::map(graph(),
                        pe::Add(pe::Const(1.0f), pe::Mul(pe::_1, pe::_2)),
                        {lrt, wdt},
                        prog);

      // var = (1.0 + (-lr) * wd) * var + (-lr) * updater
      popops::scaledAddTo(
          graph(), var, wdt, updater_t, lrt, prog, debugContext("_t_t"));
    }
  }
}

void ScaledVarUpdateOpx::growWithLrInUpdater(
    poplar::program::Sequence &prog,
    const ScaledVarUpdateOp &op,
    const poplar::Tensor &var,
    const poplar::Tensor &updater) const {
  auto updater_t = updater;
  if (var.elementType() != updater_t.elementType()) {
    updater_t = popops::cast(graph(),
                             updater,
                             var.elementType(),
                             prog,
                             debugContext("_castupdater"));
  }
  if (op.initWd.isConst()) {
    if (op.initWd.val() == 0.0f) {
      // var = var - updater
      popops::mapInPlace(graph(),
                         pe::Sub(pe::_1, pe::_2),
                         {var, updater_t},
                         prog,
                         debugContext("__0"));
    } else {
      // var = var - (wd * var + updater)
      //     = var * (1.0 - wd) - updater
      popops::scaledAddTo(graph(),
                          var,
                          1.0f - op.initWd.val(),
                          updater_t,
                          -1.0f,
                          prog,
                          debugContext("__c"));
    }
  } else {
    auto wdt = getOrCreateWdTensor();
    // var = var * (1.0 - wd) - updater
    popops::mapInPlace(
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
