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
    poplar::Tensor wd = graph().addConstant(
        lr.elementType(), lr.shape(), adaptiveVarUpdateOp.initWd.val());
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

  if (adaptiveVarUpdateOp.initLr.isConst() &&
      adaptiveVarUpdateOp.initWd.isConst()) {
    if (adaptiveVarUpdateOp.initWd.val() == 0.0f) {
      popops::scaledAddTo(graph(),
                          var,
                          updater,
                          -adaptiveVarUpdateOp.initLr.val(),
                          prog,
                          debugPrefix("_c_0"));
    } else {
      popops::scaledAddTo(graph(),
                          var,
                          1.0f - adaptiveVarUpdateOp.initLr.val() *
                                     adaptiveVarUpdateOp.initWd.val(),
                          updater,
                          -adaptiveVarUpdateOp.initLr.val(),
                          prog,
                          debugPrefix("_c_c"));
    }
  } else {
    poplar::Tensor lrt = getOrCreateLrTensor();

    // (-lr)                    (negate lr)
    lrt = popops::map(graph(), pe::Neg(pe::_1), {lrt}, prog);

    if (adaptiveVarUpdateOp.initWd.isConst() &&
        adaptiveVarUpdateOp.initWd.val() == 0.0f) {
      popops::scaledAddTo(
          graph(), var, updater, lrt, prog, debugPrefix("_t_0"));
    } else {
      poplar::Tensor wdt = getOrCreateWdTensor();

      // 1.0 + (-lr) * wd       (lr already negated)
      wdt = popops::map(graph(),
                        pe::Add(pe::Const(1.0f), pe::Mul(pe::_1, pe::_2)),
                        {lrt, wdt},
                        prog);

      // var = (1.0 + (-lr) * wd) * var + (-lr) * updater
      popops::scaledAddTo(
          graph(), var, wdt, updater, lrt, prog, debugPrefix("_t_t"));
    }
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

namespace {
OpxCreator<ScaledVarUpdateOpx>
    ScaledVarUpdateOpxCreator(Onnx::CustomOperators::ScaledVarUpdate);
}
} // namespace popx
} // namespace popart
