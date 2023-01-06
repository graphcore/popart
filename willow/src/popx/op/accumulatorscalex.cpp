// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>
#include <popart/op/accumulatorscale.hpp>
#include <popart/popx/op/accumulatorscalex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/popx/op/varupdatex.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

AccumulatorScaleOpx::AccumulatorScaleOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<AccumulatorScaleOp>(op, {Onnx::CustomOperators::AccumulatorScale});
}

void AccumulatorScaleOpx::grow(poplar::program::Sequence &prog) const {

  auto &accumulateOp = getOp<AccumulatorScaleOp>();

  auto accum = getInTensor(AccumulatorScaleOp::getVarToUpdateInIndex());

  auto factor = accumulateOp.getFactor();

  if (factor.isConst()) {
    auto val = factor.val();
    if (val == 0.0f) {
      popops::zero(graph(), accum, prog, debugContext("AccumulatorScale"));
    } else {
      popops::mulInPlace(
          graph(), accum, val, prog, debugContext("AccumulatorScale"));
    }
  } else {
    auto factor = getInTensor(AccumulatorScaleOp::getFactorInIndex());
    popops::mulInPlace(
        graph(), accum, factor, prog, debugContext("AccumulatorScale"));
  }

  if (hasInViewChangers(AccumulatorScaleOp::getVarToUpdateInIndex())) {
    setOutViewChangers(
        AccumulatorScaleOp::getUpdatedVarOutIndex(),
        getInViewChangers(AccumulatorScaleOp::getVarToUpdateInIndex()));
  }

  // reference accum returned
  setOutTensor(AccumulatorScaleOp::getUpdatedVarOutIndex(), accum);
}

namespace {
OpxCreator<AccumulatorScaleOpx>
    AccumulatorScaleOpxCreator({Onnx::CustomOperators::AccumulatorScale});
}

} // namespace popx
} // namespace popart
