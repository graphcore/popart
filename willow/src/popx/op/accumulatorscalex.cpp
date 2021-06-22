// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulatorscale.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/accumulatorscalex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

AccumulatorScaleOpx::AccumulatorScaleOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<AccumulatorScaleOp>(op, {Onnx::CustomOperators::AccumulatorScale});
}

void AccumulatorScaleOpx::grow(poplar::program::Sequence &prog) const {

  auto &accumulateOp = getOp<AccumulatorScaleOp>();

  auto accum = getInTensor(AccumulatorScaleOp::getVarToUpdateInIndex())
                   .getPoplarTensor();

  auto factor = accumulateOp.getFactor();

  if (factor.isConst()) {
    auto val = factor.val();
    if (val == 0.0f) {
      popops::zero(graph().getPoplarGraph(),
                   accum,
                   prog,
                   debugContext("AccumulatorScale"));
    } else {
      popops::mulInPlace(graph().getPoplarGraph(),
                         accum,
                         val,
                         prog,
                         debugContext("AccumulatorScale"));
    }
  } else {
    auto factor =
        getInTensor(AccumulatorScaleOp::getFactorInIndex()).getPoplarTensor();
    popops::mulInPlace(graph().getPoplarGraph(),
                       accum,
                       factor,
                       prog,
                       debugContext("AccumulatorScale"));
  }

  if (hasInViewChangers(AccumulatorScaleOp::getVarToUpdateInIndex())) {
    setOutViewChangers(
        AccumulatorScaleOp::getUpdatedVarOutIndex(),
        getInViewChangers(AccumulatorScaleOp::getVarToUpdateInIndex()));
  }

  // reference accum returned
  setOutTensor(AccumulatorScaleOp::getUpdatedVarOutIndex(),
               snap::Tensor{accum, graph()});
}

namespace {
OpxCreator<AccumulatorScaleOpx>
    AccumulatorScaleOpxCreator({Onnx::CustomOperators::AccumulatorScale});
}

} // namespace popx
} // namespace popart
