// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulatorupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/accumulatorupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

AccumulatorUpdateOpx::AccumulatorUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<AccumulatorUpdateOp>(op, {Onnx::CustomOperators::AccumulatorUpdate});
}

void AccumulatorUpdateOpx::grow(poplar::program::Sequence &prog) const {

  auto accumulateOp = getOp<AccumulatorUpdateOp>();

  auto accum = getInTensor(AccumulatorUpdateOp::getVarToUpdateInIndex());

  auto factor = accumulateOp.getFactor();

  if (!accum.isParallelWriteable()) {
    throw error("Expected accumulator {} to be writable in {}.",
                accumulateOp.inId(AccumulatorUpdateOp::getVarToUpdateInIndex()),
                accumulateOp.debugName());
  }

  if (factor.isConst()) {
    auto val = factor.val();
    if (val == 0.0f) {
      popops::zero(graph(), accum, prog, debugPrefix("accumulatorUpdate"));
    } else {
      popops::mulInPlace(
          graph(), accum, val, prog, debugPrefix("accumulatorUpdate"));
    }
  } else {
    auto factor = getInTensor(AccumulatorUpdateOp::getFactorInIndex());
    popops::mulInPlace(
        graph(), accum, factor, prog, debugPrefix("accumulatorUpdate"));
  }

  if (hasInViewChangers(AccumulatorUpdateOp::getVarToUpdateInIndex())) {
    setOutViewChangers(
        AccumulatorUpdateOp::getUpdatedVarOutIndex(),
        getInViewChangers(AccumulatorUpdateOp::getVarToUpdateInIndex()));
  }

  // reference accum returned
  setOutTensor(AccumulatorUpdateOp::getUpdatedVarOutIndex(), accum);
}

namespace {
OpxCreator<AccumulatorUpdateOpx>
    AccumulatorUpdateOpxCreator({Onnx::CustomOperators::AccumulatorUpdate});
}

} // namespace popx
} // namespace popart
