// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

  auto accum = getInTensor(VarUpdateOp::getVarToUpdateInIndex());

  popops::zero(graph(), accum, prog, debugPrefix("accumulatorUpdate"));

  // reference accum returned
  setOutTensor(VarUpdateOp::getUpdatedVarOutIndex(), accum);
}

namespace {
OpxCreator<AccumulatorUpdateOpx>
    AccumulatorUpdateOpxCreator({Onnx::CustomOperators::AccumulatorUpdate});
}

} // namespace popx
} // namespace popart
