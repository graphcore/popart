// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <popops/Zero.hpp>
#include <popart/op/init.hpp>
#include <popart/popx/op/initx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/popx/popopx.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;

InitOpx::InitOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<InitOp>(op, Onnx::CustomOperators::Init_1);
}

void InitOpx::grow(snap::program::Sequence &prog) const {
  auto &initOp          = getOp<InitOp>();
  const auto &outTensor = getOutTensor(InitOp::getOutIndex());

  switch (initOp.getInitType()) {
  case InitType::Zero: {
    popops::zero(graph().getPoplarGraph(),
                 outTensor.getPoplarTensor(),
                 prog.getPoplarSequence(),
                 debugContext("init_zero"));
    break;
  }
  case InitType::NoInit: {
    prog.getPoplarSequence().add(snap::program::WriteUndef(outTensor));
    break;
  }
  default:
    throw error("[InitOpx] Unexpected InitType.");
    // NOTE: Unreachable code. Commenting out the break so that we can enable
    // the warning so in future we can catch cases where code is unexpectedly
    // unreachable. break;
  }
}

namespace {
OpxCreator<InitOpx> InitOpxCreator(Onnx::CustomOperators::Init_1);
} // namespace
} // namespace popx
} // namespace popart
