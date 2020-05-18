// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popops/Zero.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/init.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/initx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

InitOpx::InitOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<InitOp>(op, Onnx::CustomOperators::Init_1);
}

void InitOpx::grow(poplar::program::Sequence &prog) const {
  auto &initOp          = getOp<InitOp>();
  const auto &outTensor = getOutTensor(InitOp::getOutIndex());

  switch (initOp.getInitType()) {
  case InitType::Zero: {
    popops::zero(graph(), outTensor, prog, debugPrefix("init_zero"));
    break;
  }
  case InitType::NoInit:
    prog.add(poplar::program::WriteUndef(outTensor));
    break;
  default:
    throw error("[InitOpx] Unexpected InitType.");
    break;
  }
}

namespace {
OpxCreator<InitOpx> InitOpxCreator(Onnx::CustomOperators::Init_1);
} // namespace
} // namespace popx
} // namespace popart
