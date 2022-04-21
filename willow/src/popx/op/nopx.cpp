// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/op/nop.hpp>
#include <popart/popx/op/nopx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {
namespace popx {

NopOpx::NopOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<NopOp>(op, Onnx::CustomOperators::Nop_1);
}

void NopOpx::grow(snap::program::Sequence &prog) const {
  auto input  = getInTensor(NopOp::getInIndex());
  auto output = cloneNcopy(prog, input);
  setOutTensor(NopOp::getOutIndex(), output);
}

namespace {
OpxCreator<NopOpx> nopOpxCreator(Onnx::CustomOperators::Nop_1);
} // namespace

} // namespace popx
} // namespace popart
