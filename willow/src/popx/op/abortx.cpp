// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/abort.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/abortx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <poplar/Program.hpp>

#include <snap/Program.hpp>

namespace popart {
namespace popx {

AbortOpx::AbortOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<AbortOp>(op, Onnx::CustomOperators::Abort);
}

void AbortOpx::grow(snap::program::Sequence &prog) const {
  if (hasInput(AbortOp::getInIndex())) {
    auto condition = getInTensor(AbortOp::getInIndex());
    prog.add(snap::program::AbortOnCondition(condition));
  } else {
    prog.add(snap::program::Abort(graph()));
  }
}

namespace {
OpxCreator<AbortOpx> AbortxCreator(Onnx::CustomOperators::Abort);
} // namespace

} // namespace popx
} // namespace popart
