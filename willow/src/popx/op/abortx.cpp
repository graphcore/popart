// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/abort.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/abortx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <poplar/Program.hpp>

namespace popart {
namespace popx {

AbortOpx::AbortOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<AbortOp>(op, Onnx::CustomOperators::Abort);
}

void AbortOpx::grow(poplar::program::Sequence &prog) const {
  if (hasInput(AbortOp::getInIndex())) {
    poplar::Tensor condition =
        getInTensor(AbortOp::getInIndex()).getPoplarTensor();
    prog.add(poplar::program::AbortOnCondition(condition));
  } else {
    prog.add(poplar::program::Abort());
  }
}

namespace {
OpxCreator<AbortOpx> AbortxCreator(Onnx::CustomOperators::Abort);
} // namespace

} // namespace popx
} // namespace popart
