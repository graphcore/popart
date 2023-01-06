// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <vector>
#include <poplar/Tensor.hpp>
#include <popart/op/split.hpp>
#include <popart/popx/op/splitx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

SplitOpx::SplitOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SplitOp>(op, {Onnx::Operators::Split_2, Onnx::Operators::Split_11});
}

void SplitOpx::grow(poplar::program::Sequence &prog) const {
  auto &splitOp     = getOp<SplitOp>();
  auto input        = getInTensor(SplitOp::getInIndex());
  unsigned int axis = static_cast<unsigned int>(splitOp.getAxis());

  unsigned int start = 0;
  auto splitSizes    = splitOp.getSplitSizes();
  for (int i = 0; i < splitSizes.size(); i++) {
    unsigned int end = start + static_cast<unsigned int>(splitSizes.at(i));
    auto t           = input.slice(start, end, axis);

    setOutTensor(i, cloneNcopy(prog, t));

    start = end;
  }
}

namespace {
OpxCreator<SplitOpx> splitOpxCreator({Onnx::Operators::Split_2,
                                      Onnx::Operators::Split_11});
} // namespace

} // namespace popx
} // namespace popart
