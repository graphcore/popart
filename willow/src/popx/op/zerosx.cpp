// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <snap/Graph.hpp>
#include <snap/Tensor.hpp>
#include <vector>
#include <poplar/Graph.hpp>
#include <popart/op/zeros.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/zerosx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/util.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {

ZerosOpx::ZerosOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ZerosOp>(op, Onnx::CustomOperators::Zeros_1);
}

void ZerosOpx::grow(snap::program::Sequence &) const {
  auto &op = getOp<ZerosOp>();

  auto outputInfo = op.outInfo(op.getOutIndex());

  auto poplarType = popType(op.outInfo(op.getOutIndex()));
  auto shape      = vXtoY<int64_t, std::size_t>(outputInfo.shape());

  auto zeros = graph().addConstant(poplarType, shape, 0, debugContext("zeros"));

  setOutTensor(ZerosOp::getOutIndex(), zeros);
}

namespace {
OpxCreator<ZerosOpx> zerosOpxCreator(Onnx::CustomOperators::Zeros_1);
} // namespace

} // namespace popx
} // namespace popart
