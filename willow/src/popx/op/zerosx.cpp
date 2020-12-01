// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/op/zeros.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/zerosx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

ZerosOpx::ZerosOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ZerosOp>(op, Onnx::CustomOperators::Zeros_1);
}

void ZerosOpx::grow(poplar::program::Sequence &) const {
  auto &op = getOp<ZerosOp>();

  auto outputInfo = op.outInfo(op.getOutIndex());

  auto poplarType = popType(op.outInfo(op.getOutIndex()));
  auto shape      = vXtoY<int64_t, std::size_t>(outputInfo.shape());

  auto zeros = graph().addConstant(poplarType, shape, 0, debugPrefix("zeros"));
  graph().setTileMapping(zeros, 0);

  setOutTensor(ZerosOp::getOutIndex(), zeros);
}

namespace {
OpxCreator<ZerosOpx> zerosOpxCreator(Onnx::CustomOperators::Zeros_1);
} // namespace

} // namespace popx
} // namespace popart
