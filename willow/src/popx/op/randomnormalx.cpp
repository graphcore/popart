// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprand/RandomGen.hpp>
#include <popart/op/randomnormal.hpp>
#include <popart/popx/op/randomnormalx.hpp>

namespace popart {
namespace popx {

RandomNormalOpx::RandomNormalOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<RandomNormalOp>(op, Onnx::Operators::RandomNormal_1);
}

void RandomNormalOpx::grow(poplar::program::Sequence &prog) const {
  auto &op        = getOp<RandomNormalOp>();
  auto outputInfo = op.outInfo(op.getOutIndex());
  auto shape      = vXtoY<int64_t, std::size_t>(outputInfo.shape());
  auto poplarType = popType(op.outInfo(op.getOutIndex()));

  auto refTensor = graph().addVariable(poplarType,
                                       shape,
                                       poplar::VariableMappingMethod::LINEAR,
                                       debugPrefix("refTensor"));

  auto output = poprand::normal(graph(),
                                &getInTensor(op.getSeedInIndex()),
                                op.getSeedModifier(),
                                refTensor,
                                poplarType,
                                op.getMean(),
                                op.getScale(),
                                prog);

  setOutTensor(op.getOutIndex(), output);
}

namespace {
OpxCreator<RandomNormalOpx>
    randomNormalOpxCreator(Onnx::Operators::RandomNormal_1);
}

} // namespace popx
} // namespace popart
