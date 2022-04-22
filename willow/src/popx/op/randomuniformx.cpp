// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <vector>
#include <poplar/VariableMappingMethod.hpp>
#include <poprand/RandomGen.hpp>
#include <popart/op/randomuniform.hpp>
#include <popart/popx/op/randomuniformx.hpp>

#include "popart/operators.hpp"
#include "popart/popx/devicex.hpp"
#include "popart/popx/opxmanager.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/util.hpp"

namespace popart {
class Op;

namespace popx {

RandomUniformOpx::RandomUniformOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<RandomUniformOp>(op, Onnx::Operators::RandomUniform_1);
}

void RandomUniformOpx::grow(snap::program::Sequence &prog) const {
  auto &op        = getOp<RandomUniformOp>();
  auto outputInfo = op.outInfo(op.getOutIndex());
  auto shape      = vXtoY<int64_t, std::size_t>(outputInfo.shape());
  auto poplarType = popType(op.outInfo(op.getOutIndex()));

  auto refTensor = graph().addVariable(poplarType,
                                       shape,
                                       poplar::VariableMappingMethod::LINEAR,
                                       debugContext("refTensor"));

  auto output =
      poprand::uniform(graph().getPoplarGraph(),
                       &getInTensor(op.getSeedInIndex()).getPoplarTensor(),
                       0u,
                       refTensor.getPoplarTensor(),
                       poplarType,
                       op.getLow(),
                       op.getHigh(),
                       prog.getPoplarSequence(),
                       debugContext());

  setOutTensor(op.getOutIndex(), snap::Tensor{output, graph()});
}

namespace {
OpxCreator<RandomUniformOpx>
    randomUniformOpxCreator(Onnx::Operators::RandomUniform_1);
}

} // namespace popx
} // namespace popart
