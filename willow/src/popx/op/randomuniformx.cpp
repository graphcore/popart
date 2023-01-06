// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ext/new_allocator.h>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/VariableMappingMethod.hpp>
#include <poprand/RandomGen.hpp>
#include <popart/op/randomuniform.hpp>
#include <popart/popx/op/randomuniformx.hpp>

#include "popart/operators.hpp"
#include "popart/popx/devicex.hpp"
#include "popart/popx/opx.hpp"
#include "popart/popx/opxmanager.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/util.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {

RandomUniformOpx::RandomUniformOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<RandomUniformOp>(op, Onnx::Operators::RandomUniform_1);
}

void RandomUniformOpx::grow(poplar::program::Sequence &prog) const {
  auto &op        = getOp<RandomUniformOp>();
  auto outputInfo = op.outInfo(op.getOutIndex());
  auto shape      = vXtoY<int64_t, std::size_t>(outputInfo.shape());
  auto poplarType = popType(op.outInfo(op.getOutIndex()));

  auto refTensor = graph().addVariable(poplarType,
                                       shape,
                                       poplar::VariableMappingMethod::LINEAR,
                                       debugContext("refTensor"));

  auto output = poprand::uniform(graph(),
                                 &getInTensor(op.getSeedInIndex()),
                                 0u,
                                 refTensor,
                                 poplarType,
                                 op.getLow(),
                                 op.getHigh(),
                                 prog,
                                 debugContext());

  setOutTensor(op.getOutIndex(), output);
}

namespace {
OpxCreator<RandomUniformOpx>
    randomUniformOpxCreator(Onnx::Operators::RandomUniform_1);
}

} // namespace popx
} // namespace popart
