// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <vector>
#include <poplar/Tensor.hpp>
#include <popnn/Loss.hpp>
#include <popart/popx/op/argmaxx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
namespace popx {

poplar::Tensor ArgMaxOpx::extremaOp(poplar::program::Sequence &prog,
                                    const poplar::Tensor &input) const {
  return popnn::argMax(graph(), input, prog, debugContext("argmax"));
}

namespace {
OpxCreator<ArgMaxOpx> argMaxOpxCreator({Onnx::Operators::ArgMax_1,
                                        Onnx::Operators::ArgMax_11});
} // namespace

} // namespace popx
} // namespace popart
