// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <vector>
#include <poplar/Tensor.hpp>
#include <popnn/Loss.hpp>
#include <popart/popx/op/argminx.hpp>
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

poplar::Tensor ArgMinOpx::extremaOp(poplar::program::Sequence &prog,
                                    const poplar::Tensor &input) const {
  return popnn::argMin(graph(), input, prog, debugContext("argmin"));
}

namespace {
OpxCreator<ArgMinOpx> argMinOpxCreator({Onnx::Operators::ArgMin_1,
                                        Onnx::Operators::ArgMin_11});
} // namespace

} // namespace popx
} // namespace popart
