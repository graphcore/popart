// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popnn/Loss.hpp>

#include <popart/error.hpp>
#include <popart/op/argmax.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/argmaxx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

poplar::Tensor ArgMaxOpx::extremaOp(poplar::program::Sequence &prog,
                                    const poplar::Tensor &input) const {
  return popnn::argMax(
      graph().getPoplarGraph(), input, prog, debugContext("argmax"));
}

namespace {
OpxCreator<ArgMaxOpx> argMaxOpxCreator({Onnx::Operators::ArgMax_1,
                                        Onnx::Operators::ArgMax_11});
} // namespace

} // namespace popx
} // namespace popart
