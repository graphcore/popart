// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popnn/Loss.hpp>

#include <popart/error.hpp>
#include <popart/op/argmax.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/argmaxx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

snap::Tensor ArgMaxOpx::extremaOp(snap::program::Sequence &prog,
                                  const snap::Tensor &input) const {
  return snap::Tensor{popnn::argMax(graph().getPoplarGraph(),
                                    input.getPoplarTensor(),
                                    prog.getPoplarSequence(),
                                    debugContext("argmax")),
                      graph()};
}

namespace {
OpxCreator<ArgMaxOpx> argMaxOpxCreator({Onnx::Operators::ArgMax_1,
                                        Onnx::Operators::ArgMax_11});
} // namespace

} // namespace popx
} // namespace popart
