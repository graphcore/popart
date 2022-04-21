// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popnn/Loss.hpp>

#include <popart/op/argmin.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/argminx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

snap::Tensor ArgMinOpx::extremaOp(snap::program::Sequence &prog,
                                  const snap::Tensor &input) const {
  return snap::Tensor{popnn::argMin(graph().getPoplarGraph(),
                                    input.getPoplarTensor(),
                                    prog.getPoplarSequence(),
                                    debugContext("argmin")),
                      graph()};
}

namespace {
OpxCreator<ArgMinOpx> argMinOpxCreator({Onnx::Operators::ArgMin_1,
                                        Onnx::Operators::ArgMin_11});
} // namespace

} // namespace popx
} // namespace popart
