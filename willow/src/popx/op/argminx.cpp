// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <vector>
#include <popnn/Loss.hpp>
#include <popart/popx/op/argminx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"

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
