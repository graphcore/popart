// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <gcl/Collectives.hpp>
#include <memory>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popart/popx/op/collectives/allreducex.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/region.hpp" // IWYU pragma: keep
#include "popart/tensorindex.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class AllReduceOp;

namespace popx {
class Devicex;

AllReduceOpx::AllReduceOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<AllReduceOp>(op,
                        {Onnx::CustomOperators::AllReduce,
                         Onnx::CustomGradOperators::AllReduceGrad});

  numInputs = op->input->n();
}

void AllReduceOpx::grow(poplar::program::Sequence &prog) const {
  std::vector<poplar::Tensor> inputTensors;
  for (int i = 0; i < numInputs; i++) {
    inputTensors.push_back(getInTensor(i).expand({0}));
  }

  auto reduceInput  = poplar::concat(inputTensors, 0);
  auto reduceOutput = gcl::allReduceWithinReplica(topLevelGraph(),
                                                  reduceInput,
                                                  gcl::CollectiveOperator::ADD,
                                                  prog,
                                                  "all_reduce");

  // Split result (each split is a copy on a different IPU)
  for (int i = 0; i < numInputs; i++) {
    auto t = reduceOutput.slice(i, i + 1, 0).squeeze({0});
    setOutTensor(i, t);
  }
}

InputCreatorType AllReduceOpx::getInputCreatorType(InIndex index) const {
  return InputCreatorType::CanUnwind;
}

poplar::Tensor AllReduceOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                InIndex,
                                                OutIndex) const {
  return tensor;
}

view::RegMap AllReduceOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

namespace {
OpxCreator<AllReduceOpx>
    allReduceOpxCreator({Onnx::CustomOperators::AllReduce});
OpxCreator<AllReduceOpx>
    allReduceGradOpxCreator(Onnx::CustomGradOperators::AllReduceGrad);
} // namespace

} // namespace popx
} // namespace popart
