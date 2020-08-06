// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/op/resize.hpp>
#include <popart/op/upsample.hpp>
#include <popart/patterns/upsampletoresizepattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

namespace {
ResizeMode upsampleToResizeMode(const UpsampleMode &um) {
  switch (um) {
  case UpsampleMode::Nearest:
    return ResizeMode::Nearest;
    break;
  case UpsampleMode::Linear:
    return ResizeMode::Linear;
    break;
  case UpsampleMode::N:
    return ResizeMode::N;
    break;
  default:
    throw error("Bad UpsampleMode '{}'", static_cast<int>(um));
    break;
  }
}
} // namespace

bool UpsampleToResizePattern::matches(Op *op) const {
  return op->isConvertibleTo<UpsampleOp>();
}

std::vector<const Tensor *> UpsampleToResizePattern::touches(Op *) const {
  return {};
}

bool UpsampleToResizePattern::apply(Op *op) const {
  auto &graph = op->getGraph();

  // matches must have verified the correctness before this call
  auto upsample = static_cast<UpsampleOp *>(op);

  auto in  = op->inTensor(UpsampleOp::getInIndex());
  auto out = op->outTensor(UpsampleOp::getOutIndex());

  auto resize2 = std::make_unique<ResizeOp>(
      Onnx::AiOnnx::OpSet10::Resize,
      Op::Settings(graph, upsample->name() + "_" + "Resize"),
      upsampleToResizeMode(upsample->getMode()),
      upsample->getScales());

  auto resize = resize2.get();
  transferBaseProperties(upsample, resize);
  graph.moveIntoGraph(std::move(resize2));

  // Remove the UpsampleOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // Connect up the new ops
  resize->connectInTensor(ResizeOp::getInIndex(), in->id);
  resize->connectOutTensor(ResizeOp::getOutIndex(), out->id);
  resize->setup();

  return true;
}

namespace {
static PatternCreator<popart::UpsampleToResizePattern>
    UpsampleToResizePattern(PreAliasPatternType::UpsampleToResize,
                            "UpsampleToResize");
}

} // namespace popart
