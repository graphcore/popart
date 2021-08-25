// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/split.hpp>
#include <popart/patterns/splitoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <numeric>

namespace popart {

namespace {

std::vector<Slice> calculateSlices(const SplitOp &splitOp) {
  auto inputShape = splitOp.inShape(SplitOp::getInIndex());
  auto splitAxis  = splitOp.getAxis();

  int64_t end = 0;
  std::vector<Slice> slices;
  for (auto splitSize : splitOp.getSplitSizes()) {
    auto start = end;
    end        = start + splitSize;
    slices.push_back({start, end, splitAxis});
  }

  return slices;
}

} // namespace

bool SplitOpPattern::matches(Op *op) const {
  // If running a training session. Make sure the splitGradOp has been grown
  // before removing the fwd op (SplitOp) from the IR so
  // splitgradoptoconcatpattern can run.
  if (op->getIr().canTrain() && !op->getIr().hasConstructedBackwards()) {
    return false;
  }
  return op->isConvertibleTo<SplitOp>();
}

std::vector<const Tensor *> SplitOpPattern::touches(Op *) const { return {}; }

bool SplitOpPattern::apply(Op *op) const {
  // isConvertibleTo checked in SplitOpPattern::matches method
  auto splitOp = dynamic_cast<SplitOp *>(op);
  auto slices  = calculateSlices(*splitOp);

  auto inputTensor = op->inTensor(SplitOp::getInIndex());
  auto inputShape  = inputTensor->info.shape();
  auto outputs     = op->output->tensorMap();

  std::vector<int64_t> axes(inputShape.size(), 0);
  std::iota(axes.begin(), axes.end(), 0);

  splitOp->disconnectAllInputs();
  splitOp->disconnectAllOutputs();

  for (auto &i_tensor : outputs) {
    auto idx       = i_tensor.first;
    auto outTensor = i_tensor.second;

    auto slice = slices.at(idx);

    std::vector<int64_t> starts(inputShape.size(), 0);
    starts.at(slice.axis) = slice.start;

    std::vector<int64_t> ends = inputShape;
    ends.at(slice.axis)       = slice.end;

    auto sOp     = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Slice, op);
    auto sliceOp = dynamic_cast<SliceOp *>(sOp);
    sliceOp->setStarts(starts);
    sliceOp->setEnds(ends);
    sliceOp->setAxes(axes);
    sliceOp->setSteps(std::vector<int64_t>(axes.size(), 1));
    sliceOp->connectInTensor(SliceOp::getInIndex(), inputTensor->id);
    sliceOp->connectOutTensor(SliceOp::getOutIndex(), outTensor->id);
    sliceOp->setup();
  }

  splitOp->getGraph().eraseOp(splitOp->id);
  return true;
}

namespace {
static PatternCreator<SplitOpPattern> splitOpPattern("SplitOp");
}

} // namespace popart
