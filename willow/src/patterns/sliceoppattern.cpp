// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/subsample.hpp>

#include <popart/patterns/patterns.hpp>
#include <popart/patterns/sliceoppattern.hpp>
#include <popart/tensorindex.hpp>

#include "popart/basicoptionals.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/slicestruct.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {

bool SlicePattern::matches(Op *op) const {
  if (!op->isConvertibleTo<SliceOp>()) {
    return false;
  }
  auto sliceOp = dynamic_cast<SliceOp *>(op);
  auto steps   = sliceOp->getSteps();
  for (auto step : steps) {
    if (std::abs(step) > 1) {
      return true;
    }
  }
  return false;
}

bool SlicePattern::apply(Op *op) const {
  auto sliceOp      = dynamic_cast<SliceOp *>(op);
  const auto slices = sliceOp->getSlices();
  auto steps        = sliceOp->getSteps();

  auto sliceIn  = op->inTensor(SliceOp::getInIndex());
  auto sliceOut = op->outTensor(SliceOp::getOutIndex());

  std::vector<int64_t> newAxes, newStarts, newEnds, newSteps;
  // old slice output shape
  auto inShape = sliceOp->inInfo(SliceOp::getInIndex()).shape();

  for (int i = 0; i < steps.size(); i++) {
    bool flip   = steps[i] > 0 ? false : true;
    int64_t axe = slices[i].axis;
    if (axe < 0) {
      axe = axe + inShape.size();
    }
    int64_t start = slices[i].start;
    int64_t end   = slices[i].end;

    if (flip) {
      // 1. recover data transformed in slice 'op::getSlices' func
      // 2. use negative number to express the index
      int64_t temp = start;
      start        = end - 1 - inShape[axe];
      end          = temp - 1 - inShape[axe];
    }

    newAxes.push_back(axe);
    newStarts.push_back(start);
    newEnds.push_back(end);
    newSteps.push_back(steps[i] > 0 ? 1 : -1);
  }

  // cal strides
  std::vector<int64_t> strides(inShape.size(), 1);
  for (int i = 0; i < steps.size(); i++) {
    strides[newAxes[i]] = std::abs(steps[i]);
  }

  auto &graph = op->getGraph();
  auto &ir    = graph.getIr();
  sliceOp->disconnectAllInputs();
  sliceOp->disconnectAllOutputs();

  auto newSliceOp = std::make_unique<SliceOp>(Onnx::Operators::Slice_11,
                                              newStarts,
                                              newEnds,
                                              newAxes,
                                              newSteps,
                                              op->settings);
  auto newslice   = newSliceOp.get();
  transferBaseProperties(sliceOp, newslice);
  graph.moveIntoGraph(std::move(newSliceOp));
  // connect slice in out
  newslice->createAndConnectOutTensor(
      SliceOp::getOutIndex(), ir.createIntermediateTensorId(sliceOut->id));
  newslice->connectInTensor(SliceOp::getInIndex(), sliceIn->id);

  auto subSampleOp = std::make_unique<SubsampleOp>(
      Onnx::CustomOperators::Subsample_1,
      strides,
      Op::Settings(
          graph, sliceOp->name() + "_" + "Subsample", op->debugInfo.getId()));
  auto subsample = subSampleOp.get();
  transferBaseProperties(sliceOp, subsample);
  graph.moveIntoGraph(std::move(subSampleOp));
  auto newSliceOut = newslice->outTensor(SliceOp::getOutIndex());
  subsample->connectInTensor(SubsampleOp::getInIndex(), newSliceOut->id);
  subsample->connectOutTensor(SubsampleOp::getOutIndex(), sliceOut->id);

  newslice->setup();
  subsample->setup();

  // erase original slice op
  graph.eraseOp(op->id);

  return true;
}

// Disabled by default
namespace {
static PatternCreator<SlicePattern> slicePattern("Slice2SliceSubsample", false);
} // namespace

} // namespace popart
