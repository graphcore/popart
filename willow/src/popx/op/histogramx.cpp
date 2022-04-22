// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/poputil/TileMapping.hpp>
#include <vector>
#include <poplar/ArrayRef.hpp>
#include <popops/GatherStatistics.hpp>
#include <popart/op/histogram.hpp>
#include <popart/popx/op/histogramx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/popopx.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;

void HistogramOpx::grow(snap::program::Sequence &prog) const {
  auto &op    = getOp<HistogramOp>();
  auto levels = op.getLevels();

  auto levelsT = graph().addConstant(getInTensor(op.getInIndex()).elementType(),
                                     {levels.size()},
                                     poplar::ArrayRef<float>(levels),
                                     debugContext("levels"));
  snap::poputil::mapTensorLinearly(graph(), levelsT);

  auto out = popops::histogram(
      graph().getPoplarGraph(),
      getInTensor(op.getInIndex()).flatten().getPoplarTensor(),
      levelsT.getPoplarTensor(),
      op.getAbsoluteOfInput(),
      prog.getPoplarSequence(),
      debugContext());

  setOutTensor(op.getOutIndex(), snap::Tensor{out, graph()});
}

HistogramOpx::HistogramOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<HistogramOp>(op);
}

namespace {
OpxCreator<HistogramOpx> histogramOpxCreator(Onnx::CustomOperators::Histogram);
} // namespace

} // namespace popx
} // namespace popart
