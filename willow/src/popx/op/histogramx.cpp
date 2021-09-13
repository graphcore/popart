// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op/histogram.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/histogramx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/GatherStatistics.hpp>

namespace popart {
namespace popx {

void HistogramOpx::grow(snap::program::Sequence &prog) const {
  auto &op    = getOp<HistogramOp>();
  auto levels = op.getLevels();

  auto levelsT = graph().getPoplarGraph().addConstant(
      getInTensor(op.getInIndex()).elementType(),
      {levels.size()},
      poplar::ArrayRef<float>(levels),
      debugContext("levels"));
  poputil::mapTensorLinearly(graph().getPoplarGraph(), levelsT);

  auto out = popops::histogram(
      graph().getPoplarGraph(),
      getInTensor(op.getInIndex()).flatten().getPoplarTensor(),
      levelsT,
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
