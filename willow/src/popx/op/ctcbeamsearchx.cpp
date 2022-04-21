// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>

#include <popnn/CTCInference.hpp>
#include <popart/ir.hpp>
#include <popart/op/ctcbeamsearch.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/ctcbeamsearchx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {
CtcBeamSearchDecoderOpx::CtcBeamSearchDecoderOpx(Op *op_, Devicex *devicex)
    : PopOpx(op_, devicex), plan(std::make_unique<popnn::ctc::Plan>()) {
  verifyOp<CtcBeamSearchDecoderOp>(op_,
                                   Onnx::CustomOperators::CtcBeamSearchDecoder);

  const auto &op = getOp<CtcBeamSearchDecoderOp>();
  auto logProbsTensor =
      op.input->tensor(CtcBeamSearchDecoderOp::getLogProbsInIndex());
  auto inType = popType(logProbsTensor->info.getDataTypeInfo()->type());

  *plan = popnn::ctc_infer::plan(graph().getPoplarGraph(),
                                 inType,
                                 op.getBatchSize(),
                                 op.getMaxTime(),
                                 op.getNumClasses(),
                                 op.getBeamWidth());
}

CtcBeamSearchDecoderOpx::~CtcBeamSearchDecoderOpx() = default;

void CtcBeamSearchDecoderOpx::grow(snap::program::Sequence &prog) const {
  const auto &op = getOp<CtcBeamSearchDecoderOp>();
  const auto &logProbs =
      getInTensor(CtcBeamSearchDecoderOp::getLogProbsInIndex())
          .getPoplarTensor();
  const auto &dataLengths =
      getInTensor(CtcBeamSearchDecoderOp::getDataLengthsInIndex())
          .getPoplarTensor();

  auto result = popnn::ctc_infer::beamSearchDecoderLogProbabilities(
      graph().getPoplarGraph(),
      logProbs,
      dataLengths,
      prog.getPoplarSequence(),
      op.getBlankClass(),
      op.getBeamWidth(),
      op.getTopPaths(),
      *plan,
      debugContext("ctcBeamSearchDecoder"));

  setOutTensor(CtcBeamSearchDecoderOp::getLabelProbsOutIndex(),
               snap::Tensor{std::get<0>(result), graph()});
  setOutTensor(CtcBeamSearchDecoderOp::getLabelLengthsOutIndex(),
               snap::Tensor{std::get<1>(result), graph()});
  setOutTensor(CtcBeamSearchDecoderOp::getDecodedLabelsOutIndex(),
               snap::Tensor{std::get<2>(result), graph()});
}

// Opx creator.
namespace {
static popart::popx::OpxCreator<CtcBeamSearchDecoderOpx>
    CtcBeamSearchDecoderOpxCreator(
        {Onnx::CustomOperators::CtcBeamSearchDecoder});
} // namespace
} // namespace popx
} // namespace popart
