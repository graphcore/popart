// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <tuple>
#include <popnn/CTCInference.hpp>
#include <popnn/CTCPlan.hpp>
#include <popart/op/ctcbeamsearch.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/ctcbeamsearchx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/opx.hpp"
#include "popart/tensor.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
CtcBeamSearchDecoderOpx::CtcBeamSearchDecoderOpx(Op *op_, Devicex *devicex)
    : Opx(op_, devicex), plan(std::make_unique<popnn::ctc::Plan>()) {
  verifyOp<CtcBeamSearchDecoderOp>(op_,
                                   Onnx::CustomOperators::CtcBeamSearchDecoder);

  const auto &op = getOp<CtcBeamSearchDecoderOp>();
  auto logProbsTensor =
      op.input->tensor(CtcBeamSearchDecoderOp::getLogProbsInIndex());
  auto inType = popType(logProbsTensor->info.getDataTypeInfo()->type());

  *plan = popnn::ctc_infer::plan(graph(),
                                 inType,
                                 op.getBatchSize(),
                                 op.getMaxTime(),
                                 op.getNumClasses(),
                                 op.getBeamWidth());
}

CtcBeamSearchDecoderOpx::~CtcBeamSearchDecoderOpx() = default;

void CtcBeamSearchDecoderOpx::grow(poplar::program::Sequence &prog) const {
  const auto &op = getOp<CtcBeamSearchDecoderOp>();
  const auto &logProbs =
      getInTensor(CtcBeamSearchDecoderOp::getLogProbsInIndex());
  const auto &dataLengths =
      getInTensor(CtcBeamSearchDecoderOp::getDataLengthsInIndex());

  auto result = popnn::ctc_infer::beamSearchDecoderLogProbabilities(
      graph(),
      logProbs,
      dataLengths,
      prog,
      op.getBlankClass(),
      op.getBeamWidth(),
      op.getTopPaths(),
      *plan,
      debugContext("ctcBeamSearchDecoder"));

  setOutTensor(CtcBeamSearchDecoderOp::getLabelProbsOutIndex(),
               std::get<0>(result));
  setOutTensor(CtcBeamSearchDecoderOp::getLabelLengthsOutIndex(),
               std::get<1>(result));
  setOutTensor(CtcBeamSearchDecoderOp::getDecodedLabelsOutIndex(),
               std::get<2>(result));
}

// Opx creator.
namespace {
static popart::popx::OpxCreator<CtcBeamSearchDecoderOpx>
    CtcBeamSearchDecoderOpxCreator(
        {Onnx::CustomOperators::CtcBeamSearchDecoder});
} // namespace
} // namespace popx
} // namespace popart
