// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>

#include <popart/logging.hpp>
#include <popart/op/prelu.hpp>
#include <popart/popx/op/prelux.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

namespace pe = popops::expr;

PReluOpx::PReluOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<PReluOp>(op);
}

void PReluOpx::grow(snap::program::Sequence &prog) const {
  auto inputPH = pe::_1;
  auto slopePH = pe::_2;

  // input < 0.0 ? input * slope : input;
  auto expression = pe::Select(
      pe::Mul(inputPH, slopePH), inputPH, pe::Lt(inputPH, pe::Const(0.0f)));

  auto result =
      popops::map(graph().getPoplarGraph(),
                  expression,
                  {getInTensor(PReluOp::getArg0InIndex()).getPoplarTensor(),
                   getInTensor(PReluOp::getArg1InIndex()).getPoplarTensor()},
                  prog.getPoplarSequence(),
                  debugContext("prelu"));

  setOutTensor(PReluOp::getOutIndex(), snap::Tensor{result, graph()});
}

namespace {
OpxCreator<PReluOpx> preluOpxCreator(Onnx::Operators::PRelu_9);
} // namespace

} // namespace popx
} // namespace popart
