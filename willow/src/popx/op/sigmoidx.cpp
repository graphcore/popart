// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "popart/popx/debugcontextx.hpp"
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <popnn/NonLinearity.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popart/op/sigmoid.hpp>
#include <popart/popx/op/sigmoidx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;

SigmoidInplaceOpx::SigmoidInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, SigmoidComputex::get()) {
  verifyOp<SigmoidInplaceOp>(op, Onnx::CustomOperators::SigmoidInplace);
}

SigmoidOpx::SigmoidOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, SigmoidComputex::get()) {
  verifyOp<SigmoidOp>(op, Onnx::Operators::Sigmoid_6);
}

snap::Tensor SigmoidComputex::outplace(snap::program::Sequence &p,
                                       snap::Graph &g,
                                       const snap::Tensor &t,
                                       const poplar::DebugNameAndId &dnai,
                                       const std::string &s) const {
  auto outTensor = cloneNcopy(p, g, t, dnai);
  inplace(p, g, outTensor, dnai, s);
  return outTensor;
}

void SigmoidComputex::inplace(snap::program::Sequence &p,
                              snap::Graph &g,
                              const snap::Tensor &t,
                              const poplar::DebugNameAndId &dnai,
                              const std::string &s) const {

  // apply the inplace SIGMOID
  popnn::nonLinearityInPlace(g.getPoplarGraph(),
                             popnn::NonLinearityType::SIGMOID,
                             t.getPoplarTensor(),
                             p.getPoplarSequence(),
                             {dnai, s});
}

SigmoidGradOpx::SigmoidGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SigmoidGradOp>(op, Onnx::GradOperators::SigmoidGrad);
}

void SigmoidGradOpx::grow(snap::program::Sequence &prog) const {
  auto outTensor = popnn::nonLinearityInputGradient(
      graph().getPoplarGraph(),         // graph,
      popnn::NonLinearityType::SIGMOID, // nonLinearityType,
      getInTensor(SigmoidGradOp::getFwdOutInIndex()).getPoplarTensor(), // out,
      getInTensor(SigmoidGradOp::getGradInIndex())
          .getPoplarTensor(),   // outGradient,
      prog.getPoplarSequence(), // prog,
      debugContext()            // debugContext
  );

  setOutTensor(SigmoidOp::getOutIndex(), snap::Tensor{outTensor, graph()});
}

namespace {
OpxCreator<SigmoidOpx> sigmoidOpxCreator(Onnx::Operators::Sigmoid_6);
OpxCreator<SigmoidGradOpx>
    sigmoidGradOpxCreator(Onnx::GradOperators::SigmoidGrad);
OpxCreator<SigmoidInplaceOpx>
    sigmoidxInplaceOpxCreator(Onnx::CustomOperators::SigmoidInplace);
} // namespace

} // namespace popx
} // namespace popart
