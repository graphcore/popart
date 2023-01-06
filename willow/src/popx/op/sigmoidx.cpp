// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <string>
#include <poplar/Tensor.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popart/op/sigmoid.hpp>
#include <popart/popx/op/sigmoidx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace poplar {
class Graph;
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

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

poplar::Tensor SigmoidComputex::outplace(poplar::program::Sequence &p,
                                         poplar::Graph &g,
                                         const poplar::Tensor &t,
                                         const poplar::DebugNameAndId &dnai,
                                         const std::string &s) const {
  auto outTensor = cloneNcopy(p, g, t, dnai);
  inplace(p, g, outTensor, dnai, s);
  return outTensor;
}

void SigmoidComputex::inplace(poplar::program::Sequence &p,
                              poplar::Graph &g,
                              const poplar::Tensor &t,
                              const poplar::DebugNameAndId &dnai,
                              const std::string &s) const {

  // apply the inplace SIGMOID
  popnn::nonLinearityInPlace(
      g, popnn::NonLinearityType::SIGMOID, t, p, {dnai, s});
}

SigmoidGradOpx::SigmoidGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SigmoidGradOp>(op, Onnx::GradOperators::SigmoidGrad);
}

void SigmoidGradOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = popnn::nonLinearityInputGradient(
      graph(),                                        // graph,
      popnn::NonLinearityType::SIGMOID,               // nonLinearityType,
      getInTensor(SigmoidGradOp::getFwdOutInIndex()), // out,
      getInTensor(SigmoidGradOp::getGradInIndex()),   // outGradient,
      prog,
      debugContext() // debugContext
  );

  setOutTensor(SigmoidOp::getOutIndex(), outTensor);
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
