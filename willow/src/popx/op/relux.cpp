// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/relu.hpp>
#include <popart/popx/op/relux.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include <popnn/NonLinearity.hpp>

namespace popart {
namespace popx {

ReluInplaceOpx::ReluInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, ReluComputex::get()) {
  verifyOp<ReluInplaceOp>(op, Onnx::CustomOperators::ReluInplace);
}

ReluOpx::ReluOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, ReluComputex::get()) {
  verifyOp<ReluOp>(op, Onnx::Operators::Relu_6);
}

snap::Tensor ReluComputex::outplace(snap::program::Sequence &p,
                                    snap::Graph &g,
                                    const snap::Tensor &t,
                                    const poplar::DebugNameAndId &dnai,
                                    const std::string &s) const {
  auto outTensor = cloneNcopy(p, g, t, dnai);
  inplace(p, g, outTensor, dnai, s);
  return outTensor;
}

void ReluComputex::inplace(snap::program::Sequence &p,
                           snap::Graph &g,
                           const snap::Tensor &t,
                           const poplar::DebugNameAndId &dnai,
                           const std::string &s) const {

  // apply the inplace RELU
  popnn::nonLinearityInPlace(g.getPoplarGraph(),
                             popnn::NonLinearityType::RELU,
                             t.getPoplarTensor(),
                             p.getPoplarSequence(),
                             {dnai, s});
}

ReluGradOpx::ReluGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ReluGradOp>(op, Onnx::GradOperators::ReluGrad);
}

void ReluGradOpx::grow(snap::program::Sequence &prog) const {

  ReluGradOp &rgop = getOp<ReluGradOp>();

  auto outTensor = popnn::nonLinearityInputGradient(
      graph().getPoplarGraph(),      // graph,
      popnn::NonLinearityType::RELU, // nonLinearityType,
      getInTensor(rgop.getReludInIndex()).getPoplarTensor(),     // out,
      getInTensor(rgop.getGradReludInIndex()).getPoplarTensor(), // outGradient,
      prog.getPoplarSequence(),                                  // prog,
      debugContext()                                             // debugContext
  );

  setOutTensor(0, snap::Tensor{outTensor, graph()});
}

namespace {
OpxCreator<ReluOpx> reluxOpxCreator(Onnx::Operators::Relu_6);
OpxCreator<ReluInplaceOpx>
    reluxInplaceOpxCreator(Onnx::CustomOperators::ReluInplace);
OpxCreator<ReluGradOpx> reluxGradOpxCreator(Onnx::GradOperators::ReluGrad);
} // namespace

} // namespace popx
} // namespace popart
