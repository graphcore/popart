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

poplar::Tensor ReluComputex::outplace(poplar::program::Sequence &p,
                                      poplar::Graph &g,
                                      const poplar::Tensor &t,
                                      const std::string &s) const {
  auto outTensor = cloneNcopy(p, g, t);
  inplace(p, g, outTensor, s);
  return outTensor;
}

void ReluComputex::inplace(poplar::program::Sequence &p,
                           poplar::Graph &g,
                           const poplar::Tensor &t,
                           const std::string &s) const {

  // apply the inplace RELU
  popnn::nonLinearityInPlace(g, popnn::NonLinearityType::RELU, t, p, s);
}

ReluGradOpx::ReluGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ReluGradOp>(op, Onnx::GradOperators::ReluGrad);
}

void ReluGradOpx::grow(poplar::program::Sequence &prog) const {

  ReluGradOp &rgop = getOp<ReluGradOp>();

  auto outTensor = popnn::nonLinearityInputGradient(
      graph(),                                 // graph,
      popnn::NonLinearityType::RELU,           // nonLinearityType,
      getInTensor(rgop.getReludInIndex()),     // out,
      getInTensor(rgop.getGradReludInIndex()), // outGradient,
      prog,                                    // prog,
      debugPrefix()                            // debugPrefix
  );

  setOutTensor(0, outTensor);
}

namespace {
OpxCreator<ReluOpx> reluxOpxCreator(Onnx::Operators::Relu_6);
OpxCreator<ReluInplaceOpx>
    reluxInplaceOpxCreator(Onnx::CustomOperators::ReluInplace);
OpxCreator<ReluGradOpx> reluxGradOpxCreator(Onnx::GradOperators::ReluGrad);
} // namespace

} // namespace popx
} // namespace popart
