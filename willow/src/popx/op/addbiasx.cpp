// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <poplin/Convolution.hpp>
#include <popops/Reduce.hpp>
#include <popart/error.hpp>
#include <popart/op/addbias.hpp>
#include <popart/popx/op/addbiasx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

AddBiasOpx::AddBiasOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<AddBiasOp>(op);
}

AddBiasDataGradOpx::AddBiasDataGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<AddBiasDataGradOp>(op, Onnx::CustomGradOperators::AddBiasDataGrad);
}

AddBiasInplaceOpx::AddBiasInplaceOpx(Op *op, Devicex *devicex)
    : AddBiasOpx(op, devicex) {
  verifyOp<AddBiasInplaceOp>(op, Onnx::CustomOperators::AddBiasInplace);
}

void AddBiasOpx::grow(poplar::program::Sequence &prog) const {
  // Clone & copy the input tensor because poplin::addBias is in-place.
  const auto result =
      PopOpx::cloneNcopy(prog, getInTensor(AddBiasOp::getDataInIndex()));
  poplin::addBias(graph().getPoplarGraph(),
                  result.getPoplarTensor(),
                  getInTensor(AddBiasOp::getBiasInIndex()).getPoplarTensor(),
                  prog,
                  debugContext());
  setOutTensor(AddBiasOp::getOutIndex(), result);
}

void AddBiasDataGradOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0, PopOpx::cloneNcopy(prog, getInTensor(0)));
}

std::set<TensorId> AddBiasOpx::mustExistBeforeCreate(InIndex index) const {
  if (index != AddBiasOp::getBiasInIndex()) {
    throw error("AddBiasOpx::mustExistBeforeCreate : Invalid index = " +
                std::to_string(index));
  }

  return {inId(AddBiasOp::getDataInIndex())};
}

InputCreatorType AddBiasOpx::getInputCreatorType(InIndex index) const {
  return index == AddBiasOp::getBiasInIndex() ? InputCreatorType::CanCreate
                                              : InputCreatorType::Deadend;
}

snap::Tensor
AddBiasOpx::createInputTensor(InIndex index,
                              const poplar::DebugNameAndId &dnai) const {
  if (index != AddBiasOp::getBiasInIndex()) {
    throw error("AddBiasOpx::createInput : Invalid index = " +
                std::to_string(index));
  }

  return snap::Tensor{
      poplin::createBiases(
          graph().getPoplarGraph(),
          getInTensor(AddBiasOp::getDataInIndex()).getPoplarTensor(),
          dnai),
      graph()};
}

AddBiasBiasGradOpx::AddBiasBiasGradOpx(Op *op, Devicex *devicex)
    : ReduceSumOpx(op, devicex) {
  verifyOp<AddBiasBiasGradOp>(op, Onnx::CustomGradOperators::AddBiasBiasGrad);
}

void AddBiasInplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto dataIn = getInTensor(AddBiasOp::getDataInIndex()).getPoplarTensor();
  auto biasIn = getInTensor(AddBiasOp::getBiasInIndex()).getPoplarTensor();
  poplin::addBias(
      graph().getPoplarGraph(), dataIn, biasIn, prog, debugContext());
  setOutTensor(AddBiasOp::getOutIndex(), snap::Tensor{dataIn, graph()});
}

namespace {
OpxCreator<AddBiasOpx> addBiasOpxCreator(Onnx::CustomOperators::AddBias);
OpxCreator<AddBiasInplaceOpx>
    addBiasInplaceOpxCreator(Onnx::CustomOperators::AddBiasInplace);
OpxCreator<AddBiasBiasGradOpx>
    addBiasBiasGradOpxCreator(Onnx::CustomGradOperators::AddBiasBiasGrad);
OpxCreator<AddBiasDataGradOpx>
    addBiasDataGradOpxCreator(Onnx::CustomGradOperators::AddBiasDataGrad);
} // namespace

} // namespace popx
} // namespace popart
