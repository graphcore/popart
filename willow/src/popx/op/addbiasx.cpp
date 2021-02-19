// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <poplin/Convolution.hpp>
#include <popops/Reduce.hpp>
#include <popart/error.hpp>
#include <popart/op/addbias.hpp>
#include <popart/popx/op/addbiasx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

AddBiasOpx::AddBiasOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<AddBiasOp>(op);
}

AddBiasDataGradOpx::AddBiasDataGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<AddBiasDataGradOp>(op, Onnx::CustomGradOperators::AddBiasDataGrad);
}

AddBiasInplaceOpx::AddBiasInplaceOpx(Op *op, Devicex *devicex)
    : AddBiasOpx(op, devicex) {
  verifyOp<AddBiasInplaceOp>(op, Onnx::CustomOperators::AddBiasInplace);
}

void AddBiasOpx::grow(poplar::program::Sequence &prog) const {
  // Clone & copy the input tensor because poplin::addBias is in-place.
  const auto result =
      Opx::cloneNcopy(prog, getInTensor(AddBiasOp::getDataInIndex()));
  poplin::addBias(graph(),
                  result,
                  getInTensor(AddBiasOp::getBiasInIndex()),
                  prog,
                  debugContext());
  setOutTensor(AddBiasOp::getOutIndex(), result);
}

void AddBiasDataGradOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0, Opx::cloneNcopy(prog, getInTensor(0)));
}

std::vector<TensorId> AddBiasOpx::mustExistBeforeCreate(InIndex index) const {
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

poplar::Tensor
AddBiasOpx::createInput(InIndex index,
                        const poplar::DebugNameAndId &dnai) const {
  if (index != AddBiasOp::getBiasInIndex()) {
    throw error("AddBiasOpx::createInput : Invalid index = " +
                std::to_string(index));
  }

  return poplin::createBiases(
      graph(), getInTensor(AddBiasOp::getDataInIndex()), dnai);
}

bool AddBiasOpx::createsEquiv(int, const Opx *, int) const { return false; }

AddBiasBiasGradOpx::AddBiasBiasGradOpx(Op *op, Devicex *devicex)
    : ReduceSumOpx(op, devicex) {
  verifyOp<AddBiasBiasGradOp>(op, Onnx::CustomGradOperators::AddBiasBiasGrad);
}

void AddBiasInplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto dataIn = getInTensor(AddBiasOp::getDataInIndex());
  auto biasIn = getInTensor(AddBiasOp::getBiasInIndex());
  poplin::addBias(graph(), dataIn, biasIn, prog, debugContext());
  setOutTensor(AddBiasOp::getOutIndex(), dataIn);
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
