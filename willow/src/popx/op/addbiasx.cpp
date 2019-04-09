#include <poplin/Convolution.hpp>
#include <popops/Reduce.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/addbias.hpp>
#include <poponnx/popx/op/addbiasx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

AddBiasOpx::AddBiasOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<AddBiasOp>(op, Onnx::CustomOperators::AddBias);
}

AddBiasDataGradOpx::AddBiasDataGradOpx(Op *op, Devicex *devicex)
    : IdentityOpx(op, devicex) {
  verifyOp<AddBiasDataGradOp>(op, Onnx::CustomGradOperators::AddBiasDataGrad);
}

void AddBiasOpx::grow(poplar::program::Sequence &prog) const {
  // Clone & copy the input tensor because poplin::addBias is in-place.
  const auto result =
      Opx::cloneNcopy(prog, getInTensor(AddBiasOp::getDataInIndex()));
  poplin::addBias(
      graph(), result, getInTensor(AddBiasOp::getBiasInIndex()), prog, idStr());
  setOutTensor(AddBiasOp::getOutIndex(), result);
}

std::vector<TensorId> AddBiasOpx::mustExistBeforeCreate(InIndex index) const {
  if (index != AddBiasOp::getBiasInIndex()) {
    throw error("AddBiasOpx::mustExistBeforeCreate : Invalid index = " +
                std::to_string(index));
  }

  return {inId(AddBiasOp::getDataInIndex())};
}

InputCreatorType AddBiasOpx::getInputCreatorType(InIndex index) const {
  return index == AddBiasOp::getBiasInIndex() ? InputCreatorType::CANCREATE
                                              : InputCreatorType::DEADEND;
}

poplar::Tensor AddBiasOpx::createInput(InIndex index,
                                       const std::string &name) const {
  if (index != AddBiasOp::getBiasInIndex()) {
    throw error("AddBiasOpx::createInput : Invalid index = " +
                std::to_string(index));
  }

  return poplin::createBiases(
      graph(), getInTensor(AddBiasOp::getDataInIndex()), name);
}

bool AddBiasOpx::createsEquiv(int, Opx *, int) const { return false; }

AddBiasBiasGradOpx::AddBiasBiasGradOpx(Op *op, Devicex *devicex)
    : ReduceSumOpx(op, devicex) {
  verifyOp<AddBiasBiasGradOp>(op, Onnx::CustomGradOperators::AddBiasBiasGrad);
}

namespace {
OpxCreator<AddBiasOpx> addBiasOpxCreator(Onnx::CustomOperators::AddBias);
OpxCreator<AddBiasBiasGradOpx>
    addBiasBiasGradOpxCreator(Onnx::CustomGradOperators::AddBiasBiasGrad);
OpxCreator<AddBiasDataGradOpx>
    addBiasDataGradOpxCreator(Onnx::CustomGradOperators::AddBiasDataGrad);
} // namespace

} // namespace popx
} // namespace poponnx
