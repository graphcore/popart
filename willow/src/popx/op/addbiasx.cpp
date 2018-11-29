#include <poplin/Convolution.hpp>
#include <popops/Reduce.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/addbias.hpp>
#include <poponnx/popx/op/addbiasx.hpp>

namespace poponnx {
namespace popx {

AddBiasOpx::AddBiasOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::ADDBIAS) {
    throw error("cannot create AddBiasOpx from " + op->op_type());
  }
}

AddBiasDataGradOpx::AddBiasDataGradOpx(Op *op, Devicex *devicex)
    : IdentityOpx(op, devicex) {
  if (op->opType != OpType::ADDBIASDATAGRAD) {
    throw error("cannot create AddBiasDataGradOpx from " + op->op_type());
  }
}

void AddBiasOpx::grow(poplar::program::Sequence &prog) const {
  // Clone & copy the input tensor because poplin::addBias is in-place.
  const auto result = Opx::cloneNcopy(prog, inId(AddBiasOp::getDataInIndex()));
  poplin::addBias(
      graph(), result, get(inId(AddBiasOp::getBiasInIndex())), prog, idStr());
  insert(outId(AddBiasOp::getOutIndex()), result);
}

std::vector<TensorId> AddBiasOpx::mustExistBeforeCreate(InIndex index) const {
  if (index != AddBiasOp::getBiasInIndex()) {
    throw error("AddBiasOpx::mustExistBeforeCreate : Invalid index = " +
                std::to_string(index));
  }

  return {inId(AddBiasOp::getDataInIndex())};
}

bool AddBiasOpx::canCreateInput(InIndex index) const {
  return index == AddBiasOp::getBiasInIndex();
}

poplar::Tensor AddBiasOpx::createInput(InIndex index) const {
  if (index != AddBiasOp::getBiasInIndex()) {
    throw error("AddBiasOpx::createInput : Invalid index = " +
                std::to_string(index));
  }

  return poplin::createBiases(graph(),
                              get(inId(AddBiasOp::getDataInIndex())),
                              inId(AddBiasOp::getBiasInIndex()));
}

bool AddBiasOpx::createsEquiv(int, Opx *, int) const { return false; }

AddBiasOp *AddBiasOpx::getAddBiasOp() const {
  return dynamic_cast<AddBiasOp *>(op_p);
}

AddBiasBiasGradOpx::AddBiasBiasGradOpx(Op *op, Devicex *devicex)
    : ReduceSumOpx(op, devicex) {
  if (op_p->opType != OpType::ADDBIASBIASGRAD) {
    throw error("cannot create AddBiasBiasGradOpx from " + op_p->op_type());
  }
}

} // namespace popx
} // namespace poponnx
