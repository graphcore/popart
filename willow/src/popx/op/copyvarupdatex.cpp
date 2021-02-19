// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/copyvarupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/copyvarupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

CopyVarUpdateOpx::CopyVarUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<CopyVarUpdateOp>(op, Onnx::CustomOperators::CopyVarUpdate);
}

void CopyVarUpdateOpx::grow(poplar::program::Sequence &prog) const {
  auto vu_op = getOp<CopyVarUpdateOp>();
  poplar::program::Copy copy(
      getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex()),
      getInTensor(VarUpdateOp::getVarToUpdateInIndex()),
      false,
      debugContext());
  prog.add(copy);

  // output is a reference to destination of the copy
  setOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
               getInTensor(VarUpdateOp::getVarToUpdateInIndex()));
}

poplar::Tensor
CopyVarUpdateOpx::createInput(int inIndex,
                              const poplar::DebugNameAndId &dnai) const {

  if (inIndex != VarUpdateWithUpdaterOp::getUpdaterInIndex()) {
    throw error(
        "CopyVarUpdateOpx::createInput, cannot create input at {}, it can "
        "only create the updater input Tensor",
        inIndex);
  }
  return graph().clone(getInTensor(VarUpdateOp::getVarToUpdateInIndex()), dnai);
}

InputCreatorType CopyVarUpdateOpx::getInputCreatorType(int inIndex) const {
  return inIndex == VarUpdateWithUpdaterOp::getUpdaterInIndex()
             ? InputCreatorType::CanCreate
             : Opx::getInputCreatorType(inIndex);
}

std::vector<TensorId>
CopyVarUpdateOpx::mustExistBeforeCreate(int index1) const {
  if (index1 != VarUpdateWithUpdaterOp::getUpdaterInIndex()) {
    throw internal_error(
        "CopyVarUpdate::mustExistBeforeCreate : Invalid index");
  }
  return {inId(VarUpdateOp::getVarToUpdateInIndex())};
}

namespace {
OpxCreator<CopyVarUpdateOpx>
    copyVarUpdateOpxCreator(Onnx::CustomOperators::CopyVarUpdate);
} // namespace

} // namespace popx
} // namespace popart
