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

namespace {
bool twoTensorsParallelWritable(const poplar::Tensor &a,
                                const poplar::Tensor &b) {
  if (a.rank() != b.rank()) {
    throw error("CopyVarUpdateOpx needs the tensors to be the same rank.");
  }

  if (a.rank() == 0) {
    return poplar::concat(a.flatten(), b.flatten()).isParallelWriteable();
  }

  return poplar::concat(a, b).isParallelWriteable();
}
} // namespace

CopyVarUpdateOpx::CopyVarUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<CopyVarUpdateOp>(op, Onnx::CustomOperators::CopyVarUpdate);
}

void CopyVarUpdateOpx::grow(snap::program::Sequence &prog) const {
  auto &updater  = getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());
  auto &toUpdate = getInTensor(VarUpdateOp::getVarToUpdateInIndex());

  if (twoTensorsParallelWritable(updater.getPoplarTensor(),
                                 toUpdate.getPoplarTensor())) {
    poplar::program::Copy copy(updater.getPoplarTensor(),
                               toUpdate.getPoplarTensor(),
                               false,
                               debugContext());
    prog.getPoplarSequence().add(copy);
  } else {
    auto newUpdater = cloneNcopy(prog, updater).getPoplarTensor();
    poplar::program::Copy copy(
        newUpdater, toUpdate.getPoplarTensor(), false, debugContext());
    prog.getPoplarSequence().add(copy);
  }
  // output is a reference to destination of the copy
  setOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
               getInTensor(VarUpdateOp::getVarToUpdateInIndex()));
}

snap::Tensor
CopyVarUpdateOpx::createInputTensor(int inIndex,
                                    const poplar::DebugNameAndId &dnai) const {

  if (inIndex != VarUpdateWithUpdaterOp::getUpdaterInIndex()) {
    throw error(
        "CopyVarUpdateOpx::createInput, cannot create input at {}, it can "
        "only create the updater input Tensor",
        inIndex);
  }
  return snap::Tensor{
      graph().getPoplarGraph().clone(
          getInTensor(VarUpdateOp::getVarToUpdateInIndex()).getPoplarTensor(),
          dnai),
      graph()};
}

InputCreatorType CopyVarUpdateOpx::getInputCreatorType(int inIndex) const {
  return inIndex == VarUpdateWithUpdaterOp::getUpdaterInIndex()
             ? InputCreatorType::CanCreate
             : PopOpx::getInputCreatorType(inIndex);
}

std::set<TensorId> CopyVarUpdateOpx::mustExistBeforeCreate(int index1) const {
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
