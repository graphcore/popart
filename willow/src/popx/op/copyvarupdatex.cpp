// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <set>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <popops/ExprOp.hpp>
#include <popart/error.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/popx/op/copyvarupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/varupdatex.hpp"
#include "popart/popx/opx.hpp"
#include "popart/tensordebuginfo.hpp"

namespace pe = popops::expr;

namespace popart {
class CopyVarUpdateOp;
class Op;

namespace popx {
class Devicex;

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

void CopyVarUpdateOpx::grow(poplar::program::Sequence &prog) const {
  auto &updater  = getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());
  auto &toUpdate = getInTensor(VarUpdateOp::getVarToUpdateInIndex());

  if (twoTensorsParallelWritable(updater, toUpdate)) {
    poplar::program::Copy copy(updater, toUpdate, false, debugContext());
    prog.add(copy);
  } else {
    auto newUpdater = cloneNcopy(prog, updater);
    poplar::program::Copy copy(newUpdater, toUpdate, false, debugContext());
    prog.add(copy);
  }
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
