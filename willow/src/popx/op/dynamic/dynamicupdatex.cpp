// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popart/error.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/dynamic/dynamicupdatex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/dynamic/dynamicbase.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/popx/poptensors.hpp"
#include "popart/region.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/util.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
namespace popx {

DynamicUpdateOpx::DynamicUpdateOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<DynamicBinaryBaseOp>(op);
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void DynamicUpdateOpx::grow(poplar::program::Sequence &prog) const {
  auto &op    = getOp<DynamicTernaryBaseOp>();
  auto tensor = getInTensor(DynamicTernaryBaseOp::getUpdateInIndex());
  auto index  = getInTensor(DynamicTernaryBaseOp::getIndexInIndex());
  auto slice  = getInTensor(DynamicTernaryBaseOp::getInIndex());

  std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
  std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

  auto outTensor = cloneNcopyOpt(prog, tensor);

  auto sliceShape = slice.shape();
  if (sliceShape.size() < tensor.shape().size()) {
    sliceShape.insert(sliceShape.begin(), 1);
  }

  popops::dynamicUpdate(
      graph(),
      outTensor,
      slice.reshape(sliceShape),
      popops::cast(graph(),
                   index.reshape({op.getAxes().size()}),
                   poplar::UNSIGNED_INT,
                   prog,
                   debugContext()),
      paxes,
      psizes,
      prog,
      debugContext("dynamic_update_" +
                   op.inId(DynamicTernaryBaseOp::getUpdateInIndex())));

  setOutTensor(DynamicTernaryBaseOp::getOutIndex(), outTensor);
}

InputCreatorType DynamicUpdateOpx::getInputCreatorType(InIndex index) const {
  if (index == DynamicTernaryBaseOp::getUpdateInIndex() ||
      index == DynamicTernaryBaseOp::getInIndex())
    return InputCreatorType::CanCreateOrUnwind;

  return Opx::getInputCreatorType(index);
}

poplar::Tensor
DynamicUpdateOpx::createInput(InIndex index,
                              const poplar::DebugNameAndId &dnai) const {
  auto &op = getOp<DynamicTernaryBaseOp>();

  // Create the slice we want to update for
  // NOTE: Do not confuse "InIndex index" with the Tensor we usually call
  // "index"
  if (index == DynamicTernaryBaseOp::getInIndex()) {
    if (dv_p->lowering().tensors().contains(
            op_p->input->id(DynamicTernaryBaseOp::getUpdateInIndex()))) {
      auto updateTensor = getInTensor(DynamicTernaryBaseOp::getUpdateInIndex());
      auto updateShape  = inShape(DynamicTernaryBaseOp::getUpdateInIndex());

      std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
      std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

      return popops::createSliceTensor(
                 graph(), updateTensor, paxes, psizes, 1, dnai)
          .squeeze({0})
          .reshape(inShapeSzt(index));
    }
  }

  // Create the updatee
  // NOTE: Do not confuse "InIndex index" with the Tensor we usually call
  // "index"
  if (index == DynamicTernaryBaseOp::getUpdateInIndex()) {
    if (dv_p->lowering().tensors().contains(
            op_p->input->id(DynamicTernaryBaseOp::getInIndex()))) {
      auto inTensor    = getInTensor(DynamicTernaryBaseOp::getInIndex());
      auto inShape     = inTensor.shape();
      auto updateShape = inShapeSzt(DynamicTernaryBaseOp::getUpdateInIndex());

      if (inShape.size() < updateShape.size()) {
        inShape.insert(inShape.begin(), 1);
      }

      std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
      std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

      // We ensure that the slices from createSliceableTensorFromSlice have
      // identical layout.
      // The slices will be spread across fewer tiles, but we will avoid
      // huge exchange copies as the output layout does not depend on the index
      // The output layout will match regardless of the slice size and index at
      // runtime
      std::vector<size_t> begin(updateShape.size(), 0);
      std::vector<size_t> end = vector_cast<size_t>(updateShape);
      std::vector<size_t> numSlices(paxes.size(), 0);
      for (size_t i = 0; i < paxes.size(); ++i) {
        numSlices[i]  = updateShape[paxes[i]];
        end[paxes[i]] = 1;
      }
      return popops::createSliceableTensorFromSlice(
          graph(),
          inTensor.reshape(inShape).slice(begin, end),
          paxes,
          numSlices,
          dnai);
    }
  }

  throw error("DynamicUpdateOpx::createInput : Invalid index = " +
              std::to_string(index));
}

poplar::Tensor DynamicUpdateOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                    InIndex in,
                                                    OutIndex) const {
  if (in == DynamicUpdateOp::getInIndex()) {
    DynamicTernaryBaseOp *op = dynamic_cast<DynamicTernaryBaseOp *>(this->op_p);
    std::vector<size_t> paxes(op->getAxes().begin(), op->getAxes().end());
    std::vector<size_t> psizes(op->getSizes().begin(), op->getSizes().end());

    return popops::createSliceTensor(graph(), tensor, paxes, psizes, 1)
        .squeeze({0})
        .reshape(inShapeSzt(in));
  } else if (in == DynamicUpdateOp::getUpdateInIndex()) {
    return tensor;
  } else {
    return Opx::unwindTensorLayout(tensor, in, 0);
  }
}

view::RegMap DynamicUpdateOpx::unwindRegion(InIndex index, OutIndex) const {
  DynamicTernaryBaseOp *op = dynamic_cast<DynamicTernaryBaseOp *>(this->op_p);
  auto shape               = op->inShape(index);
  return [shape](const view::Region &) {
    return view::Regions(1, view::Region::getFull(shape));
  };
}

std::set<TensorId>
DynamicUpdateOpx::mustExistBeforeCreate(InIndex index) const {
  DynamicTernaryBaseOp *op = dynamic_cast<DynamicTernaryBaseOp *>(this->op_p);

  std::set<TensorId> mustExist;

  mustExist.insert(
      op->input
          ->tensor(index == DynamicTernaryBaseOp::getUpdateInIndex()
                       ? DynamicTernaryBaseOp::getInIndex()
                       : DynamicTernaryBaseOp::getUpdateInIndex())
          ->id);

  return mustExist;
}

poplar::Tensor DynamicUpdateOpx::cloneNcopyOpt(poplar::program::Sequence &s,
                                               const poplar::Tensor &t) const {
  return cloneNcopy(s, t);
}

DynamicUpdateInplaceOpx::DynamicUpdateInplaceOpx(Op *op, Devicex *devicex)
    : DynamicUpdateOpx(op, devicex) {
  verifyOp<DynamicUpdateInplaceOp>(op);
}

poplar::Tensor
DynamicUpdateInplaceOpx::cloneNcopyOpt(poplar::program::Sequence &s,
                                       const poplar::Tensor &t) const {
  if (t.isParallelWriteable()) {
    return t;
  } else {
    // Outplace because t has internal aliases
    return cloneNcopy(s, t);
  }
}

namespace {
// Ops
OpxCreator<DynamicUpdateOpx>
    DynamicUpdateOpxCreator(Onnx::CustomOperators::DynamicUpdate_1);
OpxCreator<DynamicUpdateInplaceOpx>
    DynamicUpdateInplaceOpxCreator(Onnx::CustomOperators::DynamicUpdateInplace);
} // namespace

} // namespace popx
} // namespace popart
