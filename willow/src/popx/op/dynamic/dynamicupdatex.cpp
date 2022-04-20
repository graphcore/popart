// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
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
#include "popart/popx/popopx.hpp"
#include "popart/popx/poptensors.hpp"
#include "popart/region.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/util.hpp"

namespace popart {
namespace popx {

DynamicUpdateOpx::DynamicUpdateOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<DynamicBinaryBaseOp>(op);
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void DynamicUpdateOpx::grow(snap::program::Sequence &prog) const {
  auto &op    = getOp<DynamicTernaryBaseOp>();
  auto tensor = getInTensor(DynamicTernaryBaseOp::getUpdateInIndex());
  auto index =
      getInTensor(DynamicTernaryBaseOp::getIndexInIndex()).getPoplarTensor();
  auto slice =
      getInTensor(DynamicTernaryBaseOp::getInIndex()).getPoplarTensor();

  std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
  std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

  auto outTensor = cloneNcopyOpt(prog, tensor);

  auto sliceShape = slice.shape();
  if (sliceShape.size() < tensor.shape().size()) {
    sliceShape.insert(sliceShape.begin(), 1);
  }

  popops::dynamicUpdate(
      graph().getPoplarGraph(),
      outTensor.getPoplarTensor(),
      slice.reshape(sliceShape),
      popops::cast(graph().getPoplarGraph(),
                   index.reshape({op.getAxes().size()}),
                   poplar::UNSIGNED_INT,
                   prog.getPoplarSequence(),
                   debugContext()),
      paxes,
      psizes,
      prog.getPoplarSequence(),
      debugContext("dynamic_update_" +
                   op.inId(DynamicTernaryBaseOp::getUpdateInIndex())));

  setOutTensor(DynamicTernaryBaseOp::getOutIndex(), outTensor);
}

InputCreatorType DynamicUpdateOpx::getInputCreatorType(InIndex index) const {
  DynamicTernaryBaseOp *op = dynamic_cast<DynamicTernaryBaseOp *>(this->op_p);
  auto itUpdate            = op->settings.inferTensorMappingToFrom.find(
      DynamicTernaryBaseOp::getUpdateInIndex());
  auto itIn = op->settings.inferTensorMappingToFrom.find(
      DynamicTernaryBaseOp::getInIndex());

  bool inferUpdateFromIn =
      itUpdate != op->settings.inferTensorMappingToFrom.end() &&
      itUpdate->second == DynamicTernaryBaseOp::getInIndex();
  bool inferInFromUpdate =
      itIn != op->settings.inferTensorMappingToFrom.end() &&
      itIn->second == DynamicTernaryBaseOp::getUpdateInIndex();

  if (index == DynamicTernaryBaseOp::getUpdateInIndex()) {
    if (inferUpdateFromIn) {
      return InputCreatorType::CanCreateOrUnwind;
    } else if (inferInFromUpdate) {
      return InputCreatorType::Deadend;
    } else {
      return InputCreatorType::CanUnwind;
    }
  }

  if (index == DynamicTernaryBaseOp::getInIndex()) {
    if (inferInFromUpdate) {
      return InputCreatorType::CanCreateOrUnwind;
    } else if (inferUpdateFromIn) {
      return InputCreatorType::Deadend;
    } else {
      return InputCreatorType::CanUnwind;
    }
  }

  return PopOpx::getInputCreatorType(index);
}

snap::Tensor
DynamicUpdateOpx::createInputTensor(InIndex index,
                                    const poplar::DebugNameAndId &dnai) const {
  auto &op = getOp<DynamicTernaryBaseOp>();

  // Create the slice we want to update for
  // NOTE: Do not confuse "InIndex index" with the Tensor we usually call
  // "index"
  if (index == DynamicTernaryBaseOp::getInIndex()) {
    if (dv_p->lowering().tensors().contains(
            op_p->input->id(DynamicTernaryBaseOp::getUpdateInIndex()))) {
      auto updateTensor = getInTensor(DynamicTernaryBaseOp::getUpdateInIndex())
                              .getPoplarTensor();
      auto updateShape = inShape(DynamicTernaryBaseOp::getUpdateInIndex());

      std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
      std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

      return snap::Tensor{
          popops::createSliceTensor(
              graph().getPoplarGraph(), updateTensor, paxes, psizes, 1, dnai)
              .squeeze({0}),
          graph()}
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
      return snap::Tensor{
          popops::createSliceableTensorFromSlice(
              graph().getPoplarGraph(),
              inTensor.reshape(inShape).slice(begin, end).getPoplarTensor(),
              paxes,
              numSlices,
              dnai),
          graph()};
    }
  }

  throw error("DynamicUpdateOpx::createInput : Invalid index = " +
              std::to_string(index));
}

snap::Tensor DynamicUpdateOpx::unwindTensorLayout(snap::Tensor tensor,
                                                  InIndex in,
                                                  OutIndex) const {
  if (in == DynamicUpdateOp::getInIndex()) {
    auto &op = getOp<DynamicUpdateOp>();
    std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
    std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

    return snap::Tensor{popops::createSliceTensor(graph().getPoplarGraph(),
                                                  tensor.getPoplarTensor(),
                                                  paxes,
                                                  psizes,
                                                  1)
                            .squeeze({0}),
                        graph()}
        .reshape(inShapeSzt(in));
  } else if (in == DynamicUpdateOp::getUpdateInIndex()) {
    return tensor;
  } else {
    return PopOpx::unwindTensorLayout(tensor, in, 0);
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

  auto it = op->settings.inferTensorMappingToFrom.find(index);

  if (it != op->settings.inferTensorMappingToFrom.end() &&
      ((it->first == DynamicTernaryBaseOp::getInIndex() &&
        it->second == DynamicTernaryBaseOp::getUpdateInIndex()) ||
       (it->first == DynamicTernaryBaseOp::getUpdateInIndex() &&
        it->second == DynamicTernaryBaseOp::getInIndex()))) {
    mustExist.insert(op->input->tensor(it->second)->id);
  }

  return mustExist;
}

snap::Tensor DynamicUpdateOpx::cloneNcopyOpt(snap::program::Sequence &s,
                                             const snap::Tensor &t) const {
  return cloneNcopy(s, t);
}

DynamicUpdateInplaceOpx::DynamicUpdateInplaceOpx(Op *op, Devicex *devicex)
    : DynamicUpdateOpx(op, devicex) {
  verifyOp<DynamicUpdateInplaceOp>(op);
}

snap::Tensor
DynamicUpdateInplaceOpx::cloneNcopyOpt(snap::program::Sequence &s,
                                       const snap::Tensor &t) const {
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
