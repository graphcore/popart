// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/dynamic/dynamicupdatex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

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

  popops::dynamicUpdate(
      graph(),
      outTensor,
      slice,
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

  return Opx::getInputCreatorType(index);
}

poplar::Tensor
DynamicUpdateOpx::createInput(InIndex index,
                              const poplar::DebugNameAndId &dnai) const {
  auto &op = getOp<DynamicTernaryBaseOp>();

  if (index == DynamicTernaryBaseOp::getInIndex()) {
    if (dv_p->lowering().tensors().contains(
            op_p->input->id(DynamicTernaryBaseOp::getUpdateInIndex()))) {
      auto updateTensor = getInTensor(DynamicTernaryBaseOp::getUpdateInIndex());
      auto updateShape  = op.inShape(DynamicTernaryBaseOp::getUpdateInIndex());

      std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
      std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

      return popops::createSliceTensor(
                 graph(), updateTensor, paxes, psizes, 1, dnai)
          .squeeze({0});
    }
  }

  if (index == DynamicTernaryBaseOp::getUpdateInIndex()) {
    if (dv_p->lowering().tensors().contains(
            op_p->input->id(DynamicTernaryBaseOp::getInIndex()))) {
      auto inTensor    = getInTensor(DynamicTernaryBaseOp::getInIndex());
      auto updateShape = op.inShape(DynamicTernaryBaseOp::getUpdateInIndex());

      std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
      std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

      std::vector<size_t> numSlices(paxes.size(), 0);
      for (size_t i = 0; i < paxes.size(); ++i) {
        numSlices[i] = updateShape[paxes[i]] / psizes[i];
      }
      return popops::createSliceableTensorFromSlice(
          graph(), inTensor, paxes, numSlices, dnai);
    }
  }

  throw error("DynamicUpdateOpx::createInput : Invalid index = " +
              std::to_string(index));
}

poplar::Tensor DynamicUpdateOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                    InIndex in,
                                                    OutIndex) const {
  if (in == DynamicUpdateOp::getInIndex()) {
    auto &op = getOp<DynamicUpdateOp>();
    std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
    std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

    return popops::createSliceTensor(graph(), tensor, paxes, psizes, 1)
        .squeeze({0});
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
