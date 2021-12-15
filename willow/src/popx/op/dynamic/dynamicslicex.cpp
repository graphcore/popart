// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>

#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/dynamic/dynamicslicex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

namespace popart {
namespace popx {

DynamicSliceOpx::DynamicSliceOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<DynamicSliceBaseOp>(op);
  inputCreatorPriority = -1.0;
}

void DynamicSliceOpx::grow(snap::program::Sequence &prog) const {
  auto &op    = getOp<DynamicSliceBaseOp>();
  auto tensor = getInTensor(DynamicSliceBaseOp::getInIndex()).getPoplarTensor();
  auto index =
      getInTensor(DynamicSliceBaseOp::getIndexInIndex()).getPoplarTensor();

  std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
  std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

  auto s =
      snap::Tensor{popops::dynamicSlice(
                       graph().getPoplarGraph(),
                       tensor,
                       popops::cast(graph().getPoplarGraph(),
                                    index.reshape({op.getAxes().size()}),
                                    poplar::UNSIGNED_INT,
                                    prog.getPoplarSequence(),
                                    debugContext()),
                       paxes,
                       psizes,
                       prog.getPoplarSequence(),
                       debugContext("dynamic_slice_" +
                                    op.inId(DynamicSliceBaseOp::getInIndex()))),
                   graph()};

  auto outTensor = s;
  // Output tensor mirrors layout of the input slice provided
  if (op.hasInput(DynamicSliceOp::getSliceInIndex())) {
    outTensor =
        cloneNcopy(prog,
                   getInTensor(DynamicSliceOp::getSliceInIndex()),
                   op.inId(DynamicSliceOp::getSliceInIndex()) + "_writeable");
    prog.add(snap::program::Copy(s, outTensor));
  }

  setOutTensor(DynamicSliceBaseOp::getOutIndex(), outTensor);
}

InputCreatorType DynamicSliceOpx::getInputCreatorType(InIndex index) const {
  return (index == DynamicSliceBaseOp::getInIndex() ||
          index == DynamicSliceInplaceOp::getSliceInIndex())
             ? InputCreatorType::CanCreateOrUnwind
             : PopOpx::getInputCreatorType(index);
}

snap::Tensor
DynamicSliceOpx::createInputTensor(InIndex index,
                                   const poplar::DebugNameAndId &dnai) const {
  auto &op = getOp<DynamicSliceBaseOp>();

  // Get the tensor info and the shapes
  auto outInfo  = op.outInfo(DynamicSliceBaseOp::getOutIndex());
  auto outShape = outInfo.shape_szt();
  auto inShape  = op.inShape(DynamicSliceBaseOp::getInIndex());

  if (index == DynamicSliceBaseOp::getInIndex()) {
    // Get the axes to slice
    std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
    // Initialize the number of slices for the input tensor
    std::vector<size_t> numSlices(paxes.size(), 0);

    // We ensure that the slices from createSliceableTensorFromSlice have
    // identical layout.
    // The slices will be spread across fewer tiles, but we will avoid
    // huge exchange copies as the output layout does not depend on the index
    // The output layout will match regardless of the slice size and index at
    // runtime
    for (size_t i = 0; i < paxes.size(); ++i) {
      numSlices[i]       = inShape[paxes[i]];
      outShape[paxes[i]] = 1;
    }

    // Create the input tensor
    auto sliceTensor = snap::Tensor{
        graph().getPoplarGraph().addVariable(
            popType(outInfo),
            outShape,
            debugContext(op.inId(DynamicSliceBaseOp::getInIndex()) + "_slice")),
        graph()};

    dv_p->lowering().getLinearMapper().mapTensor(graph(), sliceTensor);

    // Create the layout for the input tensor
    return snap::Tensor{
        popops::createSliceableTensorFromSlice(graph().getPoplarGraph(),
                                               sliceTensor.getPoplarTensor(),
                                               paxes,
                                               numSlices,
                                               dnai),
        graph()};
  }

  if (index == DynamicSliceInplaceOp::getSliceInIndex()) {
    // Create the input tensor
    auto sliceTensor = snap::Tensor{
        graph().getPoplarGraph().addVariable(
            popType(outInfo),
            outShape,
            debugContext(op.inId(DynamicSliceBaseOp::getInIndex()) + "_slice")),
        graph()};
    return sliceTensor;
  }

  throw internal_error(
      "[DynamicSliceOpx::createInputTensor] Unsupported InIndex {}", index);
}

snap::Tensor DynamicSliceOpx::unwindTensorLayout(snap::Tensor tensor,
                                                 InIndex index,
                                                 OutIndex) const {
  auto &op      = getOp<DynamicSliceBaseOp>();
  auto outShape = tensor.shape();
  auto inShape  = op.inShape(DynamicSliceBaseOp::getInIndex());

  if (index == DynamicSliceOp::getInIndex()) {
    std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());

    std::vector<size_t> begin(outShape.size(), 0);
    std::vector<size_t> end = outShape;

    // We ensure that the slices from createSliceableTensorFromSlice have
    // identical layout.
    // The slices will be spread across fewer tiles, but we will avoid
    // huge exchange copies as the output layout does not depend on the index
    // The output layout will match regardless of the slice size and index at
    // runtime
    std::vector<size_t> numSlices(paxes.size(), 0);
    for (size_t i = 0; i < paxes.size(); ++i) {
      numSlices[i]  = inShape[paxes[i]];
      end[paxes[i]] = 1;
    }

    // Create the layout for the input tensor
    return snap::Tensor{popops::createSliceableTensorFromSlice(
                            graph().getPoplarGraph(),
                            tensor.slice(begin, end).getPoplarTensor(),
                            paxes,
                            numSlices),
                        graph()};
  }
  if (index == DynamicSliceInplaceOp::getSliceInIndex()) {
    return tensor;
  }

  throw internal_error(
      "[DynamicSliceOpx::unwindTensorLayout] Unsupported InIndex {}", index);
}

view::RegMap DynamicSliceOpx::unwindRegion(InIndex index, OutIndex) const {
  DynamicSliceBaseOp *op = dynamic_cast<DynamicSliceBaseOp *>(this->op_p);
  auto shape             = op->inShape(index);
  return [shape](const view::Region &) {
    return view::Regions(1, view::Region::getFull(shape));
  };
}

DynamicSliceInplaceOpx::DynamicSliceInplaceOpx(Op *op, Devicex *devicex)
    : DynamicSliceOpx(op, devicex) {
  verifyOp<DynamicSliceBaseOp>(op);
  inputCreatorPriority = -1.0;
}

void DynamicSliceInplaceOpx::grow(snap::program::Sequence &prog) const {
  auto &op    = getOp<DynamicSliceBaseOp>();
  auto tensor = getInTensor(DynamicSliceBaseOp::getInIndex()).getPoplarTensor();
  auto index =
      getInTensor(DynamicSliceBaseOp::getIndexInIndex()).getPoplarTensor();
  auto slice = getInTensor(DynamicSliceInplaceOp::getSliceInIndex());

  std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
  std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

  auto s =
      snap::Tensor{popops::dynamicSlice(
                       graph().getPoplarGraph(),
                       tensor,
                       popops::cast(graph().getPoplarGraph(),
                                    index.reshape({op.getAxes().size()}),
                                    poplar::UNSIGNED_INT,
                                    prog.getPoplarSequence(),
                                    debugContext()),
                       paxes,
                       psizes,
                       prog.getPoplarSequence(),
                       debugContext("dynamic_slice_" +
                                    op.inId(DynamicSliceBaseOp::getInIndex()))),
                   graph()};

  auto writeableSlice = slice;
  if (!writeableSlice.getPoplarTensor().isParallelWriteable()) {
    writeableSlice = cloneNcopy(
        prog,
        slice,
        op.inId(DynamicSliceInplaceOp::getSliceInIndex()) + "_writeable");
  }

  prog.add(snap::program::Copy(s, writeableSlice));

  setOutTensor(DynamicSliceBaseOp::getOutIndex(), writeableSlice);
}

std::set<TensorId> DynamicSliceOpx::mustExistBeforeCreate(InIndex index) const {
  DynamicSliceOp *op = dynamic_cast<DynamicSliceOp *>(this->op_p);

  std::set<TensorId> mustExist;

  auto it = op->settings.inferTensorMappingToFrom.find(index);

  // Check if the infer tensor mapping is requested from either:
  // 2) SliceInIndex -> InIndex
  // or:
  // 1) InIndex -> SliceInIndex
  // and insert the respective "from" tensor in mustExist
  if (it != op->settings.inferTensorMappingToFrom.end() &&
      ((it->first == DynamicSliceOp::getInIndex() &&
        it->second == DynamicSliceOp::getSliceInIndex()) ||
       (it->first == DynamicSliceOp::getSliceInIndex() &&
        it->second == DynamicSliceOp::getInIndex()))) {
    mustExist.insert(op->input->tensor(it->second)->id);
  }

  return mustExist;
}

namespace {
// Ops
OpxCreator<DynamicSliceOpx>
    dynamicSliceOpxCreator(Onnx::CustomOperators::DynamicSlice_1);
OpxCreator<DynamicSliceInplaceOpx>
    dynamicSliceInplaceOpxCreator(Onnx::CustomOperators::DynamicSliceInplace_1);
} // namespace

} // namespace popx
} // namespace popart
