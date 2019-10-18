#include <algorithm>

#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/op/slice.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/slicex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>

#include <popart/ir.hpp>
#include <popart/tensors.hpp>

#include <popops/Pad.hpp>
#include <popops/Zero.hpp>

namespace popart {
namespace popx {

BaseSliceOpx::BaseSliceOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}

InputCreatorType BaseSliceOpx::getInputCreatorType(InIndex inIndex) const {
  if (!dynamic_cast<BaseSliceOp *>(op_p)->allSlices.empty()) {
    return InputCreatorType::CANUNWIND_MULTIPLE_CREATORS;
  } else {
    return Opx::getInputCreatorType(inIndex);
  }
}
poplar::Tensor
BaseSliceOpx::unwindTensorLayout(std::vector<poplar::Tensor> tensors,
                                 InIndex,
                                 OutIndex) const {
  BaseSliceOp *op = dynamic_cast<BaseSliceOp *>(this->op_p);
  logging::opx::debug("BaseSliceOpx::unwindTensorLayout dim:{} inputs:{}",
                      op->unwindConcatDim,
                      tensors.size());
  return poplar::concat(tensors, op->unwindConcatDim);
}

std::vector<std::pair<Op *, InIndex>>
BaseSliceOpx::getCreatorCandicates(InIndex) const {

  std::vector<std::pair<Op *, InIndex>> creators;

  BaseSliceOp *op = dynamic_cast<BaseSliceOp *>(this->op_p);

  // Get the list of op's the consume the all the slices of the input tensor
  for (auto t : op->allSlices) {
    auto consumerOps = op->getIr().getTensor(t)->consumers.getOps();
    for (auto *consumer : consumerOps) {
      InIndex index = -1;
      for (auto input : consumer->input->tensorIdMap()) {
        if (input.second == t) {
          index = input.first;
          break;
        }
      }
      if (index == -1) {
        throw error("Could not determine input index for {}", t);
      }
      creators.push_back({consumer, index});
    }
  }

  return creators;
}

SliceOpx::SliceOpx(Op *op, Devicex *devicex) : BaseSliceOpx(op, devicex) {
  verifyOp<SliceOp>(op);
}

void SliceOpx::grow(poplar::program::Sequence &prog) const {
  auto t = getInTensor(SliceOp::getInIndex());
  for (auto slice : getSliceOp()->getSlices()) {
    t = t.slice(slice.start, slice.end, static_cast<unsigned>(slice.axis));
  }
  // we clone and copy t, as this in not an inplace op
  setOutTensor(SliceOp::getOutIndex(), cloneNcopy(prog, t));
}

SliceOp *SliceOpx::getSliceOp() const { return dynamic_cast<SliceOp *>(op_p); }

void SliceInplaceOpx::grow(poplar::program::Sequence &) const {
  auto t = getInTensor(SliceOp::getInIndex());
  for (auto slice : getSliceInplaceOp()->getSlices()) {
    t = t.slice(slice.start, slice.end, static_cast<unsigned>(slice.axis));
  }
  setOutTensor(SliceOp::getOutIndex(), t);
}

SliceInplaceOp *SliceInplaceOpx::getSliceInplaceOp() const {
  return dynamic_cast<SliceInplaceOp *>(op_p);
}

SliceGradOpx::SliceGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SliceGradOp>(op, Onnx::GradOperators::SliceGrad);
}

void SliceGradOpx::grow(poplar::program::Sequence &prog) const {
  poplar::Tensor inTensor = getInTensor(SliceGradOp::getInIndex());
  poplar::Tensor outTensor;

  // Look for the input (i.e. pre-sliced) tensor to the corresponding non-grad
  // op. If found, base the layout out the grad output tensor on this.
  auto found_preSlicedTensor = getPreSlicedTensorIfPossible();
  if (found_preSlicedTensor.first) {
    // Padding with zeros
    outTensor = graph().clone(found_preSlicedTensor.second);
    // Note:  If you don't set the padded region of output tensor to zero,
    // there is potential for downstream ops to change these elements in-place,
    // corrupting the output of the Opx on subsequent iterations.
    popops::zero(graph(), outTensor, prog, debugPrefix("zero"));

    // Copy input into non-padded region of output tensor
    auto nonPaddedRegion = outTensor;
    for (auto slice : dynamic_cast<SliceGradOp *>(op_p)->getSlices()) {
      nonPaddedRegion = nonPaddedRegion.slice(
          slice.start, slice.end, static_cast<unsigned>(slice.axis));
    }
    poplar::program::Copy copy(inTensor, nonPaddedRegion);
    prog.add(copy);
  }

  // Otherwise, pad with EDGE mapping
  else {
    auto sliceGradOp = dynamic_cast<SliceGradOp *>(op_p);
    outTensor        = popops::pad(graph(),
                            inTensor,
                            sliceGradOp->getLowerPadding(),
                            sliceGradOp->getUpperPadding(),
                            0,
                            popops::padding::MappingMethod::EDGE);
  }

  setOutTensor(SliceGradOp::getOutIndex(), outTensor);
}

std::pair<bool, poplar::Tensor>
SliceGradOpx::getPreSlicedTensorIfPossible() const {
  // The SliceGrad and its corresponding non-grad op:
  //
  // act_in       act_in_grad
  //    |            |
  //  Slice       SliceGrad
  //    |            |
  // act_out      act_out_grad
  //
  // 1. From 'act_out_grad', find 'act_out'
  // 2. From 'act_out', find its producer
  // 3. From its producer, find 'act_in'
  poplar::Tensor dummy;

  TensorId act_out = getNonGradId(op_p->inId(SliceGradOp::getInIndex()));
  if (!dv_p->tensors.contains(act_out)) {
    return {false, dummy};
  }

  Tensor *act_out_t = op_p->getGraph().getTensors().get(act_out);
  if (!act_out_t->hasProducer()) {
    return {false, dummy};
  }

  Op *sliceOp = act_out_t->getProducer();
  // If a transform has modified the Ir here by then give up.
  if (!dynamic_cast<BaseSliceOp *>(sliceOp)) {
    return {false, dummy};
  }

  TensorId act_in = sliceOp->inId(BaseSliceOp::getInIndex());
  if (!dv_p->tensors.contains(act_in)) {
    return {false, dummy};
  }

  return {true, get(act_in)};
}

SliceInplaceOpx::SliceInplaceOpx(Op *op_, Devicex *devicex)
    : BaseSliceOpx(op_, devicex) {
  verifyOp<SliceInplaceOp>(op_);
}

namespace {
OpxCreator<SliceOpx> sliceOpxCreator({Onnx::Operators::Slice_1,
                                      Onnx::Operators::Slice_10,
                                      Onnx::Operators::Slice_11});
OpxCreator<SliceInplaceOpx>
    sliceInplaceOpxCreator(Onnx::CustomOperators::SliceInplace);
OpxCreator<SliceGradOpx> sliceGradOpxCreator(Onnx::GradOperators::SliceGrad);
} // namespace

} // namespace popx
} // namespace popart
