// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <cstdint>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <vector>
#include <poplar/Graph.hpp>
#include <popart/op/subsample.hpp>
#include <popart/popx/op/subsamplex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/popx/popopx.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;

static snap::Tensor subsample(const snap::Tensor &t,
                              const std::vector<uint32_t> &strides) {

  auto result   = t;
  int dimension = 0;
  for (auto stride : strides) {
    result = result.subSample(stride, dimension++);
  }
  return result;
}

SubsampleInplaceOpx::SubsampleInplaceOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<SubsampleInplaceOp>(op);
}

SubsampleOpx::SubsampleOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<SubsampleOp>(op, {Onnx::CustomOperators::Subsample_1});
}

void SubsampleOpx::grow(snap::program::Sequence &prog) const {

  SubsampleOp &op = getOp<SubsampleOp>();
  auto outTensor  = getInTensor(SubsampleOp::getInIndex());
  outTensor       = subsample(outTensor, op.strides_u32());
  // Need to clone/copy a new output tensor so is not in place
  setOutTensor(SubsampleOp::getOutIndex(), cloneNcopy(prog, outTensor));
}

void SubsampleInplaceOpx::grow(snap::program::Sequence &) const {
  SubsampleInplaceOp &op = getOp<SubsampleInplaceOp>();
  auto outTensor         = getInTensor(SubsampleOp::getInIndex());
  outTensor              = subsample(outTensor, op.strides_u32());
  setOutTensor(SubsampleOp::getOutIndex(), outTensor);
}

SubsampleGradOpx::SubsampleGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<SubsampleGradOp>(op, Onnx::CustomGradOperators::SubsampleGrad);
}

// 1. Create an output tensor all 0's
// 2. Create a subsample of that tensor that matches what we did in the fwd pass
// 3. Copy the input gradients onto the subsample view of the output
// 4. Return the output tensor
void SubsampleGradOpx::grow(snap::program::Sequence &prog) const {

  SubsampleGradOp &gradOp = getOp<SubsampleGradOp>();
  auto &in                = getInTensor(SubsampleGradOp::getInIndex());

  // Design decision: make a scalar zero variable that we expand to create
  // a tensor of the same size as the output
  auto zero = getScalarVariable(in.elementType(), "zero");
  graph().getPoplarGraph().setInitialValue(zero.getPoplarTensor(), 0);

  // Create an 0'ed tensor to be a tensor of the right size
  auto output = zero;
  for (int i = 0; i < gradOp.getFwdInputShape().size(); ++i) {
    output = output.expand({0});
  }
  for (int i = 0; i < gradOp.getFwdInputShape().size(); ++i) {
    output = output.broadcast(
        static_cast<unsigned>(gradOp.getFwdInputShape()[i]), i);
  }

  // Copy the zero-view tensor into a new tensor and remap
  auto outTensor = graph().addLinearlyMappedVariable(
      output.elementType(), output.shape(), debugContext());
  prog.add(snap::program::Copy(output, outTensor, false, debugContext()));

  // Create a subsample view of the output
  auto ss_output = subsample(outTensor, gradOp.strides_u32());

  // Copy the input tensor into the subsampled view of the output
  prog.add(snap::program::Copy(in, ss_output, false, debugContext()));

  // Return the output
  setOutTensor(SubsampleGradOp::getOutIndex(), outTensor);
}

namespace {
OpxCreator<SubsampleOpx>
    subsampleOpxCreator(Onnx::CustomOperators::Subsample_1);
OpxCreator<SubsampleInplaceOpx>
    subsampleInplaceOpxCreator(Onnx::CustomOperators::SubsampleInplace);
OpxCreator<SubsampleGradOpx>
    subsampleGradOpxCreator(Onnx::CustomGradOperators::SubsampleGrad);
} // namespace

} // namespace popx
} // namespace popart
