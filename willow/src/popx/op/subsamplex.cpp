#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/subsample.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/subsamplex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

#include <ostream>

namespace popart {
namespace popx {

static poplar::Tensor subsample(poplar::Tensor &t,
                                const std::vector<uint32_t> &strides) {

  auto result   = t;
  int dimension = 0;
  for (auto stride : strides) {
    result = result.subSample(stride, dimension++);
  }
  return result;
}

SubsampleInplaceOpx::SubsampleInplaceOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<SubsampleInplaceOp>(op);
}

SubsampleOpx::SubsampleOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SubsampleOp>(op, {Onnx::CustomOperators::Subsample_1});
}

void SubsampleOpx::grow(poplar::program::Sequence &prog) const {

  SubsampleOp &op = getOp<SubsampleOp>();
  auto outTensor  = getInTensor(SubsampleOp::getInIndex());
  outTensor       = subsample(outTensor, op.strides_u32());
  // Need to clone/copy a new output tensor so is not in place
  setOutTensor(SubsampleOp::getOutIndex(), cloneNcopy(prog, outTensor));
}

void SubsampleInplaceOpx::grow(poplar::program::Sequence &) const {
  SubsampleInplaceOp &op = getOp<SubsampleInplaceOp>();
  auto outTensor         = getInTensor(SubsampleOp::getInIndex());
  outTensor              = subsample(outTensor, op.strides_u32());
  setOutTensor(SubsampleOp::getOutIndex(), outTensor);
}

SubsampleGradOpx::SubsampleGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<SubsampleGradOp>(op, Onnx::CustomGradOperators::SubsampleGrad);
}

// 1. Create an output tensor all 0's
// 2. Create a subsample of that tensor that matches what we did in the fwd pass
// 3. Copy the input gradients onto the subsample view of the output
// 4. Return the output tensor
void SubsampleGradOpx::grow(poplar::program::Sequence &prog) const {

  SubsampleGradOp &gradOp = getOp<SubsampleGradOp>();
  auto &in                = getInTensor(SubsampleGradOp::getInIndex());

  // Design decision: make a scalar zero variable that we expand to create
  // a tensor of the same size as the output
  auto zero = graph().addVariable(in.elementType(), {}, debugPrefix("zero"));
  graph().setTileMapping(zero, 0);
  graph().setInitialValue(zero, 0);

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
  auto outTensor = cloneNcopy(prog, output);
  poputil::mapTensorLinearly(graph(), outTensor);

  // Create a subsample view of the output
  auto ss_output = subsample(outTensor, gradOp.strides_u32());

  // Copy the input tensor into the subsampled view of the output
  prog.add(poplar::program::Copy(in, ss_output));

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
