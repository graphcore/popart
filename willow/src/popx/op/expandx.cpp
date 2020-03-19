#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popart/error.hpp>
#include <popart/op/expand.hpp>
#include <popart/popx/op/expandx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>
namespace popart {
namespace popx {

BaseExpandOpx::BaseExpandOpx(Op *op_, Devicex *devicex)
    : Opx(op_, devicex), op(static_cast<ExpandOp *>(op_)) {}

ExpandOpx::ExpandOpx(Op *op_, Devicex *devicex) : BaseExpandOpx(op_, devicex) {
  verifyOp<ExpandOp>(op_, {Onnx::Operators::Expand_8});
}

InputCreatorType BaseExpandOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CANUNWIND;
}

poplar::Tensor BaseExpandOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                 InIndex inIndex,
                                                 OutIndex) const {
  if (inIndex == ExpandOp::getInTensorIndex()) {
    auto output_shape = op->getOutShape();
    auto input_shape  = inShape(inIndex);
    auto dim_diff     = output_shape.size() - input_shape.size();
    for (int64_t dim = 0; dim < output_shape.size(); ++dim) {
      auto input_shape_of_current_dim =
          (dim < dim_diff) ? 1 : input_shape[dim - dim_diff];
      if (output_shape[dim] != input_shape[dim - dim_diff]) {
        tensor =
            tensor.slice(static_cast<std::size_t>(0),
                         static_cast<std::size_t>(input_shape_of_current_dim),
                         static_cast<std::size_t>(dim));
      }
    }
  }
  return tensor;
}

view::RegMap BaseExpandOpx::unwindRegion(InIndex inIndex,
                                         OutIndex outIndex) const {
  ExpandOp *cop = dynamic_cast<ExpandOp *>(this->op_p);
  return cop->bwdRegMap(inIndex, outIndex);
}

void BaseExpandOpx::expand_broadcast(const Shape output_shape,
                                     poplar::Tensor &expand) const {
  /* Make the rank of the tensor to be expanded and the output the same by
     adding 1 as the higher dimensions example: where a Tensor of shape (3,1) is
     expanded to shape (3,2,6), the tensor to be expanded is reshaped from (3,1)
     to (3,1,1)
  */
  for (int64_t rank_diff = output_shape.size() - expand.shape().size();
       rank_diff > 0;
       --rank_diff) {
    expand = expand.expand({0});
  }
  auto new_input_shape = expand.shape();
  // Broadcasting across each dimension
  // for (int dim = output_shape.size(); dim >= 0; --dim) {
  for (int dim = 0; dim < output_shape.size(); dim++) {
    if (output_shape[dim] != new_input_shape[dim]) {
      expand = expand.broadcast(static_cast<uint32_t>(output_shape[dim]), dim);
    }
  }
}

void ExpandOpx::grow(poplar::program::Sequence &prog) const {

  auto output_shape = outShape(ExpandOp::getOutIndex());
  auto expand = cloneNcopy(prog, getInTensor(ExpandOp::getInTensorIndex()));
  expand_broadcast(output_shape, expand);
  setOutTensor(ExpandOp::getOutIndex(), expand);
}

ExpandInplaceOpx::ExpandInplaceOpx(Op *op_, Devicex *devicex)
    : BaseExpandOpx(op_, devicex) {
  verifyOp<ExpandOp>(op_);
}

void ExpandInplaceOpx::grow(poplar::program::Sequence &) const {
  auto output_shape = outShape(ExpandOp::getOutIndex());
  auto expand       = getInTensor(ExpandOp::getInTensorIndex());
  expand_broadcast(output_shape, expand);
  setOutTensor(ExpandOp::getOutIndex(), expand);
}

ExpandGradOpx::ExpandGradOpx(Op *op_, Devicex *devicex) : Opx(op_, devicex) {
  verifyOp<ExpandGradOp>(op_, Onnx::GradOperators::ExpandGrad);
  auto expand_grad_op = dynamic_cast<ExpandGradOp *>(op_);
  if (expand_grad_op) {
    xShape = expand_grad_op->getXShape();
  } else {
    throw error("ExpandGradOpx::ExpandGradOpx : Constructed with invalid op ");
  }
}

void ExpandGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto dY = getInTensor(ExpandGradOp::getDYIndex());

  std::vector<size_t> axes;
  const int64_t offset = dY.rank() - xShape.size();
  for (int i = 0; i < dY.rank(); i++) {
    if (i < offset || xShape[i - offset] == 1) {
      axes.push_back(i);
    }
  }

  auto dX = popops::reduce(
      graph(), dY, axes, {popops::Operation::ADD}, prog, debugPrefix("add"));
  dX = dX.reshape(xShape);
  setOutTensor(ExpandGradOp::getOutIndex(), cloneNcopy(prog, dX));
}

namespace {
OpxCreator<ExpandOpx> expandOpxCreator({Onnx::Operators::Expand_8});
OpxCreator<ExpandInplaceOpx>
    expandInplaceOpxCreator(Onnx::CustomOperators::ExpandInplace);
OpxCreator<ExpandGradOpx> expandGradOpxCreator(Onnx::GradOperators::ExpandGrad);
} // namespace

} // namespace popx
} // namespace popart
