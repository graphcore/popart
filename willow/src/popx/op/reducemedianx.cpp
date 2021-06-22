// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <cmath>

#include <popops/Sort.hpp>
#include <popops/Zero.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/reducemedianx.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/op/sortutilx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

ReduceMedianOpx::ReduceMedianOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ReduceMedianOp>(op);
}

void ReduceMedianOpx::grow(poplar::program::Sequence &prog) const {
  const auto &op   = getOp<ReduceMedianOp>();
  const auto &axes = op.getAxes();
  const auto &input =
      getInTensor(ReduceMedianOp::getInIndex()).getPoplarTensor();
  auto output = cloneNcopy(prog, input);

  const auto &pp = reducemedianinternal::computePreprocessingParams(
      op.inShape(ReduceMedianOp::getInIndex()), axes);

  // Make sure all reduction axes are at the end and flatten the dimensions
  // over which the reduction is performed.
  output                   = output.dimShuffle(pp.dim_permute);
  size_t num_reduced_elems = 1;
  const auto output_shape  = output.shape();
  for (size_t i = pp.axes_complement.size(); i < output_shape.size(); ++i) {
    num_reduced_elems *= output_shape.at(i);
  }
  output = output.reshapePartial(
      pp.axes_complement.size(), output.rank(), {num_reduced_elems});

  // Sort the flattened dimension along with indices and select the median
  // values.
  auto indices = sortutilx::getIotaTensor(graph(),
                                          output,
                                          pp.axes_complement.size(),
                                          prog,
                                          getDebugNameAndId("iotaTensor"));
  popops::sortKeyValueInPlace(graph().getPoplarGraph(),
                              output,
                              indices,
                              pp.axes_complement.size(),
                              prog,
                              debugContext("sort"));

  size_t median_i;
  if (num_reduced_elems > 1) {
    median_i = num_reduced_elems % 2 == 0 ? floor(num_reduced_elems / 2) - 1
                                          : floor(num_reduced_elems / 2);
  } else {
    median_i = 0;
  }

  output  = output.slice(median_i, median_i + 1, pp.axes_complement.size());
  indices = indices.slice(median_i, median_i + 1, pp.axes_complement.size());

  // Reverse the dim shuffle and reshape to the specified output shape.
  output  = output.reshapePartial(pp.axes_complement.size(),
                                 output.rank(),
                                 std::vector<size_t>(axes.size(), 1));
  indices = indices.reshapePartial(pp.axes_complement.size(),
                                   indices.rank(),
                                   std::vector<size_t>(axes.size(), 1));
  output  = output.dimShuffle(pp.dim_permute_reverse);
  indices = indices.dimShuffle(pp.dim_permute_reverse);

  setOutTensor(
      ReduceMedianOp::getOutIndex(),
      snap::Tensor{
          output.reshape(outInfo(ReduceMedianOp::getOutIndex()).shape_szt()),
          graph()});
  setOutTensor(
      ReduceMedianOp::getIndicesOutIndex(),
      snap::Tensor{
          indices.reshape(
              outInfo(ReduceMedianOp::getIndicesOutIndex()).shape_szt()),
          graph()});
}

ReduceMedianGradOpx::ReduceMedianGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ReduceMedianGradOp>(op, Onnx::GradOperators::ReduceMedianGrad);
}

void ReduceMedianGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto &grad_op = getOp<ReduceMedianGradOp>();
  const auto &axes    = grad_op.getAxes();
  const auto &backward_shape =
      vector_cast<std::size_t>(grad_op.backwardShape());
  const auto &output_shape = outShape(ReduceMedianGradOp::getOutIndex());

  auto grad = graph().getPoplarGraph().addVariable(
      popType(grad_op.outInfo(ReduceMedianGradOp::getOutIndex())),
      vector_cast<std::size_t>(output_shape),
      poplar::VariableMappingMethod::LINEAR,
      debugContext("initGrad"));
  popops::zero(graph().getPoplarGraph(), grad, prog, debugContext("zeroGrad"));

  auto grad_top = cloneNcopy(
      prog, getInTensor(ReduceMedianGradOp::getInIndex()).getPoplarTensor());
  grad_top     = grad_top.reshape(backward_shape);
  auto indices = cloneNcopy(
      prog,
      getInTensor(ReduceMedianGradOp::getIndicesInIndex()).getPoplarTensor());
  indices = indices.reshape(backward_shape);

  const auto &pp =
      reducemedianinternal::computePreprocessingParams(output_shape, axes);

  // Make sure all reduction axes are at the end.
  grad     = grad.dimShuffle(pp.dim_permute);
  grad_top = grad_top.dimShuffle(pp.dim_permute);
  indices  = indices.dimShuffle(pp.dim_permute);

  // Flatten the tensors so that we can scatter using the forward pass indices.
  const auto grad_shape    = grad.shape();
  size_t num_reduced_elems = 1;
  for (size_t i = pp.axes_complement.size(); i < grad_shape.size(); ++i) {
    num_reduced_elems *= grad_shape.at(i);
  }
  grad = grad.reshapePartial(
      pp.axes_complement.size(), grad.rank(), {num_reduced_elems});
  grad_top =
      grad_top.reshapePartial(pp.axes_complement.size(), grad_top.rank(), {1});
  indices =
      indices.reshapePartial(pp.axes_complement.size(), indices.rank(), {1});

  // Scatter the gradient from the top to indices corresponding to median values
  // in the op's input tensor.
  scatterutilx::growScatter(prog,
                            graph(),
                            indices,
                            grad_top,
                            grad,
                            pp.axes_complement.size(),
                            getDebugNameAndId("scatter"));

  grad = grad.reshape(grad_shape);
  grad = grad.dimShuffle(pp.dim_permute_reverse);
  setOutTensor(ReduceMedianGradOp::getOutIndex(), snap::Tensor{grad, graph()});
}

namespace reducemedianinternal {

PreprocessingParams
computePreprocessingParams(const Shape &input_shape,
                           const std::vector<int64_t> &axes) {
  // Compute the axes that are not reduced first, this is useful for computing
  // 'dim_permute' and 'dim_permute_reverse'.
  std::vector<int64_t> axes_complement;
  axes_complement.reserve(input_shape.size() - axes.size());
  size_t axes_i = 0;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (axes_i == axes.size() || i != axes.at(axes_i)) {
      axes_complement.push_back(i);
    } else {
      axes_i++;
    }
  }

  std::vector<unsigned> dim_permute;
  dim_permute.reserve(input_shape.size());
  std::vector<unsigned> dim_permute_reverse(input_shape.size());
  for (size_t i = 0; i < input_shape.size(); ++i) {
    int axis;
    if (i < axes_complement.size()) {
      axis = axes_complement.at(i);
    } else {
      axis = axes.at(i - axes_complement.size());
    }
    dim_permute.push_back(axis);
    dim_permute_reverse.at(axis) = i;
  }

  PreprocessingParams result = {
      axes_complement, dim_permute, dim_permute_reverse};
  return result;
}

} // namespace reducemedianinternal

namespace {
OpxCreator<ReduceMedianOpx>
    ReduceMedianOpxCreator({Onnx::AiGraphcore::OpSet1::ReduceMedian});
OpxCreator<ReduceMedianGradOpx>
    ReduceMedianGradOpxCreator(Onnx::GradOperators::ReduceMedianGrad);
} // namespace

} // namespace popx
} // namespace popart
