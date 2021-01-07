// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iterator>
#include <vector>

#include <popart/error.hpp>
#include <popart/op/reduceprod.hpp>
#include <popart/popx/op/reduceprodx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

ReduceProdOpx::ReduceProdOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ReduceProdOp>(op);
}

void ReduceProdOpx::grow(poplar::program::Sequence &prog) const {
  const auto &op   = getOp<ReduceProdOp>();
  const auto input = getInTensor(ReduceProdOp::getInIndex());

  auto output_tensor = popops::reduce(graph(),
                                      input,
                                      vector_cast<std::size_t>(op.getAxes()),
                                      {popops::Operation::MUL},
                                      prog,
                                      debugContext("mul"));

  setOutTensor(
      ReduceProdOp::getOutIndex(),
      output_tensor.reshape(outInfo(ReduceProdOp::getOutIndex()).shape_szt()));
}

ReduceProdGradOpx::ReduceProdGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ReduceProdGradOp>(op, Onnx::GradOperators::ReduceProdGrad);
}

void ReduceProdGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto &op = getOp<ReduceProdGradOp>();
  auto output = cloneNcopy(prog, getInTensor(ReduceProdGradOp::getInIndex()));
  auto fwd_input =
      cloneNcopy(prog, getInTensor(ReduceProdGradOp::getFwdInInIndex()));
  auto input_shape     = inShape(ReduceProdGradOp::getInIndex());
  auto output_shape    = outShape(ReduceProdGradOp::getOutIndex());
  const auto new_shape = vector_cast<std::size_t>(op.backwardShape());

  output = output.reshape(new_shape);

  // Broadcasting across each dimension
  for (int dim = 0; dim < new_shape.size(); ++dim) {
    if (new_shape[dim] != output_shape[dim]) {
      output = output.broadcast(static_cast<uint32_t>(output_shape[dim]), dim);
    }
  }

  std::vector<unsigned> perm(output.shape().size());
  std::vector<unsigned> reverse_perm(output.shape().size());
  auto axis = op.getAxes();
  std::vector<int64_t> others(output.shape().size() - axis.size());

  // Set difference of all axis minus the reduced axis (which are sorted)
  int64_t j = others.size() - 1;
  int64_t k = axis.size() - 1;
  for (int64_t i = perm.size() - 1; i >= 0; --i) {
    if (k < 0 || axis[k] != i) {
      others[j--] = i;
    } else {
      --k;
    }
  }

  // Permute the tensor so that all the axis that are reduced come first
  j = 0;
  for (int64_t i = 0; i < perm.size(); ++i) {
    if (i < axis.size()) {
      perm[i] = static_cast<unsigned>(axis[i]);
    } else {
      perm[i] = static_cast<unsigned>(others[j++]);
    }
    reverse_perm[perm[i]] = static_cast<unsigned>(i);
  }

  output              = output.dimShuffle(perm);
  fwd_input           = fwd_input.dimShuffle(perm);
  auto shuffled_shape = output.shape();

  // Flatten all the reduced dimensions into one
  output    = output.flatten(0, static_cast<unsigned>(axis.size()));
  fwd_input = fwd_input.flatten(0, static_cast<unsigned>(axis.size()));

  auto lrcumprod = cloneNcopy(prog, output);

  // In-place left-right cumprod
  // Left-right multiplication, leaving one out,
  // e.g. [x, y, z] -> [1, x, x * y] * [y * z, z, 1] = [y * z, x * z, x * y]
  for (int64_t i = 1; i < lrcumprod.dim(0); ++i) {
    auto left  = lrcumprod.slice(i, lrcumprod.dim(0), 0);
    auto right = lrcumprod.slice(0, lrcumprod.dim(0) - i, 0);
    popops::mapInPlace(graph(),
                       pe::Mul(pe::_1, pe::_2),
                       {left, fwd_input.slice(i - 1, i, 0)},
                       prog,
                       debugContext("mul"));
    popops::mapInPlace(
        graph(),
        pe::Mul(pe::_1, pe::_2),
        {right,
         fwd_input.slice(lrcumprod.dim(0) - i, lrcumprod.dim(0) - i + 1, 0)},
        prog,
        debugContext("mul"));
  }

  // Undo flattening
  output = lrcumprod.reshape(shuffled_shape);

  // Undo permutation
  output = output.dimShuffle(reverse_perm);

  // output now matches the shape of output_shape
  setOutTensor(ReduceProdGradOp::getOutIndex(), output);
}

namespace {
OpxCreator<ReduceProdOpx> reduceProdOpxCreator(
    {Onnx::Operators::ReduceProd_1, Onnx::Operators::ReduceProd_11});
OpxCreator<ReduceProdGradOpx>
    reduceProdGradGradOpxCreator(Onnx::GradOperators::ReduceProdGrad);
} // namespace

} // namespace popx
} // namespace popart
