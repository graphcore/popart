#include <poponnx/error.hpp>
#include <poponnx/op/argmax.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/argextremax.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Sort.hpp>
#include <poputil/TileMapping.hpp>

namespace poponnx {
namespace popx {

ArgExtremaOpx::ArgExtremaOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ArgExtremaOp>(op);

  auto argextrema = dynamic_cast<ArgExtremaOp *>(op);
  axis            = static_cast<unsigned>(argextrema->getAxis());
  keepdims        = argextrema->getKeepDims() != 0;
}

// Fill a tensor with 0, 1, 2, ... on a given axis
static void AssignIota(poplar::Graph &graph,
                       poplar::program::Sequence &prog,
                       poplar::Tensor t,
                       unsigned axis) {
  prog.add(poplar::program::WriteUndef(t));

  // Create a constant of the desired shape
  const auto size = t.dim(axis);
  std::vector<int> values(size);
  std::iota(values.begin(), values.end(), 0);

  auto c =
      graph.addConstant(poplar::INT, {size}, poplar::ArrayRef<int>(values));
  poputil::mapTensorLinearly(graph, c);

  // DimShuffle the given axis to the back
  std::vector<unsigned> permutation(t.rank());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[axis], permutation.back());
  t = t.dimShuffle(permutation);

  // Reshape to a 2D tensor
  const auto shape   = t.shape();
  const auto product = std::accumulate(shape.begin(),
                                       shape.end() - 1,
                                       std::size_t(1),
                                       std::multiplies<std::size_t>());

  t = t.reshape({product, size});

  // Loop over the front dimension and copy in the constant.
  for (int i = 0; i < product; ++i) {
    prog.add(poplar::program::Copy(c, t[i]));
  }
}

void ArgExtremaOpx::grow(poplar::program::Sequence &prog) const {
  auto input   = getInTensor(ArgExtremaOp::getInIndex());
  auto indices = graph().clone(poplar::INT, input);

  AssignIota(graph(), prog, indices, axis);

  // Sort the indices with respect to the input tensor
  auto output = popops::sortKeyValue(graph(), input, indices, axis, prog);

  // Use the specialised slice
  auto result = selectSlice(output, axis);

  // Squeeze out the axis dimension?
  if (!keepdims) {
    result = result.squeeze({axis});
  }

  setOutTensor(ArgExtremaOp::getOutIndex(), result);
}

poplar::Tensor ArgExtremaOpx::createInput(InIndex) const {
  // Create an input that will minimise the amount of exchange in sort. This
  // means minimising the number of tile boundaries on the given axis.

  auto info = inInfo(ArgExtremaOp::getInIndex());

  // Put the given axis at the back of the shape.
  auto shape = info.shape_szt();
  std::swap(shape[axis], shape.back());

  // Create a new variable of the modified shape
  auto t = graph().addVariable(popType(info), shape);

  // Map it linearly
  poputil::mapTensorLinearly(graph(), t);

  // DimShuffle back to the desired shape
  std::vector<unsigned> permutation(t.rank());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[axis], permutation.back());
  return t.dimShuffle(permutation);
}

InputCreatorType ArgExtremaOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CANCREATE;
}

std::vector<TensorId> ArgExtremaOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

} // namespace popx
} // namespace poponnx
