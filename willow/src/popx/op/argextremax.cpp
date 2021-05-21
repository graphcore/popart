// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/Cast.hpp>

#include <popart/error.hpp>
#include <popart/op/argextrema.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/argextremax.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

ArgExtremaOpx::ArgExtremaOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ArgExtremaOp>(op);
}

void ArgExtremaOpx::grow(poplar::program::Sequence &prog) const {
  auto input         = getInTensor(0);
  auto dims          = input.shape().size();
  auto &argExtremaOp = getOp<ArgExtremaOp>();
  auto axis          = argExtremaOp.getAxis();

  // The axis in which to compute the arg indices should be the last axis in
  // axes. The rest of the axes should be in ascending order.
  std::vector<unsigned int> axes(dims);
  std::iota(axes.begin(), axes.end(), 0);
  axes.erase(axes.begin() + axis);
  axes.push_back(static_cast<unsigned int>(axis));
  input = input.dimShuffle(axes);

  // Reshape the input to a 2d tensor
  auto shape        = input.shape();
  std::size_t dim_0 = std::accumulate(
      shape.begin(), shape.end() - 1, 1, std::multiplies<std::size_t>());
  std::size_t dim_1 = shape.back();
  input             = input.reshape({dim_0, dim_1});

  // Do the extrema operation
  auto result = extremaOp(prog, input);

  std::vector<std::size_t> new_shape;
  std::copy(shape.begin(), shape.end() - 1, std::back_inserter(new_shape));

  if (argExtremaOp.getKeepDims()) {
    new_shape.insert(new_shape.begin() + axis, 1);
  }

  result = result.reshape(new_shape);

  result = popops::cast(graph().getPoplarGraph(),
                        result,
                        poplar::INT,
                        prog,
                        debugContext("cast"));
  setOutTensor(0, result);
}

} // namespace popx
} // namespace popart
