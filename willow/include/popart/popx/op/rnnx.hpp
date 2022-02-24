// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RNNX_HPP
#define GUARD_NEURALNET_RNNX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {

class RNNOp;

namespace popx {

class RNNOpx : public PopOpx {
public:
  RNNOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const;

private:
  // Add the 2 biases from input and set the result in bias
  // Initialize to 0 in case not provided by user
  snap::Tensor getBias(snap::program::Sequence &) const;
  // Initialize initialH to 0 in case not provided by user
  snap::Tensor getInitialH(snap::program::Sequence &) const;
};

class RNNGradOpx : public PopOpx {
public:
  RNNGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  snap::Tensor getLastOutputGrad(snap::program::Sequence &) const;
  snap::Tensor getFullOutputGrad(snap::program::Sequence &) const;
};

} // namespace popx
} // namespace popart

#endif