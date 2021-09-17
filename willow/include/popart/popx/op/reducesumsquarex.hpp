// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCESUMSQUAREX_HPP
#define GUARD_NEURALNET_REDUCESUMSQUAREX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class ReduceSumSquareOpx : public PopOpx {
public:
  ReduceSumSquareOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceSumSquareGradOpx : public PopOpx {
public:
  ReduceSumSquareGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
