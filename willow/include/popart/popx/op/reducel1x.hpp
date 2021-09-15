// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCEL1X_HPP
#define GUARD_NEURALNET_REDUCEL1X_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class ReduceL1Opx : public PopOpx {
public:
  ReduceL1Opx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceL1GradOpx : public PopOpx {
public:
  ReduceL1GradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
