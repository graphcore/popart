// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCEL2X_HPP
#define GUARD_NEURALNET_REDUCEL2X_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class ReduceL2Opx : public PopOpx {
public:
  ReduceL2Opx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

class ReduceL2GradOpx : public PopOpx {
public:
  ReduceL2GradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
