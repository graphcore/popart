// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCEMAXX_HPP
#define GUARD_NEURALNET_REDUCEMAXX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class ReduceMaxOpx : public PopOpx {
public:
  ReduceMaxOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

class ReduceMaxGradOpx : public PopOpx {
public:
  ReduceMaxGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
