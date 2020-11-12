// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCEMAXX_HPP
#define GUARD_NEURALNET_REDUCEMAXX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class ReduceMaxOpx : public Opx {
public:
  ReduceMaxOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceMaxGradOpx : public Opx {
public:
  ReduceMaxGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
