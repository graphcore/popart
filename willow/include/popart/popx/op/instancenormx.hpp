// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_INSTANCENORMX_HPP
#define GUARD_NEURALNET_INSTANCENORMX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/normx.hpp>

namespace popart {

namespace popx {

class InstanceNormOpx : public NormOpx {
public:
  InstanceNormOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
};

class InstanceNormGradOpx : public NormOpx {
public:
  InstanceNormGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
