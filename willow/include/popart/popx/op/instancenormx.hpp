// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_INSTANCENORMX_HPP
#define GUARD_NEURALNET_INSTANCENORMX_HPP

#include <popart/popx/op/normx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class InstanceNormOpx : public NormOpx {
public:
  InstanceNormOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
};

class InstanceNormGradOpx : public NormOpx {
public:
  InstanceNormGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
