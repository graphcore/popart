// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_INSTANCENORMX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_INSTANCENORMX_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_INSTANCENORMX_HPP_
