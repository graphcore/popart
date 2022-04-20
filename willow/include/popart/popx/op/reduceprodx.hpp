// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCEPRODX_HPP
#define GUARD_NEURALNET_REDUCEPRODX_HPP

#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class ReduceProdOpx : public PopOpx {
public:
  ReduceProdOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

class ReduceProdGradOpx : public PopOpx {
public:
  ReduceProdGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
