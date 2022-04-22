// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUBSAMPLEX_HPP
#define GUARD_NEURALNET_SUBSAMPLEX_HPP

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

class SubsampleOpx : public PopOpx {

public:
  SubsampleOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class SubsampleInplaceOpx : public PopOpx {

public:
  SubsampleInplaceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class SubsampleGradOpx : public PopOpx {
public:
  SubsampleGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
