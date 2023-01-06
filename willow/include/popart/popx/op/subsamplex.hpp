// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SUBSAMPLEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SUBSAMPLEX_HPP_

#include <popart/popx/opx.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class SubsampleOpx : public Opx {

public:
  SubsampleOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class SubsampleInplaceOpx : public Opx {

public:
  SubsampleInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class SubsampleGradOpx : public Opx {
public:
  SubsampleGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SUBSAMPLEX_HPP_
