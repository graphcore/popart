// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_UNSQUEEZEX_HPP
#define GUARD_NEURALNET_UNSQUEEZEX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class UnsqueezeOp;
class UnsqueezeGradOp;

namespace popx {

class UnsqueezeOpx : public Opx {
public:
  UnsqueezeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class UnsqueezeGradOpx : public Opx {
public:
  UnsqueezeGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
