// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD1ACCLUPDATEX_HPP
#define GUARD_NEURALNET_SGD1ACCLUPDATEX_HPP

#include <popart/popx/op/varupdatex.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class SGD1AcclUpdateOpx : public VarUpdateOpx {
public:
  SGD1AcclUpdateOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
