// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ERFX_HPP
#define GUARD_NEURALNET_ERFX_HPP

#include <popart/popx/op/elementwisex.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class ErfxOpx : public ElementWiseUnaryOpx {
public:
  ErfxOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class ErfxGradOpx : public ElementWiseUnaryOpx {
public:
  ErfxGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
