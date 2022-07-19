// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ERFX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ERFX_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ERFX_HPP_
