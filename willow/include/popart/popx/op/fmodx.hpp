// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_FMODX_HPP
#define GUARD_NEURALNET_FMODX_HPP

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

class FmodOpx : public ElementWiseBinaryOpx {
public:
  FmodOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override final;
};

} // namespace popx
} // namespace popart

#endif // !GUARD_NEURALNET_FMODX_HPP
