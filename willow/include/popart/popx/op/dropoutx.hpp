// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DROPOUTX_HPP
#define GUARD_NEURALNET_DROPOUTX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/popopx.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class DropoutOpx : public ElementWiseUnaryOpx {
public:
  DropoutOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
  InputCreatorType getInputCreatorType(InIndex) const override;
};

} // namespace popx
} // namespace popart

#endif
