// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_DROPOUTX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_DROPOUTX_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_DROPOUTX_HPP_
