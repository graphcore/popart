// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SUBTRACTX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SUBTRACTX_HPP_

#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/op/reducesumx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class SubtractOpx : public ElementWiseBinaryOpx {
public:
  SubtractOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class SubtractArg0GradOpx : public ReduceSumOpx {
public:
  SubtractArg0GradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SUBTRACTX_HPP_
