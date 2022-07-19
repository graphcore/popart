// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_BITWISEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_BITWISEX_HPP_

#include <popops/ExprOp.hpp>
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

class BitwiseNotOpx : public ElementWiseUnaryOpx {
public:
  BitwiseNotOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class BitwiseBinaryOpx : public ElementWiseBinaryOpx {
  popops::expr::BinaryOpType determineOpType() const;

public:
  BitwiseBinaryOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_BITWISEX_HPP_
