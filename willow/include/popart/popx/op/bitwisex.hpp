// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BITWISEX_HPP
#define GUARD_NEURALNET_BITWISEX_HPP

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

#endif // !GUARD_NEURALNET_BITWISEX_HPP
