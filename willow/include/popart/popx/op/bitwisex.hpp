// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BITWISEX_HPP
#define GUARD_NEURALNET_BITWISEX_HPP

#include <popops/ExprOp.hpp>
#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class BitwiseNotOpx : public ElementWiseUnaryOpx {
public:
  BitwiseNotOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class BitwiseBinaryOpx : public ElementWiseBinaryOpx {
  popops::expr::BinaryOpType determineOpType() const;

public:
  BitwiseBinaryOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // !GUARD_NEURALNET_BITWISEX_HPP
