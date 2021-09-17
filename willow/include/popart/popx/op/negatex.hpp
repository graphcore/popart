// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NEGATEX_HPP
#define GUARD_NEURALNET_NEGATEX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class NegateOpx : public ElementWiseUnaryOpx {
public:
  NegateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class NegateGradOpx : public ElementWiseUnaryOpx {
public:
  NegateGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
