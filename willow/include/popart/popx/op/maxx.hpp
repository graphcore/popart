// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MAXX_HPP
#define GUARD_NEURALNET_MAXX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

// Refactor needed, see T7199
class MaxOpx : public ElementWiseUnaryOpx {
public:
  MaxOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class MaxArgGradOpx : public PopOpx {
public:
  MaxArgGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
