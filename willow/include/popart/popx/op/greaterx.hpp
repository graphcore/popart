// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GREATERX_HPP
#define GUARD_NEURALNET_GREATERX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

class GreaterOp;

namespace popx {

class GreaterOpx : public BinaryComparisonOpx {
public:
  GreaterOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
