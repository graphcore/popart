// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUBTRACTX_HPP
#define GUARD_NEURALNET_SUBTRACTX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/op/negatex.hpp>
#include <popart/popx/op/reducesumx.hpp>

namespace popart {

class SubtractOp;

namespace popx {

class SubtractOpx : public ElementWiseBinaryOpx {
public:
  SubtractOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

// TODO (task T5432) should inherit from ReduceSum when we have numpy
// broadcasting
class SubtractArg0GradOpx : public ReduceSumOpx {
public:
  SubtractArg0GradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
