// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCELOGSUMX_HPP
#define GUARD_NEURALNET_REDUCELOGSUMX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class ReduceLogSumOp;

namespace popx {

class ReduceLogSumOpx : public Opx {
public:
  ReduceLogSumOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceLogSumGradOpx : public Opx {
public:
  ReduceLogSumGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
