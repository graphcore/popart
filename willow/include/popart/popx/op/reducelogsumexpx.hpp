// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCELOGSUMEXPX_HPP
#define GUARD_NEURALNET_REDUCELOGSUMEXPX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class ReduceLogSumExpOpx : public Opx {
public:
  ReduceLogSumExpOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceLogSumExpGradOpx : public Opx {
public:
  ReduceLogSumExpGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
