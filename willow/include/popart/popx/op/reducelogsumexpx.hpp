// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCELOGSUMEXPX_HPP
#define GUARD_NEURALNET_REDUCELOGSUMEXPX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class ReduceLogSumExpOpx : public PopOpx {
public:
  ReduceLogSumExpOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

class ReduceLogSumExpGradOpx : public PopOpx {
public:
  ReduceLogSumExpGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
