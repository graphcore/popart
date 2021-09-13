// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCELOGSUMX_HPP
#define GUARD_NEURALNET_REDUCELOGSUMX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class ReduceLogSumOpx : public PopOpx {
public:
  ReduceLogSumOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

class ReduceLogSumGradOpx : public PopOpx {
public:
  ReduceLogSumGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
