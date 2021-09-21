// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCEMINX_HPP
#define GUARD_NEURALNET_REDUCEMINX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class ReduceMinOpx : public PopOpx {
public:
  ReduceMinOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceMinGradOpx : public PopOpx {
public:
  ReduceMinGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
