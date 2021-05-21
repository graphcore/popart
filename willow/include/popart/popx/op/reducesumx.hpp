// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCESUMX_HPP
#define GUARD_NEURALNET_REDUCESUMX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class ReduceSumOpx : public PopOpx {
public:
  ReduceSumOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceSumGradOpx : public PopOpx {
public:
  ReduceSumGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
