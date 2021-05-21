// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MEANX_HPP
#define GUARD_NEURALNET_MEANX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class MeanOpx : public ElementWiseUnaryOpx {
public:
  MeanOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class MeanArgGradOpx : public PopOpx {
public:
  MeanArgGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
