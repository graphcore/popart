// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RANDOMNORMALX_HPP
#define GUARD_NEURALNET_RANDOMNORMALX_HPP

#include <popart/names.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/popopx.hpp>

#include <poplar/Tensor.hpp>

namespace popart {

namespace popx {

class RandomNormalOpx : public PopOpx {
public:
  RandomNormalOpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
