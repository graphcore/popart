// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RANDOMUNIFORMX_HPP
#define GUARD_NEURALNET_RANDOMUNIFORMX_HPP

#include <popart/names.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/popopx.hpp>

#include <poplar/Tensor.hpp>

namespace popart {

namespace popx {

class RandomUniformOpx : public PopOpx {
public:
  RandomUniformOpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
