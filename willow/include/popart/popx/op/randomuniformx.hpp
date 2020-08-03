// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RANDOMUNIFORMX_HPP
#define GUARD_NEURALNET_RANDOMUNIFORMX_HPP

#include <popart/names.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/normx.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <poplar/Tensor.hpp>

namespace popart {

namespace popx {

class RandomUniformOpx : public Opx {
public:
  RandomUniformOpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
