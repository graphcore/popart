// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DIVX_HPP
#define GUARD_NEURALNET_DIVX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class ModOpx : public ElementWiseBinaryOpx {
public:
  ModOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override final;
};

} // namespace popx
} // namespace popart

#endif // !GUARD_NEURALNET_DIVX_HPP
