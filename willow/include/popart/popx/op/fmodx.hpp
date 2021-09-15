// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_FMODX_HPP
#define GUARD_NEURALNET_FMODX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class FmodOpx : public ElementWiseBinaryOpx {
public:
  FmodOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override final;
};

} // namespace popx
} // namespace popart

#endif // !GUARD_NEURALNET_FMODX_HPP
