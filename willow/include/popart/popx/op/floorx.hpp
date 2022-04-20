// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_FLOORX_HPP
#define GUARD_NEURALNET_FLOORX_HPP

#include "popart/popx/debugcontextx.hpp"
#include <memory>
#include <snap/Tensor.hpp>
#include <string>
#include <popart/popx/op/elementwisex.hpp>

namespace snap {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class FloorComputex : public EwuComputex {

public:
  FloorComputex() {}

  snap::Tensor outplace(snap::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &tensor,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new FloorComputex());
  }
};

class FloorOpx : public ElementWiseUnaryOutplaceOpx {
public:
  FloorOpx(Op *, Devicex *);
};

class FloorInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  FloorInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
