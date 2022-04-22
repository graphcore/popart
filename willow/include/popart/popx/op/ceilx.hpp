// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CEILX_HPP
#define GUARD_NEURALNET_CEILX_HPP

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

class CeilComputex : public EwuComputex {

public:
  CeilComputex() {}

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
    return std::unique_ptr<EwuComputex>(new CeilComputex());
  }
};

class CeilOpx : public ElementWiseUnaryOutplaceOpx {
public:
  CeilOpx(Op *, Devicex *);
};

class CeilInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  CeilInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
