// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXPM1X_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXPM1X_HPP_

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

class Expm1Computex : public EwuComputex {

public:
  Expm1Computex() = default;

  snap::Tensor outplace(snap::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new Expm1Computex());
  }
};

class Expm1Opx : public ElementWiseUnaryOutplaceOpx {
public:
  Expm1Opx(Op *, Devicex *);
};

class Expm1InplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  Expm1InplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXPM1X_HPP_
