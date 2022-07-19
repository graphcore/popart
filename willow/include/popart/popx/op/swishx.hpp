// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SWISHX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SWISHX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <memory>
#include <snap/Tensor.hpp>
#include <string>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/popopx.hpp"

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

class SwishComputex : public EwuComputex {

public:
  SwishComputex() = default;

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
    return std::unique_ptr<EwuComputex>(new SwishComputex());
  }
};

class SwishOpx : public ElementWiseUnaryOutplaceOpx {
public:
  SwishOpx(Op *, Devicex *);
};

class SwishInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  SwishInplaceOpx(Op *, Devicex *);
};

class SwishGradOpx : public PopOpx {
public:
  SwishGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SWISHX_HPP_
