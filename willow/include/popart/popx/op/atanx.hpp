// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ATANX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ATANX_HPP_

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

class AtanComputex : public EwuComputex {

public:
  AtanComputex() = default;

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
    return std::unique_ptr<EwuComputex>(new AtanComputex);
  }
};

class AtanOpx : public ElementWiseUnaryOutplaceOpx {
public:
  AtanOpx(Op *, Devicex *);
};

class AtanInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  AtanInplaceOpx(Op *, Devicex *);
};

class AtanGradOpx : public PopOpx {
public:
  AtanGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ATANX_HPP_
