// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXPX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXPX_HPP_

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

class ExpComputex : public EwuComputex {

public:
  ExpComputex() = default;

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
    return std::unique_ptr<EwuComputex>(new ExpComputex);
  }
};

class ExpOpx : public ElementWiseUnaryOutplaceOpx {
public:
  ExpOpx(Op *, Devicex *);
};

class ExpInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  ExpInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXPX_HPP_
