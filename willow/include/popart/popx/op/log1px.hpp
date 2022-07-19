// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_LOG1PX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_LOG1PX_HPP_

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

class Log1pComputex : public EwuComputex {

public:
  Log1pComputex() = default;

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
    return std::unique_ptr<EwuComputex>(new Log1pComputex());
  }
};

class Log1pOpx : public ElementWiseUnaryOutplaceOpx {
public:
  Log1pOpx(Op *, Devicex *);
};

class Log1pInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  Log1pInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_LOG1PX_HPP_
