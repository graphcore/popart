// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ROUNDX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ROUNDX_HPP_

#include <memory>
#include <string>
#include <poplar/Tensor.hpp>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/debugcontextx.hpp"

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class RoundComputex : public EwuComputex {

public:
  RoundComputex() {}

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &tensor,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new RoundComputex());
  }
};

class RoundOpx : public ElementWiseUnaryOutplaceOpx {
public:
  RoundOpx(Op *, Devicex *);
};

class RoundInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  RoundInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ROUNDX_HPP_
