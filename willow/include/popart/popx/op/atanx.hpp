// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ATANX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ATANX_HPP_

#include <memory>
#include <string>
#include <poplar/Tensor.hpp>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/opx.hpp"

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

class AtanComputex : public EwuComputex {

public:
  AtanComputex() = default;

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
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

class AtanGradOpx : public Opx {
public:
  AtanGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ATANX_HPP_
