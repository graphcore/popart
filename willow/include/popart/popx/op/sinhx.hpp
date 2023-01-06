// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SINHX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SINHX_HPP_

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

class SinhComputex : public EwuComputex {

public:
  SinhComputex() = default;

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
    return std::unique_ptr<EwuComputex>(new SinhComputex);
  }
};

class SinhOpx : public ElementWiseUnaryOutplaceOpx {
public:
  SinhOpx(Op *, Devicex *);
};

class SinhInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  SinhInplaceOpx(Op *, Devicex *);
};

class SinhGradOpx : public Opx {
public:
  SinhGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SINHX_HPP_
