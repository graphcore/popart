// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_INCREMENTMODX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_INCREMENTMODX_HPP_

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

template <typename T> class IncrementModComputex : public EwuComputex {

public:
  IncrementModComputex(const Op *op);

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

  static std::unique_ptr<EwuComputex> get(const Op *op) {
    return std::unique_ptr<EwuComputex>(new IncrementModComputex<T>(op));
  }

public:
  const Op *op;
  T increment;
  T modulus;
};

class IncrementModOpx : public ElementWiseUnaryOutplaceOpx {
public:
  IncrementModOpx(Op *, Devicex *);
};

class IncrementModInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  IncrementModInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_INCREMENTMODX_HPP_
