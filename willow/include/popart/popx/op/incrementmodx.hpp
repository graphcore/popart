// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_INCREMENTMODX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_INCREMENTMODX_HPP_

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

template <typename T> class IncrementModComputex : public EwuComputex {

public:
  IncrementModComputex(const Op *op);

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
