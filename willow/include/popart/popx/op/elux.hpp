// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ELUX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ELUX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <memory>
#include <string>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/popopx.hpp"

namespace snap {
class Graph;
class Tensor;
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class EluComputex : public EwuComputex {
public:
  EluComputex(float alpha) : alpha_(alpha) {}

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get(float alpha) {
    return std::unique_ptr<EwuComputex>(
        new EluComputex(static_cast<float>(alpha)));
  }

  float alpha() const { return alpha_; }

private:
  float alpha_;
};

class EluOpx : public ElementWiseUnaryOutplaceOpx {
public:
  EluOpx(Op *, Devicex *);
};

class EluInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  EluInplaceOpx(Op *, Devicex *);
};

class EluGradOpx : public PopOpx {
public:
  EluGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ELUX_HPP_
