// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SELUX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SELUX_HPP_

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

class SeluComputex : public EwuComputex {
public:
  SeluComputex(float _alpha, float _gamma) : alpha(_alpha), gamma(_gamma) {}

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get(float _alpha, float _gamma) {
    return std::unique_ptr<EwuComputex>(new SeluComputex(
        static_cast<float>(_alpha), static_cast<float>(_gamma)));
  }

  float getAlpha() const { return alpha; }
  float getGamma() const { return gamma; }

private:
  float alpha;
  float gamma;
};

class SeluOpx : public ElementWiseUnaryOutplaceOpx {
public:
  SeluOpx(Op *, Devicex *);
};

class SeluInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  SeluInplaceOpx(Op *, Devicex *);
};

class SeluGradOpx : public PopOpx {
public:
  SeluGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SELUX_HPP_
