// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LRELUX_HPP
#define GUARD_NEURALNET_LRELUX_HPP

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

class LeakyReluComputex : public EwuComputex {
public:
  LeakyReluComputex(float _alpha) : alpha(_alpha) {}

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

  static std::unique_ptr<EwuComputex> get(float _alpha) {
    return std::unique_ptr<EwuComputex>(new LeakyReluComputex(_alpha));
  }

  float getAlpha() const { return alpha; }

  static float getAlphaFromLReluOp(Op *op);
  static float getAlphaFromLReluInplaceOp(Op *op);

private:
  float alpha;
};

class LeakyReluOpx : public ElementWiseUnaryOutplaceOpx {
public:
  LeakyReluOpx(Op *, Devicex *);
};

class LeakyReluInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  LeakyReluInplaceOpx(Op *, Devicex *);
};

class LeakyReluGradOpx : public PopOpx {
public:
  LeakyReluGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
