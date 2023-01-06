// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_LEAKYRELUX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_LEAKYRELUX_HPP_

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

class LeakyReluComputex : public EwuComputex {
public:
  LeakyReluComputex(float _alpha) : alpha(_alpha) {}

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

class LeakyReluGradOpx : public Opx {
public:
  LeakyReluGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_LEAKYRELUX_HPP_
