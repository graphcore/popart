// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_THRESHOLDEDRELUX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_THRESHOLDEDRELUX_HPP_

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

class ThresholdedReluComputex : public EwuComputex {
public:
  ThresholdedReluComputex(float _alpha) : alpha(_alpha) {}

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get(float _alpha) {
    return std::unique_ptr<EwuComputex>(
        new ThresholdedReluComputex(static_cast<float>(_alpha)));
  }

  float getAlpha() const { return alpha; }

private:
  float alpha;
};

class ThresholdedReluOpx : public ElementWiseUnaryOutplaceOpx {
public:
  ThresholdedReluOpx(Op *, Devicex *);
};

class ThresholdedReluInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  ThresholdedReluInplaceOpx(Op *, Devicex *);
};

class ThresholdedReluGradOpx : public PopOpx {
public:
  ThresholdedReluGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_THRESHOLDEDRELUX_HPP_
