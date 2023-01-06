// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_THRESHOLDEDRELUX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_THRESHOLDEDRELUX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <memory>
#include <string>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/opx.hpp"

namespace poplar {
class Graph;
class Tensor;
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class ThresholdedReluComputex : public EwuComputex {
public:
  ThresholdedReluComputex(float _alpha) : alpha(_alpha) {}

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
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

class ThresholdedReluGradOpx : public Opx {
public:
  ThresholdedReluGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_THRESHOLDEDRELUX_HPP_
