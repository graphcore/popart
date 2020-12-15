// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_THRESHOLDEDRELUX_HPP
#define GUARD_NEURALNET_THRESHOLDEDRELUX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

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

#endif
