// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_HARDSIGMOIDX_HPP
#define GUARD_NEURALNET_HARDSIGMOIDX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class HardSigmoidComputex : public EwuComputex {
public:
  HardSigmoidComputex(float _alpha, float _beta) : alpha(_alpha), beta(_beta) {}

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get(float _alpha, float _beta) {
    return std::unique_ptr<EwuComputex>(new HardSigmoidComputex(
        static_cast<float>(_alpha), static_cast<float>(_beta)));
  }

  float getAlpha() const { return alpha; }
  float getBeta() const { return beta; }

private:
  float alpha;
  float beta;
};

class HardSigmoidOpx : public ElementWiseUnaryOutplaceOpx {
public:
  HardSigmoidOpx(Op *, Devicex *);
};

class HardSigmoidInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  HardSigmoidInplaceOpx(Op *, Devicex *);
};

class HardSigmoidGradOpx : public PopOpx {
public:
  HardSigmoidGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
