// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SELUX_HPP
#define GUARD_NEURALNET_SELUX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class SeluComputex : public EwuComputex {
public:
  SeluComputex(float _alpha, float _gamma) : alpha(_alpha), gamma(_gamma) {}

  void inplace(poplar::program::Sequence &,
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
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
