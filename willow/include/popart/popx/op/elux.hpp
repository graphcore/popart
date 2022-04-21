// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ELUX_HPP
#define GUARD_NEURALNET_ELUX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

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

#endif
