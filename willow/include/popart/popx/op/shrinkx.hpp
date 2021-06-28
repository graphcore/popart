// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SHRINKX_HPP
#define GUARD_NEURALNET_SHRINKX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class ShrinkComputex : public EwuComputex {
public:
  ShrinkComputex(float lambd, float bias) : lambd_(lambd), bias_(bias) {}

  snap::Tensor outplace(poplar::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get(float lambd, float bias) {
    return std::unique_ptr<EwuComputex>(new ShrinkComputex(
        static_cast<float>(lambd), static_cast<float>(bias)));
  }

  float lambd() const { return lambd_; }
  float bias() const { return bias_; }

private:
  float lambd_;
  float bias_;
};

class ShrinkOpx : public ElementWiseUnaryOutplaceOpx {
public:
  ShrinkOpx(Op *, Devicex *);
};

class ShrinkInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  ShrinkInplaceOpx(Op *, Devicex *);
};

class ShrinkGradOpx : public PopOpx {
public:
  ShrinkGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
